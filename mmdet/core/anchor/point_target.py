import torch
import time
import os
from PIL import Image
from PIL import ImageDraw

from ..bbox import PseudoSampler, assign_and_sample, build_assigner
from ..utils import multi_apply

def point_target(proposals_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 img_metas,
                 cfg,
                 gt_bboxes_ignore_list=None,
                 gt_labels_list=None,
                 label_channels=1,
                 sampling=True,
                 unmap_outputs=True,
                 debug_ignore=False):
    """Compute corresponding GT box and classification targets for proposals.

    Args:
        points_list (list[list]): Multi level points of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        cfg (dict): train sample configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(proposals_list) == len(valid_flag_list) == num_imgs

    # points number of multi levels
    num_level_proposals = [points.size(0) for points in proposals_list[0]]

    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_gt, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
         point_target_single,
         proposals_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs,
         debug_ignore=debug_ignore)
    # no valid points
    if any([labels is None for labels in all_labels]):
        return None
    # sampled points of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_proposals)
    label_weights_list = images_to_levels(all_label_weights,
                                          num_level_proposals)
    bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
    proposals_list = images_to_levels(all_proposals, num_level_proposals)
    proposal_weights_list = images_to_levels(all_proposal_weights,
                                             num_level_proposals)
    return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
            proposal_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_grids):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_grids:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def point_target_single(flat_proposals,
                        valid_flags,
                        gt_bboxes,
                        gt_bboxes_ignore,
                        gt_labels,
                        cfg,
                        label_channels=1,
                        sampling=True,
                        unmap_outputs=True,
                        debug_ignore=False):
    inside_flags = valid_flags
    if not inside_flags.any():
        return (None, ) * 7
    # assign gt and sample proposals
    proposals = flat_proposals[inside_flags.type(torch.bool), :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            proposals, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(proposals, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, proposals,
                                              gt_bboxes)

    if debug_ignore and sampling_result.pos_bboxes.shape[0]:
    # assign_result: gt_inds, labels
    # sampling_result: neg_inds, neg_bboxes, pos_inds, pos_bboxes,
    #                  pos_assigned_gt_inds, num_gts
        assignerType = cfg.assigner["type"]
        pointAssigner = assignerType == "PointAssigner"
        print(
            assignerType,"-",
            "gt:", sampling_result.num_gts,
            "p:", sampling_result.pos_inds.shape[0],
            "n:", sampling_result.neg_inds.shape[0],
            "ign:", (assign_result.gt_inds == -1).sum().item()
        )
        print()

        xMin = min(map(min,(gt_bboxes[...,0],
            sampling_result.pos_bboxes[...,0])))
        yMin = min(map(min, (gt_bboxes[...,1],
            sampling_result.pos_bboxes[...,1])))
        if gt_bboxes_ignore.shape[0]:
            xMin = min(xMin, min(gt_bboxes_ignore[...,0]))
            yMin = min(yMin, min(gt_bboxes_ignore[...,1]))

        sampling_result.pos_bboxes[...,0] -= min(0,xMin)
        sampling_result.pos_bboxes[...,1] -= min(0,yMin)
        if not pointAssigner:
            sampling_result.pos_bboxes[...,2] -= min(0,xMin)
            sampling_result.pos_bboxes[...,3] -= min(0,yMin)
        aa = [gt_bboxes]
        if gt_bboxes_ignore.shape[0]:
            aa.append(gt_bboxes_ignore)
        for a in aa:
            a[...,0] -= min(0,xMin)
            a[...,2] -= min(0,xMin)
            a[...,1] -= min(0,yMin)
            a[...,3] -= min(0,yMin)

        if not pointAssigner:
            xA = sampling_result.pos_bboxes[...,2]
            yA = sampling_result.pos_bboxes[...,3]
        else:
            xA = sampling_result.pos_bboxes[...,0]
            yA = sampling_result.pos_bboxes[...,1]
        xMax = max(map(max,(gt_bboxes[...,2],xA)))
        yMax = max(map(max,(gt_bboxes[...,3],yA)))
        if gt_bboxes_ignore.shape[0]:
            xMax = max(xMax, max(gt_bboxes_ignore[...,2]))
            yMax = max(yMax, max(gt_bboxes_ignore[...,3]))

        im = Image.new("RGB", (xMax+1,yMax+1), "white")
        draw = ImageDraw.Draw(im)

        for i in range(sampling_result.pos_bboxes.shape[0]):
            x1 = int(sampling_result.pos_bboxes[i][0])
            y1 = int(sampling_result.pos_bboxes[i][1])
            if not pointAssigner:
                x2 = int(sampling_result.pos_bboxes[i][2])
                y2 = int(sampling_result.pos_bboxes[i][3])
                draw.rectangle((x1,y1,x2,y2), outline="green")
            else:
                draw.ellipse((x1-5,y1-5,x1+5,y1+5), fill="green")

        if gt_bboxes_ignore.shape[0]:
            for i in range(gt_bboxes_ignore.shape[0]):
                x1 = int(gt_bboxes_ignore[i][0])
                y1 = int(gt_bboxes_ignore[i][1])
                x2 = int(gt_bboxes_ignore[i][2])
                y2 = int(gt_bboxes_ignore[i][3])
                draw.rectangle((x1,y1,x2,y2), outline="red")

        for i in range(gt_bboxes.shape[0]):
            x1 = int(gt_bboxes[i][0])
            y1 = int(gt_bboxes[i][1])
            x2 = int(gt_bboxes[i][2])
            y2 = int(gt_bboxes[i][3])
            draw.rectangle((x1,y1,x2,y2), outline="blue")

        dir = "debug"
        os.makedirs(dir, exist_ok=True)
        im.save(os.path.join(dir, str(int(round(time.time()*1000)))+".png"))

    num_valid_proposals = proposals.shape[0]
    bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        bbox_gt[pos_inds, :] = pos_gt_bboxes
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of proposals
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals,
                                  inside_flags)

    return (labels, label_weights, bbox_gt, pos_proposals, proposals_weights,
            pos_inds, neg_inds)


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret
