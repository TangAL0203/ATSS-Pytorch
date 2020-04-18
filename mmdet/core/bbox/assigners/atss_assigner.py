# -*- coding: utf-8 -*-
# @Time    : 2020/4/17 11:55
# @Author  : shengqin.tang
# @File    : atss_assigner.py

from __future__ import absolute_import, division, print_function

import torch

from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

__all__ = ['ATSSAssigner']

class ATSSAssigner(BaseAssigner):
    """"Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection"
    https://arxiv.org/abs/1912.02424

    Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    Refer

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        k (int): top-k candidate boxes will be choosed.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 k=9,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.k = k
        self.min_pos_iou = min_pos_iou  # not used
        self.gt_max_assign_all = gt_max_assign_all  # not used
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        The assignment is done in following steps, the order matters.

        1. assign every bbox to 0.
        2. for each gt bbox, select the top-k candidate box, based on the euclidean distance
           between the center points of the candidate box and the GT box.
        3. calculate the mean and variance of the candidate box, get iou threshold.
        4. assign positive samples.
        5. update ignore candidate box.

        Args:
            bboxes (List of Tensor): Bounding boxes to be assigned, [(n1, 4), (n2, 4), (n3, 4), ..].
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        stack_bboxes = torch.cat(bboxes)
        if stack_bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        stack_bboxes = stack_bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, stack_bboxes)

        # get ignore bbox bool index tensor
        ignore_bool_index = None
        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    stack_bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, stack_bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)

            ignore_bool_index = ignore_max_overlaps > self.ignore_iof_thr

        stride_ranges = []
        for ii in range(len(bboxes)):
            if ii==0:
                cur_range = range(len(bboxes[ii]))
            else:
                cur_range = range(max(stride_ranges[ii-1])+1, max(stride_ranges[ii-1])+1+len(bboxes[ii]))
            stride_ranges.append(cur_range)

        assign_result = self.assign_wrt_atss(stack_bboxes, overlaps, stride_ranges,
                                             gt_bboxes, gt_labels, ignore_bool_index)

        return assign_result

    def assign_wrt_atss(self, stack_bboxes, overlaps, stride_ranges,
                        gt_bboxes, gt_labels=None, ignore_bool_index=None):
        """Assign w.r.t. Adaptive Training Sample Selection

        Args:
            stack_bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes, shape(k, n).
            stride_ranges (List): The index range of each stride candidate box in stack_bboxes.
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
            ignore_bool_index (Tensor): Ignored candidate box index.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)  # not used in atss

        # 1. assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             0,
                                             dtype=torch.long)
        # 2. select the top-k candidate box
        center_dst_matrix, center_gt, center_bbox = self.euclidean_dist(gt_bboxes, stack_bboxes)

        for ii in range(num_gts):
            center_dis = center_dst_matrix[ii]
            candidate_indexes = []
            index_start = 0
            for jj, stride_range in enumerate(stride_ranges):
                stride_dis = center_dis[stride_range]
                if self.k < stride_dis.shape[0]:
                    value, index = stride_dis.topk(self.k, largest=False)
                else:
                    value, index = stride_dis.topk(stride_dis.shape[0], largest=False)

                index += index_start
                candidate_indexes.append(index)
                index_start += len(stride_range)

            candidate_indexes = torch.cat(candidate_indexes)
            if len(candidate_indexes) == 0:
                continue

            # 3. calculate the mean and variance, get threshold
            candidate_overlaps = overlaps[ii][candidate_indexes]
            mg, vg = candidate_overlaps.mean(), candidate_overlaps.std()
            tg = mg + vg

            # 4. assign positive samples
            bool_inds = (overlaps[ii][candidate_indexes] >= tg) & \
                        (center_bbox[candidate_indexes][:,0] > gt_bboxes[ii][0]) & \
                        (center_bbox[candidate_indexes][:,0] < gt_bboxes[ii][2]) & \
                        (center_bbox[candidate_indexes][:,1] > gt_bboxes[ii][1]) & \
                        (center_bbox[candidate_indexes][:,1] < gt_bboxes[ii][3])

            pos_inds = candidate_indexes[bool_inds]
            assigned_gt_inds[pos_inds] = ii + 1 # 1-based

        # 5. update ignore candidate box
        assigned_gt_inds[ignore_bool_index] = -1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    @staticmethod
    def euclidean_dist(x, y):
        k, n = x.shape[0], y.shape[0]
        center_gt = torch.stack([(x[:,0]+x[:,2])/2.0,
                                 (x[:,1]+x[:,3])/2.0], dim=1)
        center_bbox = torch.stack([(y[:,0]+y[:,2])/2.0,
                                   (y[:,1]+y[:,3])/2.0], dim=1)

        xx = torch.pow(center_gt, 2).sum(1, keepdim=True).expand(k, n)
        yy = torch.pow(center_bbox, 2).sum(1, keepdim=True).expand(n, k).t()
        dist = xx + yy
        dist.addmm_(1, -2, center_gt, center_bbox.t())
        dist = dist.sqrt()

        return dist, center_gt, center_bbox
