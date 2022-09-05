# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import AssignResult, BaseAssigner, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from collections import defaultdict
from mmcv.runner.base_module import BaseModule
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from torchvision.ops import generalized_box_iou
import copy
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HTriMatcher(BaseAssigner):
    def __init__(self,
                 s_cls_cost=dict(type='ClassificationCost', weight=1.),
                 s_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 s_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 o_cls_cost=dict(type='ClassificationCost', weight=1.),
                 o_reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 o_iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 r_cls_cost=dict(type='ClassificationCost', weight=1.)):
        self.s_cls_cost = build_match_cost(s_cls_cost)
        self.s_reg_cost = build_match_cost(s_reg_cost)
        self.s_iou_cost = build_match_cost(s_iou_cost)
        self.o_cls_cost = build_match_cost(o_cls_cost)
        self.o_reg_cost = build_match_cost(o_reg_cost)
        self.o_iou_cost = build_match_cost(o_iou_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(self,
               sub_bbox_pred,
               obj_bbox_pred,
               sub_cls_score,
               obj_cls_score,
               rel_cls_score,
               gt_sub_bboxes,
               gt_obj_bboxes,
               gt_sub_labels,
               gt_obj_labels,
               gt_rel_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_sub_bboxes.size(0), sub_bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = sub_bbox_pred.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = sub_bbox_pred.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_sub_bboxes.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        s_cls_cost = self.s_cls_cost(sub_cls_score, gt_sub_labels)
        o_cls_cost = self.o_cls_cost(obj_cls_score, gt_obj_labels)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)
        # regression L1 cost
        normalize_gt_sub_bboxes = gt_sub_bboxes / factor
        normalize_gt_obj_bboxes = gt_obj_bboxes / factor
        s_reg_cost = self.s_reg_cost(sub_bbox_pred, normalize_gt_sub_bboxes)
        o_reg_cost = self.o_reg_cost(obj_bbox_pred, normalize_gt_obj_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        sub_bboxes = bbox_cxcywh_to_xyxy(sub_bbox_pred) * factor
        obj_bboxes = bbox_cxcywh_to_xyxy(obj_bbox_pred) * factor
        s_iou_cost = self.s_iou_cost(sub_bboxes, gt_sub_bboxes)
        o_iou_cost = self.o_iou_cost(obj_bboxes, gt_obj_bboxes)
        # weighted sum of above three costs
        beta_1, beta_2 = 1.2, 1
        alpha_s, alpha_o, alpha_r = 1, 1, 1
        cls_cost = (alpha_s * s_cls_cost + alpha_o * o_cls_cost +
                    alpha_r * r_cls_cost) / (alpha_s + alpha_o + alpha_r)
        bbox_cost = (s_reg_cost + o_reg_cost + s_iou_cost + o_iou_cost) / 2
        cost = beta_1 * cls_cost + beta_2 * bbox_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            sub_bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            sub_bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_labels[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_labels[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)


@BBOX_ASSIGNERS.register_module()
class IdMatcher(BaseAssigner):
    def __init__(self,
                 sub_id_cost=dict(type='ClassificationCost', weight=1.),
                 obj_id_cost=dict(type='ClassificationCost', weight=1.),
                 r_cls_cost=dict(type='ClassificationCost', weight=1.)):
        self.sub_id_cost = build_match_cost(sub_id_cost)
        self.obj_id_cost = build_match_cost(obj_id_cost)
        self.r_cls_cost = build_match_cost(r_cls_cost)

    def assign(self,
               sub_match_score,
               obj_match_score,
               rel_cls_score,
               gt_sub_ids,
               gt_obj_ids,
               gt_rel_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """gt_ids are mapped from previous Hungarian matchinmg results.

        ~[0,99]
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_rel_labels.size(0), rel_cls_score.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        sub_id_cost = self.sub_id_cost(sub_match_score, gt_sub_ids)
        obj_id_cost = self.obj_id_cost(obj_match_score, gt_obj_ids)
        r_cls_cost = self.r_cls_cost(rel_cls_score, gt_rel_labels)

        # weighted sum of above three costs
        cost = sub_id_cost + obj_id_cost + r_cls_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            rel_cls_score.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_ids[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_ids[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)

@MATCH_COST.register_module()
class IndexCost:
    """
    Args:

    sub_match_scores_flat_to_cost
    obj_match_scores_flat_to_cost
    """
    def __init__(self, weight = 1.) -> None:
        self.weight = weight
    
    def __call__(self, sub_match_scores_flat_to_cost, obj_match_scores_flat_to_cost):
        
        ent_pair_match_score = (1 - (sub_match_scores_flat_to_cost + obj_match_scores_flat_to_cost) / 2)

        return ent_pair_match_score * self.weight

@BBOX_ASSIGNERS.register_module()
class RelHungarianMatcher(BaseAssigner):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            cost_rel_class = dict(type='ClassificationCost', weight=1.5),
            cost_rel_vec = dict(type='BBoxL1Cost', weight=1.0),
            cost_class = dict(type='ClassificationCost', weight=1.5),
            cost_bbox = dict(type='BBoxL1Cost', weight=0.8),
            cost_giou = dict(type='IoUCost', iou_mode='giou', weight=1.0),
            cost_indexing = dict(type='IndexCost', weight=0.2),
            cost_foreground_ent = 0.3, #TODO
            num_entities_pairing_train = 25,
            num_entities_pairing = 3,
            num_matching_per_gt = 1,
    ):
        self.cost_rel_class = build_match_cost(cost_rel_class)
        self.cost_rel_vec = build_match_cost(cost_rel_vec)
        self.cost_class = build_match_cost(cost_class)
        self.cost_bbox = build_match_cost(cost_bbox)
        self.cost_giou = build_match_cost(cost_giou)
        self.cost_indexing = build_match_cost(cost_indexing)
        self.cost_foreground_ent = cost_foreground_ent
        self.det_match_res = None

        self.num_entities_pairing_train = num_entities_pairing_train
        self.num_entities_pairing = num_entities_pairing
        self.num_matching_per_gt = num_matching_per_gt


    def assign(self,
               outputs,
               targets,
               gt_sub_ids,
               gt_obj_ids,
               ent_match=True,
               ent_indexing=True,
               use_rel_vec_match_only=False,
               return_init_idx=False,
               det_match_res=None,
               training=True
               ):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                            classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                            box coordinates
                'pred_rel_logits': [batch_size, num_rel_queries, num_classes]
                "pred_rel_vec": [batch_size, num_rel_queries, 4]
            targets: This is a list of targets (len(targets) = batch_size), where each target
                            is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                            of ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        num_gts, num_bboxes = targets[0]['rel_labels'].size(0), outputs['pred_rel_logits'][0].size(0)
        rel_cls_score = outputs['pred_rel_logits'][0]
        # 1. assign -1 by default
        assigned_gt_inds = rel_cls_score.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        assigned_s_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_o_labels = rel_cls_score.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_s_labels), AssignResult(
                                    num_gts,
                                    assigned_gt_inds,
                                    None,
                                    labels=assigned_o_labels)
                                    
        self.det_match_res = det_match_res
        self.training = training
        self.ent_match = ent_match
        self.ent_indexing = ent_indexing
        if ent_match:
            if ent_indexing:
                if use_rel_vec_match_only:
                    match_cost, detailed_cost_dict = self.inter_vec_cost_calculation(
                        outputs, targets
                    )
                else:
                    outputs["sub_ent_indexing_rule"] = outputs["sub_entities_indexing"]
                    outputs["obj_ent_indexing_rule"] = outputs["obj_entities_indexing"]
                    (match_cost, detailed_cost_dict) = self.indexing_entities_cost_calculation(outputs, targets)
            else:
                match_cost, detailed_cost_dict = self.inter_vec_entities_cost_calculation(
                    outputs, targets
                )
        else:
            match_cost, detailed_cost_dict = self.inter_vec_cost_calculation(
                outputs, targets
            )

        match_cost_detach = [cost.detach() for cost in match_cost]
        indices = self.top_score_match(match_cost_detach, return_init_idx)

        for k in detailed_cost_dict.keys():
            if "cost" in k:
                detailed_cost_dict[k] = [c[i] for i, c in enumerate(detailed_cost_dict[k])]

        match_idx = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

        matched_row_inds = match_idx[0][0].to(
            rel_cls_score.device)
        matched_col_inds = match_idx[0][1].to(
            rel_cls_score.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_s_labels[matched_row_inds] = gt_sub_ids[matched_col_inds]
        assigned_o_labels[matched_row_inds] = gt_obj_ids[matched_col_inds]
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_s_labels), AssignResult(
                                num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_o_labels)
        # return match_idx, match_cost_each_img, detailed_cost_dict

    def inter_vec_cost_calculation(
        self, outputs, targets,
    ):
        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]

        pred_rel_score = outputs["pred_rel_logits"].flatten(0,1)
        pred_rel_vec = outputs["pred_rel_vec"].flatten(0,1)

        # Also concat the target labels and boxes
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])
        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])

        if type(self.cost_rel_class).__name__ == "FocalLossCost":
            # Compute the classification cost.
            pred_rel_prob = torch.sigmoid(
                outputs["pred_rel_logits"].flatten(0, 1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = -pred_rel_prob[:, tgt_rel_labels - 1] * self.cost_rel_class.weight
        else:
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = -pred_rel_prob[:, tgt_rel_labels] * self.cost_rel_class.weight

        cost_rel_vec = self.cost_rel_vec(pred_rel_vec, tgt_rel_vec)

        C = cost_rel_vec + cost_class

        C = C.view(bs, num_rel_queries, -1).cpu()  # bs, num_queries, all_label_in_batch

        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        detailed_cost_dict = {
            "cost_rel_vec": cost_rel_vec.view(bs, num_rel_queries, -1)
                .cpu()
                .split(sizes, -1),
            "cost_class": cost_class.view(bs, num_rel_queries, -1)
                .cpu()
                .split(sizes, -1),
        }

        return match_cost, detailed_cost_dict

    def indexing_entities_cost_calculation(
        self, outputs, targets,
    ):

        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]
        _, num_ent_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch

        pred_rel_vec = outputs["pred_rel_vec"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        # batch_size, num_rel_queries, num_ent_queries
        # pred_sub_index_scr = outputs["sub_entities_indexing"]
        # pred_obj_index_scr = outputs["obj_entities_indexing"]

        # batch_size, num_rel_queries, num_ent_queries
        sub_idxing_rule = outputs["sub_ent_indexing_rule"]
        obj_idxing_rule = outputs["obj_ent_indexing_rule"]

        if self.training:
            num_ent_pairs = self.num_entities_pairing_train
        else:
            num_ent_pairs = self.num_entities_pairing

        # batch_size, num_rel_queries, num_ent_pairs
        num_ent_pairs = num_ent_pairs if sub_idxing_rule.shape[-1] > num_ent_pairs else sub_idxing_rule.shape[-1]
        self.num_ent = sub_idxing_rule.shape[-1]
        # todo only accumulate the foreground entities that match with the GTs
        rel_match_sub_scores, rel_match_sub_ids = torch.topk(sub_idxing_rule, num_ent_pairs, dim=-1)
        rel_match_obj_scores, rel_match_obj_ids = torch.topk(obj_idxing_rule, num_ent_pairs, dim=-1)

        pred_ent_probs = outputs["pred_logits"].softmax(-1)
        pred_ent_boxes = outputs["pred_boxes"]

        def minmax_norm(data):
            return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 0.02)

        # batch_size, num_rel_queries, num_pairs ->  batch_size, num_rel_queries * num_pairs
        # sub_match_scores_flat = minmax_norm(rel_match_sub_scores).flatten(1, 2)
        # obj_match_scores_flat = minmax_norm(rel_match_obj_scores).flatten(1, 2)

        sub_match_scores_flat = rel_match_sub_scores.flatten(1, 2)
        obj_match_scores_flat = rel_match_obj_scores.flatten(1, 2)
        sub_idx_flat = rel_match_sub_ids.flatten(1, 2)
        obj_idx_flat = rel_match_obj_ids.flatten(1, 2)

        # batch_size, num_rel_queries * num_pairs -> # batch_size * num_rel_queries,  num_pairs
        pred_rel_sub_prob = torch.stack(
            [pred_ent_probs[i, sub_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1).contiguous()
        pred_rel_obj_prob = torch.stack(
            [pred_ent_probs[i, obj_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1).contiguous()

        pred_rel_sub_bbox = torch.stack(
            [pred_ent_boxes[i, sub_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1)
        pred_rel_obj_bbox = torch.stack(
            [pred_ent_boxes[i, obj_idx_flat[i]] for i in range(bs)]
        ).flatten(0, 1)

        # prepare targets
        # Also concat the target labels and boxes
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])
        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])

        # Also concat the target labels and boxes
        tgt_ent_labels = torch.cat([v["labels"] for v in targets]).contiguous()

        if targets[0].get('labels_non_masked') is not None:
            tgt_ent_labels = torch.cat([v["labels_non_masked"] for v in targets]).contiguous()

        tgt_ent_bbox = torch.cat([v["boxes"] for v in targets]).contiguous()

        num_total_gt_rel = len(tgt_rel_labels)

        # batch concate the pair index tensor with the start index padding
        tgt_rel_pair_idx = []
        start_idx = 0
        for v in targets:
            tgt_rel_pair_idx.append(v["gt_rel_pair_tensor"] + start_idx)
            start_idx += len(v["boxes"])
        tgt_rel_pair_idx = torch.cat(tgt_rel_pair_idx)

        # Compute cost of relationship vector
        # project the prediction probability vector to the GT probability vector

        if type(self.cost_rel_class).__name__ == "FocalLossCost":
            # Compute the classification cost.

            pred_rel_prob = torch.sigmoid(
                outputs["pred_rel_logits"].flatten(0, 1)
            )  # [batch_size * num_queries, num_classes]
            # alpha = 0.25
            # gamma = 2.0
            # neg_cost_class = (
            #         (1 - alpha) * (pred_rel_prob ** gamma) * (-(1 - pred_rel_prob + 1e-8).log())
            # )
            # pos_cost_class = (
            #         alpha * ((1 - pred_rel_prob) ** gamma) * (-(pred_rel_prob + 1e-8).log())
            # )
            # cost_class = pos_cost_class[:, tgt_rel_labels - 1] - neg_cost_class[:, tgt_rel_labels - 1]

            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels - 1]) * self.cost_rel_class.weight
        else:

            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels]) * self.cost_rel_class.weight

        cost_rel_vec = self.cost_rel_vec(pred_rel_vec, tgt_rel_vec)

        cost_sub_class = pred_rel_sub_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 0]]]
        cost_obj_class = pred_rel_obj_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 1]]]
        # and operation, the both giou should large, one small giou will suppress the pair matching score

        cost_ent_pair_class = 32 ** (-1 * (cost_sub_class + cost_obj_class)) * self.cost_class.weight

        cost_sub_box_l1 = self.cost_bbox(pred_rel_sub_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:,0]])

        cost_obj_box_l1 = self.cost_bbox(pred_rel_obj_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:,0]])

        cost_ent_pair_box_l1 = (cost_sub_box_l1 + cost_obj_box_l1) / 2

        # Compute the giou cost betwen boxes
        cost_sub_giou = torch.clip(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(pred_rel_sub_bbox),
                bbox_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 0]]),
            ),
            0,
        )
        cost_obj_giou = torch.clip(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(pred_rel_obj_bbox),
                bbox_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 1]]),
            ),
            0,
        )

        # batch_size * pair_num x gt_num

        # and operation, the both giou should large, one small giou will suppress the pair matching score
        # cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou))
        cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou)) * self.cost_giou.weight

        cost_foreground_ent = None
        if self.det_match_res is not None and self.cost_foreground_ent > 0:
            gt_ent_idxs = [v["gt_rel_pair_tensor"] for v in targets]

            def build_foreground_cost(ent_role_idx_flat, role_id):
                all_cost_foreground_ent = []
                gt_num = torch.cat(gt_ent_idxs).shape[0]
                for batch_idxs, ent_match_idx in enumerate(self.det_match_res):
                    ent_role_idx = ent_role_idx_flat[batch_idxs]
                    selected_gt_ent_idx = gt_ent_idxs[batch_idxs][:, role_id]
                    match_dict = {}
                    for pred_i, gt_i in zip(ent_match_idx[0], ent_match_idx[1]):
                        match_dict[gt_i.item()] = pred_i.item()

                    for gt_id in selected_gt_ent_idx:
                        cost_foreground_ent = torch.zeros_like(ent_role_idx).float()
                        matched_pred_i = match_dict[gt_id.item()]
                        cost_foreground_ent[torch.nonzero(ent_role_idx == matched_pred_i)] = -1
                        all_cost_foreground_ent.append(cost_foreground_ent)

                    for _ in range(gt_num - len(selected_gt_ent_idx)):
                        cost_foreground_ent = torch.zeros_like(ent_role_idx).float()
                        all_cost_foreground_ent.append(cost_foreground_ent)
                bz = len(self.det_match_res)
                cost_foreground_ent = torch.stack(all_cost_foreground_ent).reshape(bz, gt_num, -1)
                cost_foreground_ent = cost_foreground_ent.permute(0, 2, 1).reshape(-1, gt_num)
                return cost_foreground_ent

            cost_foreground_ent = build_foreground_cost(sub_idx_flat, 0) + build_foreground_cost(sub_idx_flat, 0)
            cost_foreground_ent[cost_foreground_ent > -1.5] = 0  # both role entities should matching with the GT
            cost_foreground_ent /= 2
        
        # batch_size * num_rel_queries, num_total_gt_rel 
        # -> batch_size * num_rel_queries * num_ent_pairs, num_total_gt_rel
        cost_rel_vec = (
            cost_rel_vec.unsqueeze(1)
                .repeat(1, num_ent_pairs, 1)
                .reshape(-1, num_total_gt_rel)
        )
        cost_class = (
            cost_class.unsqueeze(1)
                .repeat(1, num_ent_pairs, 1)
                .reshape(-1, num_total_gt_rel)
        )

        # scatter the ent_rel matching score for each gt
        # this value respect the quality of entity-rel matchin quality the triplets
        sub_match_scores_flat_to_cost = (
            sub_match_scores_flat.reshape(-1)
                .unsqueeze(1)
                .repeat(1, num_total_gt_rel)
        )

        obj_match_scores_flat_to_cost = (
            obj_match_scores_flat.reshape(-1)
                .unsqueeze(1)
                .repeat(1, num_total_gt_rel)
        )

        ent_pair_match_score = self.cost_indexing(sub_match_scores_flat_to_cost, obj_match_scores_flat_to_cost)

        # Final cost matrix
        # calculate the distance matrix across all gt in batch with prediction

        detailed_cost_dict = {
            "cost_rel_vec": cost_rel_vec,
            "cost_class": cost_class,
            "cost_ent_cls": cost_ent_pair_class,
            "cost_ent_box_l1": cost_ent_pair_box_l1,
            "cost_ent_box_giou": cost_ent_pair_giou,
            "cost_regrouping": ent_pair_match_score,
        }

        if cost_foreground_ent is not None:
            detailed_cost_dict['cost_foreground_ent'] = cost_foreground_ent

        
        # batch_size * num_rel_queries * num_ent_pairs, num_total_gt_rel
        C = torch.zeros_like(cost_rel_vec).to(cost_rel_vec.device)
        for k, v in detailed_cost_dict.items():
            if torch.isnan(v).any():
                print(k)
            if torch.isinf(v).any():
                print(k)
            C += v
        C = C.view(bs, num_rel_queries * num_ent_pairs, -1).cpu()  # bs, num_queries, num_total_gt_rel
        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        # add the non sum cost detail for further analysis
        detailed_cost_dict["cost_sub_giou"] = 1 - cost_sub_giou
        detailed_cost_dict["cost_obj_giou"] = 1 - cost_obj_giou

        detailed_cost_dict["cost_sub_box_l1"] = cost_sub_box_l1
        detailed_cost_dict["cost_obj_box_l1"] = cost_obj_box_l1

        detailed_cost_dict["cost_sub_class"] = 1 - cost_sub_class
        detailed_cost_dict["cost_obj_class"] = 1 - cost_obj_class

        # split in to batch-wise
        for k in detailed_cost_dict.keys():
            detailed_cost_dict[k] = (
                detailed_cost_dict[k]
                    .view(bs, num_rel_queries * num_ent_pairs, -1)
                    .cpu()
                    .split(sizes, -1)
            )

        detailed_cost_dict.update({
            'sub_idx': rel_match_sub_ids,
            "obj_idx": rel_match_obj_ids
        })

        return match_cost, detailed_cost_dict

    def get_ent_pred_prob(self, pred_rel_ent_logits):
        pred_rel_obj_prob = (
            pred_rel_ent_logits.flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        return pred_rel_obj_prob

    def inter_vec_entities_cost_calculation(
        self, outputs, targets,
    ):

        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        pred_rel_prob = (
            outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        pred_rel_vec = outputs["pred_rel_vec"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        pred_rel_obj_prob = self.get_ent_pred_prob(outputs["pred_rel_obj_logits"])
        # [batch_size * num_queries, num_classes]

        pred_rel_obj_bbox = outputs["pred_rel_obj_box"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        pred_rel_sub_prob = self.get_ent_pred_prob(outputs["pred_rel_sub_logits"])
        # [batch_size * num_queries, num_classes]

        pred_rel_sub_bbox = outputs["pred_rel_sub_box"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]

        tgt_rel_vec = torch.cat([v["rel_vector"] for v in targets])
        tgt_rel_labels = torch.cat([v["rel_label_no_mask"] for v in targets])

        # Also concat the target labels and boxes
        tgt_ent_labels = torch.cat([v["labels"] for v in targets]).contiguous()
        if targets[0].get('labels_non_masked') is not None:
            tgt_ent_labels = torch.cat([v["labels_non_masked"] for v in targets]).contiguous()
        tgt_ent_bbox = torch.cat([v["boxes"] for v in targets])

        # batch concate the pair index tensor with the start index padding
        tgt_rel_pair_idx = []
        start_idx = 0
        for v in targets:
            tgt_rel_pair_idx.append(v["gt_rel_pair_tensor"] + start_idx)
            start_idx += len(v["boxes"])
        tgt_rel_pair_idx = torch.cat(tgt_rel_pair_idx)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        if type(self.cost_rel_class).__name__ == "FocalLossCost":
            # Compute the classification cost.
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels - 1]) * self.cost_rel_class.weight
        else:
            pred_rel_prob = (
                outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            cost_class = 32 ** (-pred_rel_prob[:, tgt_rel_labels]) * self.cost_rel_class.weight

        # Compute the L1 cost between relationship vector
        cost_rel_vec = self.cost_rel_vec(pred_rel_vec, tgt_rel_vec)

        # Compute cost of relationship vector
        cost_sub_class = pred_rel_sub_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 0]]]
        cost_obj_class = pred_rel_obj_prob[:, tgt_ent_labels[tgt_rel_pair_idx[:, 1]]]

        cost_ent_pair_class = 32 ** (-1 * (cost_sub_class + cost_obj_class)) * self.cost_class.weight

        cost_sub_box_l1 = self.cost_bbox(pred_rel_sub_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 0]])
        cost_obj_box_l1 = self.cost_bbox(pred_rel_obj_bbox, tgt_ent_bbox[tgt_rel_pair_idx[:, 1]])

        cost_ent_pair_box_l1 = (cost_sub_box_l1 + cost_obj_box_l1) / 2

        # Compute the giou cost betwen boxes
        cost_sub_giou = torch.clip(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(pred_rel_sub_bbox),
                bbox_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 0]]),
            ),
            0,
        )
        cost_obj_giou = torch.clip(
            generalized_box_iou(
                bbox_cxcywh_to_xyxy(pred_rel_obj_bbox),
                bbox_cxcywh_to_xyxy(tgt_ent_bbox[tgt_rel_pair_idx[:, 1]]),
            ),
            0,
        )

        # batch_size * pair_num x gt_num

        # and operation, the both giou should large, one small giou will suppress the pair matching score
        # cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou))
        cost_ent_pair_giou = 32 ** (-1 * (cost_sub_giou + cost_obj_giou)) * self.cost_giou.weight

        detailed_cost_dict = {
            "cost_rel_vec": cost_rel_vec,
            "cost_class": cost_class,
            "cost_ent_cls": cost_ent_pair_class,
            "cost_ent_box_l1": cost_ent_pair_box_l1,
            "cost_ent_box_giou": cost_ent_pair_giou,
        }

        C = torch.zeros_like(cost_rel_vec).to(cost_rel_vec.device)
        for v in detailed_cost_dict.values():
            C += v
        C = C.view(bs, num_rel_queries, -1).cpu()  # bs, num_queries, all_label_in_batch

        # split the distance according to the label size of each images
        sizes = [len(v["rel_labels"]) for v in targets]
        match_cost = C.split(sizes, -1)

        # add the non sum cost detail for further analysis
        detailed_cost_dict["cost_sub_giou"] = 1 - cost_sub_giou
        detailed_cost_dict["cost_obj_giou"] = 1 - cost_obj_giou

        detailed_cost_dict["cost_sub_box_l1"] = cost_sub_box_l1 # With weight
        detailed_cost_dict["cost_obj_box_l1"] = cost_obj_box_l1

        detailed_cost_dict["cost_sub_class"] = 1 - cost_sub_class
        detailed_cost_dict["cost_obj_class"] = 1 - cost_obj_class

        for k in detailed_cost_dict.keys():
            detailed_cost_dict[k] = (
                detailed_cost_dict[k]
                    .view(bs, num_rel_queries, -1)
                    .cpu()
                    .split(sizes, -1)
            )

        return match_cost, detailed_cost_dict

    def top_score_match(self, match_cost, return_init_idx=False):
        indices_all = []
        for cost in [c[i] for i, c in enumerate(match_cost)]:
            cost_inplace = copy.deepcopy(cost)
            topk = self.num_matching_per_gt
            indice_multi = []
            for _ in range(topk):
                # selective matching:
                # We observe the the macthing is only happend in the 
                # small set of predictions that have top K cost value,
                # to this end, we optimize the matching pool by: instead 
                # matching with all possible prediction, we use the
                # top K times of GT num predictions for matching
                min_cost = cost_inplace.min(-1)[0]
                selected_range = 4096
                selected_range = (
                    selected_range
                    if selected_range < cost_inplace.shape[0]
                    else cost_inplace.shape[0]
                )
                _, idx = min_cost.topk(selected_range, largest=False)
                indices = linear_sum_assignment(cost_inplace[idx, :])
                indices = (idx[indices[0]], indices[1])
                # if one pred match with the gt, we exclude it
                cost_inplace[indices[0], :] = 1e10
                indice_multi.append(indices)

            if self.training:
                # filtering that the prediction from one query is matched with the multiple GT
                init_pred_idx = np.concatenate([each[0] for each in indice_multi])
                if self.ent_match and self.ent_indexing:
                    num_ent_pairs = self.num_entities_pairing_train if self.num_ent >= self.num_entities_pairing_train else self.num_ent
                    pred_idx = init_pred_idx // num_ent_pairs
                    # transform into the indices along the query num
                else:
                    pred_idx = init_pred_idx

                # check the matching relationship between the query id and GT id
                gt_idx = np.concatenate([each[1] for each in indice_multi])
                dup_match_dict = dict()
                for init_idx, (p_i, g_i) in enumerate(zip(pred_idx, gt_idx)):
                    if dup_match_dict.get(p_i) is not None:
                        if cost[p_i][dup_match_dict[p_i][1]] > cost[p_i][g_i]:
                            # print(cost[p_i][dup_match_dict[p_i]], cost[p_i][g_i])
                            # print(p_i, dup_match_dict[p_i], g_i)
                            dup_match_dict[p_i] = (init_idx, g_i)
                    else:
                        dup_match_dict[p_i] = (init_idx, g_i)

                init_pred_idx_sort = []
                pred_idx = []
                gt_idx = []
                for p_i, (init_idx, g_i) in dup_match_dict.items():
                    pred_idx.append(p_i)
                    gt_idx.append(g_i)
                    init_pred_idx_sort.append(init_pred_idx[init_idx])

                if return_init_idx:
                    indices_all.append((np.array(init_pred_idx_sort), np.array(gt_idx)))
                else:
                    indices_all.append((np.array(pred_idx), np.array(gt_idx)))
            else:
                indices_all.append(
                    (
                        np.concatenate([each[0] for each in indice_multi]),
                        np.concatenate([each[1] for each in indice_multi]),
                    )
                )
            # match_idx = torch.topk(-1 * cost.transpose(1, 0), topk, dim=-1)
            # pred_idx = match_idx[1].reshape(-1)
            # gt_idx = torch.arange(cost.shape[1]).unsqueeze(1).repeat(1, topk).reshape(-1)
            # indices.append((pred_idx, gt_idx))

        return indices_all