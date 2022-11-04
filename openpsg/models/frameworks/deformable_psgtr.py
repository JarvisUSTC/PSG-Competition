# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.utils.visualizer import VisImage, Visualizer
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models import DETECTORS, SingleStageDetector

from openpsg.models.relation_heads.approaches import Result
from openpsg.utils.utils import adjust_text_color, draw_text, get_colormap
from mmdet.models.builder import HEADS, build_head

import copy
import time

def triplet2Result(triplets, use_mask, eval_pan_rels=True):
    if isinstance(triplets, Result):
        return triplets
    if use_mask:
        bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
            r_labels, r_dists, pan_masks, rels, pan_labels \
            = triplets
        if isinstance(bboxes, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            bboxes = bboxes.detach().cpu().numpy()
            rel_pairs = rel_pairs.detach().cpu().numpy()
            complete_r_labels = complete_r_labels.detach().cpu().numpy()
            complete_r_dists = complete_r_dists.detach().cpu().numpy()
            r_labels = r_labels.detach().cpu().numpy()
            r_dists = r_dists.detach().cpu().numpy()
        if isinstance(pan_seg, torch.Tensor):
            pan_seg = pan_seg.detach().cpu().numpy()
            
        if isinstance(pan_rel_pairs, torch.Tensor):
            pan_rel_pairs = pan_rel_pairs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            pan_masks = pan_masks.detach().cpu().numpy()
            rels = rels.detach().cpu().numpy()
            pan_labels = pan_labels.detach().cpu().numpy()
        if eval_pan_rels:
            return Result(refine_bboxes=bboxes,
                        labels=pan_labels+1,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=pan_rel_pairs,# elif not pan: rel_pairs,
                        rel_dists=r_dists,
                        rel_labels=r_labels,
                        pan_results=pan_seg,
                        masks=pan_masks,
                        rels=rels)
        else:
            return Result(refine_bboxes=bboxes,
                        labels=labels,
                        formatted_masks=dict(pan_results=pan_seg),
                        rel_pair_idxes=rel_pairs,
                        rel_dists=complete_r_dists,
                        rel_labels=complete_r_labels,
                        pan_results=pan_seg,
                        masks=masks)
    else:
        bboxes, labels, rel_pairs, r_labels, r_dists = triplets
        labels = labels.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        rel_pairs = rel_pairs.detach().cpu().numpy()
        r_labels = r_labels.detach().cpu().numpy()
        r_dists = r_dists.detach().cpu().numpy()
        return Result(
            refine_bboxes=bboxes,
            labels=labels,
            formatted_masks=dict(pan_results=None),
            rel_pair_idxes=rel_pairs,
            rel_dists=r_dists,
            rel_labels=r_labels,
            pan_results=None,
        )


@DETECTORS.register_module()
class DeformablePSGTr(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 neck_freeze=None,
                 bbox_head=None, # For relation
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 DEBUG=False):
        super(DeformablePSGTr, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.CLASSES = self.bbox_head.object_classes
        self.PREDICATES = self.bbox_head.predicate_classes
        self.num_classes = self.bbox_head.num_classes
        self.DEBUG = DEBUG
        if neck:
            if neck_freeze:
                self.neck.eval()
                for param in self.neck.parameters():
                    param.requires_grad = False
        ## Panoptic Head
        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None):
        super(SingleStageDetector, self).forward_train(img, img_metas)

        x = self.extract_feat(img)
        # if self.bbox_head.use_mask:
        #     BS, C, H, W = img.shape
        #     new_gt_masks = []
        #     for b, each in enumerate(gt_masks):
        #         mask = each.pad(img_metas[b]['pad_shape'][:2], pad_val=0)\
        #             .to_tensor(dtype=torch.bool, device=gt_labels[b].device)
        #         new_gt_masks.append(mask)
        #     # Keep same as mask head, could be implemented in bbox_head.forward_train
        #     gt_masks_rel_head = new_gt_masks
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        losses_panoptic_head, entity_query_embedding, enc_memory, mlvl_enc_memory, entity_all_bbox_preds, entity_all_cls_scores = self.panoptic_head.forward_train(x, img_metas, copy.deepcopy(gt_bboxes),
                                            copy.deepcopy(gt_labels), copy.deepcopy(gt_masks),
                                            copy.deepcopy(gt_semantic_seg),
                                            gt_bboxes_ignore)
        # torch.cuda.synchronize()
        # end.record()
        # print('Panoptic Head:', start.elapsed_time(end))
        # start.record()
        losses = dict()
        if self.panoptic_head.topk_for_relation == -1:
            def sum_dict(list_dict):
                temp = list_dict[0]
                for each_d in list_dict[1:]:
                    for key in temp.keys() | each_d.keys():
                        temp[key] = sum([d.get(key, 0) for d in (temp, each_d)])
                for key in temp.keys():
                    temp[key] = temp[key] / len(list_dict) # average
                return temp
            # gt matched
            loss_b = []
            bbox = entity_all_bbox_preds['bbox']
            mask = entity_all_bbox_preds['mask']
            cls = entity_all_cls_scores['cls']
                        
            # & statistic the max len of queries
            query_len_list = [bbox[img_i].shape[2] for img_i in range(len(bbox))]
            max_query_num = max(query_len_list)

            for img_i in range(len(bbox)):
                entity_all_bbox_preds = dict(bbox=bbox[img_i], mask=[[mask[-1][img_i]]])
                entity_all_cls_scores = dict(cls=cls[img_i])
                x_i = [x[i][img_i:img_i+1] for i in range(len(x))]
                loss_i = self.bbox_head.forward_train(x_i, img[img_i:img_i+1], [img_metas[img_i]], entity_query_embedding[img_i], enc_memory[img_i:img_i+1], mlvl_enc_memory[:,img_i:img_i+1],
                                              entity_all_bbox_preds, entity_all_cls_scores, [gt_rels[img_i]], [gt_bboxes[img_i]],
                                              [gt_labels[img_i]], [gt_masks[img_i]], gt_semantic_seg[img_i:img_i+1],
                                              gt_bboxes_ignore)
                loss_b.append(loss_i)
            losses.update(sum_dict(loss_b))
        else:
            losses = self.bbox_head.forward_train(x, img, img_metas, entity_query_embedding, enc_memory, mlvl_enc_memory,
                                                entity_all_bbox_preds, entity_all_cls_scores, gt_rels, gt_bboxes,
                                                gt_labels, gt_masks, gt_semantic_seg,
                                                gt_bboxes_ignore)
        # torch.cuda.synchronize()
        # end.record()
        # print('Rel Head:', start.elapsed_time(end))
        losses.update(losses_panoptic_head)
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        feat = self.extract_feat(img)
        mask_cls_results, mask_pred_results, entity_query_embedding, enc_memory, mlvl_enc_memory, entity_all_bbox_preds, entity_all_cls_scores = self.panoptic_head.simple_test(
            feat, img_metas
        )
        if self.DEBUG == True:
            self.bbox_head.img = img
            self.bbox_head.img_metas = img_metas
        if self.panoptic_head.topk_for_relation == -1:
            results_list = []
            max_per_img = self.test_cfg.get('max_per_img', 100)
            bbox_placeholder = np.zeros((2 * max_per_img, 5))
            pan_results_list = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, rescale=rescale)
            for img_id, pan_results in enumerate(pan_results_list):
                # single image

                pan_seg, pan_masks, pan_labels, _, qis = pan_results['pan_results']
                if len(pan_labels) <= 1 or len(qis) <= 1:
                    pan_labels = [0]
                    pan_masks = [np.ones((1, pan_seg.shape[0], pan_seg.shape[1])).astype(bool)]
                    rel_scores = np.full((max_per_img, 57), 1 / 57).astype(np.float32)
                    s_o_indices = np.zeros((max_per_img, 2)).astype(np.int64)
                    results_list.append(Result(
                        masks=pan_masks,
                        labels=np.array(pan_labels) % INSTANCE_OFFSET + 1,
                        rel_pair_idxes=s_o_indices,
                        rel_dists=rel_scores,
                        pan_results=pan_seg, # only for PQ evaluation
                        refine_bboxes=bbox_placeholder, # placeholder
                    ))
                else:
                    bbox_all = entity_all_bbox_preds['bbox']
                    mask_all = entity_all_bbox_preds['mask']
                    cls_all = entity_all_cls_scores['cls']
                    entity_all_bbox_preds = dict(bbox=bbox_all[:,img_id:img_id+1,qis], mask=[[mask[qis,:].detach() for i, mask in enumerate(all_mask_pred)] for all_mask_pred in mask_all])
                    entity_all_cls_scores = dict(cls=cls_all[:,img_id:img_id+1,qis])
                    feat_i = [feat[i][img_id:img_id+1] for i in range(len(feat))]

                    results_list_i = self.bbox_head.simple_test_bboxes(feat_i,
                                        img_metas[img_id:img_id+1],
                                        entity_query_embedding[:,img_id:img_id+1,qis], 
                                        enc_memory[img_id:img_id+1],
                                        mlvl_enc_memory[:,img_id:img_id+1],
                                        entity_all_bbox_preds, 
                                        entity_all_cls_scores,
                                        rescale=rescale)
                    results_list.append(results_list_i[0])
                
        else:
            results_list = self.bbox_head.simple_test_bboxes(feat,
                                                    img_metas,
                                                    entity_query_embedding, 
                                                    enc_memory,
                                                    mlvl_enc_memory,
                                                    entity_all_bbox_preds, 
                                                    entity_all_cls_scores,
                                                    rescale=rescale)
        # if self.bbox_head.use_mask:
        #     results_list_tmp = []
        #     for batch_idx, results in enumerate(results_list):
        #         results_list_tmp.append(list(results))
        #         pan_seg, pan_masks, pan_labels, _, qis = pan_results_list[batch_idx]['pan_results']
        #         # if len(pan_labels) <= 1: # detect 0 or 1 entity
        #         #     pan_labels = [0]
        #         #     pan_masks = [np.ones((1, pan_seg.shape[0], pan_seg.shape[1])).astype(bool)]
        #         #   bboxes, labels, rel_pairs, masks, pan_rel_pairs, pan_seg, complete_r_labels, complete_r_dists, \
        #         #   r_labels, r_dists, pan_masks, rels, pan_labels \
        #         # results_list_tmp[-1][5] = pan_seg
        #     results_list = results_list_tmp
        sg_results = [
            triplet2Result(triplets, self.bbox_head.use_mask)
            for triplets in results_list
        ]
        # print(time.time() - s)
        return sg_results
