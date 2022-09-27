# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from openpsg.models.utils import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from mmdet.core import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from openpsg.models.utils import preprocess_panoptic_gt
from mmdet.models.utils.transformer import inverse_sigmoid

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.detr_head import DETRHead


@HEADS.register_module()
class DeformableDETRMaskHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 num_things_classes=80,
                 num_stuff_classes=0,
                 loss_mask=None,
                 loss_dice=None,
                 mixed_selection=False,
                 no_attn_mask=False,
                 mask_assigner=None,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.mixed_selection = mixed_selection
        self.no_attn_mask = no_attn_mask
        
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        super(DeformableDETRMaskHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        if not self.train_cfg:
            self.train_cfg = dict()
        
        self.num_points = self.train_cfg.get('num_points', 12544)
        self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
        self.importance_sample_ratio = self.train_cfg.get(
            'importance_sample_ratio', 0.75)
        self.mask_assigner = build_assigner(mask_assigner)
            
        

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.mask_head = Linear(self.embed_dims, self.embed_dims)
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:

            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        elif self.mixed_selection:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        
        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, enc_memory, enc_queries, mlvl_enc_memory = self.transformer(
                    mlvl_feats[1:],
                    [torch.zeros_like(mask) if self.no_attn_mask else mask for mask in mlvl_masks[1:]],
                    query_embeds,
                    mlvl_positional_encodings[1:],
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )
        
        # Generate img_feat for mask
        batch_4x_h, batch_4x_w = mlvl_feats[0].shape[-2:]
        img_feat = mlvl_feats[0] + F.interpolate(enc_memory.permute(0, 3, 1, 2), size=(batch_4x_h, batch_4x_w)) # b, c, h, w
        seg_mask_size = [(torch.logical_not(mlvl_masks[0][i]).sum(0)[0], torch.logical_not(mlvl_masks[0][i]).sum(1)[0]) for i in range(batch_size)]

        hs = hs.permute(0, 2, 1, 3) # num_layers, B, num_queries, C
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            
            query_mask_embed = self.mask_head(hs[lvl]) 
            outputs_mask = torch.einsum('bqc, bchw -> bqhw', query_mask_embed, img_feat)
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(outputs_mask)
            
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_masks = torch.stack(outputs_masks)
        outputs_masks_list = []
        n, b, q, h, w = outputs_masks.shape

        outputs_masks_list = [[outputs_masks[i, b, :, :seg_mask_size[b][0],  :seg_mask_size[b][1]] for b in range(batch_size)] for i in range(n)]

        # 'enc_mask_preds' is None
        if self.as_two_stage:
            return outputs_classes, outputs_coords, \
                enc_outputs_class[:, :], \
                enc_outputs_coord.sigmoid()[:, :], outputs_masks_list, None, hs, enc_memory.permute(0, 3, 1, 2), mlvl_enc_memory
        else:
            return outputs_classes, outputs_coords, \
                None, None, outputs_masks_list

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             all_mask_preds,
             enc_mask_preds,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        enc_gt_bboxes_list = gt_bboxes_list
        enc_gt_masks_list = gt_masks_list
        enc_gt_labels_list = gt_labels_list

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_mask_preds, 
            all_gt_bboxes_list, all_gt_labels_list, all_gt_masks_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(enc_gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, _, _ = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_mask_preds,
                                 enc_gt_bboxes_list, binary_labels_list, enc_gt_masks_list, 
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_mask_i, loss_dice_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1

        return loss_dict
    
    
    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        mask_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] if mask_preds is not None else None for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            mask_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
        )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        
        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        if mask_preds is None:
            return loss_cls, loss_bbox, loss_iou, None, None
    
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        
        loss_dice = mask_preds[0].sum() * 0
        loss_mask = mask_preds[0].sum() * 0
        if mask_weights.sum() == 0:
            # zero match
            return loss_cls, loss_bbox, loss_iou, loss_mask, loss_dice

        for b in range(len(mask_preds)):
            mask_target = mask_targets_list[b]
            mask_pred =  mask_preds[b][mask_weights_list[b] > 0]
            mask_pred = F.interpolate(
                mask_pred.unsqueeze(1),
                mask_target.shape[-2:],
                mode='bilinear',
                align_corners=False).squeeze(1)
            
            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_pred.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                mask_point_targets = point_sample(
                    mask_target.unsqueeze(1).float(), points_coords).squeeze(1)

                # shape (num_queries, h, w) -> (num_queries, num_points)
            mask_point_preds = point_sample(
                mask_pred.unsqueeze(1), points_coords).squeeze(1)

            # dice loss
            loss_dice += (self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks) / len(mask_preds)).squeeze()
            # mask loss
            # shape (num_queries, num_points) -> (num_queries * num_points, )
            mask_point_preds = mask_point_preds.reshape(-1)
            # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
            mask_point_targets = mask_point_targets.reshape(-1)
            loss_mask += (self.loss_mask(
                mask_point_preds,
                mask_point_targets,
                avg_factor=num_total_masks * self.num_points) / len(mask_preds)).squeeze()
        
        return loss_cls, loss_bbox, loss_iou, loss_mask, loss_dice
    
    def simple_test(self, feats, img_metas, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        (all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        all_mask_preds,
        enc_mask_preds,
        entity_query_embedding, 
        enc_memory,
        mlvl_enc_memory) = self(feats, img_metas)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results[0].unsqueeze(0),
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        entity_all_bbox_preds = dict(bbox=all_bbox_preds,
                              mask=all_mask_preds)
        
        entity_all_cls_scores = dict(cls=all_cls_scores,)

        return mask_cls_results, mask_pred_results, entity_query_embedding, enc_memory, mlvl_enc_memory, entity_all_bbox_preds, entity_all_cls_scores
    
    def forward_train(self,
                      feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        (all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        all_mask_preds,
        enc_mask_preds, 
        entity_query_embedding, 
        enc_memory, mlvl_enc_memory) = self(feats, img_metas)
      
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)
        
        # For stuff in Panoptic Segmentation, we generate box label with its mask
        for b in range(len(gt_labels)):
            h, w, _ = img_metas[b]['img_shape']
            if gt_labels[b].shape[0] > gt_bboxes[b].shape[0]:
                gt_bboxes[b] = torch.cat((gt_bboxes[b], *[
                    gt_bboxes[0].new_tensor([
                        gt_masks[b][i].sum(0).nonzero().min(),
                        gt_masks[b][i].sum(1).nonzero().min(),
                        gt_masks[b][i].sum(0).nonzero().max(),
                        gt_masks[b][i].sum(1).nonzero().max(),
                ]).unsqueeze(0) for i in range(gt_bboxes[b].shape[0], gt_labels[b].shape[0])]), dim=0)

        # loss
        losses = self.loss(all_cls_scores, all_bbox_preds,
                           enc_cls_scores, enc_bbox_preds, all_mask_preds, enc_mask_preds, gt_bboxes, gt_labels, gt_masks, 
                           img_metas)
        
        entity_all_bbox_preds = dict(bbox=all_bbox_preds,
                              mask=all_mask_preds)
        
        entity_all_cls_scores = dict(cls=all_cls_scores,)

        return losses, entity_query_embedding, enc_memory, mlvl_enc_memory, entity_all_bbox_preds, entity_all_cls_scores
    
    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks
    
    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        mask_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            mask_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        mask_pred,
        gt_bboxes,
        gt_labels,
        gt_masks,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        
        if mask_pred is not None:
            target_shape = mask_pred.shape[-2:]
            if gt_masks.shape[0] > 0:
                gt_masks_downsampled = F.interpolate(
                    gt_masks.unsqueeze(1).float(), target_shape,
                    mode='nearest').squeeze(1).long()
            else:
                gt_masks_downsampled = gt_masks
                
            num_bboxes = bbox_pred.size(0)
            # assigner and sampler
            assign_result = self.mask_assigner.assign(
                bbox_pred, cls_score, mask_pred, gt_bboxes, gt_labels, gt_masks_downsampled, img_meta, self.num_things_classes, gt_bboxes_ignore
            )
        else:
            num_bboxes = bbox_pred.size(0)
            # assigner and sampler
            assign_result = self.assigner.assign(
                bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta, gt_bboxes_ignore
            )
            
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # For Panoptic Segmentation, box loss is only computed for things categories in the second-stage
        # Tricks: for the first stage, the gt_labels are set to 0 so won't be filtered here
        pos_inds_bbox = pos_inds[gt_labels[sampling_result.pos_assigned_gt_inds] < self.num_things_classes]
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds_bbox] = 1.0
        
        # mask target
        if mask_pred is not None:
            mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
            mask_weights = mask_pred.new_zeros((bbox_pred.shape[0], ))
            mask_weights[pos_inds] = 1.0
        else:
            mask_targets = None
            mask_weights = None
        
        img_h, img_w, _ = img_meta["img_shape"]
        
        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, mask_targets, mask_weights, pos_inds, neg_inds)
