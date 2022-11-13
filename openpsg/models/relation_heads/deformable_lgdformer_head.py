# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from openpsg.models.utils import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from .predicate_node_generator import build_predicate_node_generator
from openpsg.models.utils import preprocess_panoptic_gt
#####imports for tools
from packaging import version
import gc
torch.autograd.set_detect_anomaly(True)
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


@HEADS.register_module()
class DeformableLGDFormerHead(AnchorFreeHead):

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_relations,
                 object_classes,
                 predicate_classes,
                 use_s_o_proj=False,
                 num_feature_levels=4,
                 num_things_classes=200,
                 num_obj_query=100,
                 num_rel_query=100,
                 num_reg_fcs=2,
                 use_mask=True,
                 temp=0.1,
                 predicate_node_generator=None,
                 n_heads=8,
                 sync_cls_avg_factor=False,
                 bg_cls_weight=0.02,
                 positional_encoding=dict(type='SinePositionalEncoding',
                                          num_feats=128,
                                          normalize=True),
                sub_loss_cls=dict(type='CrossEntropyLoss',
                                use_sigmoid=False,
                                loss_weight=1.0,
                                class_weight=1.0),
                sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                sub_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                sub_mask_loss=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                obj_loss_cls=dict(type='CrossEntropyLoss',
                                use_sigmoid=False,
                                loss_weight=1.0,
                                class_weight=1.0),
                obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                obj_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                obj_mask_loss=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    reduction='mean',
                    loss_weight=5.0),
                rel_sub_obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 rel_loss_cls=dict(type='CrossEntropyLoss',
                                   use_sigmoid=False,
                                   loss_weight=2.0,
                                   class_weight=1.0),
                 sub_id_loss=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=2.0,
                                  class_weight=1.0),
                 obj_id_loss=dict(type='CrossEntropyLoss',
                                  use_sigmoid=False,
                                  loss_weight=2.0,
                                  class_weight=1.0),
                 loss_cls=dict(type='CrossEntropyLoss',
                               use_sigmoid=False,
                               loss_weight=1.0,
                               class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                 dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                 train_cfg=dict(id_assigner=dict(
                     type='IdMatcher',
                     sub_id_cost=dict(type='ClassificationCost', weight=1.),
                     obj_id_cost=dict(type='ClassificationCost', weight=1.),
                     r_cls_cost=dict(type='ClassificationCost', weight=1.)),
                                bbox_assigner=dict(
                                    type='HungarianAssigner',
                                    cls_cost=dict(type='ClassificationCost',
                                                  weight=1.),
                                    reg_cost=dict(type='BBoxL1Cost',
                                                  weight=5.0),
                                    iou_cost=dict(type='IoUCost',
                                                  iou_mode='giou',
                                                  weight=2.0))),
                 test_cfg=dict(max_per_img=100, logit_adj_tau=0.0),
                 init_cfg=None,
                 **kwargs):

        super(AnchorFreeHead, self).__init__(init_cfg)
        self.sync_cls_avg_factor = sync_cls_avg_factor
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        assert isinstance(bg_cls_weight, float), 'Expected ' \
            'bg_cls_weight to have type float. Found ' \
            f'{type(bg_cls_weight)}.'
        self.bg_cls_weight = bg_cls_weight

        class_weight = loss_cls.get('class_weight', None)
        assert isinstance(class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(class_weight)}.'

        class_weight = torch.ones(num_classes + 1) * class_weight
        # set background class as the last indice
        class_weight[num_classes] = bg_cls_weight
        loss_cls.update({'class_weight': class_weight})

        s_class_weight = sub_loss_cls.get('class_weight', None)
        if s_class_weight is not None:
            assert isinstance(s_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(s_class_weight)}.'

            s_class_weight = torch.ones(num_classes + 1) * s_class_weight
            #NOTE set background class as the last indice
            s_class_weight[-1] = bg_cls_weight
            sub_loss_cls.update({'class_weight': s_class_weight})

        o_class_weight = obj_loss_cls.get('class_weight', None)
        if o_class_weight is not None:
            assert isinstance(o_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(o_class_weight)}.'

            o_class_weight = torch.ones(num_classes + 1) * o_class_weight
            #NOTE set background class as the last indice
            o_class_weight[-1] = bg_cls_weight
            obj_loss_cls.update({'class_weight': o_class_weight})

        r_class_weight = rel_loss_cls.get('class_weight', None)
        if r_class_weight is not None:
            assert isinstance(r_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(r_class_weight)}.'

            r_class_weight = torch.ones(num_relations + 1) * r_class_weight
            #NOTE set background class as the first indice for relations as they are 1-based
            r_class_weight[0] = bg_cls_weight
            rel_loss_cls.update({'class_weight': r_class_weight})
        if 'bg_cls_weight' in rel_loss_cls:
            rel_loss_cls.pop('bg_cls_weight')

        if train_cfg:
            assert 'id_assigner' in train_cfg, 'id_assigner should be provided '\
                'when train_cfg is set.'
            assert 'bbox_assigner' in train_cfg, 'bbox_assigner should be provided '\
                'when train_cfg is set.'
            id_assigner = train_cfg['id_assigner']
            if rel_loss_cls.get('use_sigmoid', False):
                id_assigner['r_cls_use_sigmoid'] = True
            bbox_assigner = train_cfg['bbox_assigner']
            assert loss_cls['loss_weight'] == bbox_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == bbox_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_iou['loss_weight'] == bbox_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            self.id_assigner = build_assigner(id_assigner)
            self.bbox_assigner = build_assigner(bbox_assigner)
            self.bbox_assigner_type = bbox_assigner['type']
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

            self.num_points = train_cfg.get('num_points', 12544)
            self.oversample_ratio = train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = train_cfg.get(
                'importance_sample_ratio', 0.75)
        assert num_obj_query == num_rel_query
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.use_mask = use_mask
        self.temp = temp
        self.num_classes = num_classes
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = self.num_classes - self.num_things_classes if self.num_classes - self.num_things_classes > 0 else 0
        self.num_relations = num_relations
        self.object_classes = object_classes
        self.predicate_classes = predicate_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.focal_loss = build_loss(focal_loss)
        self.dice_loss = build_loss(dice_loss)
        self.rel_loss_cls = build_loss(rel_loss_cls)

        ### id losses
        self.sub_id_loss = build_loss(sub_id_loss)
        self.obj_id_loss = build_loss(obj_id_loss)

        self.obj_loss_cls = build_loss(obj_loss_cls)
        self.obj_loss_bbox = build_loss(obj_loss_bbox)
        self.obj_loss_iou = build_loss(obj_loss_iou)

        self.sub_loss_cls = build_loss(sub_loss_cls)
        self.sub_loss_bbox = build_loss(sub_loss_bbox)
        self.sub_loss_iou = build_loss(sub_loss_iou)

        self.rel_sub_obj_loss_bbox = build_loss(rel_sub_obj_loss_bbox)
        if self.use_mask:
            # self.obj_focal_loss = build_loss(obj_focal_loss)
            self.obj_dice_loss = build_loss(obj_dice_loss)
            # self.sub_focal_loss = build_loss(sub_focal_loss)
            self.sub_dice_loss = build_loss(sub_dice_loss)
            self.obj_mask_loss = build_loss(obj_mask_loss)
            self.sub_mask_loss = build_loss(sub_mask_loss)

        if self.obj_loss_cls.use_sigmoid:
            self.obj_cls_out_channels = num_classes
        else:
            self.obj_cls_out_channels = num_classes + 1

        if self.sub_loss_cls.use_sigmoid:
            self.sub_cls_out_channels = num_classes
        else:
            self.sub_cls_out_channels = num_classes + 1

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if rel_loss_cls['use_sigmoid']:
            self.rel_cls_out_channels = num_relations
        else:
            self.rel_cls_out_channels = num_relations + 1

        self.use_s_o_proj = use_s_o_proj
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        # rel decoder
        self.predicate_node_generator = build_predicate_node_generator(predicate_node_generator)

        self.n_heads = n_heads
        self.embed_dims = self.predicate_node_generator.embed_dims
        self.num_feature_levels = num_feature_levels
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        # self.input_proj = Conv2d(self.in_channels,
        #                          self.embed_dims,
        #                          kernel_size=1)
        # self.obj_query_embed = nn.Embedding(self.num_obj_query,
        #                                     self.embed_dims)
        # self.rel_query_embed = nn.Embedding(self.num_rel_query,
        #                                     self.embed_dims)

        # self.class_embed = Linear(self.embed_dims, self.cls_out_channels)
        # self.box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        if self.predicate_node_generator.encoder is not None:
            self.level_embeds = nn.Parameter(
                torch.Tensor(self.num_feature_levels, self.embed_dims))
        else:
            self.level_embeds = None

        self.sub_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims))

        self.obj_query_update = nn.Sequential(
            Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
            Linear(self.embed_dims, self.embed_dims))

        if self.use_s_o_proj:
            import copy
            self.predicate_node_generator.s_entity_proj = copy.deepcopy(self.sub_query_update)
            self.predicate_node_generator.o_entity_proj = copy.deepcopy(self.obj_query_update)


        # self.sop_query_update = nn.Sequential(
        #     Linear(2 * self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True), Linear(self.embed_dims, self.embed_dims))

        self.rel_query_update = nn.Identity()

        self.obj_cls_embed = Linear(self.embed_dims, self.obj_cls_out_channels)
        self.obj_box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        self.sub_cls_embed = Linear(self.embed_dims, self.sub_cls_out_channels)
        self.sub_box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
        self.rel_cls_embed = Linear(self.embed_dims, self.rel_cls_out_channels)
        prior_prob = 0.01
        import math
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.obj_loss_cls.use_sigmoid:
            self.obj_cls_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.obj_box_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_box_embed.layers[-1].bias.data, 0)
        if self.sub_loss_cls.use_sigmoid:
            self.sub_cls_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.sub_box_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_box_embed.layers[-1].bias.data, 0)
        if self.rel_loss_cls.use_sigmoid:
            self.rel_cls_embed.bias.data = torch.ones(len(self.predicate_classes)) * bias_value

        self.swin = False
        if self.use_mask:
            self.sub_mask_head = Linear(self.embed_dims, self.embed_dims)
            self.obj_mask_head = Linear(self.embed_dims, self.embed_dims)
            # self.sub_bbox_attention = MHAttentionMap(self.embed_dims,
            #                                          self.embed_dims,
            #                                          self.n_heads,
            #                                          dropout=0.0)
            # self.obj_bbox_attention = MHAttentionMap(self.embed_dims,
            #                                          self.embed_dims,
            #                                          self.n_heads,
            #                                          dropout=0.0)
            # if not self.swin:
            #     self.sub_mask_head = MaskHeadSmallConv(
            #         self.embed_dims + self.n_heads, [256, 256, 256],
            #         self.embed_dims)
            #     self.obj_mask_head = MaskHeadSmallConv(
            #         self.embed_dims + self.n_heads, [256, 256, 256],
            #         self.embed_dims)
            # elif self.swin:
            #     self.sub_mask_head = MaskHeadSmallConv(
            #         self.embed_dims + self.n_heads, self.swin, self.embed_dims)
            #     self.obj_mask_head = MaskHeadSmallConv(
            #         self.embed_dims + self.n_heads, self.swin, self.embed_dims)

        # self.bbox_attention = MHAttentionMap(self.embed_dims,
        #                                      self.embed_dims,
        #                                      self.n_heads,
        #                                      dropout=0.0)
        # self.mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
        #                                    [1024, 512, 256], self.embed_dims)

    # def init_weights(self):
    #     """Initialize weights of the transformer head."""
    #     # The initialization for transformer is important
    #     self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        version = local_metadata.get('version', None)
        if (version is None or version < 2):
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.',
                '.query_embedding.': '.query_embed.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]
        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, feats, img_metas, enc_memory, mlvl_enc_memory, entity_query_embedding, entity_all_bbox_preds, entity_all_cls_scores, train_mode=False, query_masks=None):
        """
        input from panoptic head:
            memory: from deformable encoder
            outs_obj_dec: obj querys from deformable decoder
            outputs_coord: obj bboxes (could use point sampling or other algorithm) from deformable decoder prediction
        """

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        last_features = feats[1] # equal to the shape of enc_memory # feats[-1]
        batch_size = last_features.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = last_features.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0

        # last_features = self.input_proj(last_features) # Use neck and donot need project
        # interpolate masks to have the same spatial shape with feats
        masks = F.interpolate(masks.unsqueeze(1),
                              size=last_features.shape[-2:]).to(
                                  torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_obj_dec, memory \
            = entity_query_embedding, mlvl_enc_memory # (layer, b, num_query, c) and (b, h, w, c)

        outputs_coord = entity_all_bbox_preds['bbox']
        outputs_class = entity_all_cls_scores['cls']
        seg_masks = entity_all_bbox_preds['mask'][-1] # compatible with vallina psgformer

        ### new interaction
        mlvl_feats = feats
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
        
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats[1:], mlvl_masks[1:], mlvl_positional_encodings[1:])):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.level_embeds is not None:
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        if self.level_embeds is not None:
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        else:
            lvl_pos_embed_flatten = None
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks[1:]], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        rel_hs, ext_inter_feats, rel_decoder_out_res = self.predicate_node_generator(
            feat_flatten,
            mask_flatten,
            None,
            lvl_pos_embed_flatten,
            None,
            memory,
            outs_obj_dec,
            outputs_coord[-1] if not self.predicate_node_generator.no_coords_prior else None, # because of no gt_bbox for stuff, the coords are not useful for stuff...
            valid_ratios=valid_ratios,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
            ent_hs_masks=query_masks,
        )

        outs_rel_dec = rel_hs.feature
        outs_rel_dec_ent_aware_sub = ext_inter_feats[0].feature
        outs_rel_dec_ent_aware_obj = ext_inter_feats[1].feature
        ### interaction
        if self.use_s_o_proj:
            updated_sub_embed = self.predicate_node_generator.s_entity_proj(outs_obj_dec)
            updated_obj_embed = self.predicate_node_generator.o_entity_proj(outs_obj_dec)
            updated_sub_embed = self.sub_query_update(updated_sub_embed)
            updated_obj_embed = self.obj_query_update(updated_obj_embed)
        else:
            updated_sub_embed = self.sub_query_update(outs_obj_dec)
            updated_obj_embed = self.obj_query_update(outs_obj_dec)
        sub_q_normalized = F.normalize(updated_sub_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)
        obj_q_normalized = F.normalize(updated_obj_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)

        # updated_rel_embed = self.rel_query_update(outs_rel_dec)
        # rel_q_normalized = F.normalize(updated_rel_embed[-1],
        #                                p=2,
        #                                dim=-1,
        #                                eps=1e-12)
        updated_rel_sub_embed = self.rel_query_update(outs_rel_dec_ent_aware_sub)
        rel_q_normalized_sub = F.normalize(updated_rel_sub_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)

        updated_rel_obj_embed = self.rel_query_update(outs_rel_dec_ent_aware_obj)
        rel_q_normalized_obj = F.normalize(updated_rel_obj_embed[-1],
                                       p=2,
                                       dim=-1,
                                       eps=1e-12)

        #### Aux entities prediction
        sub_outputs_class_aux = self.sub_cls_embed(outs_rel_dec_ent_aware_sub)
        sub_outputs_coord_aux = self.sub_box_embed(outs_rel_dec_ent_aware_sub).sigmoid()
        obj_outputs_class_aux = self.obj_cls_embed(outs_rel_dec_ent_aware_obj)
        obj_outputs_coord_aux = self.obj_box_embed(outs_rel_dec_ent_aware_obj).sigmoid()
        rel_sub_obj_outputs_union_box_aux = []
        for layer_idx in range(self.predicate_node_generator.num_decoder_layer):
            rel_sub_obj_outputs_union_box_aux.append(self.predicate_node_generator.rel_reference_points[layer_idx](outs_rel_dec[layer_idx]).sigmoid())
        rel_sub_obj_outputs_union_box_aux = torch.stack(rel_sub_obj_outputs_union_box_aux, 0)
        if self.use_mask:
            ###########for segmentation#################
            # sub_bbox_mask = self.sub_bbox_attention(outs_rel_dec_ent_aware_sub[-1],
            #                                         memory,
            #                                         mask=masks)
            # obj_bbox_mask = self.obj_bbox_attention(outs_rel_dec_ent_aware_obj[-1],
            #                                         memory,
            #                                         mask=masks)
            # sub_seg_masks = self.sub_mask_head(last_features, sub_bbox_mask,
            #                                    [feats[2], feats[1], feats[0]])
            # outputs_sub_seg_masks_aux = sub_seg_masks.view(batch_size,
            #                                            self.num_rel_query,
            #                                            sub_seg_masks.shape[-2],
            #                                            sub_seg_masks.shape[-1])
            # obj_seg_masks = self.obj_mask_head(last_features, obj_bbox_mask,
            #                                    [feats[2], feats[1], feats[0]])
            # outputs_obj_seg_masks_aux = obj_seg_masks.view(batch_size,
            #                                            self.num_rel_query,
            #                                            obj_seg_masks.shape[-2],
            #                                            obj_seg_masks.shape[-1])
            batch_4x_h, batch_4x_w = feats[0].shape[-2:]
            img_feat = feats[0] + F.interpolate(enc_memory, size=(batch_4x_h, batch_4x_w))
            sub_query_mask_embed = self.sub_mask_head(outs_rel_dec_ent_aware_sub)
            outputs_sub_seg_masks_aux = torch.einsum('sbqc, bchw -> sbqhw', sub_query_mask_embed, img_feat)

            obj_query_mask_embed = self.obj_mask_head(outs_rel_dec_ent_aware_obj)
            outputs_obj_seg_masks_aux = torch.einsum('sbqc, bchw -> sbqhw', obj_query_mask_embed, img_feat)

        #### relation-oriented search
        # subject_scores = torch.matmul(
        #     rel_q_normalized, sub_q_normalized.transpose(1, 2)) / self.temp
        # object_scores = torch.matmul(
        #     rel_q_normalized, obj_q_normalized.transpose(1, 2)) / self.temp
        subject_scores = torch.matmul(
            rel_q_normalized_sub, sub_q_normalized.transpose(1, 2)) / self.temp
        object_scores = torch.matmul(
            rel_q_normalized_obj, obj_q_normalized.transpose(1, 2)) / self.temp
        if query_masks is not None:
            scores_mask = ~query_masks[None].repeat(self.num_rel_query, 1, 1).transpose(0,1)
            subject_scores_tmp = subject_scores.softmax(-1)*scores_mask
            object_scores_tmp = object_scores.softmax(-1)*scores_mask
            _, subject_ids = subject_scores_tmp.max(-1)
            _, object_ids = object_scores_tmp.max(-1)
        else:
            _, subject_ids = subject_scores.max(-1)
            _, object_ids = object_scores.max(-1)

        # prediction
        sub_outputs_class = torch.empty_like(outputs_class[:,:,:1].repeat(1,1,self.num_rel_query,1))
        sub_outputs_coord = torch.empty_like(outputs_coord[:,:,:1].repeat(1,1,self.num_rel_query,1))
        obj_outputs_class = torch.empty_like(outputs_class[:,:,:1].repeat(1,1,self.num_rel_query,1))
        obj_outputs_coord = torch.empty_like(outputs_coord[:,:,:1].repeat(1,1,self.num_rel_query,1))
        # outputs_sub_seg_masks = torch.empty_like(seg_masks[:,:self.num_rel_query])
        outputs_sub_seg_masks = [torch.empty_like(seg_mask[:1].repeat(self.num_rel_query,1,1)) for seg_mask in seg_masks]
        outputs_obj_seg_masks = [torch.empty_like(seg_mask[:1].repeat(self.num_rel_query,1,1)) for seg_mask in seg_masks]
        triplet_sub_ids = []
        triplet_obj_ids = []
        for i in range(len(subject_ids)):
            triplet_sub_id = subject_ids[i]
            triplet_obj_id = object_ids[i]
            sub_outputs_class[:, i] = outputs_class[:, i, triplet_sub_id, :]
            sub_outputs_coord[:, i] = outputs_coord[:, i, triplet_sub_id, :]
            obj_outputs_class[:, i] = outputs_class[:, i, triplet_obj_id, :]
            obj_outputs_coord[:, i] = outputs_coord[:, i, triplet_obj_id, :]
            outputs_sub_seg_masks[i] = seg_masks[i][triplet_sub_id, :, :]
            outputs_obj_seg_masks[i] = seg_masks[i][triplet_obj_id, :, :]
            triplet_sub_ids.append(triplet_sub_id)
            triplet_obj_ids.append(triplet_obj_id)

        all_cls_scores = dict(cls=outputs_class,
                              sub=sub_outputs_class,
                              obj=obj_outputs_class)

        rel_outputs_class = self.rel_cls_embed(outs_rel_dec)
        all_cls_scores['rel'] = rel_outputs_class
        all_cls_scores['sub_ids'] = triplet_sub_ids
        all_cls_scores['obj_ids'] = triplet_obj_ids
        all_cls_scores['subject_scores'] = subject_scores
        all_cls_scores['object_scores'] = object_scores
        all_cls_scores['sub_aux'] = sub_outputs_class_aux
        all_cls_scores['obj_aux'] = obj_outputs_class_aux

        all_bbox_preds = dict(bbox=outputs_coord,
                              sub=sub_outputs_coord,
                              obj=obj_outputs_coord,
                              mask=seg_masks,
                              sub_seg=outputs_sub_seg_masks,
                              obj_seg=outputs_obj_seg_masks,
                              sub_aux=sub_outputs_coord_aux,
                              obj_aux=obj_outputs_coord_aux,
                              sub_seg_aux=outputs_sub_seg_masks_aux,
                              obj_seg_aux=outputs_obj_seg_masks_aux,
                              rel_sub_obj_aux=rel_sub_obj_outputs_union_box_aux)

        # For debug
        if hasattr(self, 'img'):
            self.vis_reference_points(self.img.clone(), rel_decoder_out_res['reference_points'], self.img_metas, sub_outputs_coord[-1], obj_outputs_coord[-1], rel_outputs_class[-1])
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_rels_list,
             gt_bboxes_list,
             gt_labels_list,
             gt_masks_list,
             img_metas,
             gt_bboxes_ignore=None,
             query_masks=None):

        # NOTE defaultly only the outputs from the last feature scale is used.
        all_cls_scores = all_cls_scores_list
        all_bbox_preds = all_bbox_preds_list
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        ### object detection and panoptic segmentation
        ########## Do not calculate panoptic loss here
        num_dec_layers = len(all_bbox_preds['sub_aux'])

        all_od_cls_scores = all_cls_scores['cls'][-1]
        all_od_bbox_preds = all_bbox_preds['bbox'][-1]
        all_mask_preds = all_bbox_preds['mask']

        all_od_cls_scores = [all_od_cls_scores for _ in range(num_dec_layers)]
        all_od_bbox_preds = [all_od_bbox_preds for _ in range(num_dec_layers)]
        all_mask_preds = [all_mask_preds for _ in range(num_dec_layers)]

        all_s_bbox_preds = all_bbox_preds['sub'][-1]
        all_o_bbox_preds = all_bbox_preds['obj'][-1]

        all_s_bbox_preds = [all_s_bbox_preds for _ in range(num_dec_layers)]
        all_o_bbox_preds = [all_o_bbox_preds for _ in range(num_dec_layers)]

        all_s_bbox_preds_aux = all_bbox_preds['sub_aux']
        all_o_bbox_preds_aux = all_bbox_preds['obj_aux']
        all_rel_s_o_bbox_preds_aux = all_bbox_preds['rel_sub_obj_aux']
        
        if self.use_mask:
            all_s_mask_preds_aux = all_bbox_preds['sub_seg_aux']
            all_o_mask_preds_aux = all_bbox_preds['obj_seg_aux']
            if len(all_s_mask_preds_aux) == 1:
                all_s_mask_preds_aux = [all_s_mask_preds_aux for _ in range(num_dec_layers)]
                all_o_mask_preds_aux = [all_o_mask_preds_aux for _ in range(num_dec_layers)]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        all_r_cls_scores = all_cls_scores['rel']

        all_r_cls_scores_sub_aux = all_cls_scores['sub_aux']
        all_r_cls_scores_obj_aux = all_cls_scores['obj_aux']

        subject_scores = all_cls_scores['subject_scores']
        object_scores = all_cls_scores['object_scores']
        subject_scores = [subject_scores for _ in range(num_dec_layers)]
        object_scores = [object_scores for _ in range(num_dec_layers)]

        query_masks_list = [query_masks for _ in range(num_dec_layers)]

        r_losses_cls, loss_subject_match, loss_object_match, \
        s_losses_cls, o_losses_cls, s_losses_bbox, o_losses_bbox, \
        s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses, s_mask_losses, o_mask_losses, rel_s_o_losses_bbox= multi_apply(
            self.loss_single, subject_scores, object_scores,
            all_od_cls_scores, all_od_bbox_preds, all_mask_preds,
            all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
            all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_masks_list, img_metas_list, 
            all_r_cls_scores_sub_aux, all_r_cls_scores_obj_aux, 
            all_s_bbox_preds_aux, all_o_bbox_preds_aux, all_s_mask_preds_aux, 
            all_o_mask_preds_aux, all_rel_s_o_bbox_preds_aux, all_gt_bboxes_ignore_list,query_masks_list)

        # r_losses_cls, loss_subject_match, loss_object_match, \
        # s_losses_cls, o_losses_cls, s_losses_bbox, o_losses_bbox, \
        # s_losses_iou, o_losses_iou, s_dice_losses, o_dice_losses = multi_apply(
        #     self.loss_single, subject_scores, object_scores,
        #     all_od_cls_scores, all_od_bbox_preds, all_mask_preds,
        #     all_r_cls_scores, all_s_bbox_preds, all_o_bbox_preds,
        #     all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
        #     all_gt_masks_list, img_metas_list, 
        #     all_r_cls_scores_sub_aux, all_r_cls_scores_obj_aux, 
        #     all_s_bbox_preds_aux, all_o_bbox_preds_aux, all_s_mask_preds_aux, 
        #     all_o_mask_preds_aux,all_gt_bboxes_ignore_list,)


        loss_dict = dict()
        ## loss of relation-oriented matching
        loss_dict['loss_subject_match'] = loss_subject_match[-1]
        loss_dict['loss_object_match'] = loss_object_match[-1]

        ## loss of object detection and segmentation
        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        # loss_dict['loss_bbox'] = losses_bbox[-1]
        # loss_dict['loss_iou'] = losses_iou[-1]

        # loss_dict['focal_losses'] = focal_losses[-1]
        # loss_dict['dice_losses'] = dice_losses[-1]

        # # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
        #                                                losses_bbox[:-1],
        #                                                losses_iou[:-1]):
        #     loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        #     loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
        #     loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
        #     num_dec_layer += 1

        ## loss of scene graph
        # loss from the last decoder layer
        loss_dict['s_loss_cls'] = s_losses_cls[-1]
        loss_dict['o_loss_cls'] = o_losses_cls[-1]
        loss_dict['r_loss_cls'] = r_losses_cls[-1]
        loss_dict['s_loss_bbox'] = s_losses_bbox[-1]
        loss_dict['o_loss_bbox'] = o_losses_bbox[-1]
        loss_dict['s_loss_iou'] = s_losses_iou[-1]
        loss_dict['o_loss_iou'] = o_losses_iou[-1]
        loss_dict['r_s_o_loss_bbox'] = rel_s_o_losses_bbox[-1]
        if self.use_mask:
            # loss_dict['s_focal_losses'] = s_focal_losses[-1]
            # loss_dict['o_focal_losses'] = o_focal_losses[-1]
            loss_dict['s_dice_losses'] = s_dice_losses[-1]
            loss_dict['o_dice_losses'] = o_dice_losses[-1]
            loss_dict['s_mask_losses'] = s_mask_losses[-1]
            loss_dict['o_mask_losses'] = o_mask_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i, rel_s_o_loss_bbox_i, s_mask_loss_i, o_mask_loss_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                          s_losses_bbox[:-1], o_losses_bbox[:-1],
                                          s_losses_iou[:-1], o_losses_iou[:-1], rel_s_o_losses_bbox[:-1], s_mask_losses[:-1], o_mask_losses[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.r_s_o_loss_bbox'] = rel_s_o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_mask_losses'] = s_mask_loss_i
            loss_dict[f'd{num_dec_layer}.o_mask_losses'] = o_mask_loss_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    subject_scores,
                    object_scores,
                    od_cls_scores,
                    od_bbox_preds,
                    mask_preds,
                    r_cls_scores,
                    s_bbox_preds,
                    o_bbox_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    r_cls_scores_sub_aux, 
                    r_cls_scores_obj_aux, 
                    s_bbox_preds_aux, 
                    o_bbox_preds_aux, 
                    s_mask_preds_aux, 
                    o_mask_preds_aux,
                    rel_s_o_bbox_preds_aux,
                    gt_bboxes_ignore_list=None,
                    query_masks_list=None):

        ## before get targets
        num_imgs = r_cls_scores.size(0)
        if query_masks_list is not None:
            num_ent_per_img = (~query_masks_list).sum(-1)
        else:
            num_ent_per_img = [od_cls_scores.shape[1]]*num_imgs
        # obj det&seg
        cls_scores_list = [od_cls_scores[i][:num_ent_per_img[i]] for i in range(num_imgs)]
        bbox_preds_list = [od_bbox_preds[i][:num_ent_per_img[i]] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i][:num_ent_per_img[i]] for i in range(num_imgs)]

        # scene graph
        r_cls_scores_list = [r_cls_scores[i] for i in range(num_imgs)]
        s_bbox_preds_list = [s_bbox_preds[i] for i in range(num_imgs)]
        o_bbox_preds_list = [o_bbox_preds[i] for i in range(num_imgs)]

        r_cls_scores_obj_aux_list = [r_cls_scores_obj_aux[i] for i in range(num_imgs)]
        r_cls_scores_sub_aux_list = [r_cls_scores_sub_aux[i] for i in range(num_imgs)]

        s_bbox_preds_aux_list = [s_bbox_preds_aux[i] for i in range(num_imgs)]
        o_bbox_preds_aux_list = [o_bbox_preds_aux[i] for i in range(num_imgs)]

        s_mask_preds_aux_list = [s_mask_preds_aux[i] for i in range(num_imgs)]
        o_mask_preds_aux_list = [o_mask_preds_aux[i] for i in range(num_imgs)]

        rel_s_o_bbox_preds_aux_list = [rel_s_o_bbox_preds_aux[i] for i in range(num_imgs)]

        # matche scores
        subject_scores_list = [subject_scores[i][:,:num_ent_per_img[i]] for i in range(num_imgs)]
        object_scores_list = [object_scores[i][:,:num_ent_per_img[i]] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            subject_scores_list, object_scores_list, cls_scores_list,
            bbox_preds_list, mask_preds_list, r_cls_scores_list,
            s_bbox_preds_list, o_bbox_preds_list, gt_rels_list, gt_bboxes_list,
            gt_labels_list, gt_masks_list, img_metas, 
            r_cls_scores_sub_aux_list, r_cls_scores_obj_aux_list,
            s_bbox_preds_aux_list, o_bbox_preds_aux_list,
            s_mask_preds_aux_list, o_mask_preds_aux_list, rel_s_o_bbox_preds_aux_list,
            gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, num_total_od_pos, num_total_od_neg,
         mask_preds_list, r_labels_list, r_label_weights_list, num_total_pos,
         num_total_neg, filtered_subject_scores, filtered_object_scores,
         gt_subject_id_list, gt_object_id_list, s_labels_list, o_labels_list, s_label_weights_list,
         o_label_weights_list, s_bbox_targets_list, o_bbox_targets_list, s_bbox_weights_list, o_bbox_weights_list,
         s_mask_targets_list, o_mask_targets_list, s_mask_preds_list, o_mask_preds_list, rel_s_o_bbox_targets_list, rel_s_o_bbox_weights_list) = cls_reg_targets

        # Del unused tensor
        # del labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, mask_targets_list, num_total_od_pos, num_total_od_neg
        # gc.collect() # too slow
        # obj det&seg # Do not need calculate loss here
        # labels = torch.cat(labels_list, 0)
        # label_weights = torch.cat(label_weights_list, 0)

        # bbox_targets = torch.cat(bbox_targets_list, 0)
        # bbox_weights = torch.cat(bbox_weights_list, 0)

        # mask_targets = torch.cat(mask_targets_list, 0).float().flatten(1)

        # mask_preds = torch.cat(mask_preds_list, 0).flatten(1)
        # num_od_matches = mask_preds.shape[0]

        # id loss
        # filtered_subject_scores = torch.cat(
        #     filtered_subject_scores,
        #     0).reshape(len(filtered_subject_scores[0]), -1)
        # filtered_object_scores = torch.cat(filtered_object_scores, 0).reshape(
        #     len(filtered_object_scores[0]), -1)
        # gt_subject_id = torch.cat(gt_subject_id_list, 0)
        # gt_subject_id = F.one_hot(
        #     gt_subject_id, num_classes=filtered_subject_scores.shape[-1])
        # gt_object_id = torch.cat(gt_object_id_list, 0)
        # gt_object_id = F.one_hot(gt_object_id,
        #                          num_classes=filtered_object_scores.shape[-1])
        # loss_subject_match = self.sub_id_loss(filtered_subject_scores,
        #                                       gt_subject_id)
        # loss_object_match = self.obj_id_loss(filtered_object_scores,
        #                                      gt_object_id)
        # NOTE: To support bz > 1
        loss_subject_match = []
        loss_object_match = []
        for b in range(num_imgs):
            filtered_subject_scores_i = filtered_subject_scores[b].reshape(len(filtered_subject_scores[b]), -1)
            filtered_object_scores_i = filtered_object_scores[b].reshape(len(filtered_object_scores[b]), -1)
            gt_subject_id = gt_subject_id_list[b]
            gt_subject_id = F.one_hot(
                gt_subject_id, num_classes=filtered_subject_scores_i.shape[-1])
            gt_object_id = gt_object_id_list[b]
            gt_object_id = F.one_hot(gt_object_id,
                                    num_classes=filtered_object_scores_i.shape[-1])
            loss_subject_match.append(self.sub_id_loss(filtered_subject_scores_i,
                                                gt_subject_id))
            loss_object_match.append(self.obj_id_loss(filtered_object_scores_i,
                                                gt_object_id))
        loss_subject_match = sum(loss_subject_match) / len(loss_subject_match)
        loss_object_match = sum(loss_object_match) / len(loss_object_match)

        # # mask loss
        # focal_loss = self.focal_loss(mask_preds, mask_targets, num_od_matches)
        # dice_loss = self.dice_loss(mask_preds, mask_targets, num_od_matches)

        # # classification loss
        # od_cls_scores = od_cls_scores.reshape(-1, self.cls_out_channels)

        # # construct weighted avg_factor to match with the official DETR repo
        # cls_avg_factor = num_total_od_pos * 1.0 + \
        #     num_total_od_neg * self.bg_cls_weight
        # if self.sync_cls_avg_factor:
        #     cls_avg_factor = reduce_mean(
        #         od_cls_scores.new_tensor([cls_avg_factor]))
        # cls_avg_factor = max(cls_avg_factor, 1)

        # loss_cls = self.loss_cls(od_cls_scores,
        #                          labels,
        #                          label_weights,
        #                          avg_factor=cls_avg_factor)

        # # Compute the average number of gt boxes across all gpus, for
        # # normalization purposes
        # num_total_od_pos = loss_cls.new_tensor([num_total_od_pos])
        # num_total_od_pos = torch.clamp(reduce_mean(num_total_od_pos),
        #                                min=1).item()

        # # construct factors used for rescale bboxes
        # factors = []
        # for img_meta, bbox_pred in zip(img_metas, od_bbox_preds):
        #     img_h, img_w, _ = img_meta['img_shape']
        #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
        #                                    img_h]).unsqueeze(0).repeat(
        #                                        bbox_pred.size(0), 1)
        #     factors.append(factor)
        # factors = torch.cat(factors, 0)

        # # DETR regress the relative position of boxes (cxcywh) in the image,
        # # thus the learning target is normalized by the image size. So here
        # # we need to re-scale them for calculating IoU loss
        # od_bbox_preds = od_bbox_preds.reshape(-1, 4)
        # bboxes = bbox_cxcywh_to_xyxy(od_bbox_preds) * factors
        # bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(bboxes,
        #                          bboxes_gt,
        #                          bbox_weights,
        #                          avg_factor=num_total_od_pos)

        # # regression L1 loss
        # loss_bbox = self.loss_bbox(od_bbox_preds,
        #                            bbox_targets,
        #                            bbox_weights,
        #                            avg_factor=num_total_od_pos)

        # scene graph
        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)

        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        s_bbox_targets = torch.cat(s_bbox_targets_list, 0)
        o_bbox_targets = torch.cat(o_bbox_targets_list, 0)

        s_bbox_weights = torch.cat(s_bbox_weights_list, 0)
        o_bbox_weights = torch.cat(o_bbox_weights_list, 0)

        rel_s_o_bbox_targets = torch.cat(rel_s_o_bbox_targets_list, 0)
        rel_s_o_bbox_weights = torch.cat(rel_s_o_bbox_weights_list, 0)

        if self.use_mask:
            # NOTE: To Support bz > 1
            s_dice_loss = s_mask_targets_list[0].sum() * 0.0
            s_mask_loss = s_mask_targets_list[0].sum() * 0.0
            o_dice_loss = s_mask_targets_list[0].sum() * 0.0
            o_mask_loss = s_mask_targets_list[0].sum() * 0.0
            num_total_masks = reduce_mean(r_cls_scores.new_tensor([num_total_pos]))
            num_total_masks = max(num_total_masks, 1)

            for b in range(num_imgs):
                s_mask_targets = s_mask_targets_list[b].float().flatten(1)
                o_mask_targets = o_mask_targets_list[b].float().flatten(1)
                s_mask_preds = s_mask_preds_list[b].flatten(1)
                o_mask_preds = o_mask_preds_list[b].flatten(1)
                with torch.no_grad():
                    s_points_coords = get_uncertain_point_coords_with_randomness(
                        s_mask_preds.reshape(s_mask_targets_list[b].shape).unsqueeze(1), None, self.num_points,
                        self.oversample_ratio, self.importance_sample_ratio)
                    # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                    s_mask_point_targets = point_sample(
                        s_mask_targets.reshape(s_mask_targets_list[b].shape).unsqueeze(1).float(), s_points_coords).squeeze(1)

                    o_points_coords = get_uncertain_point_coords_with_randomness(
                        o_mask_preds.reshape(o_mask_targets_list[b].shape).unsqueeze(1), None, self.num_points,
                        self.oversample_ratio, self.importance_sample_ratio)
                    # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                    o_mask_point_targets = point_sample(
                        o_mask_targets.reshape(o_mask_targets_list[b].shape).unsqueeze(1).float(), o_points_coords).squeeze(1)

                s_mask_point_preds = point_sample(
                    s_mask_preds.reshape(s_mask_targets_list[b].shape).unsqueeze(1), s_points_coords).squeeze(1)

                o_mask_point_preds = point_sample(
                    o_mask_preds.reshape(o_mask_targets_list[b].shape).unsqueeze(1), o_points_coords).squeeze(1)

                # dice loss
                s_dice_loss += (self.sub_dice_loss(s_mask_point_preds, s_mask_point_targets, avg_factor=num_total_masks) / len(s_mask_preds_list)).squeeze()
                o_dice_loss += (self.obj_dice_loss(o_mask_point_preds, o_mask_point_targets, avg_factor=num_total_masks) / len(o_mask_preds_list)).squeeze()
                # mask loss
                # shape (num_queries, num_points) -> (num_queries * num_points, )
                s_mask_point_preds = s_mask_point_preds.reshape(-1)
                # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
                s_mask_point_targets = s_mask_point_targets.reshape(-1)
                s_mask_loss += (self.sub_mask_loss(
                    s_mask_point_preds,
                    s_mask_point_targets,
                    avg_factor=num_total_masks * self.num_points) / len(s_mask_preds_list)).squeeze()
                o_mask_point_preds = o_mask_point_preds.reshape(-1)
                # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
                o_mask_point_targets = o_mask_point_targets.reshape(-1)
                o_mask_loss += (self.obj_mask_loss(
                    o_mask_point_preds,
                    o_mask_point_targets,
                    avg_factor=num_total_masks * self.num_points) / len(o_mask_preds_list)).squeeze()    


            # s_mask_targets = torch.cat(s_mask_targets_list,
            #                            0).float().flatten(1)
            # o_mask_targets = torch.cat(o_mask_targets_list,
            #                            0).float().flatten(1)

            # s_mask_preds = torch.cat(s_mask_preds_list, 0).flatten(1)
            # o_mask_preds = torch.cat(o_mask_preds_list, 0).flatten(1)
            # num_matches = o_mask_preds.shape[0]
            # num_total_masks = reduce_mean(r_cls_scores.new_tensor([num_total_pos]))
            # num_total_masks = max(num_total_masks, 1)
            # with torch.no_grad():
            #     s_points_coords = get_uncertain_point_coords_with_randomness(
            #         s_mask_preds.reshape(torch.cat(s_mask_targets_list,0).shape).unsqueeze(1), None, self.num_points,
            #         self.oversample_ratio, self.importance_sample_ratio)
            #     # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            #     s_mask_point_targets = point_sample(
            #         s_mask_targets.reshape(torch.cat(s_mask_targets_list,0).shape).unsqueeze(1).float(), s_points_coords).squeeze(1)

            #     o_points_coords = get_uncertain_point_coords_with_randomness(
            #         o_mask_preds.reshape(torch.cat(o_mask_targets_list,0).shape).unsqueeze(1), None, self.num_points,
            #         self.oversample_ratio, self.importance_sample_ratio)
            #     # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            #     o_mask_point_targets = point_sample(
            #         o_mask_targets.reshape(torch.cat(o_mask_targets_list,0).shape).unsqueeze(1).float(), o_points_coords).squeeze(1)

            # s_mask_point_preds = point_sample(
            #     s_mask_preds.reshape(torch.cat(s_mask_targets_list,0).shape).unsqueeze(1), s_points_coords).squeeze(1)

            # o_mask_point_preds = point_sample(
            #     o_mask_preds.reshape(torch.cat(o_mask_targets_list,0).shape).unsqueeze(1), o_points_coords).squeeze(1)

            # # dice loss
            # s_dice_loss = (self.sub_dice_loss(s_mask_point_preds, s_mask_point_targets, avg_factor=num_total_masks) / len(s_mask_preds_list)).squeeze()
            # o_dice_loss = (self.obj_dice_loss(o_mask_point_preds, o_mask_point_targets, avg_factor=num_total_masks) / len(o_mask_preds_list)).squeeze()
            # # mask loss
            # # shape (num_queries, num_points) -> (num_queries * num_points, )
            # s_mask_point_preds = s_mask_point_preds.reshape(-1)
            # # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
            # s_mask_point_targets = s_mask_point_targets.reshape(-1)
            # s_mask_loss = (self.sub_mask_loss(
            #     s_mask_point_preds,
            #     s_mask_point_targets,
            #     avg_factor=num_total_masks * self.num_points) / len(s_mask_preds_list)).squeeze()
            # o_mask_point_preds = o_mask_point_preds.reshape(-1)
            # # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
            # o_mask_point_targets = o_mask_point_targets.reshape(-1)
            # o_mask_loss = (self.obj_mask_loss(
            #     o_mask_point_preds,
            #     o_mask_point_targets,
            #     avg_factor=num_total_masks * self.num_points) / len(o_mask_preds_list)).squeeze()            

            # # mask loss
            # # s_focal_loss = self.sub_focal_loss(s_mask_preds,s_mask_targets,num_matches)
            # s_dice_loss = self.sub_dice_loss(
            #     s_mask_preds, s_mask_targets,
            #     num_matches)

            # # o_focal_loss = self.obj_focal_loss(o_mask_preds,o_mask_targets,num_matches)
            # o_dice_loss = self.obj_dice_loss(
            #     o_mask_preds, o_mask_targets,
            #     num_matches) 
        else:
            s_dice_loss = None
            o_dice_loss = None
            # s_mask_loss = None
            # o_mask_loss = None

        # classification loss
        s_cls_scores = r_cls_scores_sub_aux.reshape(-1, self.sub_cls_out_channels)
        o_cls_scores = r_cls_scores_obj_aux.reshape(-1, self.obj_cls_out_channels)
        r_cls_scores = r_cls_scores.reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                r_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if self.sub_loss_cls.use_sigmoid:
            s_loss_cls = self.sub_loss_cls(s_cls_scores,
                                        s_labels,
                                        s_label_weights,
                                        avg_factor=num_total_pos * 1.0)
        else:
            s_loss_cls = self.sub_loss_cls(s_cls_scores,
                                        s_labels,
                                        s_label_weights,
                                        avg_factor=cls_avg_factor)

        if self.obj_loss_cls.use_sigmoid:
            o_loss_cls = self.obj_loss_cls(o_cls_scores,
                                        o_labels,
                                        o_label_weights,
                                        avg_factor=num_total_pos * 1.0)
        else:
            o_loss_cls = self.obj_loss_cls(o_cls_scores,
                                        o_labels,
                                        o_label_weights,
                                           avg_factor=cls_avg_factor)

        if self.rel_loss_cls.use_sigmoid:
            r_labels -= 1
            r_labels[r_labels==-1] = len(self.predicate_classes)
            r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                        r_labels,
                                        r_label_weights,
                                        avg_factor=num_total_pos * 1.0)
        else:
            r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                        r_labels,
                                        r_label_weights,
                                        avg_factor=cls_avg_factor)


        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = o_loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, s_bbox_preds_aux):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        s_bbox_preds_aux = s_bbox_preds_aux.reshape(-1, 4)
        s_bboxes = bbox_cxcywh_to_xyxy(s_bbox_preds_aux) * factors
        s_bboxes_gt = bbox_cxcywh_to_xyxy(s_bbox_targets) * factors

        o_bbox_preds_aux = o_bbox_preds_aux.reshape(-1, 4)
        o_bboxes = bbox_cxcywh_to_xyxy(o_bbox_preds_aux) * factors
        o_bboxes_gt = bbox_cxcywh_to_xyxy(o_bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        s_loss_iou = self.sub_loss_iou(s_bboxes,
                                       s_bboxes_gt,
                                       s_bbox_weights,
                                       avg_factor=num_total_pos)
        o_loss_iou = self.obj_loss_iou(o_bboxes,
                                       o_bboxes_gt,
                                       o_bbox_weights,
                                       avg_factor=num_total_pos)

        # regression L1 loss
        s_loss_bbox = self.sub_loss_bbox(s_bbox_preds_aux,
                                         s_bbox_targets,
                                         s_bbox_weights,
                                         avg_factor=num_total_pos)
        o_loss_bbox = self.obj_loss_bbox(o_bbox_preds_aux,
                                         o_bbox_targets,
                                         o_bbox_weights,
                                         avg_factor=num_total_pos)

        rel_s_o_bbox_preds_aux = rel_s_o_bbox_preds_aux.reshape(-1, 4)
        rel_s_o_loss_bbox = self.rel_sub_obj_loss_bbox(rel_s_o_bbox_preds_aux,
                                         rel_s_o_bbox_targets,
                                         rel_s_o_bbox_weights,
                                         avg_factor=num_total_pos)

        return r_loss_cls, loss_subject_match, loss_object_match, s_loss_cls, o_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_dice_loss, o_dice_loss, s_mask_loss, o_mask_loss, rel_s_o_loss_bbox
        # return r_loss_cls, loss_subject_match, loss_object_match, s_loss_cls, o_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_dice_loss, o_dice_loss
        # return loss_cls, loss_bbox, loss_iou, dice_loss, focal_loss, r_loss_cls, loss_subject_match, loss_object_match, s_loss_cls, o_loss_cls, s_loss_bbox, o_loss_bbox, s_loss_iou, o_loss_iou, s_dice_loss, o_dice_loss

    def get_targets(self,
                    subject_scores_list,
                    object_scores_list,
                    cls_scores_list,
                    bbox_preds_list,
                    mask_preds_list,
                    r_cls_scores_list,
                    s_bbox_preds_list,
                    o_bbox_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    r_cls_scores_sub_aux_list, 
                    r_cls_scores_obj_aux_list,
                    s_bbox_preds_aux_list, 
                    o_bbox_preds_aux_list,
                    s_mask_preds_aux_list, 
                    o_mask_preds_aux_list,
                    rel_s_o_bbox_preds_aux_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(r_cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, od_pos_inds_list, od_neg_inds_list,
         mask_preds_list, r_labels_list, r_label_weights_list, pos_inds_list,
         neg_inds_list, filtered_subject_scores, filtered_object_scores,
         gt_subject_id_list, gt_object_id_list, s_labels, o_labels, s_label_weights, o_label_weights,
         s_bbox_targets, o_bbox_targets, s_bbox_weights, o_bbox_weights,
         s_mask_targets, o_mask_targets, s_mask_preds, o_mask_preds, rel_s_o_bbox_targets, rel_s_o_bbox_weights) = multi_apply(
             self._get_target_single, subject_scores_list, object_scores_list,
             cls_scores_list, bbox_preds_list, mask_preds_list,
             r_cls_scores_list, s_bbox_preds_list, o_bbox_preds_list,
             gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
             img_metas, r_cls_scores_sub_aux_list, r_cls_scores_obj_aux_list,
             s_bbox_preds_aux_list, o_bbox_preds_aux_list,
             s_mask_preds_aux_list, o_mask_preds_aux_list, rel_s_o_bbox_preds_aux_list,
             gt_bboxes_ignore_list)

        num_total_od_pos = sum((inds.numel() for inds in od_pos_inds_list))
        num_total_od_neg = sum((inds.numel() for inds in od_neg_inds_list))

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, num_total_od_pos,
                num_total_od_neg, mask_preds_list, r_labels_list,
                r_label_weights_list, num_total_pos, num_total_neg,
                filtered_subject_scores, filtered_object_scores,
                gt_subject_id_list, gt_object_id_list, s_labels, o_labels, 
                s_label_weights, o_label_weights, s_bbox_targets, o_bbox_targets, 
                s_bbox_weights, o_bbox_weights, s_mask_targets, o_mask_targets, s_mask_preds, o_mask_preds, rel_s_o_bbox_targets, rel_s_o_bbox_weights)

    def _get_target_single(self,
                           subject_scores,
                           object_scores,
                           cls_score,
                           bbox_pred,
                           mask_preds,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           r_cls_scores_sub_aux, 
                           r_cls_scores_obj_aux,
                           s_bbox_preds_aux, 
                           o_bbox_preds_aux,
                           s_mask_preds_aux, 
                           o_mask_preds_aux,
                           rel_s_o_bbox_preds_aux,
                           gt_bboxes_ignore=None):

        assert len(gt_masks) == len(gt_bboxes)

        ###### obj det&seg
        num_bboxes = bbox_pred.size(0)
        assert len(gt_masks) == len(gt_bboxes)

        # assigner and sampler, only return human&object assign result
        if self.use_mask and 'Mask' in self.bbox_assigner_type:
            target_shape = mask_preds.shape[-2:]
            if gt_masks.shape[0] > 0:
                gt_masks_downsampled = F.interpolate(
                    gt_masks.unsqueeze(1).float(), target_shape,
                    mode='nearest').squeeze(1).long()
            else:
                gt_masks_downsampled = gt_masks
            od_assign_result = self.bbox_assigner.assign(bbox_pred, cls_score, mask_preds,
                                                        gt_bboxes, gt_labels, gt_masks_downsampled,
                                                        img_meta, self.num_things_classes,
                                                        gt_bboxes_ignore)
        else:
            od_assign_result = self.bbox_assigner.assign(bbox_pred, cls_score,
                                                        gt_bboxes, gt_labels,
                                                        img_meta,
                                                        gt_bboxes_ignore)
        sampling_result = self.sampler.sample(od_assign_result, bbox_pred,
                                              gt_bboxes)
        od_pos_inds = sampling_result.pos_inds
        od_neg_inds = sampling_result.neg_inds  #### no-rel class indices in prediction

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)  ### 0-based
        labels[od_pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # mask targets for subjects and objects
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds,
                                ...]  ###FIXME some transform might be needed
        mask_preds = mask_preds[od_pos_inds]
        mask_preds = interpolate(mask_preds[:, None],
                                 size=gt_masks.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False).squeeze(1)

        # bbox targets for subjects and objects
        od_pos_inds_bbox = od_pos_inds[gt_labels[sampling_result.pos_assigned_gt_inds] < self.num_things_classes]
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[od_pos_inds_bbox] = 1.0

        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[od_pos_inds] = pos_gt_bboxes_targets

        gt_label_assigned_query = torch.ones_like(gt_labels)
        gt_label_assigned_query[
            sampling_result.pos_assigned_gt_inds] = od_pos_inds

        ###### scene graph
        num_rels = s_bbox_preds_aux.size(0)
        # separate human boxes and object boxes from gt_bboxes and generate labels
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_rel_sub_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        gt_sub_ids = []
        gt_obj_ids = []
        gt_sub_masks = []
        gt_obj_masks = []

        for rel_id in range(gt_rels.size(0)):
            left_top = torch.stack([gt_bboxes[int(gt_rels[rel_id, 0])], gt_bboxes[int(gt_rels[rel_id, 1])]],0)[:,:2].min(0)[0]
            right_bottom = torch.stack([gt_bboxes[int(gt_rels[rel_id, 0])], gt_bboxes[int(gt_rels[rel_id, 1])]],0)[:,2:].max(0)[0]
            gt_rel_sub_obj_bboxes.append(torch.cat([left_top, right_bottom],-1))
            gt_sub_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 0])])
            gt_obj_bboxes.append(gt_bboxes[int(gt_rels[rel_id, 1])])
            gt_sub_labels.append(gt_labels[int(gt_rels[rel_id, 0])])
            gt_obj_labels.append(gt_labels[int(gt_rels[rel_id, 1])])
            gt_rel_labels.append(gt_rels[rel_id, 2])
            gt_sub_ids.append(gt_label_assigned_query[int(gt_rels[rel_id, 0])])
            gt_obj_ids.append(gt_label_assigned_query[int(gt_rels[rel_id, 1])])
            if self.use_mask:
                gt_sub_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         0])].unsqueeze(0))
                gt_obj_masks.append(gt_masks[int(gt_rels[rel_id,
                                                         1])].unsqueeze(0))

        gt_rel_sub_obj_bboxes = torch.vstack(gt_rel_sub_obj_bboxes).type_as(gt_bboxes).reshape(-1, 4)
        gt_sub_bboxes = torch.vstack(gt_sub_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_obj_bboxes = torch.vstack(gt_obj_bboxes).type_as(gt_bboxes).reshape(
            -1, 4)
        gt_sub_labels = torch.vstack(gt_sub_labels).type_as(gt_labels).reshape(
            -1)
        gt_obj_labels = torch.vstack(gt_obj_labels).type_as(gt_labels).reshape(
            -1)
        gt_rel_labels = torch.vstack(gt_rel_labels).type_as(gt_labels).reshape(
            -1)
        gt_sub_ids = torch.vstack(gt_sub_ids).type_as(gt_labels).reshape(-1)
        gt_obj_ids = torch.vstack(gt_obj_ids).type_as(gt_labels).reshape(-1)

        ########################################
        #### overwrite relation labels above####
        ########################################
        # assigner and sampler for relation-oriented id match
        s_assign_result, o_assign_result = self.id_assigner.assign(
            subject_scores, object_scores, r_cls_score, gt_sub_ids, gt_obj_ids,
            gt_rel_labels, img_meta, gt_bboxes_ignore)

        s_sampling_result = self.sampler.sample(s_assign_result, s_bbox_preds_aux,
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, o_bbox_preds_aux,
                                                gt_obj_bboxes)
        pos_inds = o_sampling_result.pos_inds
        neg_inds = o_sampling_result.neg_inds  #### no-rel class indices in prediction

        #match id targets
        gt_subject_ids = gt_sub_bboxes.new_full((num_rels, ),
                                                -1,
                                                dtype=torch.long)
        gt_subject_ids[pos_inds] = gt_sub_ids[
            s_sampling_result.pos_assigned_gt_inds]

        gt_object_ids = gt_obj_bboxes.new_full((num_rels, ),
                                               -1,
                                               dtype=torch.long)

        gt_object_ids[pos_inds] = gt_obj_ids[
            o_sampling_result.pos_assigned_gt_inds]

        # filtering unmatched subject/object id predictions
        gt_subject_ids = gt_subject_ids[pos_inds]
        # gt_subject_ids_res = torch.zeros_like(gt_subject_ids)
        # for idx, gt_subject_id in enumerate(gt_subject_ids):
        #     gt_subject_ids_res[idx] = ((od_pos_inds == gt_subject_id).nonzero(
        #         as_tuple=True)[0])
        # gt_subject_ids = gt_subject_ids_res

        gt_object_ids = gt_object_ids[pos_inds]
        # gt_object_ids_res = torch.zeros_like(gt_object_ids)
        # for idx, gt_object_id in enumerate(gt_object_ids):
        #     gt_object_ids_res[idx] = ((od_pos_inds == gt_object_id).nonzero(
        #         as_tuple=True)[0])
        # gt_object_ids = gt_object_ids_res

        filtered_subject_scores = subject_scores[pos_inds]
        # filtered_subject_scores = filtered_subject_scores[:, od_pos_inds]
        filtered_object_scores = object_scores[pos_inds]
        # filtered_object_scores = filtered_object_scores[:, od_pos_inds]

        # label targets
        s_labels = gt_sub_bboxes.new_full(
            (num_rels, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes]  as background
        s_labels[pos_inds] = gt_sub_labels[
            s_sampling_result.pos_assigned_gt_inds]
        s_label_weights = gt_sub_bboxes.new_ones(num_rels)
        # s_label_weights[pos_inds] = 1.0

        o_labels = gt_obj_bboxes.new_full(
            (num_rels, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes] as background
        o_labels[pos_inds] = gt_obj_labels[
            o_sampling_result.pos_assigned_gt_inds]
        o_label_weights = gt_obj_bboxes.new_ones(num_rels)
        # o_label_weights[pos_inds] = 1.0

        r_labels = gt_obj_bboxes.new_full((num_rels, ), 0,
                                          dtype=torch.long)  ### 1-based

        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_rels)

        if self.use_mask:

            gt_sub_masks = torch.cat(gt_sub_masks, axis=0).type_as(gt_masks[0])
            gt_obj_masks = torch.cat(gt_obj_masks, axis=0).type_as(gt_masks[0])

            assert gt_sub_masks.size() == gt_obj_masks.size()
            # mask targets for subjects and objects
            s_mask_targets = gt_sub_masks[
                s_sampling_result.pos_assigned_gt_inds,
                ...]  
            s_mask_preds = s_mask_preds_aux[pos_inds]
            

            o_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = o_mask_preds_aux[pos_inds]
            
            s_mask_preds = interpolate(s_mask_preds[:, None],
                                       size=gt_sub_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)

            o_mask_preds = interpolate(o_mask_preds[:, None],
                                       size=gt_obj_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)
        else:
            s_mask_targets = None
            s_mask_preds = None
            o_mask_targets = None
            o_mask_preds = None

        # bbox targets for subjects and objects
        pos_inds_sub_bbox = pos_inds[gt_sub_labels[s_sampling_result.pos_assigned_gt_inds] < self.num_things_classes]
        s_bbox_targets = torch.zeros_like(s_bbox_preds_aux)
        s_bbox_weights = torch.zeros_like(s_bbox_preds_aux)
        s_bbox_weights[pos_inds_sub_bbox] = 1.0

        pos_inds_obj_bbox = pos_inds[gt_obj_labels[s_sampling_result.pos_assigned_gt_inds] < self.num_things_classes]
        o_bbox_targets = torch.zeros_like(o_bbox_preds_aux)
        o_bbox_weights = torch.zeros_like(o_bbox_preds_aux)
        o_bbox_weights[pos_inds_obj_bbox] = 1.0

        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = o_bbox_preds_aux.new_tensor([img_w, img_h, img_w,
                                         img_h]).unsqueeze(0)

        pos_gt_s_bboxes_normalized = s_sampling_result.pos_gt_bboxes / factor
        pos_gt_s_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_s_bboxes_normalized)
        s_bbox_targets[pos_inds] = pos_gt_s_bboxes_targets

        pos_gt_o_bboxes_normalized = o_sampling_result.pos_gt_bboxes / factor
        pos_gt_o_bboxes_targets = bbox_xyxy_to_cxcywh(
            pos_gt_o_bboxes_normalized)
        o_bbox_targets[pos_inds] = pos_gt_o_bboxes_targets

        # bbox targets for rel
        pos_rel_s_o_gt_bboxes = gt_rel_sub_obj_bboxes / factor
        rel_s_o_bbox_targets = torch.zeros_like(rel_s_o_bbox_preds_aux)
        rel_s_o_bbox_weights = torch.zeros_like(rel_s_o_bbox_preds_aux)
        rel_s_o_bbox_weights[pos_inds] = 1.0
        rel_s_o_bbox_targets[pos_inds] = pos_rel_s_o_gt_bboxes[o_sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, od_pos_inds, od_neg_inds, mask_preds, r_labels,
                r_label_weights, pos_inds, neg_inds, filtered_subject_scores,
                filtered_object_scores, gt_subject_ids, gt_object_ids,
                s_labels, o_labels, s_label_weights, o_label_weights,
                s_bbox_targets, o_bbox_targets, s_bbox_weights, o_bbox_weights,
                s_mask_targets, o_mask_targets, s_mask_preds, o_mask_preds, rel_s_o_bbox_targets, rel_s_o_bbox_weights
                )  ###return the interpolated predicted masks

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img,
                      img_metas,
                      entity_query_embedding, 
                      enc_memory,
                      mlvl_enc_memory,
                      entity_all_bbox_preds, 
                      entity_all_cls_scores,
                      gt_rels,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      query_masks=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            query_masks (Tensor): Support variable query numbers in different images

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas, enc_memory, mlvl_enc_memory, entity_query_embedding, entity_all_bbox_preds, entity_all_cls_scores, query_masks=query_masks)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_masks, img_metas)
        else:
            gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, query_masks=query_masks)
        return losses

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

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.

        # Logit Adjustment https://arxiv.org/pdf/2007.07314.pdf
        psg_rel_freq = [0.15031376, 0.04144597, 0.17209725, 0.20244892,
                            0.03754787, 0.078283  , 0.01474781, 0.00047389, 0.00002675,
                            0.00042038, 0.00072229, 0.02722937, 0.00436434, 0.00070701,
                            0.07036451, 0.00561403, 0.02021279, 0.00320638, 0.00027898,
                            0.00062293, 0.01094143, 0.03900774, 0.00881276, 0.01948285,
                            0.0003172 , 0.00006115, 0.00475033, 0.00043567, 0.00024459,
                            0.00059236, 0.00050446, 0.00003057, 0.00129554, 0.00013758,
                            0.00003822, 0.00006115, 0.00712741, 0.00360765, 0.00026369,
                            0.0012    , 0.00002675, 0.00007261, 0.00152102, 0.00067643,
                            0.00050828, 0.00225478, 0.00762422, 0.02531471, 0.02144337,
                            0.00214778, 0.00027898, 0.00278217, 0.00018344, 0.00009936,
                            0.00195287, 0.00307262] # for psg dataset
        if self.rel_cls_out_channels > len(psg_rel_freq):
            psg_rel_freq = [1.] + psg_rel_freq
        psg_rel_freq = torch.tensor(psg_rel_freq, dtype=torch.float, device=cls_scores['rel'].device)
        self.psg_rel_freq = psg_rel_freq.log()
        self.logit_adj_tau = 0.0
        self.test_cfg.get('logit_adj_tau', self.logit_adj_tau)



        result_list = []
        for img_id in range(len(img_metas)):
            # od_cls_score = cls_scores['cls'][-1, img_id, ...]
            # bbox_pred = bbox_preds['bbox'][-1, img_id, ...]
            # mask_pred = bbox_preds['mask'][img_id, ...]
            all_cls_score = cls_scores['cls'][-1, img_id, ...]
            all_masks = bbox_preds['mask'][img_id]

            s_cls_score = cls_scores['sub'][-1, img_id, ...]
            o_cls_score = cls_scores['obj'][-1, img_id, ...]
            r_cls_score = cls_scores['rel'][-1, img_id, ...]
            s_bbox_pred = bbox_preds['sub'][-1, img_id, ...]
            o_bbox_pred = bbox_preds['obj'][-1, img_id, ...]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            s_mask_pred = bbox_preds['sub_seg'][img_id]
            o_mask_pred = bbox_preds['obj_seg'][img_id]
            triplet_sub_ids = cls_scores['sub_ids'][img_id]
            triplet_obj_ids = cls_scores['obj_ids'][img_id]
            triplets = self._get_bboxes_single(all_masks, all_cls_score,
                                               s_cls_score, o_cls_score,
                                               r_cls_score, s_bbox_pred,
                                               o_bbox_pred, s_mask_pred,
                                               o_mask_pred, img_shape,
                                               triplet_sub_ids,
                                               triplet_obj_ids,
                                               scale_factor, rescale)
            result_list.append(triplets)

        return result_list

    def _get_bboxes_single(self,
                           all_masks,
                           all_cls_score,
                           s_cls_score,
                           o_cls_score,
                           r_cls_score,
                           s_bbox_pred,
                           o_bbox_pred,
                           s_mask_pred,
                           o_mask_pred,
                           img_shape,
                           triplet_sub_ids,
                           triplet_obj_ids,
                           scale_factor,
                           rescale=False):

        assert len(s_cls_score) == len(o_cls_score)
        assert len(s_cls_score) == len(s_bbox_pred)
        assert len(s_cls_score) == len(o_bbox_pred)

        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        max_per_img = self.test_cfg.get('max_per_img', self.num_obj_query)

        # assert self.rel_loss_cls.use_sigmoid == False
        assert len(s_cls_score) == len(r_cls_score)

        # 0-based label input for objects and self.num_classes as default background cls
        if self.sub_loss_cls.use_sigmoid == True:
            s_logits = s_cls_score.sigmoid()
        else:
            s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
        
        if self.obj_loss_cls.use_sigmoid == True:
            o_logits = o_cls_score.sigmoid()
        else:
            o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]

        s_scores, s_labels = s_logits.max(-1)
        o_scores, o_labels = o_logits.max(-1)

        # r_lgs = F.softmax(r_cls_score, dim=-1) - self.logit_adj_tau * self.psg_rel_freq
        if self.rel_loss_cls.use_sigmoid == True:
            r_logits = r_cls_score.sigmoid()
            r_lgs = torch.cat([torch.zeros_like(r_logits[...,:1]), r_logits],-1)
        else:
            r_lgs = F.softmax(r_cls_score - self.logit_adj_tau * self.psg_rel_freq, dim=-1)
            r_logits = r_lgs[..., 1:]
        r_scores, r_indexes = r_logits.reshape(-1).topk(max_per_img*2)
        r_labels = r_indexes % self.num_relations + 1
        triplet_index = r_indexes // self.num_relations

        s_scores = s_scores[triplet_index]
        s_labels = s_labels[triplet_index] + 1
        s_bbox_pred = s_bbox_pred[triplet_index]

        o_scores = o_scores[triplet_index]
        o_labels = o_labels[triplet_index] + 1
        o_bbox_pred = o_bbox_pred[triplet_index]

        r_dists = r_lgs.reshape(
            -1, self.num_relations +
            1)[triplet_index]  #### NOTE: to match the evaluation in vg
        # one hot for evaluation
        r_dists = torch.zeros_like(r_dists)
        for ids in range(len(triplet_index)):
            r_dists[ids][r_labels[ids]] = 1

        labels = torch.cat((s_labels, o_labels), 0)
        complete_labels = labels
        complete_r_labels = r_labels
        complete_r_dists = r_dists

        if self.use_mask:
            object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
            iou_thr = self.test_cfg.get('iou_thr', 0.8)
            filter_low_score = self.test_cfg.get('filter_low_score', False)
            s_mask_pred = s_mask_pred[triplet_index]
            o_mask_pred = o_mask_pred[triplet_index]
            s_mask_pred = F.interpolate(s_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            s_mask_pred = torch.sigmoid(s_mask_pred) > 0.85
            o_mask_pred = F.interpolate(o_mask_pred.unsqueeze(1),
                                        size=mask_size).squeeze(1)
            o_mask_pred = torch.sigmoid(o_mask_pred) > 0.85
            output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)

            all_scores, all_labels = all_cls_score.sigmoid().max(-1) # use sigmoid
            all_masks = F.interpolate(all_masks.unsqueeze(1),
                                    size=mask_size,mode='bilinear',
                                    align_corners=False).squeeze(1)
            #### for panoptic postprocessing ####
            triplet_sub_ids = triplet_sub_ids[triplet_index].view(-1,1)
            triplet_obj_ids = triplet_obj_ids[triplet_index].view(-1,1)
            pan_rel_pairs = torch.cat((triplet_sub_ids,triplet_obj_ids), -1).to(torch.int).to(all_masks.device)
            # tri_obj_unique = pan_rel_pairs.unique() # equal to: pan_rel_pairs.view(-1).unique()
            # keep = (all_labels != self.num_classes) #& (all_scores > object_mask_thr) # why minus 1
            # tmp = torch.zeros_like(keep, dtype=torch.bool)
            # for id in tri_obj_unique:
            #     tmp[id] = True
            # keep = keep & tmp

            keep_ent = (all_labels != self.num_classes) & (all_scores > object_mask_thr)
            all_labels = all_labels[keep_ent]
            all_masks = all_masks[keep_ent]
            all_scores = all_scores[keep_ent]
            h, w = all_masks.shape[-2:]

            no_obj_filter = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
            for triplet_id in range(pan_rel_pairs.shape[0]):
                if keep_ent[pan_rel_pairs[triplet_id,0]] and keep_ent[pan_rel_pairs[triplet_id,1]]:
                    no_obj_filter[triplet_id]=True
            pan_rel_pairs = pan_rel_pairs[no_obj_filter]
            # if keep.sum() != len(keep):
            #     for new_id, past_id in enumerate(keep.nonzero().view(-1)):
            #         pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(past_id), new_id)
            if keep_ent.sum() != len(keep_ent):
                for new_id, past_id in enumerate(keep_ent.nonzero().view(-1)):
                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(past_id), new_id)
            r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]

            if all_labels.numel() == 0:
                pan_img = torch.ones(mask_size).cpu().to(torch.long)
                pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
                # pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(pan_masks.device).reshape(2, -1).T
                pan_rel_pairs = torch.zeros((max_per_img, 2), dtype=torch.int).to(pan_masks.device)
                rels = torch.tensor([0,0,0]).view(-1,3).to(pan_masks.device)
                pan_labels = torch.tensor([0]).to(pan_masks.device)
                r_labels = torch.tensor([0]).to(pan_masks.device)
                r_dists = torch.ones((max_per_img, r_dists.size()[-1]), dtype=torch.float32).to(pan_masks.device) * 1/57
            else:
                all_masks = all_masks.flatten(1)
                stuff_equiv_classes = defaultdict(lambda: [])
                thing_classes = defaultdict(lambda: [])
                thing_dedup = defaultdict(lambda: [])
                for k, label in enumerate(all_labels):
                    if label.item() >= 80:
                        stuff_equiv_classes[label.item()].append(k)
                    else:
                        thing_classes[label.item()].append(k)


                def dedup_things(pred_ids, binary_masks):
                    while len(pred_ids) > 1:
                        base_mask = binary_masks[pred_ids[0]].unsqueeze(0)
                        other_masks = binary_masks[pred_ids[1:]]
                        # calculate ious
                        ious = base_mask.mm(other_masks.transpose(0,1))/((base_mask+other_masks)>0).sum(-1)
                        ids_left = []
                        thing_dedup[pred_ids[0]].append(pred_ids[0])
                        for iou, other_id in zip(ious[0],pred_ids[1:]):
                            if iou>0.5:
                                thing_dedup[pred_ids[0]].append(other_id)
                            else:
                                ids_left.append(other_id)
                        pred_ids = ids_left
                    if len(pred_ids) == 1:
                        thing_dedup[pred_ids[0]].append(pred_ids[0])
                
                all_binary_masks = (torch.sigmoid(all_masks) > 0.5).to(torch.float)
                # create dict that groups duplicate masks
                for thing_pred_ids in thing_classes.values():
                    if len(thing_pred_ids) > 1:
                      dedup_things(thing_pred_ids, all_binary_masks)
                    else:
                        thing_dedup[thing_pred_ids[0]].append(thing_pred_ids[0])

                def get_ids_area(all_masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                    # This helper function creates the final panoptic segmentation image
                    # It also returns the area of the masks that appears on the image

                    m_id = all_masks.transpose(0, 1).sigmoid()

                    if m_id.shape[-1] == 0:
                        # We didn't detect any mask :(
                        m_id = torch.zeros((h, w),
                                           dtype=torch.long,
                                           device=m_id.device)
                    else:
                        m_id = m_id.argmax(-1).view(h, w)

                    if dedup:
                        # Merge the masks corresponding to the same stuff class
                        for equiv in stuff_equiv_classes.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(eq_id), equiv[0])
                        # Merge the masks corresponding to the same thing instance
                        for equiv in thing_dedup.values():
                            if len(equiv) > 1:
                                for eq_id in equiv:
                                    m_id.masked_fill_(m_id.eq(eq_id), equiv[0])
                                    pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(eq_id), equiv[0])
                    m_ids_remain,_ = m_id.unique().sort()
                    no_obj_filter2 = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
                    for triplet_id in range(pan_rel_pairs.shape[0]):
                        if pan_rel_pairs[triplet_id,0] in m_ids_remain and pan_rel_pairs[triplet_id,1] in m_ids_remain:
                            no_obj_filter2[triplet_id]=True
                    pan_rel_pairs = pan_rel_pairs[no_obj_filter2]
                    r_labels, r_dists = r_labels[no_obj_filter2], r_dists[no_obj_filter2]

                    pan_labels = [] 
                    pan_masks = []
                    for i, m_id_remain in enumerate(m_ids_remain):
                        pan_masks.append(m_id.eq(m_id_remain).unsqueeze(0))
                        pan_labels.append(all_labels[m_id_remain].unsqueeze(0))
                        m_id.masked_fill_(m_id.eq(m_id_remain), i)
                        pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(m_id_remain), i)
                    pan_masks = torch.cat(pan_masks, 0)
                    pan_labels = torch.cat(pan_labels, 0)

                    seg_img = m_id * INSTANCE_OFFSET + pan_labels[m_id]
                    seg_img = seg_img.view(h, w).cpu().to(torch.long)
                    m_id = m_id.view(h, w).cpu()
                    area = []
                    for i in range(len(all_masks)):
                        area.append(m_id.eq(i).sum().item())
                    return area, seg_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels

                area, pan_img, pan_rel_pairs, pan_masks, r_labels, r_dists, pan_labels = \
                    get_ids_area(all_masks, pan_rel_pairs, r_labels, r_dists, dedup=True)

                if r_labels.numel() == 0:
                    rels = torch.tensor([0,0,0]).view(-1,3).to(pan_masks.device)
                else:
                    rels = torch.cat((pan_rel_pairs,r_labels.unsqueeze(-1)),-1)
                    # dedup_rels = rels.unique(dim=0)
                    # dedup_r_labels = dedup_rels[...,-1]
                    dedup_rel_index = torch.zeros_like(r_labels, dtype=torch.bool)
                    dedup_rels_hand = []
                    rels_numpy = rels.cpu().numpy()
                    for r_index, rel in enumerate(rels_numpy):
                        if rel.tolist() not in dedup_rels_hand:
                            dedup_rels_hand.append(rel.tolist())
                            dedup_rel_index[r_index] = True
                    
                    # rels = torch.as_tensor(dedup_rels_hand, dtype=rels.dtype, device=rels.device)
                    rels = rels[dedup_rel_index][:max_per_img]
                    r_labels = r_labels[dedup_rel_index][:max_per_img]
                    r_dists = r_dists[dedup_rel_index][:max_per_img]
                    pan_rel_pairs = pan_rel_pairs[dedup_rel_index][:max_per_img]



                # if all_labels.numel() > 0:
                #     # We know filter empty masks as long as we find some
                #     while True:
                #         filtered_small = torch.as_tensor(
                #             [area[i] <= 4 for i, c in enumerate(all_labels)],
                #             dtype=torch.bool,
                #             device=keep.device)
                #         if filtered_small.any().item():
                #             all_scores = all_scores[~filtered_small]
                #             all_labels = all_labels[~filtered_small]
                #             all_masks = all_masks[~filtered_small]
                #             area, pan_img = get_ids_area(all_masks, all_scores)
                #         else:
                #             break

        s_det_bboxes = bbox_cxcywh_to_xyxy(s_bbox_pred)
        s_det_bboxes[:, 0::2] = s_det_bboxes[:, 0::2] * img_shape[1]
        s_det_bboxes[:, 1::2] = s_det_bboxes[:, 1::2] * img_shape[0]
        s_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        s_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            s_det_bboxes /= s_det_bboxes.new_tensor(scale_factor)
        s_det_bboxes = torch.cat((s_det_bboxes, s_scores.unsqueeze(1)), -1)

        o_det_bboxes = bbox_cxcywh_to_xyxy(o_bbox_pred)
        o_det_bboxes[:, 0::2] = o_det_bboxes[:, 0::2] * img_shape[1]
        o_det_bboxes[:, 1::2] = o_det_bboxes[:, 1::2] * img_shape[0]
        o_det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        o_det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            o_det_bboxes /= o_det_bboxes.new_tensor(scale_factor)
        o_det_bboxes = torch.cat((o_det_bboxes, o_scores.unsqueeze(1)), -1)

        det_bboxes = torch.cat((s_det_bboxes, o_det_bboxes), 0)
        rel_pairs = torch.arange(len(det_bboxes),
                                 dtype=torch.int).reshape(2, -1).T

        if self.use_mask:
            return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                pan_img, complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks, rels, pan_labels
        else:
            return det_bboxes, labels, rel_pairs, r_scores, r_labels, r_dists

    def simple_test_bboxes(self, feats, img_metas, entity_query_embedding, enc_memory, mlvl_enc_memory, entity_all_bbox_preds, entity_all_cls_scores, rescale=False):
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas, enc_memory, mlvl_enc_memory, entity_query_embedding, entity_all_bbox_preds, entity_all_cls_scores)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def vis_reference_points(self, img, reference_points, img_metas, sub_outputs_coord, obj_outputs_coord, rel_outputs_class):
        import cv2
        import numpy as np
        predicate_classes = [
            'over',
            'in front of',
            'beside',
            'on',
            'in',
            'attached to',
            'hanging from',
            'on back of',
            'falling off',
            'going down',
            'painted on',
            'walking on',
            'running on',
            'crossing',
            'standing on',
            'lying on',
            'sitting on',
            'flying over',
            'jumping over',
            'jumping from',
            'wearing',
            'holding',
            'carrying',
            'looking at',
            'guiding',
            'kissing',
            'eating',
            'drinking',
            'feeding',
            'biting',
            'catching',
            'picking',
            'playing with',
            'chasing',
            'climbing',
            'cleaning',
            'playing',
            'touching',
            'pushing',
            'pulling',
            'opening',
            'cooking',
            'talking to',
            'throwing',
            'slicing',
            'driving',
            'riding',
            'parked on',
            'driving on',
            'about to hit',
            'kicking',
            'swinging',
            'entering',
            'exiting',
            'enclosing',
            'leaning on',
        ]
        for im, reference_points_per, img_meta, sub_coords, obj_coords, rel_cls in zip(img, reference_points[-1], img_metas, sub_outputs_coord, obj_outputs_coord, rel_outputs_class):
            if self.rel_loss_cls.use_sigmoid == True:
                r_logits = rel_cls.sigmoid()
            else:
                r_lgs = F.softmax(rel_cls, dim=-1)
                r_logits = r_lgs[..., 1:]
            # r_scores, r_indexes = r_logits.reshape(-1).topk(10)
            # r_labels = r_indexes % self.num_relations + 1
            # triplet_index = r_indexes // self.num_relations
            r_scores, r_indexes = r_logits.max(-1)
            topk_scores, topk_indexes = r_scores.topk(10)
            r_labels = r_indexes[topk_indexes] + 1
            triplet_index = topk_indexes
            sub_coords_xyxy = bbox_cxcywh_to_xyxy(sub_coords)
            obj_coords_xyxy = bbox_cxcywh_to_xyxy(obj_coords)
            for i,index in enumerate(triplet_index):
                img_name = 'work_dirs/debugs/' + img_meta['filename'].split('/')[-1].split('.')[0]+f'_{index}.jpg'
                unorm = UnNormalize(mean=img_meta['img_norm_cfg']['mean'], std=img_meta['img_norm_cfg']['std'])
                im_output = unorm(im.clone()).cpu().clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
                im_output = cv2.cvtColor(im_output, cv2.COLOR_RGB2BGR)

                if reference_points_per.shape[-1] == 2:
                    ref = reference_points_per[index:index+1,0,:].clone().detach().cpu().numpy()
                    im_size = np.array(img_meta['img_shape'][:2])
                    ori_coord = np.array(ref*im_size[::-1],dtype=np.int)

                    for ref in ori_coord:
                        cv2.circle(im_output,ref,5,(0,0,255),-1)
                else:
                    ref = reference_points_per[index:index+1,0,:].clone().detach().cpu().numpy()
                    im_size = np.array(img_meta['img_shape'][:2])
                    ori_coord = np.array(ref*np.concatenate((im_size[::-1],im_size[::-1]),axis=-1),dtype=np.int)

                    for ref in ori_coord:
                        cv2.rectangle(im_output, ref[:2], ref[2:], (0,0,255), 2)
                
                sub = sub_coords_xyxy[index].clone().detach().cpu().numpy()
                im_size = np.array(img_meta['img_shape'][:2])
                ori_coord = np.array(sub*np.concatenate((im_size[::-1],im_size[::-1]),axis=-1),dtype=np.int)
                cv2.rectangle(im_output, ori_coord[:2], ori_coord[2:], (255,0,0), 2)

                obj = obj_coords_xyxy[index].clone().detach().cpu().numpy()
                im_size = np.array(img_meta['img_shape'][:2])
                ori_coord = np.array(obj*np.concatenate((im_size[::-1],im_size[::-1]),axis=-1),dtype=np.int)
                cv2.rectangle(im_output, ori_coord[:2], ori_coord[2:], (255,0,0), 2)

                cv2.putText(im_output, predicate_classes[r_labels[i]-1], (ref[:2] + ref[2:]) // 2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                cv2.imwrite(img_name, im_output)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN) Copied from
    hoitr."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """Simple convolutional head, using group norm.

    Upsampling is done using a FPN approach
    """
    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim, context_dim // 2, context_dim // 4, context_dim // 8,
            context_dim // 16, context_dim // 64
        ]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        x = torch.cat(
            [_expand(x, bbox_mask.shape[1]),
             bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode='nearest')
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax
    (no multiplication by value)"""
    def __init__(self,
                 query_dim,
                 hidden_dim,
                 num_heads,
                 dropout=0.0,
                 bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads)**-0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k,
                     self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
                     self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads,
                    self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads,
                    self.hidden_dim // self.num_heads, k.shape[-2],
                    k.shape[-1])
        weights = torch.einsum('bqnc,bnchw->bqnhw', qh * self.normalize_fact,
                               kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


def interpolate(input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
    """Equivalent to nn.functional.interpolate, but with support for empty
    batch sizes.

    This will eventually be supported natively by PyTorch, and this class can
    go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor,
                                                   mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor,
                                                mode, align_corners)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor