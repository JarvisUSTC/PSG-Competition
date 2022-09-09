# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import build_transformer
from .predicate_node_generator import build_predicate_node_generator
from .entities_indexing_head import build_entities_indexing_head
from openpsg.utils.metrics.cosine_sim import cosine_similarity
from torchvision.ops import generalized_box_iou
#####imports for tools
from packaging import version
import copy

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


@HEADS.register_module()
class SGTRHead(AnchorFreeHead):

    _version = 2

    def __init__(self,
                num_classes,
                in_channels,
                num_relations,
                object_classes,
                predicate_classes,
                num_obj_query=100,
                num_rel_query=100,
                num_reg_fcs=2,
                use_mask=True,
                temp=0.1,
                transformer=None,
                predicate_node_generator=None,
                entities_aware_indexing_head=None,
                indexing_module_type="rule_base",
                n_heads=8,
                sync_cls_avg_factor=False,
                bg_cls_weight=0.02,
                positional_encoding=dict(type='SinePositionalEncoding',
                                        num_feats=128,
                                        normalize=True),
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
                obj_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
                obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                obj_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                sub_loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0,
                              class_weight=1.0),
                sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
                sub_dice_loss=dict(type='DiceLoss', loss_weight=1.0),
                loss_rel_vec=dict(type='L1Loss', loss_weight=2.0),
                train_cfg=dict(id_assigner=dict(
                                    type='IdMatcher',
                                    sub_id_cost=dict(type='ClassificationCost', weight=1.),
                                    obj_id_cost=dict(type='ClassificationCost', weight=1.),
                                    r_cls_cost=dict(type='ClassificationCost', weight=1.)),
                    rel_assigner=dict(type='RelHungarianMatcher',
                                cost_rel_class = dict(type='ClassificationCost', weight=1.5),
                                cost_rel_vec = dict(type='BBoxL1Cost', weight=1.0),
                                cost_class = dict(type='ClassificationCost', weight=1.5),
                                cost_bbox = dict(type='BBoxL1Cost', weight=0.8),
                                cost_giou = dict(type='IoUCost', iou_mode='giou', weight=1.0),
                                cost_indexing = dict(type='IndexCost', weight=0.2),
                                cost_foreground_ent = 0.3,
                                num_entities_pairing_train = 25,
                                num_entities_pairing = 3,
                                num_matching_per_gt = 1,),
                            bbox_assigner=dict(
                                type='HungarianAssigner',
                                cls_cost=dict(type='ClassificationCost',
                                                weight=1.),
                                reg_cost=dict(type='BBoxL1Cost',
                                                weight=5.0),
                                iou_cost=dict(type='IoUCost',
                                                iou_mode='giou',
                                                weight=2.0))),
                test_cfg=dict(max_per_img=100),
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
        assert isinstance(s_class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(s_class_weight)}.'

        s_class_weight = torch.ones(num_classes + 1) * s_class_weight
        #NOTE set background class as the last indice
        s_class_weight[-1] = bg_cls_weight
        sub_loss_cls.update({'class_weight': s_class_weight})

        o_class_weight = obj_loss_cls.get('class_weight', None)
        assert isinstance(o_class_weight, float), 'Expected ' \
            'class_weight to have type float. Found ' \
            f'{type(o_class_weight)}.'

        o_class_weight = torch.ones(num_classes + 1) * o_class_weight
        #NOTE set background class as the last indice
        o_class_weight[-1] = bg_cls_weight
        obj_loss_cls.update({'class_weight': o_class_weight})

        r_class_weight = rel_loss_cls.get('class_weight', None)
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
            bbox_assigner = train_cfg['bbox_assigner']
            rel_assigner = train_cfg['rel_assigner']
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
            self.rel_assigner = build_assigner(rel_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        assert num_obj_query == num_rel_query
        self.num_obj_query = num_obj_query
        self.num_rel_query = num_rel_query
        self.use_mask = use_mask
        self.temp = temp
        self.num_classes = num_classes
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

        self.obj_loss_cls = build_loss(obj_loss_cls)
        self.obj_loss_bbox = build_loss(obj_loss_bbox)
        self.obj_loss_iou = build_loss(obj_loss_iou)

        self.sub_loss_cls = build_loss(sub_loss_cls)
        self.sub_loss_bbox = build_loss(sub_loss_bbox)
        self.sub_loss_iou = build_loss(sub_loss_iou)

        self.rel_vec_loss = build_loss(loss_rel_vec)
        if self.use_mask:
            # self.obj_focal_loss = build_loss(obj_focal_loss)
            self.obj_dice_loss = build_loss(obj_dice_loss)
            # self.sub_focal_loss = build_loss(sub_focal_loss)
            self.sub_dice_loss = build_loss(sub_dice_loss)

        ### id losses
        self.sub_id_loss = build_loss(sub_id_loss)
        self.obj_id_loss = build_loss(obj_id_loss)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.obj_loss_cls.use_sigmoid:
            self.obj_cls_out_channels = num_classes
        else:
            self.obj_cls_out_channels = num_classes + 1

        if self.sub_loss_cls.use_sigmoid:
            self.sub_cls_out_channels = num_classes
        else:
            self.sub_cls_out_channels = num_classes + 1

        if rel_loss_cls['use_sigmoid']:
            self.rel_cls_out_channels = num_relations
        else:
            self.rel_cls_out_channels = num_relations + 1

        

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        self.n_heads = n_heads
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'

        # rel decoder
        self.predicate_node_generator = build_predicate_node_generator(predicate_node_generator)
        self.num_pred_edges = 1
        if predicate_node_generator.get("entities_aware_decoder"):
            # Hyper parameter preset
            self.entities_aware_head = True
            self.normed_rel_vec_dist = False
            self.num_entities_pairing = 3
            ###########
            self.obj_class_embed = nn.Linear(self.embed_dims, self.obj_cls_out_channels)
            self.obj_bbox_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
            self.sub_class_embed = nn.Linear(self.embed_dims, self.sub_cls_out_channels)
            self.sub_bbox_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)
            num_layers = predicate_node_generator['rel_decoder']['num_layers']
            def initialize_ent_pred(class_embed, bbox_embed):
                class_embed = nn.ModuleList([class_embed for _ in range(num_layers)])
                bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_layers)])

                return class_embed, bbox_embed

            (self.obj_class_embed, self.obj_bbox_embed) = initialize_ent_pred(self.obj_class_embed, self.obj_bbox_embed)
            (self.sub_class_embed, self.sub_bbox_embed) = initialize_ent_pred(self.sub_class_embed, self.sub_bbox_embed)

            # could add mask head
            # self.obj_mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
            #                                [1024, 512, 256], self.embed_dims)
            # self.sub_mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
            #                                [1024, 512, 256], self.embed_dims)
            # self.predicate_node_generator.sub_class_embed = self.sub_class_embed
            # self.predicate_node_generator.sub_bbox_embed = self.sub_bbox_embed

            # self.predicate_node_generator.obj_bbox_embed = self.obj_bbox_embed
            # self.predicate_node_generator.obj_class_embed = self.obj_class_embed
        
        # post entities decoder
        if entities_aware_indexing_head is not None:
            self.use_entities_indexing_ranking = False
            self.entities_indexing_heads = build_entities_indexing_head(entities_aware_indexing_head)
            self.indexing_module_type = indexing_module_type
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        # Entity DETR
        self.input_proj = Conv2d(self.in_channels,
                                 self.embed_dims,
                                 kernel_size=1)
        self.obj_query_embed = nn.Embedding(self.num_obj_query,
                                            self.embed_dims)
        self.class_embed = Linear(self.embed_dims, self.cls_out_channels)
        self.box_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)

        # Relation DETR
        self.rel_query_embed = nn.Embedding(self.num_rel_query,
                                            self.embed_dims)
        self.rel_query_pos_embed = nn.Embedding(self.num_rel_query, self.embed_dims)
        self.rel_input_proj = nn.Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1
        )

        self.rel_cls_embed = Linear(self.embed_dims, self.rel_cls_out_channels)

        self.rel_vector_embed = nn.Linear(self.embed_dims, 4)

        # Keep same with ori psgformer, it could be replaced by MaskDINO
        self.rel_bbox_sub_attention = MHAttentionMap(self.embed_dims,
                                             self.embed_dims,
                                             self.n_heads,
                                             dropout=0.0)

        self.rel_mask_sub_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
                                           [1024, 512, 256], self.embed_dims)

        self.rel_bbox_obj_attention = copy.deepcopy(self.rel_bbox_sub_attention)

        self.rel_mask_obj_head = copy.deepcopy(self.rel_mask_sub_head)
        ####################


        self.rel_position_embedding = self.positional_encoding

        # self.sub_query_update = nn.Sequential(
        #     Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims))

        # self.obj_query_update = nn.Sequential(
        #     Linear(self.embed_dims, self.embed_dims), nn.ReLU(inplace=True),
        #     Linear(self.embed_dims, self.embed_dims))

        # self.sop_query_update = nn.Sequential(
        #     Linear(2 * self.embed_dims, self.embed_dims),
        #     nn.ReLU(inplace=True), Linear(self.embed_dims, self.embed_dims))

        # self.rel_query_update = nn.Identity()

        self.bbox_attention = MHAttentionMap(self.embed_dims,
                                             self.embed_dims,
                                             self.n_heads,
                                             dropout=0.0)
        self.mask_head = MaskHeadSmallConv(self.embed_dims + self.n_heads,
                                           [1024, 512, 256], self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

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
                # '.decoder2.norm.': '.decoder2.post_norm.',
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


    def forward(self, feats, img_metas, train_mode=False):

        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        last_features = feats[-1]
        batch_size = last_features.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = last_features.new_ones((batch_size, input_img_h, input_img_w))
        image_sizes = []
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            masks[img_id, :img_h, :img_w] = 0
            image_sizes.append(torch.as_tensor([img_h, img_w]).to(last_features.device))
        image_sizes = torch.stack(image_sizes)

        last_features = self.input_proj(last_features)
        # interpolate masks to have the same spatial shape with feats
        masks = F.interpolate(masks.unsqueeze(1),
                              size=last_features.shape[-2:]).to(
                                  torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_obj_dec, memory \
            = self.transformer(last_features, masks,
                               self.obj_query_embed.weight,
                               pos_embed)
        ent_hs = outs_obj_dec
        outputs_class = self.class_embed(outs_obj_dec)
        outputs_coord = self.box_embed(outs_obj_dec).sigmoid()
        ent_box_pred = outputs_coord[-1]
        ent_cls_pred = outputs_class[-1]
        bbox_mask = self.bbox_attention(outs_obj_dec[-1], memory, mask=masks)
        seg_masks = self.mask_head(last_features, bbox_mask,
                                   [feats[2], feats[1], feats[0]])
        seg_masks = seg_masks.view(batch_size, self.num_obj_query,
                                   seg_masks.shape[-2], seg_masks.shape[-1])

        ent_predictions = dict(
            cls = outputs_class,
            bbox = outputs_coord,
            mask = seg_masks
        )

        ### new interaction
        rel_hs, ext_inter_feats, rel_decoder_out_res = self.predicate_node_generator(
            last_features,
            masks,
            self.rel_query_embed.weight,
            pos_embed,
            None,
            memory,
            outs_obj_dec,
            ent_box_pred
        )

        pred_rel_logits = self.rel_cls_embed(rel_hs.feature) # n_lyr, batch_size, num_queries, N

        pred_rel_vec = self.rel_vector_embed(rel_hs.feature).sigmoid() # (cx1,cy1,cx2,cy2) [B, N, 4]

        if self.predicate_node_generator.entities_aware_decoder:
            rel_hs_ent_aware_sub = ext_inter_feats[0].feature
            rel_hs_ent_aware_obj = ext_inter_feats[1].feature

            # Now it just predicts mask for last layer
            pred_rel_sub_bbox_mask = self.rel_bbox_sub_attention(rel_hs_ent_aware_sub[-1], memory, mask = masks)
            pred_rel_sub_seg_masks = self.rel_mask_sub_head(last_features, pred_rel_sub_bbox_mask,
                                   [feats[2], feats[1], feats[0]])
            pred_rel_sub_seg_masks = pred_rel_sub_seg_masks.view(batch_size, self.num_obj_query,
                                   pred_rel_sub_seg_masks.shape[-2], pred_rel_sub_seg_masks.shape[-1])

            pred_rel_obj_bbox_mask = self.rel_bbox_obj_attention(rel_hs_ent_aware_obj[-1], memory, mask = masks)
            pred_rel_obj_seg_masks = self.rel_mask_obj_head(last_features, pred_rel_obj_bbox_mask,
                                   [feats[2], feats[1], feats[0]])
            pred_rel_obj_seg_masks = pred_rel_obj_seg_masks.view(batch_size, self.num_obj_query,
                                   pred_rel_obj_seg_masks.shape[-2], pred_rel_obj_seg_masks.shape[-1])

            pred_rel_sub_seg_masks = [pred_rel_sub_seg_masks for _ in range(len(pred_rel_vec))]
            pred_rel_obj_seg_masks = [pred_rel_obj_seg_masks for _ in range(len(pred_rel_vec))]

        #  pack prediction results
        semantic_predictions = {
            "pred_logits": ent_cls_pred,
            "pred_boxes": ent_box_pred,
            "mask": seg_masks,
            "pred_rel_logits": pred_rel_logits[-1],
            # layer, batch_size, num_queries, 4 => batch_size, num_queries, 4
            "pred_rel_vec": pred_rel_vec[-1],
            # take the output from the last layer
            "pred_rel_sub_mask": pred_rel_sub_seg_masks[-1],
            "pred_rel_obj_mask": pred_rel_obj_seg_masks[-1]
        }

        rel_hs_ent_aware_sub = None
        rel_hs_ent_aware_obj = None

        pred_rel_sub_box = None
        pred_rel_obj_box = None
        pred_rel_obj_logits = None
        pred_rel_sub_logits = None
        
        if self.entities_indexing_heads:
            (ent_hs,
                pred_rel_obj_box, pred_rel_obj_logits,
                pred_rel_sub_box, pred_rel_sub_logits,
                rel_hs_ent_aware_obj, rel_hs_ent_aware_sub, 
                pred_ent_rel_vec) = self.predict_rel_ent_semantics(
                ent_hs, ext_inter_feats, image_sizes, rel_hs, semantic_predictions, rel_decoder_out_res
            )

        pred_rel_confidence = None

        rel_aux_out = self.generate_aux_out(image_sizes, ent_hs, ent_cls_pred, ent_box_pred,
                                            pred_rel_logits, pred_rel_vec,
                                            rel_hs_ent_aware_sub, rel_hs_ent_aware_obj,
                                            pred_rel_sub_box, pred_rel_obj_box,
                                            pred_rel_obj_logits, pred_rel_sub_logits, rel_decoder_out_res, pred_rel_sub_seg_masks, pred_rel_obj_seg_masks)

        return rel_aux_out, semantic_predictions, ent_predictions

    # from raw prediction to semantic results
    def predict_rel_ent_semantics(self, ent_hs, ext_inter_feats, image_sizes, rel_hs,
                                  semantic_predictions, rel_decoder_out_res):

        # memory is for mask head
        # default entities representation is overide by the unify relationship representation
        rel_hs_ent_aware_sub = rel_hs.feature
        rel_hs_ent_aware_obj = rel_hs.feature
        if self.predicate_node_generator.entities_aware_decoder:
            rel_hs_ent_aware_sub = ext_inter_feats[0].feature
            rel_hs_ent_aware_obj = ext_inter_feats[1].feature

            pred_rel_sub_box = []
            pred_rel_obj_box = []
            pred_rel_obj_logits = []
            pred_rel_sub_logits = []

            for lid in range(len(rel_hs_ent_aware_sub)):
                # TODO mask head
                pred_rel_sub_logits.append(self.sub_class_embed[lid](rel_hs_ent_aware_sub[lid]))
                pred_rel_sub_box.append(self.sub_bbox_embed[lid](rel_hs_ent_aware_sub[lid]).sigmoid())
                pred_rel_obj_logits.append(self.obj_class_embed[lid](rel_hs_ent_aware_obj[lid]))
                pred_rel_obj_box.append(self.obj_bbox_embed[lid](rel_hs_ent_aware_obj[lid]).sigmoid())

            pred_rel_sub_logits = torch.stack(pred_rel_sub_logits)
            pred_rel_sub_box = torch.stack(pred_rel_sub_box)
            pred_rel_obj_logits = torch.stack(pred_rel_obj_logits)
            pred_rel_obj_box = torch.stack(pred_rel_obj_box)  # layer, bz, num_q, dim
        
        pred_ent_rel_vec = torch.cat((pred_rel_sub_box[..., :2], pred_rel_obj_box[..., :2]), dim=-1)

        semantic_predictions.update(
            {
                "pred_rel_obj_logits": pred_rel_obj_logits[-1],
                "pred_rel_obj_box": pred_rel_obj_box[-1],
                "pred_rel_sub_logits": pred_rel_sub_logits[-1],
                "pred_rel_sub_box": pred_rel_sub_box[-1],
                "pred_ent_rel_vec": pred_ent_rel_vec[-1]
            }
        )


        if self.entities_indexing_heads is not None:
            ent_hs = ent_hs[-1]

            (
                sub_idxing, obj_idxing,
            ) = self.graph_assembling(
                semantic_predictions, image_sizes, ent_hs,
                rel_hs_ent_aware_sub[-1], rel_hs_ent_aware_obj[-1],
            )
            semantic_predictions.update({
                "sub_entities_indexing": sub_idxing,
                "obj_entities_indexing": obj_idxing,
            })

        return ent_hs, pred_rel_obj_box, pred_rel_obj_logits, pred_rel_sub_box, pred_rel_sub_logits, rel_hs_ent_aware_obj, rel_hs_ent_aware_sub, pred_ent_rel_vec

    def graph_assembling(
            self, out, image_sizes, ent_hs, rel_hs_ent_aware_sub, rel_hs_ent_aware_obj
    ):
        if self.indexing_module_type in ["rule_base", "pred_att", 'rel_vec']:

            if self.indexing_module_type in ["rule_base", 'rel_vec']:
                sub_idxing, obj_idxing = self.entities_indexing_heads(
            out, image_sizes
            )
            elif self.indexing_module_type == "pred_att":
                sub_idxing, obj_idxing = self.entities_indexing_heads(out)

        elif self.indexing_module_type == "feat_att":
            sub_idxing = self.entities_indexing_heads["sub"](
                ent_hs, rel_hs_ent_aware_sub
            )
            obj_idxing = self.entities_indexing_heads["obj"](
                ent_hs, rel_hs_ent_aware_obj
            )

        return sub_idxing, obj_idxing

    def generate_aux_out(self, image_sizes, ent_hs, outputs_class, outputs_coord, pred_rel_logits, pred_rel_vec,
                         rel_hs_ent_aware_sub, rel_hs_ent_aware_obj, pred_rel_sub_box, pred_rel_obj_box,
                         pred_rel_obj_logits, pred_rel_sub_logits, rel_decoder_out_res, pred_rel_sub_seg_masks, pred_rel_obj_seg_masks):
        aux_out = []
        for ir in range(len(pred_rel_logits) - 1):
            tmp_out = {
                # take the output from the last layer
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                # layer, batch_size, num_queries, 4
                #     => batch_size, num_queries, 4
                "pred_rel_logits": pred_rel_logits[ir],
                "pred_rel_vec": pred_rel_vec[ir],
            }

            if self.predicate_node_generator.entities_aware_decoder:
                tmp_out.update({
                    "pred_rel_obj_logits": pred_rel_obj_logits[ir],
                    "pred_rel_obj_box": pred_rel_obj_box[ir],
                    "pred_rel_sub_logits": pred_rel_sub_logits[ir],
                    "pred_rel_sub_box": pred_rel_sub_box[ir],
                    "pred_rel_sub_mask": pred_rel_sub_seg_masks[ir],
                    "pred_rel_obj_mask": pred_rel_obj_seg_masks[ir],
                })

                if rel_decoder_out_res.get('unify_assembling') is not None:
                    uni_ass_res = rel_decoder_out_res['unify_assembling'][-1]
                    for idx, role in enumerate(['sub', 'obj']):
                        for att_name in ['boxs', 'logits', "index"]:
                            tmp_out.update({
                                f'uni_ass_{role}_{att_name}': uni_ass_res[f'agg_{att_name}'][idx]
                            })

            if self.entities_indexing_heads is not None:
                (sub_idxing, obj_idxing,) = self.graph_assembling(
                    tmp_out, image_sizes, ent_hs,
                    rel_hs_ent_aware_sub[ir], rel_hs_ent_aware_obj[ir],
                )

                tmp_out.update({
                    "sub_entities_indexing": sub_idxing,
                    "obj_entities_indexing": obj_idxing,
                })

                # if rel_decoder_out_res.get('unify_assembling') is not None:
                #     uni_ass_res = rel_decoder_out_res['unify_assembling'][-1]
                #     for idx, role in enumerate(['sub', 'obj']):
                #         tmp_out.update({f'{role}_entities_indexing': uni_ass_res[f'agg_index'][idx]})

            aux_out.append(tmp_out)

        return aux_out

    def loss(
        self,
        rel_aux_out, 
        semantic_predictions, 
        ent_predictions,
        gt_rels_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore=None):

        all_ent_cls_scores = ent_predictions['cls']
        all_ent_bbox_preds = ent_predictions['bbox']
        all_mask_preds = ent_predictions['mask']

        num_ent_dec_layers = len(all_ent_cls_scores)

        all_mask_preds = [all_mask_preds for _ in range(num_ent_dec_layers)] # just predict mask for last layer now TODO: if using MaskDINO, could predict for all layers

        # relation
        all_rel_out = rel_aux_out + [semantic_predictions]
        num_rel_dec_layers = len(all_rel_out)

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_rel_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_rel_dec_layers)]
        all_gt_rels_list = [gt_rels_list for _ in range(num_rel_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_rel_dec_layers)
        ]
        all_gt_masks_list = [gt_masks_list for _ in range(num_rel_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_rel_dec_layers)]

        losses_cls, losses_bbox, losses_iou, dice_losses, focal_losses, \
        r_losses_cls, s_losses_cls, o_losses_cls, rel_vector_loss, \
        s_losses_iou, o_losses_iou,s_losses_bbox,o_losses_bbox, \
        s_dice_losses, o_dice_losses = multi_apply(
            self.loss_single, all_rel_out,
            all_ent_cls_scores, all_ent_bbox_preds, all_mask_preds,
            all_gt_rels_list, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_masks_list, img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        ## loss of relation-oriented matching
        # loss_dict['loss_subject_match'] = loss_subject_match[-1]
        # loss_dict['loss_object_match'] = loss_object_match[-1]

        ## loss of object detection and segmentation
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        loss_dict['focal_losses'] = focal_losses[-1]
        loss_dict['dice_losses'] = dice_losses[-1]

        # loss from the last decoder layer
        loss_dict['s_loss_cls'] = s_losses_cls[-1]
        loss_dict['o_loss_cls'] = o_losses_cls[-1]
        loss_dict['r_loss_cls'] = r_losses_cls[-1]
        loss_dict['s_loss_bbox'] = s_losses_bbox[-1]
        loss_dict['o_loss_bbox'] = o_losses_bbox[-1]
        loss_dict['s_loss_iou'] = s_losses_iou[-1]
        loss_dict['o_loss_iou'] = o_losses_iou[-1]
        loss_dict['rel_vector_loss'] = rel_vector_loss[-1]
        if self.use_mask:
            # loss_dict['s_focal_losses'] = s_focal_losses[-1]
            # loss_dict['o_focal_losses'] = o_focal_losses[-1]
            loss_dict['s_dice_losses'] = s_dice_losses[-1]
            loss_dict['o_dice_losses'] = o_dice_losses[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for s_loss_cls_i, o_loss_cls_i, r_loss_cls_i, \
            s_loss_bbox_i, o_loss_bbox_i, \
            s_loss_iou_i, o_loss_iou_i, rel_vector_loss_i in zip(s_losses_cls[:-1], o_losses_cls[:-1], r_losses_cls[:-1],
                                          s_losses_bbox[:-1], o_losses_bbox[:-1],
                                          s_losses_iou[:-1], o_losses_iou[:-1], rel_vector_loss[:-1]):
            loss_dict[f'd{num_dec_layer}.s_loss_cls'] = s_loss_cls_i
            loss_dict[f'd{num_dec_layer}.o_loss_cls'] = o_loss_cls_i
            loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
            loss_dict[f'd{num_dec_layer}.s_loss_bbox'] = s_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.o_loss_bbox'] = o_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.s_loss_iou'] = s_loss_iou_i
            loss_dict[f'd{num_dec_layer}.o_loss_iou'] = o_loss_iou_i
            loss_dict[f'd{num_dec_layer}.rel_vector_loss'] = rel_vector_loss_i
            num_dec_layer += 1


        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        ## loss of scene graph
        # loss from the last decoder layer
        # loss_dict['r_loss_cls'] = r_losses_cls[-1]

        # # loss from other decoder layers
        # num_dec_layer = 0
        # for r_loss_cls_i in r_losses_cls[:-1]:
        #     loss_dict[f'd{num_dec_layer}.r_loss_cls'] = r_loss_cls_i
        #     num_dec_layer += 1
        return loss_dict


    def loss_single(self,
                    all_rel_out,
                    od_cls_scores,
                    od_bbox_preds,
                    mask_preds,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        ## before get targets
        num_imgs = od_cls_scores.size(0)
        # obj det&seg
        cls_scores_list = [od_cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [od_bbox_preds[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        # scene graph
        rel_out_list = [dict()]*num_imgs
        for key, value in all_rel_out.items():
            for i in range(num_imgs):
                rel_out_list[i][key] = value[i]

        rel_out_list_for_match = [dict()]*num_imgs
        for key, value in all_rel_out.items():
            for i in range(num_imgs):
                rel_out_list_for_match[i][key] = value[i][None]


        cls_reg_targets = self.get_targets(
             rel_out_list, rel_out_list_for_match, cls_scores_list,
            bbox_preds_list, mask_preds_list,  gt_rels_list, gt_bboxes_list,
            gt_labels_list, gt_masks_list, img_metas, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, num_total_od_pos, num_total_od_neg,
         mask_preds_list, s_labels_list, s_label_weights_list, 
         o_labels_list, o_label_weights_list, r_labels_list, r_label_weights_list, num_total_pos,
         num_total_neg, rel_vector_targets_list, rel_vector_weights_list,
        subject_bbox_targets_list, subject_bbox_weights_list,
        object_bbox_targets_list, object_bbox_weights_list,
        subject_mask_targets_list, subject_mask_pred_list,
        object_mask_targets_list, object_mask_pred_list
         ) = cls_reg_targets

        # obj det&seg
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)

        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        mask_targets = torch.cat(mask_targets_list, 0).float().flatten(1)

        mask_preds = torch.cat(mask_preds_list, 0).flatten(1)
        num_od_matches = mask_preds.shape[0]

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

        # mask loss
        focal_loss = self.focal_loss(mask_preds, mask_targets, num_od_matches)
        dice_loss = self.dice_loss(mask_preds, mask_targets, num_od_matches)

        # classification loss
        od_cls_scores = od_cls_scores.reshape(-1, self.cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_od_pos * 1.0 + \
            num_total_od_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                od_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(od_cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_od_pos = loss_cls.new_tensor([num_total_od_pos])
        num_total_od_pos = torch.clamp(reduce_mean(num_total_od_pos),
                                       min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, od_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        od_bbox_preds = od_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(od_bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_od_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(od_bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_od_pos)

        # scene graph

        s_labels = torch.cat(s_labels_list, 0)
        o_labels = torch.cat(o_labels_list, 0)
        r_labels = torch.cat(r_labels_list, 0)
        s_label_weights = torch.cat(s_label_weights_list, 0)
        o_label_weights = torch.cat(o_label_weights_list, 0)
        r_label_weights = torch.cat(r_label_weights_list, 0)

        # classification loss
        s_cls_scores = all_rel_out['pred_rel_obj_logits'].reshape(-1, self.sub_cls_out_channels)
        o_cls_scores = all_rel_out['pred_rel_sub_logits'].reshape(-1, self.obj_cls_out_channels)        
        r_cls_scores = all_rel_out['pred_rel_logits'].reshape(-1, self.rel_cls_out_channels)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                r_cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        s_loss_cls = self.sub_loss_cls(s_cls_scores,
                                       s_labels,
                                       s_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        o_loss_cls = self.obj_loss_cls(o_cls_scores,
                                       o_labels,
                                       o_label_weights,
                                       avg_factor=num_total_pos * 1.0)

        r_loss_cls = self.rel_loss_cls(r_cls_scores,
                                       r_labels,
                                       r_label_weights,
                                       avg_factor=cls_avg_factor)
        

        num_total_pos = r_loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos),
                                       min=1).item()
        # rel_entities_ware_loss
        rel_vector_targets = torch.cat(rel_vector_targets_list, 0)
        rel_vector_weights = torch.cat(rel_vector_weights_list, 0)
        rel_vector_preds = all_rel_out['pred_rel_vec'].reshape(-1, 4)

        rel_vector_loss = self.rel_vec_loss(rel_vector_preds, 
                            rel_vector_targets, 
                            rel_vector_weights, 
                            avg_factor=num_total_pos)

        subject_bbox_targets = torch.cat(subject_bbox_targets_list, 0)
        subject_bbox_weights = torch.cat(subject_bbox_weights_list, 0)
        subject_bbox_preds = all_rel_out['pred_rel_sub_box'].reshape(-1, 4)
        unscale_subject_bbox_preds = bbox_cxcywh_to_xyxy(subject_bbox_preds) * factor
        unscale_subject_bbox_targets = bbox_cxcywh_to_xyxy(subject_bbox_targets) * factor

        # regression IoU loss, defaultly GIoU loss
        loss_iou_sub = self.loss_iou(unscale_subject_bbox_preds,
                                 unscale_subject_bbox_targets,
                                 subject_bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox_sub = self.loss_bbox(subject_bbox_preds,
                                   subject_bbox_targets,
                                   subject_bbox_weights,
                                   avg_factor=num_total_pos)

        object_bbox_targets = torch.cat(object_bbox_targets_list, 0)
        object_bbox_weights = torch.cat(object_bbox_weights_list, 0)
        object_bbox_preds = all_rel_out['pred_rel_obj_box'].reshape(-1, 4)
        unscale_object_bbox_preds = bbox_cxcywh_to_xyxy(object_bbox_preds) * factor
        unscale_object_bbox_targets = bbox_cxcywh_to_xyxy(object_bbox_targets) * factor

        # regression IoU loss, defaultly GIoU loss
        loss_iou_obj = self.loss_iou(unscale_object_bbox_preds,
                                 unscale_object_bbox_targets,
                                 object_bbox_weights,
                                 avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox_obj = self.loss_bbox(object_bbox_preds,
                                   object_bbox_targets,
                                   object_bbox_weights,
                                   avg_factor=num_total_pos)

        subject_mask_targets = torch.cat(subject_mask_targets_list, 0).float().flatten(1)
        subject_mask_preds = torch.cat(subject_mask_pred_list, 0).flatten(1)

        object_mask_targets = torch.cat(object_mask_targets_list, 0).float().flatten(1)
        object_mask_preds = torch.cat(object_mask_pred_list, 0).flatten(1)

        num_matches = subject_mask_preds.shape[0]

        # mask loss
        # s_focal_loss = self.sub_focal_loss(s_mask_preds,s_mask_targets,num_matches)
        s_dice_loss = self.sub_dice_loss(
            subject_mask_preds, subject_mask_targets,
            num_matches)

        # o_focal_loss = self.obj_focal_loss(o_mask_preds,o_mask_targets,num_matches)
        o_dice_loss = self.obj_dice_loss(
            object_mask_preds, object_mask_targets,
            num_matches) 

        return loss_cls, loss_bbox, loss_iou, dice_loss, focal_loss, \
        r_loss_cls, s_loss_cls, o_loss_cls, rel_vector_loss, \
        loss_iou_sub, loss_iou_obj,loss_bbox_sub,loss_bbox_obj, \
        s_dice_loss, o_dice_loss

    def get_targets(self,
                    rel_out_list,
                    rel_out_list_for_match,
                    cls_scores_list,
                    bbox_preds_list,
                    mask_preds_list,
                    gt_rels_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, od_pos_inds_list, od_neg_inds_list,
         mask_preds_list, s_labels_list, s_label_weights_list, o_labels_list, o_label_weights_list,
         r_labels_list, r_label_weights_list, pos_inds_list, neg_inds_list, 
         rel_vector_targets_list, rel_vector_weights_list,
        subject_bbox_targets_list, subject_bbox_weights_list,
        object_bbox_targets_list, object_bbox_weights_list,
        subject_mask_targets_list, subject_mask_pred_list,
        object_mask_targets_list, object_mask_pred_list
         ) = multi_apply(
             self._get_target_single, rel_out_list,rel_out_list_for_match,
             cls_scores_list, bbox_preds_list, mask_preds_list,
             gt_rels_list, gt_bboxes_list, gt_labels_list, gt_masks_list,
             img_metas, gt_bboxes_ignore_list)

        num_total_od_pos = sum((inds.numel() for inds in od_pos_inds_list))
        num_total_od_neg = sum((inds.numel() for inds in od_neg_inds_list))

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, num_total_od_pos,
                num_total_od_neg, mask_preds_list, 
                s_labels_list, s_label_weights_list, o_labels_list, o_label_weights_list,
                r_labels_list, r_label_weights_list, num_total_pos, num_total_neg,
                rel_vector_targets_list, rel_vector_weights_list,
                subject_bbox_targets_list, subject_bbox_weights_list,
                object_bbox_targets_list, object_bbox_weights_list,
                subject_mask_targets_list, subject_mask_pred_list,
                object_mask_targets_list, object_mask_pred_list)
    
    def convert_tgt_format(self, gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks, img_meta):
        target = {}
        h, w, _ = img_meta['img_shape']

        boxes_xyxy = gt_bboxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(gt_bboxes.device)
        boxes = bbox_xyxy_to_cxcywh(boxes_xyxy).to(gt_bboxes.device)

        target["boxes"] = boxes
        target["boxes_init"] = bbox_xyxy_to_cxcywh(gt_bboxes).to(gt_bboxes.device)

        target["boxes_xyxy_init"] = gt_bboxes
        target["boxes_xyxy"] = boxes_xyxy.to(gt_bboxes.device)

        target["labels"] = gt_labels

        target["masks"] = gt_masks

        target["size"] = torch.tensor([h, w], device=gt_bboxes.device)

        ## relationship parts
        target["rel_labels"] = gt_rels[...,2].long()
        target["rel_label_no_mask"] = gt_rels[...,2].long()

        rel_pair_tensor = gt_rels.long()
        target["gt_rel_pair_tensor"] = rel_pair_tensor

        target["rel_vector"] = torch.cat(
                (boxes[rel_pair_tensor[:, 0], :2], boxes[rel_pair_tensor[:, 1], :2]),
                dim=1,
            ).to(
                gt_bboxes.device
            )  # Kx2 + K x2 => K x 4

        return [target]
        

    def _get_target_single(self,
                           rel_out,
                           rel_out_for_match,
                           cls_score,
                           bbox_pred,
                           mask_preds,
                           gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks,
                           img_meta,
                           gt_bboxes_ignore=None):

        assert len(gt_masks) == len(gt_bboxes)

        ###### obj det&seg
        num_bboxes = bbox_pred.size(0)
        assert len(gt_masks) == len(gt_bboxes)

        # assigner and sampler, only return human&object assign result
        # could using MaskDINO to replace
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
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[od_pos_inds] = 1.0

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

        ##### scene graph
        num_rels = rel_out['pred_rel_logits'].size(0)
        # separate human boxes and object boxes from gt_bboxes and generate labels
        gt_sub_bboxes = []
        gt_obj_bboxes = []
        gt_sub_labels = []
        gt_obj_labels = []
        gt_rel_labels = []
        gt_sub_ids = []
        gt_obj_ids = []
        if self.use_mask:
            gt_sub_masks = []
            gt_obj_masks = []

        for rel_id in range(gt_rels.size(0)):
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


        ###### SGTR
        targets = self.convert_tgt_format(gt_rels,
                           gt_bboxes,
                           gt_labels,
                           gt_masks, img_meta)
        s_assign_result, o_assign_result = self.rel_assigner.assign(
            rel_out_for_match, targets, gt_sub_ids, gt_obj_ids
        )

        # ########################################
        # #### overwrite relation labels above####
        # ########################################
        # # assigner and sampler for relation-oriented id match
        # s_assign_result, o_assign_result = self.id_assigner.assign(
        #     subject_scores, object_scores, r_cls_score, gt_sub_ids, gt_obj_ids,
        #     gt_rel_labels, img_meta, gt_bboxes_ignore)

        s_sampling_result = self.sampler.sample(s_assign_result, rel_out['pred_rel_sub_box'],
                                                gt_sub_bboxes)
        o_sampling_result = self.sampler.sample(o_assign_result, rel_out['pred_rel_obj_box'],
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


        #match bbox targets
        subject_bbox_targets = torch.zeros_like(rel_out['pred_rel_sub_box'])
        subject_bbox_weights = torch.zeros_like(rel_out['pred_rel_sub_box'])
        subject_bbox_weights[pos_inds] = 1.0
        gt_subject_bboxes = gt_sub_bboxes[s_sampling_result.pos_assigned_gt_inds]
        gt_subject_bboxes = bbox_xyxy_to_cxcywh(gt_subject_bboxes / factor)
        subject_bbox_targets[pos_inds] = gt_subject_bboxes

        object_bbox_targets = torch.zeros_like(rel_out['pred_rel_obj_box'])
        object_bbox_weights = torch.zeros_like(rel_out['pred_rel_obj_box'])
        object_bbox_weights[pos_inds] = 1.0
        gt_object_bboxes = gt_obj_bboxes[o_sampling_result.pos_assigned_gt_inds]
        gt_object_bboxes = bbox_xyxy_to_cxcywh(gt_object_bboxes / factor)
        object_bbox_targets[pos_inds] = gt_object_bboxes

        #match mask targets
        if self.use_mask:

            gt_sub_masks = torch.cat(gt_sub_masks, axis=0).type_as(gt_masks[0])
            gt_obj_masks = torch.cat(gt_obj_masks, axis=0).type_as(gt_masks[0])

            assert gt_sub_masks.size() == gt_obj_masks.size()
            # mask targets for subjects and objects
            subject_mask_targets = gt_sub_masks[
                s_sampling_result.pos_assigned_gt_inds,
                ...]  
            s_mask_preds = rel_out['pred_rel_sub_mask'][pos_inds]
            

            object_mask_targets = gt_obj_masks[
                o_sampling_result.pos_assigned_gt_inds, ...]
            o_mask_preds = rel_out['pred_rel_obj_mask'][pos_inds]
            
            s_mask_preds = interpolate(s_mask_preds[:, None],
                                       size=gt_sub_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)

            o_mask_preds = interpolate(o_mask_preds[:, None],
                                       size=gt_obj_masks.shape[-2:],
                                       mode='bilinear',
                                       align_corners=False).squeeze(1)
        else:
            subject_mask_targets = None
            s_mask_preds = None
            object_mask_targets = None
            o_mask_preds = None

        #match rel_vector targets
        rel_vector_targets = torch.zeros_like(rel_out['pred_rel_sub_box'])
        rel_vector_weights = torch.zeros_like(rel_out['pred_rel_sub_box'])
        rel_vector_weights[pos_inds] = 1.0
        rel_vector_targets[pos_inds] = targets[0]["rel_vector"][o_sampling_result.pos_assigned_gt_inds]

        # Aux Entities Indexing
        # num_ent_pairs = self.rel_assigner.num_entities_pairing_train

        # topk_pred = self.rel_assigner.num_entities_pairing
        # topk_all = num_ent_pairs

        # def cal_entities_index(matching_matrix, role_name):
        #     role_id = 0 if role_name == "sub" else 1
        #     num_ent_pairs = self.rel_assigner.num_entities_pairing_train
        #     num_ent_pairs = num_ent_pairs if num_ent_pairs < matching_matrix.shape[-1] else matching_matrix.shape[-1]
        #     matching_ref_mat_collect = []

        #     _, rel_match_pred_ent_ids = torch.topk(
        #         matching_matrix, num_ent_pairs, dim=-1
        #     )

        #     valid_rel_idx = (
        #         []
        #     )  # the index of relationship prediction match with the and also has the fg indexing matching

        #     entities_match_cache = []
        #     if self.entities_match_cache is None:
        #         self.entities_match_cache = dict()
            
        #     if self.entities_match_cache.get(role_name) is None:
        #         pred_boxes = bbox_cxcywh_to_xyxy(rel_out['pred_boxes'])
        #         tgt_boxes = bbox_cxcywh_to_xyxy(targets[0]['boxes'])

        #         box_giou = generalized_box_iou(pred_boxes, tgt_boxes).detach()
        #         box_match_idx = box_giou >= 0.7

        #         inst_loc_hit_idx = torch.nonzero(box_match_idx)
        #         pred_box_loc_hit_idx = inst_loc_hit_idx[:, 0]
        #         gt_box_loc_hit_idx = inst_loc_hit_idx[:, 1]

        #         loc_box_matching_results = defaultdict(set)
        #         for idx in range(len(gt_box_loc_hit_idx)):
        #             loc_box_matching_results[gt_box_loc_hit_idx[idx].item()].add(
        #                 pred_box_loc_hit_idx[idx].item()
        #             )
                
        #         pred_labels = rel_out['pred_logits'][:,:-1].max(-1)[1]
        #         tgt_labels = targets[0]['labels']
        #         gt_det_label_to_cmp = pred_labels[pred_box_loc_hit_idx]
        #         pred_det_label_to_cmp = tgt_labels[gt_box_loc_hit_idx]

        #         pred_det_hit_stat = pred_det_label_to_cmp == gt_det_label_to_cmp

        #         pred_box_det_hit_idx = pred_box_loc_hit_idx[pred_det_hit_stat]
        #         gt_box_det_hit_idx = gt_box_loc_hit_idx[pred_det_hit_stat]

        #         det_box_matching_results = defaultdict(set)
        #         for idx in range(len(gt_box_det_hit_idx)):
        #             det_box_matching_results[gt_box_det_hit_idx[idx].item()].add(
        #                 pred_box_det_hit_idx[idx].item()
        #             )
        #         # merge the entities set matching results
        #         # det_box_matching_results = defaultdict(set)
        #         if self.bbox_assigner is not None:
        #             gt_ent_idxs = sampling_result.pos_assigned_gt_inds
        #             pred_ent_idxs = od_pos_inds
        #             for idx in range(len(gt_ent_idxs)):
        #                 gt_ent_idx = gt_ent_idxs[idx].item()
        #                 pred_ent_idx = pred_ent_idxs[idx].item()
        #                 det_box_matching_results[gt_ent_idx].add(
        #                     pred_ent_idx
        #                 )
        #         # loc_box_matching_results = det_box_matching_results

        #         entities_match_cache.append({
        #             'loc_box_matching_results': loc_box_matching_results,
        #             'det_box_matching_results': det_box_matching_results
        #         })

        #     else:
        #         loc_box_matching_results = self.entities_match_cache[role_name][
        #             'loc_box_matching_results']
        #         det_box_matching_results = self.entities_match_cache[role_name][
        #             'det_box_matching_results']
            
        #     ent_pred_idx = rel_match_pred_ent_ids[pos_inds,:]

        #     rel_pred_num = rel_out['pred_rel_obj_logits'].shape[0]
        #     ent_pred_num = rel_out['pred_logits'].shape[0]
        #     matching_ref_mat = torch.zeros(
        #         (rel_pred_num, ent_pred_num), device = rel_out['pred_boxes'].device
        #     ).long()

        #     matching_ref_mat_loc = torch.zeros(
        #         (rel_pred_num, ent_pred_num), device=rel_out["pred_boxes"].device
        #     ).long()

        #     gt_rel_pair_tensor = targets[0]["gt_rel_pair_tensor"]
        #     for idx, (rel_idx, gt_rel_idx) in enumerate(
        #         zip(pos_inds.cpu().numpy(), s_sampling_result.pos_assigned_gt_inds.cpu().numpy())
        #     ):
        #         gt_box_idx = gt_rel_pair_tensor[gt_rel_idx, role_id].item()
        #         for ent_idx in ent_pred_idx[idx].cpu().numpy():
        #             if ent_idx in loc_box_matching_results[gt_box_idx]:
        #                 matching_ref_mat_loc[rel_idx, ent_idx] = 1
        #             if ent_idx in det_box_matching_results[gt_box_idx]:
        #                 matching_ref_mat[rel_idx, ent_idx] = 1
            
        #     matching_ref_mat_collect.append(matching_ref_mat.long())

        #     valid_rel_idx.append(pos_inds.unique())

        #     # if self.entities_match_cache.get(role_name) is None:
        #     #     self.entities_match_cache[role_name] = entities_match_cache

        #     matching_ref_mat = torch.cat(matching_ref_mat_collect, dim=0).float()
        #     valid_idx = matching_ref_mat.sum(-1) > 0
        #     # print(valid_idx.sum()/ valid_idx.shape[0])
        #     # bs*rel_num_pred, ent_num_pred
        #     matching_matrix = matching_matrix.view(-1, matching_matrix.shape[-1])

        #     return matching_matrix[valid_idx], matching_ref_mat[valid_idx]

        # filtering unmatched subject/object id predictions
        # gt_subject_ids = gt_subject_ids[pos_inds]
        # gt_subject_ids_res = torch.zeros_like(gt_subject_ids)
        # for idx, gt_subject_id in enumerate(gt_subject_ids):
        #     gt_subject_ids_res[idx] = ((od_pos_inds == gt_subject_id).nonzero(
        #         as_tuple=True)[0])
        # gt_subject_ids = gt_subject_ids_res

        # gt_object_ids = gt_object_ids[pos_inds]
        # gt_object_ids_res = torch.zeros_like(gt_object_ids)
        # for idx, gt_object_id in enumerate(gt_object_ids):
        #     gt_object_ids_res[idx] = ((od_pos_inds == gt_object_id).nonzero(
        #         as_tuple=True)[0])
        # gt_object_ids = gt_object_ids_res

        # filtered_subject_scores = rel_out['pred_rel_sub_logits'][pos_inds]
        # filtered_subject_scores = filtered_subject_scores[:, od_pos_inds]
        # filtered_object_scores = rel_out['pred_rel_obj_logits'][pos_inds]
        # filtered_object_scores = filtered_object_scores[:, od_pos_inds]

        # label targets
        s_labels = gt_sub_bboxes.new_full(
            (num_rels, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes]  as background
        s_labels[pos_inds] = gt_sub_labels[
            s_sampling_result.pos_assigned_gt_inds]
        # s_label_weights = gt_sub_bboxes.new_ones(num_rels)
        s_label_weights = gt_sub_bboxes.new_zeros(num_rels)
        s_label_weights[pos_inds] = 1.0

        o_labels = gt_obj_bboxes.new_full(
            (num_rels, ), self.num_classes,
            dtype=torch.long)  ### 0-based, class [num_classes] as background
        o_labels[pos_inds] = gt_obj_labels[
            o_sampling_result.pos_assigned_gt_inds]
        # o_label_weights = gt_sub_bboxes.new_ones(num_rels)
        o_label_weights = gt_obj_bboxes.new_zeros(num_rels)
        o_label_weights[pos_inds] = 1.0

        r_labels = gt_obj_bboxes.new_full((num_rels, ), 0,
                                          dtype=torch.long)  ### 1-based

        assert 0 not in gt_rel_labels[o_sampling_result.pos_assigned_gt_inds]

        r_labels[pos_inds] = gt_rel_labels[
            o_sampling_result.pos_assigned_gt_inds]
        r_label_weights = gt_obj_bboxes.new_ones(num_rels)

        return (labels, label_weights, bbox_targets, bbox_weights,
                mask_targets, od_pos_inds, od_neg_inds, mask_preds, 
                s_labels, s_label_weights, o_labels, o_label_weights,
                r_labels, r_label_weights, pos_inds, neg_inds, 
                rel_vector_targets, rel_vector_weights,
                subject_bbox_targets, subject_bbox_weights,
                object_bbox_targets, object_bbox_weights,
                subject_mask_targets, s_mask_preds,
                object_mask_targets, o_mask_preds
                )  ###return the interpolated predicted masks



    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_rels,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
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

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_rels, gt_bboxes, gt_labels, gt_masks,
                                  img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def get_bboxes(self, rel_aux_out, semantic_predictions, ent_predictions, img_metas, rescale=False):

        result_list = []
        for img_id in range(len(img_metas)):
            semantic_prediction_single = dict()
            for key, value in semantic_predictions.items():
                semantic_prediction_single[key] = value[-1]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            triplets = self._get_bboxes_single(semantic_prediction_single, img_shape, scale_factor)
            result_list.append(triplets)

        return result_list
    
    def _get_bboxes_single(self, semantic_prediction_single, img_shape, scale_factor, rescale=False):
        
        mask_size = (round(img_shape[0] / scale_factor[1]),
                     round(img_shape[1] / scale_factor[0]))
        # PostProcess Ent Predictions
        ent_logits = semantic_prediction_single['pred_logits']
        ent_output_bbox = semantic_prediction_single['pred_boxes']

        # do not consider mask match now
        ent_mask = semantic_prediction_single['mask']

        ent_prob = F.softmax(ent_logits, -1)
        ent_scores, ent_labels = ent_prob[..., :-1].max(-1)

        ent_bbox_norm = bbox_cxcywh_to_xyxy(ent_output_bbox)
        ent_bboxes = torch.zeros_like(ent_bbox_norm)

        ent_bboxes[:, 0::2] = ent_bbox_norm[:, 0::2] * img_shape[1]
        ent_bboxes[:, 1::2] = ent_bbox_norm[:, 1::2] * img_shape[0]
        ent_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        ent_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        ent_masks = F.interpolate(ent_mask.unsqueeze(1), size=mask_size).squeeze(1)
        ent_masks_logits = ent_masks

        ent_masks = torch.sigmoid(ent_masks) > 0.85

        if rescale:
            ent_bboxes /= ent_bboxes.new_tensor(scale_factor)

        ent_det_res = {'scores': ent_scores, "labels": ent_labels, "boxes":ent_bboxes, \
                        "prob": ent_prob, "boxes_norm": ent_bbox_norm, "masks": ent_masks,\
                             "masks_logits": ent_masks_logits}

        # PostProcess Rel Prediction
        post_proc_filtering = True
        overlap_thres = 0.8
        max_proposal_pairs = 300
        pred_rel_logits = semantic_prediction_single['pred_rel_logits']
        pred_rel_vec = semantic_prediction_single['pred_rel_vec']

        device = pred_rel_vec.device

        if self.rel_loss_cls.use_sigmoid:
            pred_rel_probs = torch.sigmoid(pred_rel_logits)
        else:
            pred_rel_probs = torch.softmax(pred_rel_logits, -1)

        pred_rel_vec[:,0::2] = pred_rel_vec[:,0::2] * img_shape[1]
        pred_rel_vec[:,1::2] = pred_rel_vec[:,1::2] * img_shape[0]

        scale_fct = torch.as_tensor([img_shape[1], img_shape[0], img_shape[1], img_shape[0]]).to(device)

        ent_score = ent_det_res["scores"]
        ent_label = ent_det_res["labels"]
        ent_box = ent_det_res["boxes"]
        ent_box_normed = ent_det_res["boxes_norm"]
        ent_mask = ent_det_res["masks"]

        ent_box_cnter_normed = ent_box_normed[..., :2]       
        rel_vec_flat_normed =  semantic_prediction_single['pred_rel_vec']

        ent_box_cnter = bbox_xyxy_to_cxcywh(ent_box)[:, :2]

        rel_vec_flat = pred_rel_vec

        if self.entities_aware_head:
            pred_rel_obj_box = (
                bbox_cxcywh_to_xyxy(semantic_prediction_single['pred_rel_obj_box']) * scale_fct[None, :]
            )
            pred_rel_obj_box = torch.squeeze(pred_rel_obj_box)

            pred_rel_sub_box = (
                bbox_cxcywh_to_xyxy(semantic_prediction_single['pred_rel_sub_box']) * scale_fct[None, :]
            )
            pred_rel_sub_box = torch.squeeze(pred_rel_sub_box)

            pred_rel_obj_mask = F.interpolate(semantic_prediction_single['pred_rel_obj_mask'].unsqueeze(1),\
                                 size=mask_size).squeeze(1)

            pred_rel_obj_mask = torch.sigmoid(pred_rel_obj_mask) > 0.85

            pred_rel_sub_mask = F.interpolate(semantic_prediction_single['pred_rel_obj_mask'].unsqueeze(1),\
                                 size=mask_size).squeeze(1)

            pred_rel_sub_mask = torch.sigmoid(pred_rel_sub_mask) > 0.85

            pred_rel_sub_dist = F.softmax(
                semantic_prediction_single["pred_rel_sub_logits"], dim=-1
            )[..., :-1]
            pred_rel_sub_score, pred_rel_sub_label = torch.max(
                pred_rel_sub_dist, dim=-1
            )

            pred_rel_obj_dist = F.softmax(
                semantic_prediction_single["pred_rel_obj_logits"], dim=-1
            )[..., :-1]
            pred_rel_obj_score, pred_rel_obj_label = torch.max(
                pred_rel_obj_dist, dim=-1
            )

            if self.num_classes == ent_det_res["prob"].shape[-1]:
                ent_prob = ent_det_res["prob"]
            else:
                ent_prob = ent_det_res["prob"][..., :-1]

            match_cost_details = {}

            ent_num = len(ent_prob)
            rel_num = len(pred_rel_sub_box)

            match_scr_sub = torch.zeros((rel_num, ent_num), device=device)
            match_scr_obj = torch.zeros((rel_num, ent_num), device=device)

            if self.indexing_module_type == "rule_base":
                (
                    match_scr_sub_r,
                    match_scr_obj_r,
                    match_cost_details_r,
                ) = get_matching_scores_entities_aware(
                    s_cetr=ent_box_cnter,
                    o_cetr=ent_box_cnter,
                    s_scores=ent_score,
                    o_scores=ent_score,
                    rel_vec=rel_vec_flat,
                    s_cetr_normed=ent_box_cnter_normed,
                    o_cetr_normed=ent_box_cnter_normed,
                    rel_vec_normed=rel_vec_flat_normed,
                    ent_box=ent_box,
                    ent_mask=ent_mask.to(torch.float).flatten(1),
                    ent_box_normed=bbox_cxcywh_to_xyxy(ent_box_normed),
                    s_dist=ent_prob,
                    o_dist=ent_prob,
                    rel_ent_s_box=pred_rel_sub_box,
                    rel_ent_o_box=pred_rel_obj_box,
                    rel_ent_s_mask=pred_rel_sub_mask.to(torch.float).flatten(1),
                    rel_ent_o_mask=pred_rel_obj_mask.to(torch.float).flatten(1),
                    rel_s_dist=pred_rel_sub_dist,
                    rel_o_dist=pred_rel_obj_dist,
                    normed_rel_vec_dist=self.normed_rel_vec_dist,
                )

                match_scr_sub = match_scr_sub + match_scr_sub_r
                match_scr_obj = match_scr_obj + match_scr_obj_r
                match_cost_details.update(match_cost_details_r)
            else:
                assert NotImplementedError

            # num_rel_queries, num_ent_queries
            # one relationship prediction may matching with multiple entities
            init_max_match_ent = None
            if init_max_match_ent is None:
                init_max_match_ent = self.num_entities_pairing
            max_match_ent = (
                init_max_match_ent
                if match_scr_sub.shape[-1] > init_max_match_ent
                else match_scr_sub.shape[-1]
            )
            rel_match_sub_scores, rel_match_sub_ids = torch.topk(
                match_scr_sub, max_match_ent, dim = -1
            )

            # num_rel_queries; num_rel_queries
            max_match_ent = (
                init_max_match_ent
                if match_scr_obj.shape[-1] > init_max_match_ent
                else match_scr_obj.shape[-1]
            )
            rel_match_obj_scores, rel_match_obj_ids = torch.topk(
                match_scr_obj, max_match_ent, dim=-1
            )

            if self.rel_loss_cls.use_sigmoid:
                # Focal loss
                pred_rel_prob = pred_rel_probs
                num_q, cls_num = pred_rel_prob.shape

                pred_num_per_edge = self.num_pred_edges

                topk = num_q * pred_num_per_edge

                topk_values_all, topk_indexes_all = torch.sort(
                    pred_rel_prob.reshape(-1), dim=-1, descending=True
                )  # num_query * num_cls

                pred_rel_prob = topk_values_all[
                                :topk
                                ]  # scores for each relationship predictions
                # (num of query * pred_num_per_edge)
                total_pred_idx = topk_indexes_all[:topk] // cls_num
                #pytorch >=1.8 torch.div(topk_indexes_all[:topk], cls_num, rounding_mode='trunc')
                pred_rel_labels = topk_indexes_all[:topk] % cls_num
                pred_rel_labels += 1

                # =>  (num_queries * num_pred_rel,  num_group_entities)
                rel_match_sub_ids = rel_match_sub_ids[total_pred_idx]
                rel_match_obj_ids = rel_match_obj_ids[total_pred_idx]

                total_pred_idx = (
                    total_pred_idx.contiguous()
                        .unsqueeze(1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_prob = (
                    pred_rel_prob.reshape(-1, 1).repeat(1, max_match_ent).view(-1)
                )
                pred_rel_labels = (
                    pred_rel_labels.reshape(-1, 1)
                        .repeat(1, max_match_ent)
                        .view(-1)
                        .unsqueeze(1)
                )
            
            else:
                # CrossEntropy

                pred_rel_prob = pred_rel_probs[:, 1:]

                num_rel_queries = pred_rel_prob.shape[0]

                pred_num_per_edge = 1

                pred_rel_prob, pred_rel_labels = pred_rel_prob.sort(-1, descending=True)

                pred_rel_labels += 1

                pred_rel_labels = pred_rel_labels[:, :pred_num_per_edge]
                pred_rel_prob = pred_rel_prob[:, :pred_num_per_edge]

                # (num_queries * num_categories)
                # => (num_queries * num_categories, 1)
                # =>  (num_queries * num_pred_rel * num_group_entities)
                pred_rel_prob = pred_rel_prob.reshape(-1, 1)
                pred_rel_prob = pred_rel_prob.repeat(1, max_match_ent).view(-1)

                # (num_queries * num_categories)
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                pred_rel_labels = pred_rel_labels.reshape(-1, 1)
                pred_rel_labels = (
                    pred_rel_labels.repeat(1, max_match_ent).view(-1).unsqueeze(1)
                )

                total_pred_idx = (
                    torch.arange(num_rel_queries)
                        .unsqueeze(1)
                        .repeat(1, pred_num_per_edge)
                )
                # =>  (num_queries * num_pred_rel * num_group_entities, 1)
                total_pred_idx = total_pred_idx.reshape(-1, 1)
                total_pred_idx = (
                    total_pred_idx.repeat(1, max_match_ent)
                        .view(-1)
                        .contiguous()
                        .unsqueeze(1)
                )

            rel_match_sub_ids_flat = rel_match_sub_ids.view(-1).contiguous()
            rel_match_obj_ids_flat = rel_match_obj_ids.view(-1).contiguous()

            rel_trp_scores = (
                pred_rel_prob
                * ent_score[rel_match_sub_ids_flat]
                * ent_score[rel_match_obj_ids_flat]
            ) # (num_rel_queries, 1)

            if self.use_entities_indexing_ranking:
                rel_match_sub_scores_flat = rel_match_sub_scores.view(-1).contiguous()
                rel_match_obj_scores_flat = rel_match_obj_scores.view(-1).contiguous()
                matching_score = (
                        rel_match_sub_scores_flat * rel_match_obj_scores_flat
                ).view(-1)
                rel_trp_scores = rel_trp_scores * matching_score  # (num_queries,  1)                

            # (num_queries, num_categories)
            # =>  (num_queries * num_pred_rel * num_group_entities, 1)
            pred_rel_pred_score = pred_rel_prob.contiguous().unsqueeze(1)
            rel_trp_scores = rel_trp_scores.unsqueeze(1)

            rel_match_sub_ids2cat = rel_match_sub_ids_flat.unsqueeze(1)
            rel_match_obj_ids2cat = rel_match_obj_ids_flat.unsqueeze(1)

            SUB_IDX = 0
            OBJ_IDX = 1
            REL_LABEL = 2
            REL_TRP_SCR = 3
            REL_PRED_SCR = 4
            INIT_PROP_IDX = 5

            pred_rel_triplet = torch.cat(
                (
                    rel_match_sub_ids2cat.long().to(device),
                    rel_match_obj_ids2cat.long().to(device),
                    pred_rel_labels.long().to(device),
                    rel_trp_scores.to(device),
                    pred_rel_pred_score.to(device),
                    total_pred_idx.to(device),
                ),
                1,
            )
            
            pred_rel_triplet = pred_rel_triplet.to("cpu")
            init_pred_rel_triplet = pred_rel_triplet.to("cpu")
            # init_pred_rel_triplet = pred_rel_triplet.to(device)

            # removed the self connection
            self_iou = generalized_box_iou(
                bbox_cxcywh_to_xyxy(ent_box), bbox_cxcywh_to_xyxy(ent_box)
            )

            ent_mask_for_iou = ent_mask.to(torch.float).flatten(1)
            self_mask_iou = torch.ones((ent_mask_for_iou.shape[0],ent_mask_for_iou.shape[0])).to(device)
            for index in range(ent_mask_for_iou.shape[0]):
                self_mask_iou[index, :] = ent_mask_for_iou[index:index+1,:].mm(ent_mask_for_iou.transpose(0,1)) / \
                    ((ent_mask_for_iou[index:index+1,:] + ent_mask_for_iou)>0).sum(-1)

            # self_mask_iou = ent_mask_for_iou.mm(ent_mask_for_iou.transpose(0,1))/ \
            #     ((ent_mask_for_iou[:,None].repeat(1,ent_mask_for_iou.shape[0],1) + ent_mask_for_iou[None,:].repeat(ent_mask_for_iou.shape[0], 1, 1))>0).sum(-1)
            non_self_conn_idx = (rel_match_obj_ids_flat - rel_match_sub_ids_flat) != 0
            sub_obj_iou_check = self_iou[pred_rel_triplet[:, SUB_IDX].long(), pred_rel_triplet[:, OBJ_IDX].long()] < 0.95
            sub_obj_mask_check = self_mask_iou[pred_rel_triplet[:, SUB_IDX].long(), pred_rel_triplet[:, OBJ_IDX].long()] < 0.85
            non_self_conn_idx = torch.logical_and(non_self_conn_idx, sub_obj_iou_check)
            non_self_conn_idx = torch.logical_and(non_self_conn_idx, sub_obj_mask_check)

            # first stage filtering
            if post_proc_filtering:
                pred_rel_triplet = init_pred_rel_triplet[non_self_conn_idx]

                _, top_rel_idx = torch.sort(
                    pred_rel_triplet[:, REL_TRP_SCR], descending=True
                )
                top_rel_idx = top_rel_idx[:4096]

                pred_rel_triplet = pred_rel_triplet[top_rel_idx]
            
            # Why cpu
            ent_label = ent_det_res["labels"].detach().cpu()
            ent_box = ent_det_res["boxes"]
            ent_score = ent_det_res["scores"].detach().cpu()

            self_iou = self_iou.detach().cpu()
            self_mask_iou = self_mask_iou.detach().cpu()

            sub_idx = pred_rel_triplet[:, SUB_IDX].long().detach().cpu()
            obj_idx = pred_rel_triplet[:, OBJ_IDX].long().detach().cpu()

            rel_pred_label = (
                pred_rel_triplet[:, REL_LABEL].long().detach().cpu()
            )
            rel_pred_score = pred_rel_triplet[:, REL_TRP_SCR].detach().cpu()

            def rel_prediction_filtering(pred_rel_triplet):
                """
                Args:
                    pred_idx_set:
                    new_come_pred_idx:
                Returns:
                """
                pred_idx_set = []
                for new_come_pred_idx in range(len(pred_rel_triplet)):

                    new_come_sub_idx = sub_idx[new_come_pred_idx]
                    new_come_obj_idx = obj_idx[new_come_pred_idx]

                    new_come_sub_label = ent_label[new_come_sub_idx]
                    new_come_obj_label = ent_label[new_come_obj_idx]

                    new_come_pred_label = rel_pred_label[new_come_pred_idx]
                    new_come_pred_score = rel_pred_score[new_come_pred_idx] * ent_score[new_come_sub_idx] * ent_score[new_come_obj_idx]

                    pred_idx = torch.Tensor(pred_idx_set).long()
                    curr_sub_idx = sub_idx[pred_idx]
                    curr_obj_idx = obj_idx[pred_idx]

                    curr_sub_label = ent_label[curr_sub_idx]
                    curr_obj_label = ent_label[curr_obj_idx]

                    curr_pred_label = rel_pred_label[pred_idx]
                    curr_pred_score = rel_pred_score[pred_idx] * ent_score[curr_sub_idx] * ent_score[curr_obj_idx]

                    entities_indx_match = torch.logical_and(
                        curr_sub_idx == new_come_sub_idx,
                        curr_obj_idx == new_come_obj_idx
                    )

                    new_come_sub_idx = (torch.ones(len(pred_idx)) * new_come_sub_idx).long()
                    new_come_obj_idx = (torch.ones(len(pred_idx)) * new_come_obj_idx).long()

                    sub_iou = self_iou[new_come_sub_idx, curr_sub_idx]
                    obj_iou = self_iou[new_come_obj_idx, curr_obj_idx]

                    entities_pred_match = torch.logical_and(
                            torch.logical_and(sub_iou > overlap_thres, obj_iou > overlap_thres),
                            torch.logical_and(curr_sub_label == new_come_sub_label, curr_obj_label == new_come_obj_label)
                    )

                    sub_mask_iou = self_mask_iou[new_come_sub_idx, curr_sub_idx]
                    obj_mask_iou = self_mask_iou[new_come_obj_idx, curr_obj_idx]

                    entities_pred_match = torch.logical_and(
                            torch.logical_and(sub_mask_iou > 0.5, obj_mask_iou > 0.5),
                            entities_pred_match
                            #torch.logical_and(curr_sub_label == new_come_sub_label, curr_obj_label == new_come_obj_label)
                    )
                    # box iou > 0.8 and mask iou > 0.5, prevent stuff

                    entity_match = torch.logical_or(entities_pred_match, entities_indx_match)
                    

                    if entity_match.any():
                        pred_match = curr_pred_label == new_come_pred_label
                        rel_match = torch.logical_and(entity_match, pred_match)

                        if rel_match.any():
                            is_existed = new_come_pred_score < curr_pred_score[rel_match]
                            if not is_existed.any():
                                pred_idx_set.append(new_come_pred_idx)
                        else:
                            pred_idx_set.append(new_come_pred_idx)
                        
                    else:
                        pred_idx_set.append(new_come_pred_idx)

                pred_idx_set = torch.Tensor(pred_idx_set).long().to(device)
                bin_mask = torch.zeros((pred_rel_triplet.shape[0]), dtype=torch.bool).to(
                    device
                )
                bin_mask[pred_idx_set] = True
                pred_rel_triplet_selected = pred_rel_triplet[bin_mask]

                return pred_rel_triplet_selected

            if post_proc_filtering and overlap_thres > 0:
                pred_rel_triplet_selected = rel_prediction_filtering(
                    pred_rel_triplet
                )
            else:
                pred_rel_triplet_selected = pred_rel_triplet
                non_max_suppressed_idx = None

            # top K selection
            _, top_rel_idx = torch.sort(
                pred_rel_triplet_selected[:, REL_TRP_SCR], descending=True
            )
            pred_rel_triplet_selected = pred_rel_triplet_selected[
                top_rel_idx[:max_proposal_pairs]
            ]

            def res2dict(pred_rel_triplet):
                ret = {
                    "rel_trp": pred_rel_triplet[:, :3].long(),
                    "rel_pred_label": pred_rel_triplet[:, REL_LABEL].long(),
                    "rel_score": pred_rel_triplet[:, REL_PRED_SCR],
                    "rel_trp_score": pred_rel_triplet[:, REL_TRP_SCR],
                    "pred_prob_dist": pred_rel_probs[
                        pred_rel_triplet[:, INIT_PROP_IDX].long()
                    ],
                    "rel_vec": rel_vec_flat[pred_rel_triplet[:, INIT_PROP_IDX].long()],
                    "init_prop_indx": pred_rel_triplet[:, INIT_PROP_IDX].long(),
                }

                return ret

            init_pred_dict = res2dict(init_pred_rel_triplet)

            init_rel_proposals_predict = init_pred_dict

            rel_proposals_predict = res2dict(pred_rel_triplet_selected)
        
        # Post Process for panoptic segmentation
        s_scores = ent_scores[pred_rel_triplet_selected[:,0].view(-1).long()]
        s_labels = ent_labels[pred_rel_triplet_selected[:,0].view(-1).long()] + 1
        s_bbox_pred = ent_bboxes[pred_rel_triplet_selected[:,0].view(-1).long()]

        o_scores = ent_scores[pred_rel_triplet_selected[:,1].view(-1).long()]
        o_labels = ent_labels[pred_rel_triplet_selected[:,1].view(-1).long()] + 1
        o_bbox_pred = ent_bboxes[pred_rel_triplet_selected[:,1].view(-1).long()]

        r_labels = pred_rel_triplet_selected[:,2].view(-1).to(device)

        r_dists = pred_rel_probs[
                        pred_rel_triplet_selected[:, INIT_PROP_IDX].long()
                    ]

        labels = torch.cat((s_labels, o_labels), 0)
        complete_labels = labels
        complete_r_labels = r_labels
        complete_r_dists = r_dists

        s_mask_pred = ent_masks[pred_rel_triplet_selected[:,0].view(-1).long()]
        o_mask_pred = ent_masks[pred_rel_triplet_selected[:,1].view(-1).long()]

        output_masks = torch.cat((s_mask_pred, o_mask_pred), 0)

        all_scores = ent_scores
        all_labels = ent_labels

        all_masks = ent_masks_logits

        triplet_sub_ids = pred_rel_triplet_selected[:,0].view(-1,1)
        triplet_obj_ids = pred_rel_triplet_selected[:,1].view(-1,1)
        pan_rel_pairs = torch.cat((triplet_sub_ids,triplet_obj_ids), -1).to(torch.int).to(all_masks.device)
        tri_obj_unique = pan_rel_pairs.unique()
        keep = all_labels != (ent_logits.shape[-1] - 1)
        tmp = torch.zeros_like(keep, dtype=torch.bool)
        for id in tri_obj_unique:
            tmp[id] = True
        keep = keep & tmp

        all_labels = all_labels[keep]
        all_masks = all_masks[keep]
        all_scores = all_scores[keep]
        h, w = all_masks.shape[-2:]

        no_obj_filter = torch.zeros(pan_rel_pairs.shape[0],dtype=torch.bool)
        for triplet_id in range(pan_rel_pairs.shape[0]):
            if keep[pan_rel_pairs[triplet_id,0]] and keep[pan_rel_pairs[triplet_id,1]]:
                no_obj_filter[triplet_id]=True
        pan_rel_pairs = pan_rel_pairs[no_obj_filter]
        if keep.sum() != len(keep):
            for new_id, past_id in enumerate(keep.nonzero().view(-1)):
                pan_rel_pairs.masked_fill_(pan_rel_pairs.eq(past_id), new_id)
        r_labels, r_dists = r_labels[no_obj_filter], r_dists[no_obj_filter]

        if all_labels.numel() == 0:
            pan_img = torch.ones(mask_size).cpu().to(torch.long)
            pan_masks = pan_img.unsqueeze(0).cpu().to(torch.long)
            pan_rel_pairs = torch.arange(len(labels), dtype=torch.int).to(pan_masks.device).reshape(2, -1).T
            rels = torch.tensor([0,0,0]).view(-1,3)
            pan_labels = torch.tensor([0])
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

            all_binary_masks = (torch.sigmoid(all_masks) > 0.85).to(torch.float)
            # create dict that groups duplicate masks
            for thing_pred_ids in thing_classes.values():
                if len(thing_pred_ids) > 1:
                    dedup_things(thing_pred_ids, all_binary_masks)
                else:
                    thing_dedup[thing_pred_ids[0]].append(thing_pred_ids[0])

            def get_ids_area(all_masks, pan_rel_pairs, r_labels, r_dists, dedup=False):
                # This helper function creates the final panoptic segmentation image
                # It also returns the area of the masks that appears on the image

                m_id = all_masks.transpose(0, 1).softmax(-1)

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
                rels = torch.tensor([0,0,0]).view(-1,3)
            else:
                rels = torch.cat((pan_rel_pairs,r_labels.unsqueeze(-1)),-1)
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

        s_det_bboxes = torch.cat((s_bbox_pred, s_scores.unsqueeze(1)), -1)
        o_det_bboxes = torch.cat((o_bbox_pred, o_scores.unsqueeze(1)), -1)
        det_bboxes = torch.cat((s_det_bboxes, o_det_bboxes), 0)

        rel_pairs = torch.arange(len(det_bboxes),
                                dtype=torch.int).reshape(2, -1).T
        return det_bboxes, complete_labels, rel_pairs, output_masks, pan_rel_pairs, \
                pan_img, complete_r_labels, complete_r_dists, r_labels, r_dists, pan_masks, rels, pan_labels


    def simple_test_bboxes(self, feats, img_metas, rescale=False):

        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list


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

def get_matching_scores_entities_aware(
        s_cetr,
        o_cetr,
        s_scores,
        o_scores,
        rel_vec,
        s_cetr_normed,
        o_cetr_normed,
        rel_vec_normed,
        ent_box,
        ent_mask,
        ent_box_normed,
        s_dist,
        o_dist,
        rel_ent_s_box,
        rel_ent_o_box,
        rel_ent_s_mask,
        rel_ent_o_mask,
        rel_s_dist,
        rel_o_dist,
        normed_rel_vec_dist=False,
):
    """
    Args:
        s_cetr: image size normed
        o_cetr: image size normed
        s_scores:
        o_scores:
        rel_vec: image size normed
        ent_box:
        s_dist:
        o_dist:
        rel_ent_s_box:
        rel_ent_o_box:
        rel_s_dist:
        rel_o_dist:
    Returns:
    """
    def rev_vec_abs_dist(rel_vec, s_cetr, o_cetr):
        rel_s_centr = rel_vec[..., :2].unsqueeze(-1).repeat(1, 1, s_cetr.shape[0])
        rel_o_centr = rel_vec[..., 2:].unsqueeze(-1).repeat(1, 1, o_cetr.shape[0])
        s_cetr = s_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)
        o_cetr = o_cetr.unsqueeze(0).repeat(rel_vec.shape[0], 1, 1)

        dist_s_x = abs(rel_s_centr[..., 0, :] - s_cetr[..., 0])
        dist_s_y = abs(rel_s_centr[..., 1, :] - s_cetr[..., 1])
        dist_o_x = abs(rel_o_centr[..., 0, :] - o_cetr[..., 0])
        dist_o_y = abs(rel_o_centr[..., 1, :] - o_cetr[..., 1])
        return dist_s_x, dist_s_y, dist_o_x, dist_o_y

    if not normed_rel_vec_dist:
        (dist_s_x, dist_s_y, dist_o_x, dist_o_y) = rev_vec_abs_dist(
            rel_vec, s_cetr, o_cetr
        )
    else:
        (dist_s_x, dist_s_y, dist_o_x, dist_o_y) = rev_vec_abs_dist(
            rel_vec_normed, s_cetr_normed, o_cetr_normed
        )
    match_rel_vec_sub = 1 / (dist_s_x + dist_s_y + 1)
    match_rel_vec_obj = 1 / (dist_o_x + dist_o_y + 1)

    s_scores = s_scores ** 0.6
    o_scores = o_scores ** 0.6
    s_scores = s_scores.repeat(rel_vec.shape[0], 1)
    o_scores = o_scores.repeat(rel_vec.shape[0], 1)
    
    # match_vec_n_conf_sub = s_scores
    # match_vec_n_conf_obj = o_scores
    match_vec_n_conf_sub = s_scores * match_rel_vec_sub
    match_vec_n_conf_obj = o_scores * match_rel_vec_obj

    match_cost_details = {
        "match_rel_vec_sub": match_rel_vec_sub,
        "match_rel_vec_obj": match_rel_vec_obj,
        "match_sub_conf": s_scores,
        "match_obj_conf": o_scores,
        "match_vec_n_conf_sub": match_vec_n_conf_sub,
        "match_vec_n_conf_obj": match_vec_n_conf_obj,
    }

    match_rel_sub_cls = cosine_similarity(rel_s_dist, s_dist)
    match_rel_obj_cls = cosine_similarity(rel_o_dist, o_dist)
    
    # match_rel_sub_cls = torch.squeeze(torch.cdist(rel_s_dist.unsqueeze(0), s_dist.unsqueeze(0), p=2)) / s_dist.shape[-1]
    # match_rel_obj_cls = torch.squeeze(torch.cdist(rel_o_dist.unsqueeze(0), o_dist.unsqueeze(0), p=2)) / s_dist.shape[-1]

    match_rel_sub_cls = match_rel_sub_cls ** 0.6
    match_rel_obj_cls = match_rel_obj_cls ** 0.6
    match_cost_details["match_rel_sub_cls"] = match_rel_sub_cls
    match_cost_details["match_rel_obj_cls"] = match_rel_obj_cls


    match_sub_giou = torch.clip(generalized_box_iou(rel_ent_s_box, ent_box), 0)
    match_obj_giou = torch.clip(generalized_box_iou(rel_ent_o_box, ent_box), 0)

    match_sub_mask_ious = torch.ones((rel_ent_s_mask.shape[0], ent_mask.shape[0])).to(rel_ent_s_mask.device)
    for index in range(rel_ent_s_mask.shape[0]):
        match_sub_mask_ious[index:index+1,:] = rel_ent_s_mask[index:index+1,:].mm(ent_mask.transpose(0,1))/((rel_ent_s_mask[index:index+1,:] + ent_mask)>0).sum(-1)
    
    match_obj_mask_ious = torch.ones((rel_ent_o_mask.shape[0], ent_mask.shape[0])).to(rel_ent_o_mask.device)
    for index in range(rel_ent_o_mask.shape[0]):
        match_obj_mask_ious[index:index+1,:] = rel_ent_o_mask[index:index+1,:].mm(ent_mask.transpose(0,1))/((rel_ent_o_mask[index:index+1,:] + ent_mask)>0).sum(-1)
    
    # match_sub_mask_ious = rel_ent_s_mask.mm(ent_mask.transpose(0,1))/((rel_ent_s_mask[:,None].repeat(1,ent_mask.shape[0],1) + ent_mask[None,:].repeat(rel_ent_s_mask.shape[0], 1, 1))>0).sum(-1)
    # match_obj_mask_ious = rel_ent_o_mask.mm(ent_mask.transpose(0,1))/((rel_ent_o_mask[:,None].repeat(1,ent_mask.shape[0],1) + ent_mask[None,:].repeat(rel_ent_o_mask.shape[0], 1, 1))>0).sum(-1)
    ## cuda out of memory

    match_cost_details["match_sub_giou"] = match_sub_giou
    match_cost_details["match_obj_giou"] = match_obj_giou

    match_cost_details["match_sub_mask_ious"] = match_sub_mask_ious
    match_cost_details["match_obj_mask_ious"] = match_obj_mask_ious

    # TODO consider to replace match_sub_giou with match_sub_mask_ious
    match_scr_sub = match_rel_sub_cls * 32 ** (match_sub_giou) * match_vec_n_conf_sub * 32 ** (match_sub_mask_ious)
    match_scr_obj = match_rel_obj_cls * 32 ** (match_obj_giou) * match_vec_n_conf_obj * 32 ** (match_obj_mask_ious)

    # match_scr_sub = minmax_norm(match_scr_sub)
    # match_scr_obj = minmax_norm(match_scr_obj)

    match_cost_details["match_scr_sub"] = match_scr_sub
    match_cost_details["match_scr_obj"] = match_scr_obj

    return match_scr_sub, match_scr_obj, match_cost_details