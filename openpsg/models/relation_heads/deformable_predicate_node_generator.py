# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from mmcv.runner.base_module import BaseModule
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
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_attention

from mmcv.utils import Registry, build_from_cfg
from openpsg.models.relation_heads.predicate_node_generator import PREDICATE_NODE_GENERATOR
#####imports for tools
from packaging import version
import copy

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class HiddenInstance:
    def __init__(self, feature=None, box=None, seg=None, logits=None):

        self.feature = feature
        self.box = box
        self.seg = seg
        self.logits = logits

        if isinstance(feature, list):
            if len(feature) > 0:
                self.feature = torch.stack(self.feature)
            else:
                self.feature = None
        if isinstance(box, list):
            if len(box) > 0:
                self.box = torch.stack(self.box)
            else:
                self.box = None
        if isinstance(seg, list):
            if len(seg) > 0:
                self.seg = torch.stack(self.seg)
            else:
                self.seg = None
        if isinstance(logits, list):
            if len(logits) > 0:
                self.logits = torch.stack(self.logits)
            else:
                self.logits = None

@PREDICATE_NODE_GENERATOR.register_module()
class Deformable_Predicate_Node_Generator(BaseModule):

    def __init__(
        self,
        num_classes = 80,
        rel_encoder = None,
        rel_decoder = None,
        rel_q_gen_decoder = None,
        entities_aware_decoder = None,
        num_rel_query = 100,
        entities_enhance_decoder = None,
        intra_self_attention = None,
        update_query_by_rel_hs=False,
        no_coords_prior = False,
        two_stage = False,
        supervision_for_init = False,
        init_cfg = None,
    ):
        super(Deformable_Predicate_Node_Generator, self).__init__(init_cfg=init_cfg)
        if rel_encoder is not None:
            # Shared memory with entities encoder
            self.encoder = build_transformer_layer_sequence(rel_encoder)
        else:
            self.encoder = None

        self.predicate_decoder = build_transformer_layer_sequence(rel_decoder)
        self.embed_dims = self.predicate_decoder.embed_dims
        self.predicate_sub_decoder_layers = copy.deepcopy(self.predicate_decoder.layers)
        self.rel_decoder_norm = None
        del self.predicate_decoder
        self.predicate_decoder = True

        self.dynamic_query_on = True
        self.queries_cache = {}
        self.num_rel_queries = num_rel_query

        # Predicate Query
        self.rel_q_gen = build_transformer_layer_sequence(rel_q_gen_decoder)
        self.rel_query_embed = nn.Embedding(self.num_rel_queries,
                                            self.embed_dims)

        # self.coord_points_embed = nn.Sequential(nn.Linear(4, self.embed_dims)) # Do not use it....
        self.num_classes = num_classes
        # self.logits_embed = nn.Linear(self.num_classes + 1, self.embed_dims)
        self.no_coords_prior = no_coords_prior
        if not self.no_coords_prior:
            self.ent_pos_sine_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.split_query = nn.Sequential(nn.ReLU(True), nn.Linear(self.embed_dims, self.embed_dims * 3))

        # Entity aware decoder
        self.entities_aware_decoder = build_transformer_layer_sequence(entities_aware_decoder)
        if entities_enhance_decoder is not None:
            self.entities_enhance_decoder = build_transformer_layer_sequence(entities_enhance_decoder)
            self.rel_entity_enhance_decoder_layers_obj = copy.deepcopy(self.entities_enhance_decoder.layers)
            self.rel_entity_enhance_decoder_layers_sub = copy.deepcopy(self.entities_enhance_decoder.layers)
            self.ent_sub_reference_points = nn.Linear(self.embed_dims, 2)
            self.ent_obj_reference_points = nn.Linear(self.embed_dims, 2)
            del self.entities_enhance_decoder
            self.entities_enhance_decoder = True
        else:
            self.entities_enhance_decoder = False

        # self.ent_pred_fuse_layernorm = nn.LayerNorm(self.embed_dims)

        self.rel_query_embed_sub = nn.Embedding(self.num_rel_queries, self.embed_dims)
        self.rel_query_embed_obj = nn.Embedding(self.num_rel_queries, self.embed_dims)

        self.rel_entity_cross_decoder_layers_obj = copy.deepcopy(self.entities_aware_decoder.layers)
        self.rel_entity_cross_decoder_layers_sub = copy.deepcopy(self.entities_aware_decoder.layers)

        self.num_decoder_layer = len(self.entities_aware_decoder.layers)
        if self.entities_aware_decoder.post_norm is None:
            self.rel_entity_cross_decoder_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.rel_entity_cross_decoder_norm = None
        del self.entities_aware_decoder
        self.entities_aware_decoder = True

        self.update_query_by_rel_hs = update_query_by_rel_hs # It should be updated
        if self.update_query_by_rel_hs:
            self.update_sub_query_layernorm = nn.LayerNorm(self.embed_dims)
            self.update_obj_query_layernorm = nn.LayerNorm(self.embed_dims)
        self.ent_dec_each_lvl = True # From ori SGTR

        self.ent_rel_fuse_fc_obj = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims))

        self.ent_rel_fuse_fc_sub = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims)) # copy.deepcopy(self.ent_rel_fuse_fc_obj)

        self.ent_rel_fuse_x = nn.Linear(self.embed_dims, self.embed_dims)
        self.ent_rel_fuse_y = copy.deepcopy(self.ent_rel_fuse_x)

        self.ent_rel_fuse_norm_obj = nn.LayerNorm(self.embed_dims)
        self.ent_rel_fuse_norm_sub = nn.LayerNorm(self.embed_dims)

        self.ent_rel_fuse_norm = nn.LayerNorm(self.embed_dims)

        if intra_self_attention is not None:
            intra_self_attention = build_attention(intra_self_attention)
            intra_self_attention_layernorm = nn.LayerNorm(self.embed_dims)
            self.intra_self_attention = _get_clones(intra_self_attention, self.num_decoder_layer)
            self.intra_self_attention_layernorm = _get_clones(intra_self_attention_layernorm, self.num_decoder_layer)
        else:
            self.intra_self_attention = None

        self.rel_sub_obj_box_embed = Linear(self.embed_dims, 4) # Predict the union box
        if supervision_for_init:
            self.rel_reference_points = _get_clones(self.rel_sub_obj_box_embed, self.num_decoder_layer+1)
        else:
            self.rel_reference_points = _get_clones(self.rel_sub_obj_box_embed, self.num_decoder_layer)
            self.rel_reference_points_proposal = Linear(self.embed_dims, 2)
        del self.rel_sub_obj_box_embed

        self.two_stage = two_stage
        if self.two_stage:
            # For 2D ref points
            self.pos_trans_2d = nn.Linear(self.embed_dims, self.embed_dims)
            self.pos_trans_norm_2d = nn.LayerNorm(self.embed_dims)

            # For 4D ref boxes
            self.pos_trans_4d = nn.Linear(self.embed_dims * 2, self.embed_dims)
            self.pos_trans_norm_4d = nn.LayerNorm(self.embed_dims)
        self.supervision_for_init = supervision_for_init
        self.s_entity_proj = None # ~ this is a hack
        self.o_entity_proj = None # ~ this is a hack

    def forward(
        self,
        src,
        mask,
        query_embed,
        src_pos_embed,
        rel_query_pos_embed,
        shared_encoder_memory=None,
        ent_hs=None,
        ent_coords=None,
        ent_cls=None,
        valid_ratios=None,
        spatial_shapes=None,
        level_start_index=None,
        reference_points=None,
        ent_hs_masks=None,
    ):
        """
            Args:
                src: the backbone features
                mask: mask for backbone features sequence
                query_embed: relationship prediction query embedding (N_q, dim)
                src_pos_embed: position_embedding for the src backbone features (W*H, dim)
                query_pos_embed: position_embedding for relationship prediction query (N_q, dim)
                shared_encoder_memory: the output of entities encoder (W*H, bz, dim)
                ent_hs: entities transformer outputs (lys, bz, num_q, dim)
        """
        # [bz, channel, w, h]
        if src_pos_embed is not None:
            src_pos_embed = src_pos_embed.flatten(2).permute(2,0,1)
        # [bz, w, h]
        mask_flatten = mask.flatten(1)

        rel_memory = self.rel_encoder(
            src, src_pos_embed, shared_encoder_memory, mask_flatten, spatial_shapes, reference_points, level_start_index, valid_ratios
        )

        # initialize the rel mem features HWXNXC
        ent_hs_input = ent_hs[-1].permute(1, 0, 2)  # seq_len, bz, dim

        # initialize the triplets query
        (
            query_embed_obj_init,
            query_embed_sub_init,
            query_embed_rel_init,
        ) = self.query_tgt_initialization(ent_hs, ent_coords, ent_hs_masks=ent_hs_masks)

        (rel_tgt, ent_obj_tgt, ent_sub_tgt) = self.reset_tgt()

        if self.supervision_for_init:
            rel_feat_all = [rel_tgt]
            ent_aware_rel_hs_sub = [ent_sub_tgt]
            ent_aware_rel_hs_obj = [ent_obj_tgt]
        else:
            rel_feat_all = []
            ent_aware_rel_hs_sub = []
            ent_aware_rel_hs_obj = []
        reference_points_output = [] # for debug

        # outputs placeholder & container
        intermediate = []
        inter_rel_hs = []

        decoder_out_dict = {}
        ent_sub_dec_outputs = {}
        predicate_sub_dec_output_dict = None

        start = 0
        end = self.num_decoder_layer

        for idx in range(self.num_decoder_layer):

            output_dict = {}

            # predicate sub-decoder
            rel_hs_out = None

            if self.predicate_decoder is not None:
                
                if self.intra_self_attention is not None:
                    concat_sop_tgt = torch.cat([rel_tgt[:,:,None], ent_obj_tgt[:,:,None], ent_sub_tgt[:,:,None]], 2) # N_Q, B, 3, Dim
                    concat_sop_pos_embed = torch.cat([query_embed_rel_init[:,:,None], query_embed_obj_init[:,:,None], query_embed_sub_init[:,:,None]], 2)
                    concat_sop_embed = concat_sop_tgt + concat_sop_pos_embed
                    concat_sop_embed = self.intra_self_attention[idx](concat_sop_embed.flatten(0,1).transpose(0,1)).transpose(0,1).reshape(concat_sop_embed.shape)
                    concat_sop_embed = self.intra_self_attention_layernorm[idx](concat_sop_embed)
                    rel_tgt = concat_sop_embed[:,:,0]
                    ent_obj_tgt = concat_sop_embed[:,:,1]
                    ent_sub_tgt = concat_sop_embed[:,:,2]
                
                # rel_memory: multi-scale feature maps from mask deformable detr head
                # reference_points: predicate from query_pos
                # spatial_shapes, level_start_index, valid_ratios: from mask deformable detr head
                # reg_branches: None

                if idx == 0 and not self.supervision_for_init:
                    rel_reference_points = self.rel_reference_points_proposal(rel_tgt).sigmoid().permute(1,0,2).contiguous()
                # Predict the union box of object and subject
                elif self.rel_reference_points is not None:
                    rel_reference_points = self.rel_reference_points[idx](rel_tgt).sigmoid().permute(1,0,2).contiguous() # rel_tgt include prior information
                else:
                    raise NotImplementedError
                if rel_reference_points.shape[-1] == 4:
                    reference_points_input = rel_reference_points[:, :, None] * torch.cat([valid_ratios,valid_ratios],-1)[:, None]
                    if self.two_stage:
                        query_embed_rel_init = self.pos_trans_norm_4d(
                                                    self.pos_trans_4d(
                                                        torch.cat([gen_sineembed_for_position(rel_reference_points[...,:2], self.embed_dims//2), \
                                                        gen_sineembed_for_position(rel_reference_points[...,2:], self.embed_dims//2)], -1)
                                                    )
                                                ).permute(1,0,2)
                else:
                    reference_points_input = rel_reference_points[:, :, None] * valid_ratios[:, None]
                    if self.two_stage:
                        query_embed_rel_init = self.pos_trans_norm_2d(
                                                    self.pos_trans_2d(gen_sineembed_for_position(rel_reference_points[...,:2], self.embed_dims//2))
                                                ).permute(1,0,2)
                reference_points_output.append(reference_points_input)
                predicate_sub_dec_output_dict = self.predicate_sub_decoder_layers[idx](
                    query=rel_tgt,
                    key=None,
                    value=rel_memory,
                    query_pos=query_embed_rel_init if self.intra_self_attention is None else None,
                    key_padding_mask=mask_flatten,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )
                # w/o attn_weight and value sum
                if self.rel_decoder_norm is not None:
                    rel_tgt = self.rel_decoder_norm(predicate_sub_dec_output_dict)
                else:
                    rel_tgt = predicate_sub_dec_output_dict

                inter_rel_hs.append(rel_tgt)

                output_dict["rel_hs"] = rel_tgt

                rel_hs_out = inter_rel_hs[-1]
            
            if self.entities_aware_decoder is not None and idx >= start:

                # entity indicator sub-decoder
                ent_sub_dec_outputs = self.entities_sub_decoder(
                    ent_hs_input, 
                    query_embed_obj_init if self.intra_self_attention is None else None,
                    query_embed_sub_init if self.intra_self_attention is None else None,
                    ent_obj_tgt, ent_sub_tgt, start, idx, rel_hs_out, shared_encoder_memory, 
                    src_pos_embed, mask_flatten, valid_ratios, spatial_shapes, level_start_index,
                    ent_hs_masks=ent_hs_masks,
                )
                ent_obj_tgt = ent_sub_dec_outputs['obj_ent_hs']
                ent_sub_tgt = ent_sub_dec_outputs['sub_ent_hs']
                rel_tgt = ent_sub_dec_outputs["ent_aug_rel_hs"]

                output_dict.update(ent_sub_dec_outputs)
                # only return needed intermediate hs
                intermediate.append(output_dict)

        for outs in intermediate:
            if "ent_aug_rel_hs" in outs.keys():
                rel_feat_all.append(outs["ent_aug_rel_hs"])
            elif "rel_hs" in outs.keys():
                rel_feat_all.append(outs["rel_hs"])

            if "obj_ent_hs" in outs.keys():
                ent_aware_rel_hs_sub.append(outs["sub_ent_hs"])  # layer x [Nq, bz dim]
                ent_aware_rel_hs_obj.append(outs["obj_ent_hs"])

        assert len(rel_feat_all) > 0

        rel_rep = HiddenInstance(feature=rel_feat_all)
        rel_rep.feature.transpose_(1, 2)

        sub_ent_rep = HiddenInstance(feature=ent_aware_rel_hs_sub)
        obj_ent_rep = HiddenInstance(feature=ent_aware_rel_hs_obj)

        sub_ent_rep.feature.transpose_(1, 2)
        obj_ent_rep.feature.transpose_(1, 2)

        dynamic_query = None
        if self.queries_cache.get("dynamic_query") is not None:
            dynamic_query = self.queries_cache.get("dynamic_query")
            dynamic_query = HiddenInstance(feature=dynamic_query)
            dynamic_query.feature.transpose_(1, 2)

        decoder_out_dict['reference_points'] = reference_points_output
        return (
            rel_rep,
            (sub_ent_rep, obj_ent_rep, dynamic_query),
            decoder_out_dict,
        )

        

    def rel_encoder(self, src, src_pos_embed, shared_encoder_memory, mask_flatten, spatial_shapes, reference_points, level_start_index, valid_ratios):
        if self.encoder is not None:
            rel_memory = self.encoder(
                query=shared_encoder_memory,
                key=None,
                value=None,
                query_pos=src_pos_embed.permute(2,1,0),
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,)
        else:
            if len(src.shape) == 4 and len(shared_encoder_memory.shape):
                bs, c, h, w = src.shape
                # flatten NxCxHxW to HWxNxC for following decoder
                rel_memory = shared_encoder_memory.view(bs, c, h * w).permute(2, 0, 1)
            else:
                # not satisfy the reshape: directly use
                # must in shape (len, bz, dim)
                rel_memory = shared_encoder_memory
        return rel_memory

    def query_tgt_initialization(self, ent_hs, ent_coords, ent_hs_masks=None):
        """
        apply the dynamic query into the rel_query
        """
        # static query weights (N_q, dim) -> (N_q, bz, dim)
        self.queries_cache = {}
        bs = ent_hs.shape[1]
        query_embed_obj_init_w = self.rel_query_embed_obj.weight.unsqueeze(1).repeat(
            1, bs, 1
        )
        query_embed_sub_init_w = self.rel_query_embed_sub.weight.unsqueeze(1).repeat(
            1, bs, 1
        )
        query_embed_rel_init_w = self.rel_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # query pos embedding
        query_embed_sub_init = query_embed_sub_init_w
        query_embed_obj_init = query_embed_obj_init_w
        query_embed_rel_init = query_embed_rel_init_w

        self.queries_cache.update(
            {
                "query_embed_obj_init_w": query_embed_obj_init_w,
                "query_embed_sub_init_w": query_embed_sub_init_w,
                "query_embed_rel_init_w": query_embed_rel_init_w,
            }
        )

        if self.dynamic_query_on:
            dynamic_query = self.dynamic_predicate_query_generation(
                query_embed_rel_init_w, ent_hs, ent_coords, ent_hs_masks=ent_hs_masks
            )
            seq_len, bs, dim = dynamic_query.shape  # seq_len, bz, dim
            dynamic_query = dynamic_query.reshape(seq_len, bs, 3, dim // 3).transpose(
                3, 2
            )
            d_query_embed_sub_input_init = dynamic_query[..., 0,]
            d_query_embed_obj_input_init = dynamic_query[..., 1,]
            d_query_embed_rel_input_init = dynamic_query[..., 2,]
            dynamic_query = dynamic_query.permute(3, 0, 1, 2)

            self.queries_cache.update(
                {
                    "dynamic_query": dynamic_query,
                    "d_query_embed_sub_input_init": d_query_embed_sub_input_init,
                    "d_query_embed_obj_input_init": d_query_embed_obj_input_init,
                    "d_query_embed_rel_input_init": d_query_embed_rel_input_init,
                }
            )

        self.queries_cache.update(
            {
                "query_embed_obj_init": query_embed_obj_init,
                "query_embed_sub_init": query_embed_sub_init,
                "query_embed_rel_init": query_embed_rel_init,
            }
        )

        return query_embed_obj_init, query_embed_sub_init, query_embed_rel_init

    def dynamic_predicate_query_generation(self, query_embed, ent_hs, ent_coords=None, rel_q_gen=None, ent_hs_masks=None):
        if ent_coords is not None:
            ent_coords_embd = self.ent_pos_sine_proj(
                gen_sineembed_for_position(ent_coords[..., :2], self.embed_dims // 2)
            ).contiguous()
            ent_hs = ent_hs[-1].transpose(1, 0)
            ent_coords_embd = ent_coords_embd.transpose(1, 0)
            if rel_q_gen is None:
                rel_q_gen = self.rel_q_gen
            query_embed = rel_q_gen(query_embed, ent_hs + ent_coords_embd, ent_hs + ent_coords_embd, key_padding_mask=ent_hs_masks)[0]
        else:
            # The box of Stuff will confuse the query feature
            ent_hs = ent_hs[-1].transpose(1, 0)
            if rel_q_gen is None:
                rel_q_gen = self.rel_q_gen
            query_embed = rel_q_gen(query_embed, ent_hs, ent_hs)[0]
        query_embed = self.split_query(query_embed)

        return query_embed  # seq_len, bz, dim

    def reset_tgt(self):
        # keys & tgt:
        #   initialization by the dynamic query
        if self.dynamic_query_on:
            d_query_embed_sub_input_init = self.queries_cache[
                "d_query_embed_sub_input_init"
            ]
            d_query_embed_obj_input_init = self.queries_cache[
                "d_query_embed_obj_input_init"
            ]
            d_query_embed_rel_input_init = self.queries_cache[
                "d_query_embed_rel_input_init"
            ]

            ent_sub_tgt = d_query_embed_sub_input_init.clone()
            ent_obj_tgt = d_query_embed_obj_input_init.clone()
            rel_tgt = d_query_embed_rel_input_init.clone()
        else:
            query_embed_rel_init = self.queries_cache["query_embed_rel_init"]
            ent_sub_tgt = torch.zeros_like(query_embed_rel_init)
            ent_obj_tgt = torch.zeros_like(query_embed_rel_init)
            rel_tgt = torch.zeros_like(query_embed_rel_init)

        return rel_tgt, ent_obj_tgt, ent_sub_tgt

    def entities_sub_decoder(
            self,
            ent_hs_input,
            query_pos_embed_obj,
            query_pos_embed_sub,
            ent_obj_tgt,
            ent_sub_tgt,
            start,
            idx,
            rel_hs_out,
            enc_memory,
            src_pos_embed, 
            mask_flatten,
            valid_ratios, 
            spatial_shapes, 
            level_start_index,
            ent_hs_masks=None,
    ):
        rel_hs_out_obj_hs = []
        rel_hs_out_sub_hs = []

        _sub_ent_dec_layers = self.rel_entity_cross_decoder_layers_sub
        _obj_ent_dec_layers = self.rel_entity_cross_decoder_layers_obj

        if self.ent_dec_each_lvl:
            _sub_ent_dec_layers = [self.rel_entity_cross_decoder_layers_sub[idx - start]]
            _obj_ent_dec_layers = [self.rel_entity_cross_decoder_layers_obj[idx - start]]

        if self.entities_enhance_decoder:
            enhance_sub_ent_dec_layers = [self.rel_entity_enhance_decoder_layers_sub[idx - start]]
            enhance_obj_ent_dec_layers = [self.rel_entity_enhance_decoder_layers_obj[idx - start]]

            for layeri, (ent_dec_layer_sub, ent_dec_layer_obj, enhance_ent_dec_layer_sub, enhance_ent_dec_layer_obj,) in enumerate(zip(
                    _sub_ent_dec_layers, _obj_ent_dec_layers, enhance_sub_ent_dec_layers, enhance_obj_ent_dec_layers
            )):
                # seq_len, bs, dim = rel_hs_out.shape
                # self.debug_print('ent_dec_layers id' + str(layeri))
                ent_sub_reference_points = self.ent_sub_reference_points(query_pos_embed_sub).sigmoid() # rel_tgt include prior information
                reference_points_input = ent_sub_reference_points[:, :, None] * valid_ratios[:, None]
                ent_sub_output_dict_enh = enhance_ent_dec_layer_sub(
                    query=ent_sub_tgt,
                    key=None,
                    value=enc_memory,
                    query_pos=query_pos_embed_sub,
                    key_padding_mask=mask_flatten,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )

                ent_obj_reference_points = self.ent_obj_reference_points(query_pos_embed_obj).sigmoid() # rel_tgt include prior information
                reference_points_input = ent_obj_reference_points[:, :, None] * valid_ratios[:, None]
                ent_obj_output_dict_enh = enhance_ent_dec_layer_obj(
                    query=ent_obj_tgt,
                    key=None,
                    value=enc_memory,
                    query_pos=query_pos_embed_obj,
                    key_padding_mask=mask_flatten,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )

                ent_sub_output_dict = ent_dec_layer_sub(
                    ent_sub_output_dict_enh,
                    self.s_entity_proj(ent_hs_input) if self.s_entity_proj else ent_hs_input,
                    self.s_entity_proj(ent_hs_input) if self.s_entity_proj else ent_hs_input,
                    key_padding_mask=ent_hs_masks,
                    key_pos=None,
                    query_pos=query_pos_embed_sub,
                )

                ent_obj_output_dict = ent_dec_layer_obj(
                    ent_obj_output_dict_enh,
                    self.o_entity_proj(ent_hs_input) if self.o_entity_proj else ent_hs_input,
                    self.o_entity_proj(ent_hs_input) if self.o_entity_proj else ent_hs_input,
                    key_padding_mask=ent_hs_masks,
                    key_pos=None,
                    query_pos=query_pos_embed_obj,
                )

                ent_obj_tgt = ent_obj_output_dict
                ent_sub_tgt = ent_sub_output_dict

                if self.rel_entity_cross_decoder_norm is not None:
                    rel_hs_sub = self.rel_entity_cross_decoder_norm(ent_sub_tgt)
                    rel_hs_obj = self.rel_entity_cross_decoder_norm(ent_obj_tgt)
                else:
                    rel_hs_sub = ent_sub_tgt
                    rel_hs_obj = ent_obj_tgt

                rel_hs_out_obj_hs.append(rel_hs_obj)
                rel_hs_out_sub_hs.append(rel_hs_sub)
        
        else:

            for layeri, (ent_dec_layer_sub, ent_dec_layer_obj) in enumerate(zip(
                    _sub_ent_dec_layers, _obj_ent_dec_layers,
            )):
                # seq_len, bs, dim = rel_hs_out.shape
                # self.debug_print('ent_dec_layers id' + str(layeri))

                ent_sub_output_dict = ent_dec_layer_sub(
                    ent_sub_tgt,
                    self.s_entity_proj(ent_hs_input) if self.s_entity_proj else ent_hs_input,
                    self.s_entity_proj(ent_hs_input) if self.s_entity_proj else ent_hs_input,
                    key_padding_mask=ent_hs_masks,
                    key_pos=None,
                    query_pos=query_pos_embed_sub,
                )

                ent_obj_output_dict = ent_dec_layer_obj(
                    ent_obj_tgt,
                    self.o_entity_proj(ent_hs_input) if self.o_entity_proj else ent_hs_input,
                    self.o_entity_proj(ent_hs_input) if self.o_entity_proj else ent_hs_input,
                    key_padding_mask=ent_hs_masks,
                    key_pos=None,
                    query_pos=query_pos_embed_obj,
                )

                ent_obj_tgt = ent_obj_output_dict
                ent_sub_tgt = ent_sub_output_dict

                if self.rel_entity_cross_decoder_norm is not None:
                    rel_hs_sub = self.rel_entity_cross_decoder_norm(ent_sub_tgt)
                    rel_hs_obj = self.rel_entity_cross_decoder_norm(ent_obj_tgt)
                else:
                    rel_hs_sub = ent_sub_tgt
                    rel_hs_obj = ent_obj_tgt

                rel_hs_out_obj_hs.append(rel_hs_obj)
                rel_hs_out_sub_hs.append(rel_hs_sub)

        ent_sub_tgt = rel_hs_out_sub_hs[-1]
        ent_obj_tgt = rel_hs_out_obj_hs[-1]

        # merge the final representation for prediction: TODO How to design the fusion module
        # ent_aug_rel_hs_out = (F.relu(self.ent_rel_fuse_fc_sub(ent_sub_tgt)
        #                              + self.ent_rel_fuse_fc_obj(ent_obj_tgt)))
        # if rel_hs_out is not None:
        #     ent_aug_rel_hs_out = rel_hs_out + ent_aug_rel_hs_out

        # Modify from Structural Sparse RCNN E2R
        ent_aug_rel_hs_out = self.ent_rel_fuse_x(F.relu(self.ent_rel_fuse_norm_sub(self.ent_rel_fuse_fc_sub(ent_sub_tgt)))) \
                     + self.ent_rel_fuse_y(F.relu(self.ent_rel_fuse_norm_obj(self.ent_rel_fuse_fc_obj(ent_obj_tgt))))
        if rel_hs_out is not None:
            ent_aug_rel_hs_out = self.ent_rel_fuse_norm(rel_hs_out + ent_aug_rel_hs_out)

        if self.update_query_by_rel_hs and rel_hs_out is not None:
            ent_sub_tgt = self.update_sub_query_layernorm(ent_sub_tgt + rel_hs_out)
            ent_obj_tgt = self.update_obj_query_layernorm(ent_obj_tgt + rel_hs_out)

        return {
            "ent_aug_rel_hs": ent_aug_rel_hs_out,
            "sub_ent_hs": ent_sub_tgt,
            "obj_ent_hs": ent_obj_tgt,
            "ent_sub_output_dict": ent_sub_output_dict,
            "ent_obj_output_dict": ent_obj_output_dict,
        }


def gen_sineembed_for_position(pos_tensor, feat_dim):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    import math
    scale = 2 * math.pi
    dim_t = torch.arange(feat_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / feat_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos