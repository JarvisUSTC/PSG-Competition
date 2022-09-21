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

PREDICATE_NODE_GENERATOR = Registry('Predicate_Node_Generator')

def build_predicate_node_generator(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, PREDICATE_NODE_GENERATOR, default_args)

@PREDICATE_NODE_GENERATOR.register_module()
class Predicate_Node_Generator(BaseModule):

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
        init_cfg = None,
    ):
        super(Predicate_Node_Generator, self).__init__(init_cfg=init_cfg)
        if rel_encoder is not None:
            # Shared memory with entities encoder
            self.encoder = build_transformer_layer_sequence(rel_encoder)
        else:
            self.encoder = None

        self.predicate_decoder = build_transformer_layer_sequence(rel_decoder)
        self.embed_dims = self.predicate_decoder.embed_dims
        self.predicate_sub_decoder_layers = copy.deepcopy(self.predicate_decoder.layers)
        self.rel_decoder_norm = nn.LayerNorm(self.embed_dims)

        self.dynamic_query_on = True
        self.queries_cache = {}
        self.num_rel_queries = num_rel_query

        # Predicate Query
        self.rel_q_gen = build_transformer_layer_sequence(rel_q_gen_decoder)
        self.rel_query_embed = nn.Embedding(self.num_rel_queries,
                                            self.embed_dims)

        self.coord_points_embed = nn.Sequential(nn.Linear(4, self.embed_dims)) # Do not use it....
        self.num_classes = num_classes
        self.logits_embed = nn.Linear(self.num_classes + 1, self.embed_dims)
        self.ent_pos_sine_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.split_query = nn.Sequential(nn.ReLU(True), nn.Linear(self.embed_dims, self.embed_dims * 3))

        # Entity aware decoder
        self.entities_aware_decoder = build_transformer_layer_sequence(entities_aware_decoder)
        if entities_enhance_decoder is not None:
            self.entities_enhance_decoder = build_transformer_layer_sequence(entities_enhance_decoder)
            self.rel_entity_enhance_decoder_layers_obj = copy.deepcopy(self.entities_enhance_decoder.layers)
            self.rel_entity_enhance_decoder_layers_sub = copy.deepcopy(self.entities_enhance_decoder.layers)
        else:
            self.entities_enhance_decoder = None

        self.ent_pred_fuse_layernorm = nn.LayerNorm(self.embed_dims)

        self.rel_query_embed_sub = nn.Embedding(self.num_rel_queries, self.embed_dims)
        self.rel_query_embed_obj = nn.Embedding(self.num_rel_queries, self.embed_dims)

        self.rel_entity_cross_decoder_layers_obj = copy.deepcopy(self.entities_aware_decoder.layers)
        self.rel_entity_cross_decoder_layers_sub = copy.deepcopy(self.entities_aware_decoder.layers)

        self.rel_entity_cross_decoder_norm = nn.LayerNorm(self.embed_dims)

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

        self.num_decoder_layer = len(self.entities_aware_decoder.layers)

        if intra_self_attention is not None:
            intra_self_attention = build_attention(intra_self_attention)
            intra_self_attention_layernorm = nn.LayerNorm(self.embed_dims)
            self.intra_self_attention = _get_clones(intra_self_attention, self.num_decoder_layer)
            self.intra_self_attention_layernorm = _get_clones(intra_self_attention_layernorm, self.num_decoder_layer)
        else:
            self.intra_self_attention = None

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
    ):
        """
            Args:
                src: the backbone features
                mask: mask for backbone features sequence
                query_embed: relationship prediction query embedding (N_q, dim)
                src_pos_embed: position_embedding for the src backbone features (W*H, dim)
                query_pos_embed: position_embedding for relationship prediction query (N_q, dim)
                shared_encoder_memory: the output of entities encoder (bz, dim, W*H )
                ent_hs: entities transformer outputs (lys, bz, num_q, dim)
        """
        # [bz, channel, w, h]
        src_pos_embed = src_pos_embed.flatten(2).permute(2,0,1)
        # [bz, w, h]
        mask_flatten = mask.flatten(1)

        device = query_embed.device

        bs, h, w, rel_memory = self.rel_encoder(
            src, src_pos_embed, shared_encoder_memory, mask_flatten
        )

        # initialize the rel mem features HWXNXC
        ent_hs_input = ent_hs[-1].permute(1, 0, 2)  # seq_len, bz, dim
        enc_featmap = rel_memory.permute(1, 2, 0).reshape(bs, -1, h, w)

        # initialize the triplets query
        (
            query_embed_obj_init,
            query_embed_sub_init,
            query_embed_rel_init,
        ) = self.query_tgt_initialization(ent_hs, ent_coords)

        (rel_tgt, ent_obj_tgt, ent_sub_tgt) = self.reset_tgt()

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
                
                predicate_sub_dec_output_dict = self.predicate_sub_decoder_layers[idx](
                    rel_tgt,
                    rel_memory,
                    rel_memory,
                    key_padding_mask=mask_flatten,
                    key_pos=src_pos_embed,
                    query_pos=query_embed_rel_init if self.intra_self_attention is None else None,
                )
                # w/o attn_weight and value sum
                if self.predicate_decoder.post_norm is None:
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
                    ent_obj_tgt, ent_sub_tgt, start, idx, rel_hs_out, shared_encoder_memory, src_pos_embed, mask_flatten
                )
                ent_obj_tgt = ent_sub_dec_outputs['obj_ent_hs']
                ent_sub_tgt = ent_sub_dec_outputs['sub_ent_hs']
                rel_tgt = ent_sub_dec_outputs["ent_aug_rel_hs"]

                output_dict.update(ent_sub_dec_outputs)
                # only return needed intermediate hs
                intermediate.append(output_dict)

        rel_feat_all = []
        ent_aware_rel_hs_sub = []
        ent_aware_rel_hs_obj = []

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

        return (
            rel_rep,
            (sub_ent_rep, obj_ent_rep, dynamic_query),
            decoder_out_dict,
        )

        

    def rel_encoder(self, src, src_pos_embed, shared_encoder_memory, mask_flatten):
        if self.encoder is not None:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = shared_encoder_memory.shape
            shared_encoder_memory = shared_encoder_memory.flatten(2).permute(2, 0, 1)
            rel_memory = self.encoder(
                shared_encoder_memory,
                key=None,
                value=None,
                query_key_padding_mask=mask_flatten,
                query_pos=src_pos_embed,
            )
        else:
            if len(src.shape) == 4 and len(shared_encoder_memory.shape):
                bs, c, h, w = src.shape
                # flatten NxCxHxW to HWxNxC for following decoder
                rel_memory = shared_encoder_memory.view(bs, c, h * w).permute(2, 0, 1)
            else:
                # not satisfy the reshape: directly use
                # must in shape (len, bz, dim)
                rel_memory = shared_encoder_memory
                bs, c, h_w = rel_memory.shape
        return bs, h, w, rel_memory

    def query_tgt_initialization(self, ent_hs, ent_coords):
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
                query_embed_rel_init_w, ent_hs, ent_coords
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

    def dynamic_predicate_query_generation(self, query_embed, ent_hs, ent_coords, rel_q_gen=None):
        ent_coords_embd = self.ent_pos_sine_proj(
            gen_sineembed_for_position(ent_coords[..., :2], self.embed_dims // 2)
        ).contiguous()
        ent_hs = ent_hs[-1].transpose(1, 0)
        ent_coords_embd = ent_coords_embd.transpose(1, 0)
        if rel_q_gen is None:
            rel_q_gen = self.rel_q_gen
        query_embed = rel_q_gen(query_embed, ent_hs + ent_coords_embd, ent_hs + ent_coords_embd)[0]
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
            mask_flatten
    ):
        rel_hs_out_obj_hs = []
        rel_hs_out_sub_hs = []

        _sub_ent_dec_layers = self.rel_entity_cross_decoder_layers_sub
        _obj_ent_dec_layers = self.rel_entity_cross_decoder_layers_obj

        if self.ent_dec_each_lvl:
            _sub_ent_dec_layers = [self.rel_entity_cross_decoder_layers_sub[idx - start]]
            _obj_ent_dec_layers = [self.rel_entity_cross_decoder_layers_obj[idx - start]]

        if self.entities_enhance_decoder is not None:
            enhance_sub_ent_dec_layers = [self.rel_entity_enhance_decoder_layers_sub[idx - start]]
            enhance_obj_ent_dec_layers = [self.rel_entity_enhance_decoder_layers_obj[idx - start]]
            bs, c, h, w = enc_memory.shape
            enc_memory = enc_memory.flatten(2).permute(2, 0, 1)

            for layeri, (ent_dec_layer_sub, ent_dec_layer_obj, enhance_ent_dec_layer_sub, enhance_ent_dec_layer_obj,) in enumerate(zip(
                    _sub_ent_dec_layers, _obj_ent_dec_layers, enhance_sub_ent_dec_layers, enhance_obj_ent_dec_layers
            )):
                # seq_len, bs, dim = rel_hs_out.shape
                # self.debug_print('ent_dec_layers id' + str(layeri))

                ent_sub_output_dict = ent_dec_layer_sub(
                    ent_sub_tgt,
                    ent_hs_input,
                    ent_hs_input,
                    key_padding_mask=None,
                    key_pos=None,
                    query_pos=query_pos_embed_sub,
                )

                ent_obj_output_dict = ent_dec_layer_obj(
                    ent_obj_tgt,
                    ent_hs_input,
                    ent_hs_input,
                    key_padding_mask=None,
                    key_pos=None,
                    query_pos=query_pos_embed_obj,
                )

                ent_obj_tgt = ent_obj_output_dict
                ent_sub_tgt = ent_sub_output_dict

                ent_sub_output_dict = enhance_ent_dec_layer_sub(
                    ent_sub_tgt,
                    enc_memory,
                    enc_memory,
                    key_padding_mask=mask_flatten,
                    key_pos=src_pos_embed,
                    query_pos=query_pos_embed_sub,
                )

                ent_obj_output_dict = enhance_ent_dec_layer_obj(
                    ent_obj_tgt,
                    enc_memory,
                    enc_memory,
                    key_padding_mask=mask_flatten,
                    key_pos=src_pos_embed,
                    query_pos=query_pos_embed_obj,
                )

                ent_obj_tgt = ent_obj_output_dict
                ent_sub_tgt = ent_sub_output_dict

                if self.entities_aware_decoder.post_norm is None:
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
                    ent_hs_input,
                    ent_hs_input,
                    key_padding_mask=None,
                    key_pos=None,
                    query_pos=query_pos_embed_sub,
                )

                ent_obj_output_dict = ent_dec_layer_obj(
                    ent_obj_tgt,
                    ent_hs_input,
                    ent_hs_input,
                    key_padding_mask=None,
                    key_pos=None,
                    query_pos=query_pos_embed_obj,
                )

                ent_obj_tgt = ent_obj_output_dict
                ent_sub_tgt = ent_sub_output_dict

                if self.entities_aware_decoder.post_norm is None:
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