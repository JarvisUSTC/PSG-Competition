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
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from mmcv.utils import Registry, build_from_cfg

from openpsg.utils.metrics.cosine_sim import cosine_similarity
from torchvision.ops import generalized_box_iou

#####imports for tools
from packaging import version
import copy
import pickle

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

ENTITIES_INDEXING_HEAD = Registry('EntitiesIndexingHead')

def build_entities_indexing_head(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, ENTITIES_INDEXING_HEAD, default_args)

@ENTITIES_INDEXING_HEAD.register_module()
class EntitiesIndexingHead(BaseModule):

    def __init__(
        self,
        embed_dims,
        num_classes,
        init_cfg = None,
    ):
        super(EntitiesIndexingHead, self).__init__(init_cfg=init_cfg)
        self.vis_feat_input_dim = embed_dims
        self.hidden_dim = self.vis_feat_input_dim * 2

        self.ent_input_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.vis_feat_input_dim, self.hidden_dim)
        )

        self.rel_input_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.vis_feat_input_dim, self.hidden_dim)
        )

    def forward(self, entities_features, rel_features):
        rel_feat2cmp = self.rel_input_fc(rel_features)

        ent_feat2cmp = self.ent_input_fc(entities_features)

        scaling = float(self.hidden_dim) ** -0.5
        attn_output_weights = rel_feat2cmp @ ent_feat2cmp.permute(0, 2, 1) * scaling

        return attn_output_weights

@ENTITIES_INDEXING_HEAD.register_module()
class EntitiesIndexingHeadHOTR(BaseModule):
    def __init__(
        self, 
        embed_dims,
        num_classes,
        init_cfg = None,
    ):
        super(EntitiesIndexingHead, self).__init__(init_cfg=init_cfg)

        self.tau = 0.05

        self.hidden_dim = embed_dims

        self.H_Pointer_embed   = MLP(self.hidden_dim , self.hidden_dim , self.hidden_dim , 3)
        self.O_Pointer_embed   = MLP(self.hidden_dim , self.hidden_dim , self.hidden_dim , 3)

    def forward(self, entities_features, rel_features):

        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(rel_features), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(rel_features), p=2, dim=-1)
        outputs_hidx = [(torch.bmm(H_Pointer_repr, entities_features.transpose(1,2))) / self.tau for H_Pointer_repr in H_Pointer_reprs]
        outputs_oidx = [(torch.bmm(O_Pointer_repr, entities_features.transpose(1,2))) / self.tau for O_Pointer_repr in O_Pointer_reprs]

        return outputs_hidx, outputs_oidx

@ENTITIES_INDEXING_HEAD.register_module()
class EntitiesIndexingHeadRuleBased(BaseModule):
    def __init__(
        self, 
        embed_dims,
        num_classes,
        init_cfg = None,
    ):
        super(EntitiesIndexingHeadRuleBased, self).__init__(init_cfg=init_cfg)
        self.num_ent_class = num_classes
        self.normed_rel_vec_dist = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        ent_box_all = outputs["pred_boxes"]
        if outputs.get("pred_probs") is None:
            ent_prob_all = F.softmax(outputs["pred_logits"], dim=-1)

        obj_dist_mat = []
        sub_dist_mat = []

        for ind in range(len(outputs["pred_rel_obj_logits"])):
            ent_box = (
                bbox_cxcywh_to_xyxy(ent_box_all[ind]) * scale_fct[ind, None, :]
            )
            ent_box_cnter = bbox_xyxy_to_cxcywh(ent_box)[..., :2]

            ent_box_normed = ent_box_all[ind]
            ent_box_cnter_normed = ent_box_normed[..., :2]

            if self.num_ent_class == outputs["pred_rel_obj_logits"][ind].shape[-1]:
                pred_rel_obj_dist = torch.sigmoid(outputs["pred_rel_obj_logits"][ind])
                pred_rel_sub_dist = torch.sigmoid(outputs["pred_rel_sub_logits"][ind])
            else:
                pred_rel_obj_dist = F.softmax(
                    outputs["pred_rel_obj_logits"][ind], dim=-1
                )[..., :-1]
                pred_rel_sub_dist = F.softmax(
                    outputs["pred_rel_sub_logits"][ind], dim=-1
                )[..., :-1]

            pred_rel_obj_box = bbox_cxcywh_to_xyxy(
                outputs["pred_rel_obj_box"][ind]
            )
            pred_rel_obj_box = torch.squeeze(pred_rel_obj_box * scale_fct[ind, None, :])

            pred_rel_sub_box = bbox_cxcywh_to_xyxy(
                outputs["pred_rel_sub_box"][ind]
            )
            pred_rel_sub_box = torch.squeeze(pred_rel_sub_box * scale_fct[ind, None, :])

            # print((pred_rel_sub_box[:, 2:] < pred_rel_sub_box[:, :2]).sum())
            # print((pred_rel_obj_box[:, 2:] < pred_rel_obj_box[:, :2]).sum())
            # print((ent_box[:, 2:] <= ent_box[:, :2]).sum())
            if not (pred_rel_sub_box[:, 2:] >= pred_rel_sub_box[:, :2]).all():
                with open("box_tmp.pkl", 'wb') as f:
                    pickle.dump((pred_rel_sub_box, pred_rel_obj_box, ent_box), )
                

            rel_vec_flat_normed = outputs["pred_rel_vec"][ind]
            rel_vec_flat = rel_vec_flat_normed * scale_fct[ind, None, :]

            ent_prob = ent_prob_all[ind]

            if self.num_ent_class != ent_prob.shape[-1]:
                ent_prob = ent_prob[..., :-1]
            ent_score = ent_prob.max(-1)[0]

            (dist_s, dist_o, match_cost_details) = get_matching_scores_entities_aware(
                s_cetr=ent_box_cnter,
                o_cetr=ent_box_cnter,
                s_scores=ent_score,
                o_scores=ent_score,
                rel_vec=rel_vec_flat,
                s_cetr_normed=ent_box_cnter_normed,
                o_cetr_normed=ent_box_cnter_normed,
                rel_vec_normed=rel_vec_flat_normed,
                ent_box=ent_box,
                ent_box_normed=bbox_cxcywh_to_xyxy(ent_box_normed),
                s_dist=ent_prob,
                o_dist=ent_prob,
                rel_ent_s_box=pred_rel_sub_box,
                rel_ent_o_box=pred_rel_obj_box,
                rel_s_dist=pred_rel_sub_dist,
                rel_o_dist=pred_rel_obj_dist,
                normed_rel_vec_dist=self.normed_rel_vec_dist,
            )

            for k,v in match_cost_details.items():
                if torch.isnan(v).any():
                    print(k)
                if torch.isinf(v).any():
                    print(k)
            # if self.training:
            #     # suppress the low quality matching
            #     dist_s[match_cost_details["match_sub_giou"] < 0.7] *= 0.1
            #     dist_o[match_cost_details["match_obj_giou"] < 0.7] *= 0.1

            obj_dist_mat.append(dist_o)
            sub_dist_mat.append(dist_s)

        return torch.stack(sub_dist_mat).detach(), torch.stack(obj_dist_mat).detach()

@ENTITIES_INDEXING_HEAD.register_module()
class EntitiesIndexingHeadPredAtt(BaseModule):
    def __init__(self, 
        embed_dims,
        num_classes,
        init_cfg = None):
        super(EntitiesIndexingHeadPredAtt, self).__init__(init_cfg=init_cfg)

        self.hidden_dim = embed_dims

        self.cls_num = num_classes

        self.rel_geo_info_encode = nn.Sequential(nn.Linear(4 + 4, self.hidden_dim))

        self.ent_geo_info_encode = nn.Sequential(nn.Linear(4, self.hidden_dim))

        self.cls_info_encode_fc = nn.Sequential(
            nn.BatchNorm1d(self.cls_num),
            nn.Linear(self.cls_num, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.ent_sub_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.ent_obj_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.rel_obj_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        self.rel_sub_input_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def cls_info_encode(self, pred_dist):
        bs, num_q, cls_num = pred_dist.shape
        return self.cls_info_encode_fc(pred_dist.reshape(-1, cls_num)).reshape(
            bs, num_q, self.hidden_dim
        )

    def forward(self, outputs):
        pred_rel_vec = outputs["pred_rel_vec"]
        rel_obj_ent = []
        rel_sub_ent = []
        ent = []

        for ind, ent_det in enumerate(pred_rel_vec):
            ent_box = outputs["pred_boxes"][ind]
            num_ent = len(ent_box)
            ent_box_cnter = ent_box[..., :2]

            pred_rel_obj_dist = F.softmax(outputs["pred_rel_obj_logits"][ind], dim=-1)[
                ..., :-1
            ]
            pred_rel_obj_box = outputs["pred_rel_obj_box"][ind]
            pred_rel_obj_box = torch.squeeze(pred_rel_obj_box)

            pred_rel_sub_dist = F.softmax(outputs["pred_rel_sub_logits"][ind], dim=-1)[
                ..., :-1
            ]
            pred_rel_sub_box = outputs["pred_rel_sub_box"][ind]
            pred_rel_sub_box = torch.squeeze(pred_rel_sub_box)
            ent_prob = F.softmax(outputs["pred_logits"][ind], dim=-1)[..., :-1]

            rel_vec_flat = outputs["pred_rel_vec"][ind]

            rel_obj_ent.append(
                torch.cat((pred_rel_obj_dist, pred_rel_obj_box, rel_vec_flat), -1)
            )
            rel_sub_ent.append(
                torch.cat((pred_rel_sub_dist, pred_rel_sub_box, rel_vec_flat), -1)
            )
            ent.append(torch.cat((ent_prob, ent_box), -1))

        # todo word embedding

        rel_obj_ent_input = torch.stack(rel_obj_ent)
        rel_sub_ent_input = torch.stack(rel_sub_ent)
        ent_input = torch.stack(ent)

        rel_feat2cmp_obj = self.rel_obj_input_fc(
            torch.cat(
                (
                    self.rel_geo_info_encode(rel_obj_ent_input[:, :, -8:]),
                    self.cls_info_encode(rel_obj_ent_input[:, :, :-8]),
                ),
                dim=-1,
            )
        )

        rel_feat2cmp_sub = self.rel_sub_input_fc(
            torch.cat(
                (
                    self.rel_geo_info_encode(rel_sub_ent_input[:, :, -8:]),
                    self.cls_info_encode(rel_sub_ent_input[:, :, :-8]),
                ),
                dim=-1,
            )
        )

        ent_feat2cmp_obj = self.ent_obj_input_fc(
            torch.cat(
                (
                    self.ent_geo_info_encode(ent_input[:, :, -4:]),
                    self.cls_info_encode(ent_input[:, :, :-4]),
                ),
                dim=-1,
            )
        )
        ent_feat2cmp_sub = self.ent_sub_input_fc(
            torch.cat(
                (
                    self.ent_geo_info_encode(ent_input[:, :, -4:]),
                    self.cls_info_encode(ent_input[:, :, :-4]),
                ),
                dim=-1,
            )
        )

        scaling = float(self.hidden_dim) ** -0.5
        obj_attn_output = rel_feat2cmp_obj @ ent_feat2cmp_obj.permute(0, 2, 1) * scaling
        sub_attn_output = rel_feat2cmp_sub @ ent_feat2cmp_sub.permute(0, 2, 1) * scaling

        return obj_attn_output, sub_attn_output

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
        ent_box_normed,
        s_dist,
        o_dist,
        rel_ent_s_box,
        rel_ent_o_box,
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

    match_cost_details["match_sub_giou"] = match_sub_giou
    match_cost_details["match_obj_giou"] = match_obj_giou

    match_scr_sub = match_rel_sub_cls * 32 ** (match_sub_giou) * match_vec_n_conf_sub
    match_scr_obj = match_rel_obj_cls * 32 ** (match_obj_giou) * match_vec_n_conf_obj

    # match_scr_sub = minmax_norm(match_scr_sub)
    # match_scr_obj = minmax_norm(match_scr_obj)

    match_cost_details["match_scr_sub"] = match_scr_sub
    match_cost_details["match_scr_obj"] = match_scr_obj

    return match_scr_sub, match_scr_obj, match_cost_details

def minmax_norm(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data) + 0.02)

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