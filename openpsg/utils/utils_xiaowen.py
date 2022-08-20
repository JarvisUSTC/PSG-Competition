from typing import Tuple
import os.path as osp
import PIL
import mmcv
import mmcv.ops as ops
import numpy as np
import torch
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import VisImage, Visualizer
from detectron2.data.detection_utils import read_image
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
import matplotlib.pyplot as plt

# from mmcv.ops.nms import batched_nms


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def adjust_text_color(color: Tuple[float, float, float],
                      viz: Visualizer) -> Tuple[float, float, float]:
    color = viz._change_color_brightness(color, brightness_factor=0.7)
    color = np.maximum(color, 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))
    return color


def draw_text(
    viz_img: VisImage = None,
    text: str = None,
    x: float = None,
    y: float = None,
    color: Tuple[float, float, float] = [0, 0, 0],
    size: float = 10,
    padding: float = 5,
    box_color: str = 'black',
    font: str = None,
) -> float:
    text_obj = viz_img.ax.text(
        x,
        y,
        text,
        size=size,
        # family="sans-serif",
        bbox={
            'facecolor': box_color,
            'alpha': 0.8,
            'pad': padding,
            'edgecolor': 'none',
        },
        verticalalignment='top',
        horizontalalignment='left',
        color=color,
        zorder=10,
        rotation=0,
    )
    viz_img.get_image()
    text_dims = text_obj.get_bbox_patch().get_extents()

    return text_dims.width


def multiclass_nms_alt(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    return_dist=False,
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        return_dist (bool): whether to return score dist.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        # (N_b, N_c, 4)
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr  # (N_b, N_c)
    valid_box_idxes = torch.nonzero(valid_mask)[:, 0].view(-1)
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    score_dists = scores[valid_box_idxes, :]
    # add bg column for later use.
    score_dists = torch.cat(
        (torch.zeros(score_dists.size(0), 1).to(score_dists), score_dists),
        dim=-1)
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        if return_dist:
            return bboxes, (labels, multi_bboxes.new_zeros(
                (0, num_classes + 1)))
        else:
            return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(ops, nms_type)
    dets, keep = nms_op(bboxes_for_nms, scores, **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]
    score_dists = score_dists[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        score_dists = score_dists[inds]

    if return_dist:
        # score_dists has bg_column
        return torch.cat([bboxes, scores[:, None]],
                         1), (labels.view(-1), score_dists)
    else:
        return torch.cat([bboxes, scores[:, None]], 1), labels.view(-1)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged', 'background'
]

PREDICATES = [
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

def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def show_result_xiaowen(img,
                result,
                seg_info,
                seg_map,
                rels,
                is_one_stage,
                out_file=None,
                num_topk=20):
    """
    A shit mountain stiched function to generate better visualization images and save it.
    Inputs:
        img: path of the ground truth image
        result: from result.pkl
        seg_info: key 'segments_info' in annotations
        seg_map: path of the ground truth segmentation mask
        rels: key 'relations' in annotations
        is_one_stage: just consider it to be true
        out_file: the path to wrtie outputs
        num_topk: top k predictions to consider
    Return:
        None
    """
    
    # Load image
    img = mmcv.imread(img)
    img_ = img.copy()
    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]
    
    # Process masks and labels
    seg_map = read_image(seg_map, format="RGB")
    from panopticapi.utils import rgb2id
    seg_map = rgb2id(seg_map)
    
    # prepare to write groundtruth relations
    masks = []
    labels_coco = []
    label_indices_for_tri = []
    for i, s in enumerate(seg_info):
        label_indices_for_tri.append(s["category_id"])
        label = CLASSES[s["category_id"]]
        labels_coco.append(label)
        masks.append(seg_map == s["id"])

    colormap_coco = get_colormap(len(seg_info))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    viz_with_seg = Visualizer(img)
    viz_with_seg.overlay_instances(
        labels=labels_coco,
        masks=masks,
        assigned_colors=colormap_coco,
    )
    viz_with_seg = viz_with_seg.get_output().get_image()

    # Decrease contrast
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)
    '''if out_file is not None:
        mmcv.imwrite(np.asarray(img), 'bw'+out_file)'''

    # Draw masks
    pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [CLASSES[l] for l in labels]

    #For psgtr
    rel_obj_labels = result.labels
    rel_obj_labels = [CLASSES[l - 1] for l in rel_obj_labels]

    # (N_m, H, W)
    segms = pan_results[None] == ids[:, None, None]
    # Resize predicted masks
    segms = [
        mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    ]
    # One stage segmentation
    res_masks = result.masks

    # Choose colors for each instance in coco
    pred_colormap_coco = get_colormap(len(res_masks)) if is_one_stage else get_colormap(len(segms))
    pred_colormap_coco = (np.array(pred_colormap_coco) / 255).tolist()

    # Viualize masks
    viz = Visualizer(img)
    viz.overlay_instances(
        labels=rel_obj_labels if is_one_stage else labels,
        masks=res_masks if is_one_stage else segms,
        assigned_colors=pred_colormap_coco,
    )
    viz_img = viz.get_output().get_image()
    '''if out_file is not None:
        mmcv.imwrite(viz_img, out_file)'''

    # match predictions with ground truths
    # generate groundtruth triplets
    gt_triplets, gt_triplet_det_results, _ = _triplet_panseg(np.array(rels), np.array(label_indices_for_tri), masks)
    rel_scores = result.rel_dists

    # generate prediction triplets
    pred_rels = np.column_stack(
        (result.rel_pair_idxes, 1 + rel_scores[:, 1:].argmax(1)))
    pred_scores = rel_scores[:, 1:].max(1)
    pred_triplets, pred_triplet_det_results, _ = _triplet_panseg(pred_rels, result.labels, res_masks, pred_scores, result.refine_bboxes[:, -1])

    #use torch to get topk prediction results, NOTE that only predicate score is considered
    scores_pt = torch.from_numpy(_[:,1]) #only predicate scores
    topk_values, topk_indices = scores_pt.topk(num_topk, sorted=True)
    pred_triplets = pred_triplets[topk_indices]
    pred_triplet_det_results = pred_triplet_det_results[topk_indices]
    pred_triplets = pred_triplets - 1 # remove background class
    pred_obj_indices = result.rel_pair_idxes[topk_indices] #link topk predicted relations to object masks

    iou_thrs = 0.5
    pred_to_gt = _compute_pred_matches_panseg( #pred_to_gt marks gt to prediction, n x [gt_index, prediction_index]
        gt_triplets,
        pred_triplets,
        gt_triplet_det_results,
        pred_triplet_det_results,
        iou_thrs,
        phrdet=False, #seperately calculate iou for object and subject
    )

    # Filter out relations
    relations = pred_triplets
    n_rels = len(relations)
    
    viz_img = mmcv.image.imresize(viz_img, (int(img_w*1.5), int(img_h*1.5)))
    img_w = int(img_w*1.5)
    img_h = int(img_h*1.5)

    # write groundtruth relations    
    viz_with_seg = mmcv.image.imresize(viz_with_seg, (img_w, img_h))

    for i, r in enumerate(rels):
        s_idx, o_idx, rel_id = r
        s_label = labels_coco[s_idx]
        o_label = labels_coco[o_idx]
        rel_label = PREDICATES[rel_id]
        viz_graph = VisImage(np.full((40, img_w, 3), 255))
        curr_x = 2
        curr_y = 2
        text_size = 25
        text_padding = 20
        font = 36

        #add matching tag
        text_width = draw_text(
            viz_img=viz_graph,
            text=str(i) + " ",
            x=curr_x,
            y=curr_y,
            color=(1,1,1),
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width

        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width
        # Draw relation text
        text_width = draw_text(
            viz_img=viz_graph,
            text=rel_label,
            x=curr_x,
            y=curr_y,
            size=text_size,
            padding=text_padding,
            box_color='gainsboro',
            font=font,
        )
        curr_x += text_width

        # Draw object text
        text_width = draw_text(
            viz_img=viz_graph,
            text=o_label,
            x=curr_x,
            y=curr_y,
            color=colormap_coco[o_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        viz_with_seg = np.vstack([viz_with_seg, viz_graph.get_image()])

    # prepare variables to write topk predicted relations
    top_padding = 20
    bottom_padding = 20
    left_padding = 20
    text_size = 10
    text_padding = 5
    text_height = text_size + 2 * text_padding
    row_padding = 10
    height = (top_padding + bottom_padding + n_rels *
              (text_height + row_padding) - row_padding)
    width = img_w
    curr_x = left_padding
    curr_y = top_padding
    
    viz_graph = VisImage(np.full((height, width, 3), 255))
    
    # write topk predicted relations
    for i, r in enumerate(relations):
        rel_id = r[1]
        s_idx, o_idx = pred_obj_indices[i] # get corresponding object id, NOTE that r contains class id, which cannot be associated with specific predicted masks 
        s_label = rel_obj_labels[s_idx] 
        o_label = rel_obj_labels[o_idx]
        rel_label = PREDICATES[rel_id]
        viz_graph = VisImage(np.full((40, width, 3), 255))
        curr_x = 2
        curr_y = 2
        text_size = 25
        text_padding = 20
        font = 36

        if len(pred_to_gt[i]) != 0:
            tx = str(pred_to_gt[i][0])
        else:
            tx = "  "
        text_width = draw_text(
            viz_img=viz_graph,
            text=tx,
            x=curr_x,
            y=curr_y,
            color=(1,1,1),
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width


        text_width = draw_text(
            viz_img=viz_graph,
            text=s_label,
            x=curr_x,
            y=curr_y,
            color=pred_colormap_coco[s_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        curr_x += text_width
        # Draw relation text
        text_width = draw_text(
            viz_img=viz_graph,
            text=rel_label,
            x=curr_x,
            y=curr_y,
            size=text_size,
            padding=text_padding,
            box_color='gainsboro',
            font=font,
        )
        curr_x += text_width

        # Draw object text
        text_width = draw_text(
            viz_img=viz_graph,
            text=o_label,
            x=curr_x,
            y=curr_y,
            color=pred_colormap_coco[o_idx],
            size=text_size,
            padding=text_padding,
            font=font,
        )
        viz_img = np.vstack([viz_img, viz_graph.get_image()])

    # stack up three columns of images and write the final result
    vis_h, vis_w = viz_img.shape[:-1]
    img_ = mmcv.image.imresize(img_, (img_w, img_h))
    original_img = np.full((vis_h, vis_w, 3), 255)
    original_img[:img_h, :img_w, :] = img_
    masked_img = np.full((vis_h, vis_w, 3), 255)
    masked_img[:viz_with_seg.shape[0], :img_w, :] = viz_with_seg
    viz_img = np.hstack([original_img, masked_img, viz_img])
    if out_file is not None:
        mmcv.imwrite(viz_img, out_file)


def intersect_2d(x1, x2):
    """Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each
    entry is True if those rows match.

    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError('Input arrays must have same #columns')

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def _triplet_panseg(relations,
                    classes,
                    masks,
                    predicate_scores=None,
                    class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        masks (#objs, )
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplet_masks(#rel, 2, , )
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:,
                                                                            2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    masks = np.array(masks)
    triplet_masks = np.stack((masks[sub_id], masks[ob_id]), axis=1)

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id],
            predicate_scores,
            class_scores[ob_id],
        ))

    return triplets, triplet_masks, triplet_scores

def _compute_pred_matches_panseg(gt_triplets,
                                 pred_triplets,
                                 gt_masks,
                                 pred_masks,
                                 iou_thrs,
                                 phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    #here gt_has_match means a gt triplet matches a prediction in S,P,O classes, didn't consider mask yet
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_masks.shape[0])]

    for gt_ind, gt_mask, keep_inds in zip(
            np.where(gt_has_match)[0],
            gt_masks[gt_has_match],
            keeps[gt_has_match],
    ):
        #print(gt_ind, gt_mask.shape, keep_inds)
        pred_mask = pred_masks[keep_inds]

        sub_gt_mask = gt_mask[0]
        ob_gt_mask = gt_mask[1]
        sub_pred_mask = pred_mask[:, 0]
        ob_pred_mask = pred_mask[:, 1]

        if phrdet:
            # Evaluate where the union mask > 0.5
            inds = []
            gt_mask_union = np.logical_or(sub_gt_mask, ob_gt_mask)
            pred_mask_union = np.logical_or(sub_pred_mask, ob_pred_mask)
            #print(pred_mask_union.shape)
            for pred_mask in pred_mask_union:
                iou = mask_iou(gt_mask_union, pred_mask)
                #print(iou)
                inds.append(iou >= iou_thrs)
        else:
            sub_inds = []
            for pred_mask in sub_pred_mask:
                sub_iou = mask_iou(sub_gt_mask, pred_mask)
                sub_inds.append(sub_iou >= iou_thrs)
            ob_inds = []
            for pred_mask in ob_pred_mask:
                ob_iou = mask_iou(ob_gt_mask, pred_mask)
                ob_inds.append(ob_iou >= iou_thrs)

            inds = np.logical_and(sub_inds, ob_inds)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def mask_iou(mask1, mask2):
    assert mask1.shape == mask2.shape
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou