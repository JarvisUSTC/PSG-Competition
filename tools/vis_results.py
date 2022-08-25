# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from pathlib import Path

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset, replace_ImageToTensor
from openpsg.utils.vis_tools.preprocess import load_json

from openpsg.utils.utils import show_result
from openpsg.utils.utils_xiaowen import show_result_xiaowen


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('prediction_path',
                        help='prediction path where test pkl result')
    parser.add_argument('show_dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--img_idx',
                        default=[0,1],
                        nargs='+',
                        type=int,
                        help='which image to show')
    parser.add_argument('--wait-time',
                        type=float,
                        default=0,
                        help='the interval of show (s), 0 is block')
    parser.add_argument('--topk',
                        default=20,
                        type=int,
                        help='saved Number of the highest topk '
                        'and lowest topk after index sorting')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0,
                        help='score threshold (default: 0.)')
    parser.add_argument('--one_stage', default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    #print(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    #print(dataset)
    outputs = mmcv.load(args.prediction_path)
    psg_dataset_file = load_json(Path(cfg.data.test.ann_file))
    test_jsons = {}
    for js in psg_dataset_file['data']:
        if js['image_id'] in psg_dataset_file['test_image_ids']:
            test_jsons[js['file_name'].split(".jpg")[0].split("/")[-1]] = js

    for idx in range(0, 100):
        print(idx, flush=True)
        img = dataset[idx]['img_metas'][0].data['filename']
        data_ = test_jsons[img.split(".jpg")[0].split("/")[-1]]
        #print(data_)
        #print(dataset[idx]['img_metas'][0])
        seg_info = data_['segments_info']
        seg_map = osp.join(cfg.data.test.seg_prefix,data_['pan_seg_file_name'])
        rels = data_['relations']
        result = outputs[idx]
        out_filepath = osp.join(args.show_dir, f'{idx}.png')
        show_result_xiaowen(img,
                    result,
                    seg_info,
                    seg_map,
                    rels,
                    is_one_stage=args.one_stage,
                    num_rel=args.topk,
                    out_dir = osp.join(args.show_dir, f'{idx}/'),
                    out_file=out_filepath)


if __name__ == '__main__':
    main()
