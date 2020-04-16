# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

import cv2
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from configs.config import add_T1_config



CLASS_NAMES = ["Fast food", "Shorts","Bus","Laptop","Vehicle registration plate"]

# 数据集路径
# --------------------
DATASET_ROOT = '/data2/open_images_v6_imgs/'
ANN_ROOT = '/data1/bb/src/detectron2/datasets/convert/'

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'test')

TRAIN_JSON = os.path.join(ANN_ROOT, 'seg_test_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'seg_test_test.json')


# 数据集的子集
# ----------------------
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_coco_instances(name=key, metadata={} ,
                                   json_file=json_file,
                                   image_root=image_root)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    add_T1_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

# 查看数据集标注
def checkout_dataset_annotation(name="coco_my_val"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        if i == 200:
            break

def main(args):
    cfg = setup(args)

    # 注册数据集
    register_dataset()
    # plain_register_dataset()

    # 检测数据集注释是否正确
    # checkout_dataset_annotation()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # 对象检测配置
    # args.config_file = "/data1/bb/src/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    # 对象分割配置
    args.config_file = "/data1/bb/src/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # args.resume = Trueopmkl[l[olp]
    #
    # args.num_gpus = 2
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
