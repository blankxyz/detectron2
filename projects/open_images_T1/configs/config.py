# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN
from detectron2 import model_zoo

def add_T1_config(cfg):
    """
    Add config for PointRend.
    """
    cfg.DATASETS.TRAIN = ("coco_my_train",)
    cfg.DATASETS.TEST = ('coco_my_val',)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        # "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATALOADER.NUM_WORKERS = 2
    # 1 gpu 2, 0.0025
    # 2 gpu 4, 0.005
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR =  0.00025

    cfg.SOLVER.MAX_ITER = (
        10000  # 1000 iterations is a good start, for better accuracy increase this value
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        512  # (default: 512), select smaller if faster training is needed
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)
    # 学习率衰减方式-----------
    # cfg.SOLVER.WEIGHT_DECAY = 0.0001  # 权重衰减
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0001  # 权重衰减系数
    # cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

    cfg.OUTPUT_DIR = '/data1/bb/src/detectron2/datasets/seg_T1_output'
