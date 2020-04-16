
## Getting Started with Detectron2

This document provides a brief intro of the usage of builtin command-line tools in detectron2.

For a tutorial that involves actual coding with the API,
see our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
which covers how to run inference with an
existing model, and how to train a builtin model on a custom dataset.

For more advanced tutorials, refer to our [documentation](https://detectron2.readthedocs.io/tutorials/extend.html).

本文档简要介绍了detectron2中内置命令行工具的用法。

有关涉及使用API​​进行实际编码的教程，请参阅我们的Colab笔记本 ，其中涵盖了如何对现有模型进行推理，以及如何在自定义数据集上训练内置模型。

有关更高级的教程，请参阅我们的文档。

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
	[model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md),
	for example, `mask_rcnn_R_50_FPN_3x.yaml`.
	
	从模型Zoo中选择一个模型及其配置文件 ，例如mask_rcnn_R_50_FPN_3x.yaml。
	
2. We provide `demo.py` that is able to run builtin standard models. Run it with:

   我们提供demo.py能够运行内置标准模型的工具。使用以下命令运行它：
```
python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

这些配置是为了进行培训而设计的，因此我们需要指定MODEL.WEIGHTS来自模型动物园的模型进行评估。此命令将运行推断并在OpenCV窗口中显示可视化效果。

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:

有关命令行参数的详细信息，请参阅或查看其源代码以了解其行为。一些常见的参数是：demo.py -h

* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


### Training & Evaluation in Command Line （命令行中的培训与评估）

We provide a script in "tools/{,plain_}train_net.py", that is made to train
all the configs provided in detectron2.

我们在“ tools / {，plain_} train_net.py”中提供了一个脚本，该脚本用于训练detectron2中提供的所有配置。您可能希望将其用作编写新研究的训练脚本的参考。

You may want to use it as a reference to write your own training script for a new research.

To train a model with "train_net.py", first
setup the corresponding datasets following

要使用“ train_net.py”训练模型，请首先在datasets / README.md之后设置相应的数据 集，然后运行：
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:
```
python tools/train_net.py --num-gpus 8 \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```

The configs are made for 8-GPU training. To train on 1 GPU, change the batch size with:

这些配置是为8-GPU训练而设计的。要在1个GPU上进行训练，请使用以下命令更改批量大小：

```
python tools/train_net.py \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

For most models, CPU training is not supported.

(Note that we applied the [linear learning rate scaling rule](https://arxiv.org/abs/1706.02677)
when changing the batch size.)
（请注意， 更改批次大小时，我们应用了线性学习率缩放规则。）


To evaluate a model's performance, use

要评估模型的性能，请使用
```
python tools/train_net.py \
	--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python tools/train_net.py -h`.

### Use Detectron2 APIs in Your Code

See our [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn how to use detectron2 APIs to:
1. run inference with an existing model
2. train a builtin model on a custom dataset

See [detectron2/projects](https://github.com/facebookresearch/detectron2/tree/master/projects)
for more ways to build your project on detectron2.
