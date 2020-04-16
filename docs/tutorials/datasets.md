# Use Custom Datasets

If you want to use a custom dataset while also reusing detectron2's data loaders,
you will need to

1. Register your dataset (i.e., tell detectron2 how to obtain your dataset).
2. Optionally, register metadata for your dataset.

Next, we explain the above two concepts in details.

The [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
has a working example of how to register and train on a dataset of custom formats.

如果要使用自定义数据集，同时还重复使用detectron2的数据加载器，则需要

注册您的数据集（即，告诉detectron2如何获取您的数据集）。
（可选）为数据集注册元数据。
接下来，我们详细解释上述两个概念。

该Colab笔记本 有如何在自定义格式的数据集注册和培训工作的例子。

### Register a Dataset

To let detectron2 know how to obtain a dataset named "my_dataset", you will implement
a function that returns the items in your dataset and then tell detectron2 about this
function:

为了让detectron2知道如何获取名为“ my_dataset”的数据集，您将实现一个函数，该函数返回数据集中的项目，然后将此函数告知detectron2：

```python
def get_dicts():
  ...
  return list[dict] in the following format

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", get_dicts)
```

在此，代码段将数据集“ my_dataset”与返回数据的函数相关联。在该过程存在之前，注册将一直有效。

该函数可以将数据从其原始格式处理为以下任意一种：

Detectron2的标准数据集字典，如下所述。它可以与detectron2中的许多其他内置功能一起使用，因此建议在足以完成任务时使用它。
您的自定义数据集字典。您还可以以自己的格式返回任意字典，例如为新任务添加额外的键。然后，您还需要在下游正确处理它们。请参阅下面的更多细节。

Here, the snippet associates a dataset "my_dataset" with a function that returns the data.
The registration stays effective until the process exists.

The function can processes data from its original format into either one of the following:
1. Detectron2's standard dataset dict, described below. This will work with many other builtin
	 features in detectron2, so it's recommended to use it when it's sufficient for your task.
2. Your custom dataset dict. You can also returns arbitrary dicts in your own format,
	 such as adding extra keys for new tasks.
	 Then you will need to handle them properly in the downstream as well.
	 See below for more details.

#### Standard Dataset Dicts (标准数据集字典)

For standard tasks
(instance detection, instance/semantic/panoptic segmentation, keypoint detection),
we load the original dataset into `list[dict]` with a specification similar to COCO's json annotations.
This is our standard representation for a dataset.

对于标准任务（实例检测，实例/语义/全景分割，关键点检测），我们将原始数据集加载到list[dict]具有类似于COCO json注释的规范中。这是我们对数据集的标准表示。

Each dict contains information about one image.
The dict may have the following fields.
The fields are often optional, and some functions may be able to
infer certain fields from others if needed, e.g., the data loader
will load the image from "file_name" and load "sem_seg" from "sem_seg_file_name".

每个字典包含有关一个图像的信息。该词典可能具有以下字段。这些字段通常是可选的，如果需要，某些功能可能能够从其他字段推断某些字段，例如，数据加载器将从“ file_name”加载图像，并从“ sem_seg_file_name”加载“ sem_seg”。

+ `file_name`: the full path to the image file. Will apply rotation and flipping if the image has such exif information.
+ `sem_seg_file_name`: the full path to the ground truth semantic segmentation file.
    基本事实语义分段文件的完整路径。
+ `sem_seg`: semantic segmentation ground truth in a 2D `torch.Tensor`. Values in the array represent
   category labels starting from 0. 语义分割在2D中的真实性torch.Tensor。数组中的值表示从0开始的类别标签。
+ `height`, `width`: integer. The shape of image.
+ `image_id` (str or int): a unique id that identifies this image(标识此图像的唯一ID。). Used
	during evaluation to identify the images, but a dataset may use it for different purposes.
+ `annotations` (list[dict]): each dict corresponds to annotations of one instance
  in this image. 每个字典对应此图片中一个实例的注释。Images with empty `annotations` will by default be removed from training,
	but can be included using `DATALOADER.FILTER_EMPTY_ANNOTATIONS`.
	Each dict may contain the following keys:
  + `bbox` (list[float]): list of 4 numbers representing the bounding box of the instance.
  + `bbox_mode` (int): the format of bbox.
    It must be a member of
    [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
    Currently supports: `BoxMode.XYXY_ABS`, `BoxMode.XYWH_ABS`.
  + `category_id` (int): an integer in the range [0, num_categories) representing the category label.
    The value num_categories is reserved to represent the "background" category, if applicable.
  + `segmentation` (list[list[float]] or dict):
    + If `list[list[float]]`, it represents a list of polygons, one for each connected component
      of the object. Each `list[float]` is one simple polygon in the format of `[x1, y1, ..., xn, yn]`.
      The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
      depend on whether "bbox_mode" is relative.
    + If `dict`, it represents the per-pixel segmentation mask in COCO's RLE format. The dict should have
			keys "size" and "counts". You can convert a uint8 segmentation mask of 0s and 1s into
			RLE format by `pycocotools.mask.encode(np.asarray(mask, order="F"))`.
  + `keypoints` (list[float]): in the format of [x1, y1, v1,..., xn, yn, vn].
    v[i] means the [visibility](http://cocodataset.org/#format-data) of this keypoint.
    `n` must be equal to the number of keypoint categories.
    The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates,
    depend on whether "bbox_mode" is relative.

    Note that the coordinate annotations in COCO format are integers in range [0, H-1 or W-1].
    By default, detectron2 adds 0.5 to absolute keypoint coordinates to convert them from discrete
    pixel indices to floating point coordinates.
  + `iscrowd`: 0 or 1. Whether this instance is labeled as COCO's "crowd
    region". Don't include this field if you don't know what it means.

The following keys are used by Fast R-CNN style training, which is rare today.

+ `proposal_boxes` (array): 2D numpy array with shape (K, 4) representing K precomputed proposal boxes for this image.
+ `proposal_objectness_logits` (array): numpy array with shape (K, ), which corresponds to the objectness
  logits of proposals in 'proposal_boxes'.
+ `proposal_bbox_mode` (int): the format of the precomputed proposal bbox.
  It must be a member of
  [structures.BoxMode](../modules/structures.html#detectron2.structures.BoxMode).
  Default is `BoxMode.XYXY_ABS`.


If your dataset is already a json file in COCO format, you can simply register it by
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```
which will take care of everything (including metadata) for you.

If your dataset is in COCO format with custom per-instance annotations,
the [load_coco_json](../modules/data.html#detectron2.data.datasets.load_coco_json) function can be used.

#### Custom Dataset Dicts(自定义数据集字典)

In the `list[dict]` that your dataset function return, the dictionary can also has arbitrary custom data.
This can be useful when you're doing a new task and needs extra information not supported
by the standard dataset dicts. In this case, you need to make sure the downstream code can handle your data
correctly. Usually this requires writing a new `mapper` for the dataloader (see [Use Custom Dataloaders](data_loading.html))

When designing your custom format, note that all dicts are stored in memory
(sometimes serialized and with multiple copies).
To save memory, each dict is meant to contain small but sufficient information
about each sample, such as file names and annotations.
Loading full samples typically happens in the data loader.

For attributes shared among the entire dataset, use `Metadata` (see below).
To avoid exmemory, do not save such information repeatly for each sample.

在list[dict]数据集函数返回的目录中，字典也可以具有任意的自定义数据。当您要执行新任务并且需要标准数据集字典不支持的其他信息时，此功能很有用。在这种情况下，您需要确保下游代码可以正确处理您的数据。通常，这需要mapper为数据加载器编写新的内容（请参阅使用自定义数据加载器）。

在设计自定义格式时，请注意所有字典都存储在内存中（有时会序列化并带有多个副本）。为了节省内存，每个字典旨在包含有关每个样本的少量但足够的信息，例如文件名和注释。加载完整样本通常在数据加载器中进行。

对于在整个数据集中共享的属性，请使用Metadata（请参阅下文）。为避免内存不足，请勿为每个样本重复保存此类信息。

### "Metadata" for Datasets (数据集的“元数据” )

Each dataset is associated with some metadata, accessible through
`MetadataCatalog.get(dataset_name).some_metadata`.
Metadata is a key-value mapping that contains information that's shared among
the entire dataset, and usually is used to interpret what's in the dataset, e.g.,
names of classes, colors of classes, root of files, etc.
This information will be useful for augmentation, evaluation, visualization, logging, etc.
The structure of metadata depends on the what is needed from the corresponding downstream code.


If you register a new dataset through `DatasetCatalog.register`,
you may also want to add its corresponding metadata through
`MetadataCatalog.get(dataset_name).set(name, value)`, to enable any features that need metadata.
You can do it like this (using the metadata field "thing_classes" as an example):

```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
```

Here is a list of metadata keys that are used by builtin features in detectron2.
If you add your own dataset without these metadata, some features may be
unavailable to you:

* `thing_classes` (list[str]): Used by all instance detection/segmentation tasks.
  A list of names for each instance/thing category.
  If you load a COCO format dataset, it will be automatically set by the function `load_coco_json`.

* `thing_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each thing category.
  Used for visualization. If not given, random colors are used.

* `stuff_classes` (list[str]): Used by semantic and panoptic segmentation tasks.
  A list of names for each stuff category.

* `stuff_colors` (list[tuple(r, g, b)]): Pre-defined color (in [0, 255]) for each stuff category.
  Used for visualization. If not given, random colors are used.

* `keypoint_names` (list[str]): Used by keypoint localization. A list of names for each keypoint.

* `keypoint_flip_map` (list[tuple[str]]): Used by the keypoint localization task. A list of pairs of names,
  where each pair are the two keypoints that should be flipped if the image is
  flipped during augmentation.
* `keypoint_connection_rules`: list[tuple(str, str, (r, g, b))]. Each tuple specifies a pair of keypoints
  that are connected and the color to use for the line between them when visualized.

Some additional metadata that are specific to the evaluation of certain datasets (e.g. COCO):

* `thing_dataset_id_to_contiguous_id` (dict[int->int]): Used by all instance detection/segmentation tasks in the COCO format.
  A mapping from instance class ids in the dataset to contiguous ids in range [0, #class).
  Will be automatically set by the function `load_coco_json`.

* `stuff_dataset_id_to_contiguous_id` (dict[int->int]): Used when generating prediction json files for
  semantic/panoptic segmentation.
  A mapping from semantic segmentation class ids in the dataset
  to contiguous ids in [0, num_categories). It is useful for evaluation only.

* `json_file`: The COCO annotation json file. Used by COCO evaluation for COCO-format datasets.
* `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
* `evaluator_type`: Used by the builtin main training script to select
   evaluator. No need to use it if you write your own main script.
   You can just provide the [DatasetEvaluator](../modules/evaluation.html#detectron2.evaluation.DatasetEvaluator)
   for your dataset directly in your main script.

NOTE: For background on the concept of "thing" and "stuff", see
[On Seeing Stuff: The Perception of Materials by Humans and Machines](http://persci.mit.edu/pub_pdfs/adelson_spie_01.pdf).
In detectron2, the term "thing" is used for instance-level tasks,
and "stuff" is used for semantic segmentation tasks.
Both are used in panoptic segmentation.


每个数据集都与一些元数据相关联，可通过访问这些元数据 MetadataCatalog.get(dataset_name).some_metadata。元数据是一个键值映射，其中包含在整个数据集中共享的信息，通常用于解释数据集中的内容，例如，类的名称，类的颜色，文件的根目录等。此信息对于元数据的结构取决于相应下游代码的需求。

如果通过来注册新的数据集DatasetCatalog.register，则可能还需要通过来添加其相应的元数据 ，以启用需要元数据的任何功能。您可以这样做（以元数据字段“ thing_classes”为例）：MetadataCatalog.get(dataset_name).set(name, value)

from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]
这是detectron2中的内置功能使用的元数据密钥的列表。如果您添加没有这些元数据的自己的数据集，则某些功能可能对您不可用：

thing_classes（list [str]）：由所有实例检测/分段任务使用。每个实例/事物类别的名称列表。如果加载COCO格式的数据集，它将由函数自动设置load_coco_json。
thing_colors（list [tuple（r，g，b）]）：每个事物类别的预定义颜色（在[0，255]中）。用于可视化。如果未给出，则使用随机颜色。
stuff_classes（list [str]）：用于语义和全景分割任务。每个物料类别的名称列表。
stuff_colors（list [tuple（r，g，b）]）：每个填充类别的预定义颜色（在[0，255]中）。用于可视化。如果未给出，则使用随机颜色。
keypoint_names（list [str]）：由关键点本地化使用。每个关键点的名称列表。
keypoint_flip_map（list [tuple [str]]）：由关键点本地化任务使用。名称对列表，其中每对是在增强过程中翻转图像时应翻转的两个关键点。
keypoint_connection_rules：列表[tuple（str，str，（r，g，b））]。每个元组指定一对已连接的关键点，以及在可视化时用于它们之间的线的颜色。
一些特定于某些数据集评估的其他元数据（例如COCO）：

thing_dataset_id_to_contiguous_id（dict [int-> int]）：由COCO格式的所有实例检测/分段任务使用。从数据集中的实例类ID到[0，#class）范围内的连续ID的映射。将由该功能自动设置load_coco_json。
stuff_dataset_id_to_contiguous_id（dict [int-> int]）：在生成用于语义/全景分割的预测json文件时使用。从数据集中的语义细分类ID到[0，num_categories）中的连续ID的映射。它仅对评估有用。
json_file：COCO注释json文件。由COCO评估用于COCO格式的数据集。
panoptic_root，panoptic_json：由全景评估中使用。
evaluator_type：由内置的主要培训脚本用来选择评估者。如果您编写自己的主脚本，则无需使用它。您可以 直接在主脚本中为数据集提供DatasetEvaluator。
注意：有关“事物”和“材料”概念的背景知识，请参见 “看到材料：人与机器对材料的感知”。在detectron2中，术语“事物”用于实例级任务，而“东西”用于语义分割任务。两者都用于全景分割。

### Update the Config for New Datasets

Once you've registered the dataset, you can use the name of the dataset (e.g., "my_dataset" in
example above) in `DATASETS.{TRAIN,TEST}`.
There are other configs you might want to change to train or evaluate on new datasets:

* `MODEL.ROI_HEADS.NUM_CLASSES` and `MODEL.RETINANET.NUM_CLASSES` are the number of thing classes
	for R-CNN and RetinaNet models.
* `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` sets the number of keypoints for Keypoint R-CNN.
  You'll also need to set [Keypoint OKS](http://cocodataset.org/#keypoints-eval)
	with `TEST.KEYPOINT_OKS_SIGMAS` for evaluation.
* `MODEL.SEM_SEG_HEAD.NUM_CLASSES` sets the number of stuff classes for Semantic FPN & Panoptic FPN.
* If you're training Fast R-CNN (with precomputed proposals), `DATASETS.PROPOSAL_FILES_{TRAIN,TEST}`
	need to match the datasts. The format of proposal files are documented
	[here](../modules/data.html#detectron2.data.load_proposals_into_dataset).
