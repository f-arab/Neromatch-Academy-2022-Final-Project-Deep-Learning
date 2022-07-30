from django.urls import include
import torchvision
import torch
from .detection.mask_rcnn import MaskRCNN
from .detection.faster_rcnn import FastRCNNPredictor
from .detection.mask_rcnn import MaskRCNNPredictor

from efficientnet_pytorch import EfficientNet

from trashdetect_engine.models.backbone import resnet

# from https://github.com/lukemelas/EfficientNet-PyTorch
def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def get_instance_segmentation_model(
    num_classes, model_name="maskrcnn_resnet50_fpn", max_size=512, min_size=512
):
    # load a pre-trained model for classification
    # and return only the features
    if model_name.startswith("efficientnet"):
        backbone = EfficientNet.from_pretrained(
            model_name, num_classes=num_classes, include_top=False
        )
        # number of output channels
        backbone.out_channels = int(round_filters(1280, backbone._global_params))
        model = MaskRCNN(backbone, num_classes, min_size=min_size, max_size=max_size)
    else:
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.__dict__[model_name](
            pretrained=True, min_size=min_size, max_size=max_size
        )
        # backbone = resnet.resnet50(pretrained=True, include_top=False)
        # backbone.out_channels = (
        #     2048  # int(round_filters(1280, backbone._global_params))
        # )

    # model = MaskRCNN(backbone, num_classes, min_size=min_size, max_size=max_size)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if model_name.startswith("mask") or model_name.startswith("efficientnet"):
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    return model
