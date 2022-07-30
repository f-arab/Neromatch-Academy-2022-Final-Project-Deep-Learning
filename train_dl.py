import time
import os
import torch
import argparse
from trashdetect_engine.models.segmentation_models import (
    get_instance_segmentation_model,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Prepare instance segmentation task with Mask R-CNN"
    )
    parser.add_argument(
        "--output_dir",
        help="path to save checkpoints",
        default=f"/mnt/ssd1T/TACO/detect-waste/MaskRCNN/output",
        type=str,
    )
    parser.add_argument(
        "--images_dir",
        help="path to images directory",
        default="/mnt/ssd1T/TACO/TACO/data",
        type=str,
    )
    parser.add_argument(
        "--anno_name",
        help="path to annotation json (part name)",
        default="/mnt/ssd1T/TACO/detect-waste/annotations/annotations_binary",
        type=str,
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # Devices
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--test-batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--gpu_id", default=0, type=int)
    # Learning
    parser.add_argument("--num_epochs", default=26, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument(
        "--lr-step-size", default=0, type=int, help="decrease lr every step-size epochs"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--optimizer",
        help="Chose type of optimization algorithm, SGD as default",
        default="SGD",
        choices=["AdamW", "SGD"],
        type=str,
    )
    # Model
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument(
        "--model",
        default="maskrcnn_resnet50_fpn",
        type=str,
        choices=[
            "maskrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "retinanet_resnet50_fpn",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
        ],
    )
    ##
    parser.add_argument("--wandb", action="store_true")

    return parser


from trashdetect_engine.engine import WasteDetectModelDL
from trashdetect_engine.data import WasteDatasetDL
from pytorch_lightning import Trainer

if __name__ == "__main__":
    parser = get_args_parser()

    args = parser.parse_args()
    args.lr = 0.001
    args.optimizer = "AdamW"
    args.test_batch_size = 1
    args.batch_size = 4

    data_module_dl = WasteDatasetDL(args, return_masks=True)

    # if args.test_dataloader:
    #     for batch in train_dataloader:
    #         pass

    # our dataset has two classes only - background and waste
    num_classes = args.num_classes

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes, args.model)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )

    # and a learning rate scheduler
    if args.lr_step_size != 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
        )

    from trashdetect_engine.data import get_coco_api_from_dataset
    from trashdetect_engine.engine import _get_iou_types
    from trashdetect_engine.coco_eval import CocoEvaluator
    from trashdetect_engine import utils

    coco = get_coco_api_from_dataset(data_module_dl.dataset_val)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    model_dl = WasteDetectModelDL(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        coco_evaluator=coco_evaluator,
        args=args,
    )
    # define model
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # from pytorch_lightning.loggers import WandbLogger
    # wandb_logger = WandbLogger()

    # trainer = Trainer(logger=wandb_logger)

    trainer = Trainer(
        devices=args.gpu_id + 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=args.num_epochs,
        callbacks=[],
        # fast_dev_run=5,  # Runs 5 batches
        # limit_train_batches=0.01,
    )
    trainer.fit(model_dl, data_module_dl)
