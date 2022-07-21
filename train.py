import time
import os
import torch
from trashdetect_engine.engine import train_one_epoch, evaluate
import argparse
from trashdetect_engine import utils

from trashdetect_engine.data import build
from trashdetect_engine.models.segmentation_models import (
    get_instance_segmentation_model,
)

from datetime import datetime
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Prepare instance segmentation task with Mask R-CNN"
    )
    parser.add_argument(
        "--output_dir",
        help="path to save checkpoints",
        default="/mnt/ssd1T/TACO/detect-waste/MaskRCNN/output",
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


def train(args):
    start_epoch = 0
    return_masks = False
    if args.wandb and (not args.resume):
        import wandb

        exp_logger = wandb.init(
            project="wastedetect",
            entity="nma2022-wastedetect",
            name=f"experiment_{utils.generate_datetime()}",
        )
        wandb.config = vars(args)

    else:
        exp_logger = None

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if args.model.startswith("mask"):
        return_masks = True

    # use our dataset and defined transformations
    dataset_train = build("train", args.images_dir, args.anno_name, return_masks)
    dataset_val = build("val", args.images_dir, args.anno_name, return_masks)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True
    )

    # define model
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # our dataset has two classes only - background and waste
    num_classes = args.num_classes

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes, args.model)

    # move model to the right device
    model.to(device)

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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        # evaluate on the test dataset
        print("Start evaluating")
        dataset_val = build("test", args.images_dir, args.anno_name)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
        )
        evaluate(model, data_loader_test, device=device)
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, args.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(
                model,
                optimizer,
                data_loader,
                device,
                epoch,
                print_freq=10,
                exp_logger=exp_logger,
            )
            # update the learning rate
            lr_scheduler.step()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(output_dir, f"checkpoint{epoch:04}.pth"),
            )
            # evaluate on the test dataset
            coco_evaluator = evaluate(model, data_loader_test, device=device)

            if exp_logger is not None:
                exp_logger.log(
                    {
                        "valid/bbox-mAP@0.5:0.95": coco_evaluator.coco_eval[
                            "bbox"
                        ].stats[0],
                        "valid/bbox-mAP@0.5": coco_evaluator.coco_eval["bbox"].stats[1],
                    }
                )
                if "segm" in coco_evaluator.coco_eval:
                    exp_logger.log(
                        {
                            "valid/segm-mAP@0.5:0.95": coco_evaluator.coco_eval[
                                "segm"
                            ].stats[0],
                            "valid/segm-mAP@0.5": coco_evaluator.coco_eval[
                                "segm"
                            ].stats[1],
                        }
                    )
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    args.lr = 0.001
    for bs in [2, 4, 8]:
        print('Run with args: ', args)
        args.batch_size = bs
        train(args)