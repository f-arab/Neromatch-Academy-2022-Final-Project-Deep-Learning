import math
import sys
import time

# from pyrsistent import v
import torch

import torchvision.models.detection.mask_rcnn

# from trashdetect_engine.data import get_coco_api_from_dataset
from trashdetect_engine.coco_eval import CocoEvaluator
from trashdetect_engine import utils


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq, exp_logger=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if exp_logger is not None:
            exp_logger.log({"train/loss": loss_value})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, exp_logger=None):
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = get_iou_types(model)
    coco_evaluator = CocoEvaluator(data_loader.dataset.coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 5, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        # 'boxes', 'labels', 'scores', 'masks'
        # boxes: [BS, N, 4] ~ x, y, w, h
        # boxes: [BS, N, 4] ~ x, y, w, h
        outputs = model(images)
        # if exp_logger is not None:
        #     exp_logger.log_metric({"valid/loss": loss_value})
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(
            model_time=model_time,
            evaluator_time=evaluator_time,
        )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # Accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # torch.set_num_threads(n_threads)
    return coco_evaluator


# ===============================================================================
# **Pytorch Lightning Adoption**
# ===============================================================================
import pytorch_lightning as pl
import torch


class WasteDetectModelDL(pl.LightningModule):
    def __init__(self, model, optimizer, lr_scheduler, coco_evaluator, args, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.coco_evaluator = coco_evaluator
        self.args = args

    def configure_optimizers(self):
        return [[self.optimizer], [self.lr_scheduler]]

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)

        # dict_keys(['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
        loss = sum(loss for loss in loss_dict.values())

        # loss_value = losses.item()
        self.log(
            "loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)

        targets_cpu = []
        outputs_cpu = []

        for target, output in zip(targets, outputs):
            t_cpu = {k: v.cpu() for k, v in target.items()}
            o_cpu = {k: v.cpu() for k, v in output.items()}
            targets_cpu.append(t_cpu)
            outputs_cpu.append(o_cpu)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets_cpu, outputs_cpu)
        }
        self.coco_evaluator.update(res)

    def validation_epoch_end(self, outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        # coco main metric
        metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        self.log("val", metric)

        self.coco_evaluator.reset()

    def predict(self, images):
        pass
