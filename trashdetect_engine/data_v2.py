import os
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from . import transforms as T
import albumentations as A


class DetectWasteMultiDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks=True):
        super().__init__()

        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.convert_poly_to_mask = ConvertCocoPolysToMask(return_masks)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.img_folder, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        target = {"image_id": image_id, "annotations": target}

        image, target = self.convert_poly_to_mask(image, target)

        if self.transforms is not None:
            """
            image_dict = {'image': numpy.asarray(img),
                          'bboxes': target['boxes'],
                          'masks': numpy.asarray(target['masks']),
                          'labels': target['labels']}
            image_dict = self._transforms(**image_dict)
            img = torch.as_tensor(image_dict['image'])
            target['boxes'] = torch.as_tensor(image_dict['bboxes'])
            target['masks'] = torch.as_tensor(image_dict['mask'])
            """
            transformed_data = self.transforms(
                image=image,
                masks=target["masks"],
                bboxes=target["boxes"],
                labels=target["labels"],
            )
            image = transformed_data["image"]
            target["masks"] = transformed_data["masks"]
            target["boxes"] = transformed_data["bboxes"]
            target["labels"] = transformed_data["labels"]
        return image, target


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=True):
        self.return_masks = return_masks

    @classmethod
    def convert_coco_poly_to_mask(cls, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)

            if len(mask.shape) < 3:
                mask = mask[..., None]
            # mask = np.array(mask, dtype=np.uint8)
            mask = np.any(mask, axis=2).astype(np.uint8)
            masks.append(mask)
        if len(masks) == 0:  # Return dummy mask
            masks = list(np.zeros((0, height, width), dtype=np.uint8))
        return masks

    def __call__(self, image, target):
        w, h = image.size
        image = np.asarray(image)

        image_id = target["image_id"]
        image_id = np.array([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # Guard against no boxes via resizing
        # boxes: x, y, w, h
        boxes = np.array(boxes, dtype=np.int64).reshape(-1, 4)

        # No need to convert to x, y, xx, yy
        # boxes[:, 2:] += boxes[:, :2]  # x, y, w, h --> x, y, xx, yy
        # boxes[:, 0::2].clamp_(min=0, max=w)
        # boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = np.array(classes, dtype=np.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = np.array(keypoints, dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((num_keypoints, -1, 3))

        # Convert to (x, y, xx, yy) to check validity
        boxes1 = boxes.copy()
        boxes1[:, 2:] += boxes1[:, :2]  # x, y, w, h --> x, y, xx, yy
        # boxes1[:, 0::2].clip_(min=0, max=w)
        # boxes1[:, 1::2].clip_(min=0, max=h)
        boxes1[:, 0::2] = np.clip(boxes1[:, 0::2], a_min=0, a_max=w)
        boxes1[:, 1::2] = np.clip(boxes1[:, 1::2], a_min=0, a_max=h)

        # Check validity
        keep = (boxes1[:, 3] > boxes1[:, 1]) & (boxes1[:, 2] > boxes1[:, 0])
        boxes = boxes[keep]

        classes = classes[keep]
        if self.return_masks:
            masks = list(np.array(masks)[keep])

        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.array([obj["area"] for obj in anno])
        iscrowd = np.array([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = np.array([int(h), int(w)])
        target["size"] = np.array([int(h), int(w)])

        return image, target


# from albumentations.augmentations.geometric.resize import LongestMaxSize
from albumentations.pytorch import ToTensorV2

# References:
# [1] https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
def get_transform_albumentation(mode: str, max_size: int = 720):
    print("> Get_transform_albumentation")
    transforms = []

    if mode == "train":
        transforms.extend([A.HorizontalFlip(p=0.5)])

    transforms.append(A.LongestMaxSize(max_size=max_size))
    transforms.append(ToTensorV2(p=1))  # -> [0, 1]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
    )


# Old transforms from trashdetect repo
def get_transform(mode):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if mode == "train":
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def build(
    image_set: str,
    images_path: str,
    annotation_path: str,
    return_masks: bool = True,
    max_size: int = 1333,
):
    assert image_set in ["train", "val", "test"]
    PATHS = {
        "train": (images_path, f"{annotation_path}_train.json"),
        "val": (images_path, f"{annotation_path}_test.json"),
        "test": (images_path, f"{annotation_path}_test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = DetectWasteMultiDataset(
        img_folder,
        ann_file,
        transforms=get_transform_albumentation(mode=image_set, max_size=max_size),
        return_masks=return_masks,
    )
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 0
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for i in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


# ===============================================================================
# **Pytorch Lightning Adoption**
# ===============================================================================
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from trashdetect_engine import utils


class WasteDatasetDL(pl.LightningDataModule):
    def __init__(self, args, return_masks=True):
        super().__init__()
        self.args = args
        self.return_masks = return_masks
        self.dataset_train = build(
            "train", self.args.images_dir, self.args.anno_name, self.return_masks
        )
        self.dataset_val = build(
            "val", self.args.images_dir, self.args.anno_name, self.return_masks
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=utils.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=utils.collate_fn,
        )
