import torch
from trashdetect_engine.data import build
from trashdetect_engine.models.segmentation_models import (
    get_instance_segmentation_model,
)
from trashdetect_engine import utils
from trashdetect_engine.models.segmentation_models import (
    get_instance_segmentation_model,
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from tqdm import tqdm
import numpy as np

images_dir = "/mnt/ssd1T/TACO/TACO/data"
anno_name = "/mnt/ssd1T/TACO/detect-waste/annotations/annotations_binary"
num_workers = 2
dataset_val = build("test", images_dir, anno_name)
test_batch_size = 1
num_classes = 2
model = "maskrcnn_resnet50_fpn"
data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=utils.collate_fn,
)


min_size = 800
max_size = 1333
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
print(transform)

# model = get_instance_segmentation_model(num_classes, model)
# model.eval()

all_shapes = []
for batch in tqdm(data_loader_test):
    # ['boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd', 'orig_size', 'size']
    orig_size = np.array(batch[1][0]["orig_size"]).astype(np.int)
    img_tensor = transform(batch[0])[0]
    image_size = img_tensor.tensors.shape[1:]
    image_size = np.asarray(image_size).astype(np.int)
    all_shapes.append((orig_size, image_size))

import mipkit

mipkit.debug.set_trace()
exit()
