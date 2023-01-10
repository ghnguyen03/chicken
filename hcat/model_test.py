import torch
import torchvision.utils
from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)

model.eval()

img = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(img)