from .base import *
from .cityscapes import CityscapesSegmentation
from .pascalcontext import PascalContextSegmentation
from .camvid import CamVidSegmentation

datasets = {
    'cityscapes': CityscapesSegmentation,
    'pascalcontext': PascalContextSegmentation,
    'camvid': CamVidSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
