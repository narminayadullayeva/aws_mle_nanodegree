# https://aws-ml-blog-artifacts.s3.us-east-2.amazonaws.com/inference.py

import base64
import io
import json
import logging
import os

import numpy as np
import smdebug
import torch
import torchvision
from PIL import Image
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()


def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, "model.pth"), map_location=device)
    model.eval()
    model.to(device)
    return model


def predict_fn(image, model):
    return model(image)


def input_fn(request_body, content_type):
    iobytes = io.BytesIO(request_body)
    img = Image.open(iobytes)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5232, 0.4421, 0.3526], std=[0.1355, 0.1196, 0.0899]),
        ]
    )
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model
    return input_batch.to(device)
