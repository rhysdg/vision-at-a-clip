import os
import pytest
from PIL import Image
from clip.model import OnnxClip


def test_clip_instantiation():
    onnx_model = OnnxClip(batch_size=16)
    assert os.path.isfile('./clip/data/clip_image_model_vitb32.onnx')

