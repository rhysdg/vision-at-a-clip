import os
import pytest
from PIL import Image
from clip.model import OnnxClip


def _test_clip_instantiation():
    onnx_model = OnnxClip(batch_size=16, device='cpu')
    assert os.path.isfile('./clip/data/clip_image_model_vitb32.onnx')
    assert os.path.isfile('./clip/data/clip_text_model_vitb32.onnx')


def _test_siglip_instantiation():
    onnx_model = OnnxClip(batch_size=16, type='siglip', device='cpu')
    assert os.path.isfile('./clip/data/siglip_image_fp16.onnx')
    assert os.path.isfile('./clip/data/siglip_text_fp16.onnx')


