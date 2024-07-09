import os
import pytest
from PIL import Image
from clip.model import OnnxLip


def test_clip_instantiation():
    onnx_model = OnnxLip(batch_size=16, type='clip', device='cpu', trt=False)
    assert os.path.isfile('./clip/data/clip_image_model_vitb32.onnx')
    assert os.path.isfile('./clip/data/clip_text_model_vitb32.onnx')


def test_siglip_instantiation():
    onnx_model = OnnxLip(batch_size=16, type='siglip', device='cpu', trt=False)
    assert os.path.isfile('./clip/data/siglip_image_384_fp16.onnx')
    assert os.path.isfile('./clip/data/siglip_text_384_fp16.onnx')


