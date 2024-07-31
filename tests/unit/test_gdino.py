import os
import pytest
from PIL import Image
from gdino.model import OnnxGDINO


def test_gdino_instantiation():
    onnx_model = OnnxGDINO()
    assert os.path.isfile('./gdino/data/gdino_fp32.onnx')



