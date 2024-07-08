import os
import pytest
from PIL import Image
from sam.model import OnnxSAM


def test_sam_instantiation():
    onnx_model = OnnxSAM()
    assert os.path.isfile('./sam/data/sam_vit_l_0b3195.decoder.quant.onnx')
    assert os.path.isfile('./sam/data/sam_vit_l_0b3195.encoder.quant.onnx')

