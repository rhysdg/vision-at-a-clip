
#import logging
from PIL import Image
from sam.model import OnnxSAM
from clip.model import OnnxLip, softmax, get_probabilities


#logging.basicConfig(level=logging.INFO)

images = [Image.open("images/dog.jpg").convert("RGB")]

texts = {"classification":  ["a photo of space",
                            "a photo of a dog",
                            "a photo of a dog with flowers laying on grass",
                            "a photo of a brown and white dog with blue flowers laying on grass",
                            "a photo of a brown and white dog with yellow flowers laying on grass"],
    }

#type='clip' is also avvaiilable with this usage    
onnx_model = OnnxLip(batch_size=16, type='siglip_full')
probs, _ = onnx_model.inference(images, texts)

for k,v in texts.items():
    print(f'\ncontext: {k}\n')
    for text, p in zip(texts[k], probs[k]):
        print(f"Probability that the image is '{text}': {p:.3f}")
