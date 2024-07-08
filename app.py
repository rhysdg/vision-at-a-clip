import gradio as gr
from PIL import Image
from clip.model import OnnxClip, softmax, get_similarity_scores

example =  ["a photo of space",
            "a photo of a man",
            "a photo of a man in dungarees",
            "a photo of a sad man in dungarees",
            "a photo of a sad man in dungarees with short hair and a orange container to the right",
            "a photo of a sad man in dungarees with short hair",
            "a photo of a happy man in dungarees",
            "A photo of Christopher Nolan"]


def classify(image, text):
    images = [image]
    

    texts = {"classification": text.split(',')
        }

    #type='clip' is also avvailable with this usage    
    onnx_model = OnnxClip(batch_size=16, type='siglip_full')
    probs, _ = onnx_model.inference(images, texts)
      
    probs = [float(p) for p in probs['classification']]
 
    return {label: prob for label, prob in zip(texts['classification'],probs)}


demo = gr.Interface(
    classify,
    [
        gr.Image(label="Image", type="pil"),
        gr.Textbox(label="Labels", info="Comma-separated list of class labels"),
    ],
    gr.Label(label="Result"),
    examples=[['clip/data/interstellar.jpg', ','.join(example)]],
)
try:
    demo.launch(debug=True, height=1000)
except Exception:
    demo.launch(share=True, debug=True, height=1000)
