import numpy as np
import gradio as gr

def process_image(image, BGR):
    if BGR:
        image = np.ascontiguousarray(image[..., ::-1])
    return image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image", sources="upload"),
        gr.Checkbox(label="BGR"),
    ],
    outputs=gr.Image(type="numpy", label="Processed Image"),
    title="Image BGR Augmentation",
    allow_flagging="never"
)

interface.launch()