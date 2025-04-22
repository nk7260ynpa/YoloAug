import math

import numpy as np
import cv2
import gradio as gr


def process_image(image, flip_horizontal, flip_vertical):
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
    return image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image", sources="upload"),
        gr.Checkbox(label="Flip Horizontally"),
        gr.Checkbox(label="Flip Vertically")
    ],
    outputs=gr.Image(type="numpy", label="Processed Image"),
    title="Image Flip Augmentation",
    allow_flagging="never"
)

interface.launch()