import numpy as np
from ultralytics.data.augment import RandomHSV
import gradio as gr
import cv2


def adjust_hsv(image, h_gain, s_gain, v_gain):
    hsv_augment = RandomHSV(h_gain=h_gain, s_gain=s_gain, v_gain=v_gain)
    augmented_image = hsv_augment(image)
    return augmented_image

def process_image(image, h_gain, s_gain, v_gain):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented_image = adjust_hsv(image, h_gain, s_gain, v_gain)
    return augmented_image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy"),
        gr.Slider(0, 1, step=0.01, label="Hue Gain"),
        gr.Slider(0, 1, step=0.01, label="Saturation Gain"),
        gr.Slider(0, 1, step=0.01, label="Value Gain"),
    ],
    outputs=gr.Image(type="numpy"),
    title="HSV Adjustment",
    description="Upload an image and adjust HSV values using the sliders."
)

interface.launch()

