import numpy as np
import random
import cv2
import gradio as gr
from PIL import Image

def mosaic4(imgs):
    """
    Create a 2x2 mosaic of images with random placement and size.

    Args:
        imgs (list): List of 4 images to be placed in the mosaic.

    Returns:
        img4 (numpy.ndarray): The resulting mosaic image.

    Example:
        >>> img1 = cv2.imread('image1.jpg')
        >>> img2 = cv2.imread('image2.jpg')
        >>> img3 = cv2.imread('image3.jpg')
        >>> img4 = cv2.imread('image4.jpg')
        >>> mosaic_image = mosaic4([img1, img2, img3, img4])
        >>> cv2.imshow('Mosaic Image', mosaic_image)
        >>> cv2.waitKey(0)
        >>> cv2.destroyAllWindows()  
    """
    s = 480  # size of the mosaic
    border = (0, 0)  # border size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in border)

    for i in range(4):
        # Load image
        img = imgs[i]
        img = cv2.resize(img, (s*2, s*2))  # resize to s x s
        h, w = img.shape[:2]  # height, width

        # Place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
    return img4

def process_images(img1, img2, img3, img4):
    """
    Process_images function to create a mosaic of 4 images.

    Args:
        img1 (numpy.ndarray): First input image.
        img2 (numpy.ndarray): Second input image.
        img3 (numpy.ndarray): Third input image.
        img4 (numpy.ndarray): Fourth input image.

    Returns:
        img (numpy.ndarray): The resulting mosaic image.

    Example:
        >>> img1 = cv2.imread('image1.jpg')
        >>> img2 = cv2.imread('image2.jpg')
        >>> img3 = cv2.imread('image3.jpg')
        >>> img4 = cv2.imread('image4.jpg')
        >>> mosaic_image = process_images(img1, img2, img3, img4)
        >>> cv2.imshow('Mosaic Image', mosaic_image)
        >>> cv2.waitKey(0)
        >>> cv2.destroyAllWindows()
    """
    img = mosaic4([img1, img2, img3, img4])
    return img

with gr.Blocks() as demo:
    
    gr.Markdown("## Upload 4 Images and Submit")
    
    with gr.Row():
        img1 = gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image1", sources="upload")
        img2 = gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image2", sources="upload")
        img3 = gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image3", sources="upload")
        img4 = gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image4", sources="upload")
         
    submit_btn = gr.Button("Submit")
    output = gr.Image(label="Processed Mosaic Image", type="numpy", elem_id="output_image")
    submit_btn.click(process_images, inputs=[img1, img2, img3, img4], outputs=output)
    
if __name__ == "__main__":
    demo.launch()