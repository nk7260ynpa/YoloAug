import numpy as np
from ultralytics.data.augment import RandomHSV
import gradio as gr
import cv2

class HSV(RandomHSV):
    def __call__(self, img):  # Convert RGB to BGR for OpenCV
        """
        Apply random HSV augmentation to the input image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format.

        Returns:
            numpy.ndarray: Augmented image in BGR format.
        
        Example:
            >>> import cv2
            >>> img = cv2.imread('path_to_image.jpg')
            >>> hsv_augment = HSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> augmented_img = hsv_augment(img)
        """
        dtype = img.dtype  # uint8
        r = np.array([self.hgain, self.sgain, self.vgain])  # random gains
        x = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img

def adjust_hsv(image, h_gain, s_gain, v_gain):
    """
    adjust the HSV values of the input image.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        h_gain (float): Hue gain value.
        s_gain (float): Saturation gain value.
        v_gain (float): Value gain value.

    Returns:
        numpy.ndarray: Augmented image in BGR format.
        
    Example:
        >>> import cv2
        >>> img = cv2.imread('path_to_image.jpg')
        >>> h_gain = 0.5
        >>> s_gain = 0.5
        >>> v_gain = 0.5
        >>> augmented_img = adjust_hsv(img, h_gain, s_gain, v_gain)
    """
    hsv_augment = HSV(hgain=h_gain, sgain=s_gain, vgain=v_gain)
    augmented_image = hsv_augment(image)
    return augmented_image

def process_image(image, h_gain, s_gain, v_gain):
    """
    Process the input image and apply HSV augmentation.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        h_gain (float): Hue gain value.
        s_gain (float): Saturation gain value.
        v_gain (float): Value gain value.

    Returns:
        numpy.ndarray: Augmented image in BGR format.

    Example:
        >>> import cv2
        >>> img = cv2.imread('path_to_image.jpg')
        >>> h_gain = 0.5
        >>> s_gain = 0.5
        >>> v_gain = 0.5
        >>> augmented_img = process_image(img, h_gain, s_gain, v_gain) 
    """
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)  # Return a black image
    augmented_image = adjust_hsv(image, h_gain, s_gain, v_gain)
    return augmented_image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image", sources="upload"),
        gr.Slider(-1, 1, step=0.01, value=0.0, label="Hue Gain"),
        gr.Slider(-1, 1, step=0.01, value=0.0, label="Saturation Gain"),
        gr.Slider(-1, 1, step=0.01, value=0.00, label="Value Gain"),
    ],
    outputs=gr.Image(type="numpy"),
    title="HSV Augmentation",
    description="Upload an image and adjust HSV values using the sliders.",
    allow_flagging="never"
)

interface.launch(debug=True)

