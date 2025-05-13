import math
import numpy as np
import cv2
import gradio as gr
from ultralytics.data.augment import RandomPerspective
from ultralytics.data.augment import RandomHSV

class Perspective(RandomPerspective):
    """
    Apply random perspective transformation to the input image.
    
    Args:
        degrees (float): Rotation angle in degrees.
        translate (float): Translation factor.
        scale (float): Scale factor.
        shear (float): Shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (tuple): Border size for the image.
        pre_transform (callable, optional): Pre-transform function to apply before the perspective transformation.

    Example:
        >>> import cv2
        >>> img = cv2.imread('path_to_image.jpg')
        >>> perspective_augment = Perspective(degrees=10, translate=0.1, scale=0.1, shear=5, perspective=0.001)
        >>> augmented_img = perspective_augment(img)
    """
    def affine_transform(self, img, border):
        # Center
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = self.perspective  # x perspective (about y)
        P[2, 1] = self.perspective  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = self.degrees
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = 1 + self.scale
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(self.shear * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(self.shear * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = (0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s
    
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

def process_image(image, *params):
    """
    Process the input image with affine transformations and HSV adjustments.

    Args:
        image (numpy.ndarray): Input image in BGR format.
        params (tuple): Parameters for affine transformations and HSV adjustments.
            - bgr (bool): Whether to convert the image to BGR format.
            - degrees (float): Rotation angle in degrees.
            - translate (float): Translation factor.
            - scale (float): Scale factor.
            - shear (float): Shear angle in degrees.
            - perspective (float): Perspective distortion factor.
            - h_gain (float): Hue gain value.
            - s_gain (float): Saturation gain value.
            - v_gain (float): Value gain value.
            - flip_horizontal (bool): Whether to flip the image horizontally.
            - flip_vertical (bool): Whether to flip the image vertically.
    Returns:
        numpy.ndarray: Processed image in BGR format.

    Example:
        >>> import cv2
        >>> img = cv2.imread('path_to_image.jpg')
        >>> params = (True, 10, 0.1, 0.1, 5, 0.001, -1, 1, 0.5, True, False)
        >>> processed_img = process_image(img, *params)
    """
    if image is None:
        return None
    bgr, degrees, translate, scale, shear, perspective, h_gain, s_gain, v_gain, flip_horizontal, flip_vertical = params

    if bgr:
        image = np.ascontiguousarray(image[..., ::-1])
    affine_augment = Perspective(degrees=degrees, translate=translate, scale=scale, shear=shear, 
                                 perspective=perspective, border=(0, 0), pre_transform=None)
    image, _, _ = affine_augment.affine_transform(image, (0, 0))

    hsv_augment = HSV(hgain=h_gain, sgain=s_gain, vgain=v_gain)
    image = hsv_augment(image)

    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)
        
    return image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image", sources="upload"),
        gr.Checkbox(label="BGR"),
        gr.Slider(-180, 180, step=0.1, value=0.0, label="Degrees"),
        gr.Slider(-0.5, 0.5, step=0.01, value=0.0, label="Translate"),
        gr.Slider(-0.5, 0.5, step=0.01, value=0.0, label="Scale"),
        gr.Slider(-180, 180, step=0.1, value=0.0, label="Shear"),
        gr.Slider(0, 0.001, step=0.0001, value=0.0, label="Perspective"),
        gr.Slider(-1, 1, step=0.01, value=0.0, label="Hue Gain"),
        gr.Slider(-1, 1, step=0.01, value=0.0, label="Saturation Gain"),
        gr.Slider(-1, 1, step=0.01, value=0.00, label="Value Gain"),
        gr.Checkbox(label="Flip Horizontally"),
        gr.Checkbox(label="Flip Vertically")
    ],
    outputs=gr.Image(type="numpy"),
    title="Affine Augmentation",
    description="Upload an image and adjust the sliders to apply affine transformations.",
    allow_flagging="never",
    live=True  
)

interface.launch(debug=True)