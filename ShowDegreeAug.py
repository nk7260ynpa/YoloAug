import math

import numpy as np
import cv2
import gradio as gr

from ultralytics.data.augment import RandomPerspective

class Perspective(RandomPerspective):
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
    
def adjust_affine(image, degrees=0, translate=0, scale=0, shear=0, perspective=0):
    affine_augment = Perspective(degrees=degrees, translate=translate, scale=scale, shear=shear, 
                                 perspective=perspective, border=(0, 0), pre_transform=None)
    augmented_image, _, _ = affine_augment.affine_transform(image, (0, 0))
    return augmented_image

def process_image(image, degrees, translate, scale, shear, perspective):
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)  # Return a black image
    augmented_image = adjust_affine(image, degrees, translate, scale, shear, perspective)
    return augmented_image

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", value="ultralytics/assets/zidane.jpg", label="Input Image", sources="upload"),
        gr.Slider(-180, 180, step=0.1, value=0.0, label="Degrees"),
        gr.Slider(-0.5, 0.5, step=0.01, value=0.0, label="Translate"),
        gr.Slider(-0.5, 0.5, step=0.01, value=0.0, label="Scale"),
        gr.Slider(-180, 180, step=0.1, value=0.0, label="Shear"),
        gr.Slider(0, 0.001, step=0.0001, value=0.0, label="Perspective"),
    ],
    outputs=gr.Image(type="numpy"),
    title="Affine Augmentation",
    description="Upload an image and apply affine transformations using the sliders.",
    allow_flagging="never"
)

interface.launch(debug=True)