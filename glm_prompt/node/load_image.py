import os
from PIL import Image
import numpy as np

class LoadImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "image_path")
    FUNCTION = "load_image_path"
    CATEGORY = "JFD/image"

    def load_image_path(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"文件不存在: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        return (image_np, image_path)

# NODE_CLASS_MAPPINGS = {
#     "LoadImage": LoadImageNode
# }