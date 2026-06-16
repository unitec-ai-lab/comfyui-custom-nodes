from .fabric_lipsync import FabricLipsync
from .fabric_text_to_video import FabricTextToVideo

NODE_CLASS_MAPPINGS = {
    "FabricLipsync": FabricLipsync,
    "FabricTextToVideo": FabricTextToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FabricLipsync": "Fabric 1.0 Lipsync (Image + Audio)",
    "FabricTextToVideo": "Fabric 1.0 Text to Video (Image + Text)",
}
