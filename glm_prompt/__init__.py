# __init__.py (推荐的写法，无需修改)
from .node.glm import GLM_Text_Chat
from .node.glm import GLM_Vision_ImageToPrompt
from .node.aliyun_oss_node import AliyunOSSDownloadNode
from .node.aliyun_oss_node import AliyunOSSUploadNode
from .node.load_image import LoadImageNode

NODE_CLASS_MAPPINGS = {
    "AliyunOSSUploadNode": AliyunOSSUploadNode,
    "AliyunOSSDownloadNode": AliyunOSSDownloadNode,
    "GLM_Text_Chat": GLM_Text_Chat,
    "GLM_Vision_ImageToPrompt": GLM_Vision_ImageToPrompt,
    "LoadImage": LoadImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AliyunOSSUploadNode": "Aliyun OSS Upload",
    "AliyunOSSDownloadNode": "Aliyun OSS Download",
    "GLM_Text_Chat": "GLM提示词扩写",
    "GLM_Vision_ImageToPrompt": "GLM提示词反推",
}
