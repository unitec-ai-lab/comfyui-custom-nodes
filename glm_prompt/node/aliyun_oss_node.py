from pathlib import Path
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from itertools import islice
import os
import logging
import time
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AliyunOSSUploadNode:
    """
    上传本地文件到阿里云OSS，并返回文件URL
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "access_key_id": ("STRING", {"multiline": False}),
                "access_key_secret": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False}),  # 例如：https://oss-cn-hangzhou.aliyuncs.com
                "bucket_name": ("STRING", {"multiline": False}),
                "local_file_path": ("STRING", {"multiline": False}),
                "object_name": ("STRING", {"multiline": False}),  # 上传到OSS的路径，例如 "uploads/test.jpg"
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("oss_url",)
    FUNCTION = "upload_file"
    CATEGORY = "JFD/aliyun_oss"

    def upload_file(self, access_key_id, access_key_secret, endpoint, bucket_name, local_file_path, object_name):
        try:
            # 添加超时设置
            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, endpoint, bucket_name)
            
            # 验证本地文件
            if not os.path.exists(local_file_path):
                return (f"Error: local file {local_file_path} not found",)
            
            # 上传文件
            bucket.put_object_from_file(object_name, local_file_path)
            result = bucket.get_bucket_location()
            # https://oss-cn-shanghai.aliyuncs.com
            # 构建URLhttps://drawbookai.oss-cn-shanghai.aliyuncs.com/comfyui/1755572562.png
            oss_url = f"https://{bucket_name}.{result.location}.aliyuncs.com/{object_name}"
            return (oss_url,)
            
        except oss2.exceptions.RequestError as e:
            return (f"Network error: {str(e)}",)
        except oss2.exceptions.ServerError as e:
            return (f"Server error: {str(e)}",)
        except Exception as e:
            return (f"Upload failed: {str(e)}",)


class AliyunOSSDownloadNode:
    """
    从阿里云OSS下载文件到本地，并返回本地路径
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "access_key_id": ("STRING", {"multiline": False}),
                "access_key_secret": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False}),
                "bucket_name": ("STRING", {"multiline": False}),
                "oss_file_path": ("STRING", {"multiline": False}),
                "local_save_path": ("STRING", {"multiline": False}),  # 保存到本地的路径
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_file",)
    FUNCTION = "download_file"
    CATEGORY = "JFD/aliyun_oss"

    def download_file(self, access_key_id, access_key_secret, endpoint, bucket_name, oss_file_path, local_save_path):
        try:
            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, endpoint, bucket_name)

            save_dir = Path(local_save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            bucket.get_object_to_file(oss_file_path, local_save_path)

            return (str(local_save_path),)
        except Exception as e:
            return (f"Download failed: {str(e)}",)


# # 节点注册
# NODE_CLASS_MAPPINGS = {
#     "AliyunOSSUploadNode": AliyunOSSUploadNode,
#     "AliyunOSSDownloadNode": AliyunOSSDownloadNode,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "AliyunOSSUploadNode": "Aliyun OSS Upload",
#     "AliyunOSSDownloadNode": "Aliyun OSS Download",
# }
