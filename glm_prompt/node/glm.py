import os
import json
import base64
import random
from zhipuai import ZhipuAI
from PIL import Image
import numpy as np
import io

# 全局配置文件加载
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_NAME = './config.json'

# 提示词预设文件
TEXT_PROMPTS_FILE_NAME = '../prompt/text_prompts.txt'
IMAGE_PROMPTS_FILE_NAME = '../prompt/image_prompts.txt'

#文本模型列表
TEXT_MODL_LIST = [
    "GLM-4.5",
    "GLM-4.5-air",
    "GLM-4.5-x",
    "GLM-4.5-airx",
    "GLM-4.5-Flash",
    "GLM-4-plus",
    "GLM-4-air-250414",
    "GLM-4-airx",
    "GLM-4-Flashx",
    "GLM-4-Flashx-250414",
    "GLM-z1-air",
    "GLM-z1-airx",
    "GLM-z1-Flash",
    "GLM-z1-Flashx",
    ]

#视觉模型列表
VISION_MODL_LIST = [
    "GLM-4.5v",
    "GLM-4v-plus-0111",
    "GLM-4v-flash",
    "GLM-4.1v-thinking-flashx",
    "GLM-4.1v-thinking-flash",
    ]

SUPPORTED_TRANSLATION_LANGS = [
    'zh', 'en',
]

#日志输出
def _log_info(message):
    """统一的日志输出函数"""
    print(f"[GLM_Nodes] 信息：{message}")

def _log_warning(message):
    """统一的警告输出函数"""
    print(f"[GLM_Nodes] 警告：{message}")

def _log_error(message):
    """统一的错误输出函数"""
    print(f"[GLM_Nodes] 错误：{message}")

def get_zhipuai_api_key():
    env_api_key = os.getenv("ZHIPUAI_API_KEY")
    if env_api_key:
        _log_info("使用环境变量 API Key。")
        return env_api_key

    config_path = os.path.join(CURRENT_DIR, CONFIG_FILE_NAME)
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            api_key = config.get("ZHIPUAI_API_KEY")
            if api_key:
                _log_info(f"从 {CONFIG_FILE_NAME} 读取 API Key。")
                return api_key
            else:
                _log_warning(f"在 {CONFIG_FILE_NAME} 中未找到 ZHIPUAI_API_KEY。")
                return ""
        else:
            _log_warning(f"未找到 API Key 配置文件 {CONFIG_FILE_NAME}。")
            return ""
    except json.JSONDecodeError:
        _log_error(f"配置文件 {CONFIG_FILE_NAME} 格式不正确。")
        return ""
    except Exception as e:
        _log_error(f"读取config.json文件时发生错误: {e}")
        return ""

def load_prompts_from_txt(file_path, default_built_in_prompts):
    prompts = {}
    current_prompt_name = None
    current_prompt_content = []

    if not os.path.exists(file_path):
        _log_warning(f"提示词文件 '{os.path.basename(file_path)}' 不存在，使用内置默认提示词。")
        return default_built_in_prompts

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip() # 移除行首行尾空白
                if not line: # 跳过空行
                    continue

                if line.startswith('[') and line.endswith(']'):
                    # 新的提示词名称
                    if current_prompt_name and current_prompt_content:
                        prompts[current_prompt_name] = "\n".join(current_prompt_content).strip()
                
                    current_prompt_name = line[1:-1].strip() # 提取名称
                    current_prompt_content = [] # 重置内容
                elif current_prompt_name is not None:
                    # 添加内容到当前提示词
                    current_prompt_content.append(line)
                # else: 忽略文件开头在第一个 [ ] 之前的行

            # 处理文件末尾的最后一个提示词
            if current_prompt_name and current_prompt_content:
                prompts[current_prompt_name] = "\n".join(current_prompt_content).strip()

        if not prompts:
            _log_warning(f"提示词文件 '{os.path.basename(file_path)}' 内容为空或格式不正确，使用内置默认提示词。")
            return default_built_in_prompts

        #_log_info(f"从 '{os.path.basename(file_path)}' 加载提示词成功。")
        return prompts

    except Exception as e:
        _log_error(f"读取提示词文件 '{os.path.basename(file_path)}' 失败: {e}。使用默认提示词。")
        return default_built_in_prompts 

#GLM文本补全节点
class GLM_Text_Chat:
    CATEGORY = "JFD/GLM_Prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "glm_chat_function"

    # 内置的默认系统提示词 (当TXT文件不存在或解析失败时作为备用)
    _BUILT_IN_TEXT_PROMPTS = {
        "默认视频扩写提示 (内置)": """
你是一个专业的视频脚本和提示词生成助手。你的任务是根据用户提供的核心概念，将其扩写成一个详细、具体、富有画面感的视频生成提示词。请严格遵循以下结构和要求：

1.  **主体描述：** 详细刻画视频中的主要对象或人物的外观、特征、状态。
2.  **场景描述：** 细致描绘主体所处环境，包括时间、地点、背景元素、光线、天气等。
3.  **运动描述：** 明确主体的动作细节（幅度、速率、效果）。
4.  **镜头语言：** 指定景别（如特写、近景、中景、全景）、视角（如平视、仰视、俯视）、镜头类型（如广角、长焦）、运镜方式（如推、拉、摇、移、跟、升、降）。
5.  **氛围词：** 定义画面的情感与气氛。
6.  **风格化：** 设定画面的艺术风格（如写实、卡通、赛博朋克、水墨画、电影感、抽象）。

**输出格式要求：**
-   只输出最终扩写后的视频提示词，不要包含任何解释性文字或额外的对话。
-   将所有要素融合为一段连贯的描述性文字，确保逻辑流畅。
-   最终提示词应该尽可能详细，包含丰富的细节，以便AI模型能准确理解并生成高质量视频。

**举例：**
用户输入：一只小狗在草地上玩耍。
你的输出：一只毛茸茸的金毛幼犬，披着阳光般金色的毛发，眼神好奇而活泼，在阳光明媚的广阔草地上奔跑。它欢快地追逐着一只飞舞的蝴蝶，时而跳跃，时而打滚，草屑和泥土溅起细小的弧线。中景，低角度仰拍，镜头随着小狗的奔跑而平稳地横向移动，展现出草地的广阔和小狗的活力。画面充满温暖、快乐、生机勃勃的氛围，色彩鲜艳，如田园诗般的卡通风格。
"""
    }

    @classmethod
    def get_text_prompts(cls):
        return load_prompts_from_txt(
            os.path.join(CURRENT_DIR, TEXT_PROMPTS_FILE_NAME),
            cls._BUILT_IN_TEXT_PROMPTS
        )

    @classmethod
    def INPUT_TYPES(s):
        available_prompts = s.get_text_prompts()
        prompt_keys = list(available_prompts.keys())
        default_selection = prompt_keys[0] if prompt_keys else "无可用提示词"
        default_model = "GLM-4.5-Flash"
        return {
            "required": {
                "text_system_prompt_preset": (prompt_keys, {"default": default_selection}),
                "system_prompt_override": ("STRING", {"multiline": True, "default": "", "placeholder": "系统提示词 (最高优先级，留空则从预设加载)"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "可选：智谱AI API Key (留空则尝试从环境变量或config.json读取)"}),
                "model_name": (TEXT_MODL_LIST, {"default": default_model, "tooltip": "选择大模型"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "0=随机种子。"}),
                "text_input": ("STRING", {"multiline": True, "default": "a girl。", "placeholder": "请输入需要扩写的视频提示词内容"}),
            }
        }

    def glm_chat_function(self, text_input, api_key, model_name, seed, temperature, top_p, max_tokens, system_prompt_override, text_system_prompt_preset):
        
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)

        _log_info("初始化智谱AI客户端。")

        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        final_system_prompt = ""
        available_prompts = self.get_text_prompts()

        if system_prompt_override and system_prompt_override.strip():
            final_system_prompt = system_prompt_override.strip()
            _log_info("使用 'system_prompt_override'。")
        elif text_system_prompt_preset in available_prompts:
            final_system_prompt = available_prompts[text_system_prompt_preset]
            _log_info(f"使用预设提示词: '{text_system_prompt_preset}'。")
        else:
            if available_prompts:
                final_system_prompt = list(available_prompts.values())[0]
                _log_warning(f"预设 '{text_system_prompt_preset}' 未找到，使用第一个可用预设。")
            else:
                final_system_prompt = list(self._BUILT_IN_TEXT_PROMPTS.values())[0]
                _log_warning("无可用预设提示词，使用内置备用。")


        if not final_system_prompt:
            _log_error("系统提示词不能为空。")
            return ("系统提示词不能为空。",)

        if not isinstance(final_system_prompt, str):
            _log_warning(f"系统提示词类型异常: {type(final_system_prompt)}。尝试转换为字符串。")
            final_system_prompt = str(final_system_prompt)

        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        random.seed(effective_seed)

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": text_input}
        ]
        _log_info(f"调用 GLM-4 ({model_name})...")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            # response_text = response.choices[0].message.content
            # _log_info("GLM-4 响应成功。")
            # return (response_text,)
            response_text = str(response.choices[0].message.content)
            if "<|begin_of_box|>" in response_text and "<|end_of_box|>" in response_text:
                start = response_text.find("<|begin_of_box|>") + len("<|begin_of_box|>")
                end = response_text.find("<|end_of_box|>")
                response_text = response_text[start:end].strip()
            _log_info(f"GLM_vsion响应成功。({response_text})...")
            return (response_text,)
        except Exception as e:
            error_message = f"GLM-4 API 调用失败: {e}"
            return (error_message,)


# GLM提示词反推节点
class GLM_Vision_ImageToPrompt:
    CATEGORY = "JFD/GLM_Prompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "generate_prompt"

    _BUILT_IN_IMAGE_PROMPTS = {
        "通用高质量英文描述 (内置)": "你是一个专业的图像描述专家，能够将图片内容转化为高质量的英文提示词，用于文本到图像的生成模型。请仔细观察提供的图片，并生成一段详细、具体、富有创造性的英文短语，描述图片中的主体对象、场景、动作、光线、材质、色彩、构图和艺术风格。要求：语言：严格使用英文。细节：尽可能多地描绘图片细节，包括但不限于物体、人物、背景、前景、纹理、表情、动作、服装、道具等。角度：尽可能从多个角度丰富描述，例如特写、广角、俯视、仰视等，但不要直接写“角度”。连接：使用逗号（,）连接不同的短语，形成一个连贯的提示词。人物：描绘人物时，使用第三人称（如 'a woman', 'the man'）。质量词：在生成的提示词末尾，务必添加以下质量增强词：', best quality, high resolution, 4k, high quality, masterpiece, photorealistic'"
    }

    @classmethod
    def get_image_prompts(cls):
        """加载外部或内置的图像提示词字典。"""
        return load_prompts_from_txt(
            os.path.join(CURRENT_DIR, IMAGE_PROMPTS_FILE_NAME),
            cls._BUILT_IN_IMAGE_PROMPTS
        )

    @classmethod
    def INPUT_TYPES(cls):
        available_prompts = cls.get_image_prompts()
        prompt_keys = list(available_prompts.keys())
        default_selection = prompt_keys[0] if prompt_keys else "无可用提示词"
        default_model = "GLM-4v-flash"

        return {
            "required": {
                "image_prompt_preset": (prompt_keys, {"default": default_selection}),
                "prompt_override": ("STRING", {"default": "", "multiline": True, "placeholder": "请输入用于描述图片的文本提示词 (最高优先级，留空则从上方预设加载)"}),
                "model_name": (VISION_MODL_LIST, {"default": default_model, "tooltip": "选择大模型"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "placeholder": "可选：智谱AI API Key (留空则尝试从环境变量或config.json读取)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "0=随机种子。"}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "", "placeholder": "请输入图片URL (与Base64/IMAGE三选一)"}),
                "image_input": ("IMAGE", {"optional": True, "tooltip": "直接输入ComfyUI IMAGE对象 (与URL/Base64三选一)"}),
            }
        }
    def generate_prompt(self, api_key, prompt_override, seed, model_name, image_url="", image_base64="", image_prompt_preset="", image_input=None):
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)
        
        _log_info("初始化GLM...")
        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        image_url_provided = bool(image_url and image_url.strip())
        image_base64_provided = bool(image_base64 and image_base64.strip())
        image_input_provided = image_input is not None

        if not (image_url_provided or image_base64_provided or image_input_provided):
            _log_error("必须提供图片URL、Base64数据或IMAGE对象。")
            return ("必须提供图片URL、Base64数据或IMAGE对象。",)
            
        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        random.seed(effective_seed)

        #处理图片输入优先级：IMAGE > Base64 > URL
        final_image_data = None
        if image_input_provided:
            _log_info("检测到 IMAGE 对象输入，正在转换为 Base64。")
            try:
                # ComfyUI的IMAGE是PyTorch张量，范围[0,1]，形状[B, H, W, C]
                # 转换为PIL Image，范围[0,255]，形状[H, W, C]
                i = 255. * image_input.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0]) # 取第一个batch的图片
                
                buffered = io.BytesIO()
                img.save(buffered, format="PNG") # 通常PNG是无损且支持透明度
                final_image_data = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')
                _log_info("IMAGE 对象成功转换为 Base64。")
            except Exception as e:
                _log_error(f"将 IMAGE 对象转换为 Base64 失败: {e}")
                return (f"将 IMAGE 对象转换为 Base64 失败: {e}",)
        elif image_base64_provided:
            _log_info("检测到 Base64 字符串输入。")
            if image_base64.startswith("data:image/"):
                final_image_data = image_base64
            else:
                _log_warning("Base64字符串缺少前缀，尝试添加默认JPEG前缀。")
                try:
                    # 尝试解码验证有效性，并添加常见前缀
                    base64.b64decode(image_base64.split(',')[-1])
                    final_image_data = f"data:image/jpeg;base64,{image_base64}"
                except Exception as decode_e:
                    _log_error(f"Base64解码失败: {decode_e}")
                    return ("提供的Base64图片数据无效。",)
        elif image_url_provided:
            _log_info(f"检测到图片URL输入: {image_url}")
            final_image_data = image_url

        if not final_image_data:
            _log_error("未能获取有效的图片数据。")
            return ("未能获取有效的图片数据。",)

        #识图提示词确定优先级
        final_prompt_text = ""
        available_prompts = self.get_image_prompts()

        if prompt_override and prompt_override.strip():
            final_prompt_text = prompt_override.strip()
            _log_info("使用 'prompt_override'。")
        elif image_prompt_preset in available_prompts:
            final_prompt_text = available_prompts[image_prompt_preset]
            _log_info(f"使用预设识图提示词: '{image_prompt_preset}'。")
        else:
            if available_prompts:
                final_prompt_text = list(available_prompts.values())[0]
                _log_warning(f"预设 '{image_prompt_preset}' 未找到，使用第一个可用预设。")
            else:
                final_prompt_text = list(self._BUILT_IN_IMAGE_PROMPTS.values())[0]
                _log_warning("无可用预设识图提示词，使用内置备用。")


        if not final_prompt_text:
            _log_error("识图提示词不能为空。")
            return ("识图提示词不能为空。",)
        if not isinstance(final_prompt_text, str):
            _log_warning(f"识图提示词类型异常: {type(final_prompt_text)}。尝试转换为字符串。")
            final_prompt_text = str(final_prompt_text)

        # 构建消息内容
        content_parts = [{"type": "text", "text": final_prompt_text}]
        content_parts.append({"type": "image_url", "image_url": {"url": final_image_data}})

        _log_info(f"调用 GLM-4V ({model_name})...")
    
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}]
            )
            response_content = str(response.choices[0].message.content)
            if "<|begin_of_box|>" in response_content and "<|end_of_box|>" in response_content:
                start = response_content.find("<|begin_of_box|>") + len("<|begin_of_box|>")
                end = response_content.find("<|end_of_box|>")
                response_content = response_content[start:end].strip()
            _log_info(f"GLM_vsion响应成功。({response_content})...")
            return (response_content,)
        except Exception as e:
            error_message = f"GLM-4V API 调用失败: {e}"
            _log_error(error_message)
            return (error_message,)

# # --- ComfyUI 节点映射 ---
# NODE_CLASS_MAPPINGS = {
#     "GLM_Text_Chat": GLM_Text_Chat,
#     "GLM_Vision_ImageToPrompt": GLM_Vision_ImageToPrompt,
# }

# # ComfyUI 节点显示名称映射
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "GLM_Text_Chat": "GLM提示词扩写",
#     "GLM_Vision_ImageToPrompt": "GLM提示词反推",
# }

