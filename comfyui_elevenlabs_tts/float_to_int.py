import math

class FloatToInt:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0}),
                "mode":  (["round", "ceil", "floor"], {"default": "ceil"}),
            }
        }

    RETURN_TYPES  = ("INT",)
    RETURN_NAMES  = ("int_value",)
    FUNCTION      = "convert"
    CATEGORY      = "utils"

    def convert(self, value, mode):
        if mode == "round":
            return (int(round(value)),)
        elif mode == "ceil":
            return (math.ceil(value),)
        else:
            return (math.floor(value),)


NODE_CLASS_MAPPINGS = {
    "FloatToInt": FloatToInt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatToInt": "Float To Int",
}
