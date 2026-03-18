from .dashengtokenizer_node import (
    DashengTokenizerDecode,
    DashengTokenizerEncode,
    DashengTokenizerLoadModel,
)

NODE_CLASS_MAPPINGS = {
    "DashengTokenizerLoadModel": DashengTokenizerLoadModel,
    "DashengTokenizerEncode": DashengTokenizerEncode,
    "DashengTokenizerDecode": DashengTokenizerDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DashengTokenizerLoadModel": "Load DashengTokenizer Model",
    "DashengTokenizerEncode": "DashengTokenizer Encode",
    "DashengTokenizerDecode": "DashengTokenizer Decode",
}
