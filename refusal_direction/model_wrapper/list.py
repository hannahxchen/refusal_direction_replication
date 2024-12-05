from .llama2 import Llama2Model
from .llama3 import Llama3Model
from .gemma import GemmaModel
from .qwen import QwenModel
from .yi import YiModel

def get_supported_model_class(model_name):
    if "Llama-2" in model_name:
        return Llama2Model(model_name)
    elif "Llama-3" in model_name:
        return Llama3Model(model_name)
    elif "gemma" in model_name:
        return GemmaModel(model_name)
    elif "Qwen" in model_name:
        return QwenModel(model_name)
    elif "yi" in model_name:
        return YiModel(model_name)
    else:
        raise Exception("No supported model found.")