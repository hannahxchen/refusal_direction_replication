from operator import attrgetter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import ModelBase
from nnsight import LanguageModel


SYSTEM_PROMPT = """You are a helpful assistant."""
CHAT_TEMPLATE = """\
{% for message in messages %}\n{% if message['role'] == 'system' %}\n{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>' }}\n\
{% elif message['role'] == 'user' %}\n{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>' }}\n\
{% elif message['role'] == 'assistant' %}\n{{ '<|im_start|>assistant\n'  + message['content'] + '<|im_end|>' }}\n\
{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|im_start|>assistant' }}\n{% endif %}\n{% endfor %}\
"""
QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

class QwenModel(ModelBase):
    def __init__(self, model_name="Qwen/Qwen-1_8B-chat"):
        super().__init__(model_name, torch_dtype=torch.float16)
        assert self.model.config.architectures[0] == 'QWenLMHeadModel'

    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, clean_up_tokenization_spaces=True, **kwargs)
        # The model never sees or computes the pad token, so you may use any known token.
        # Reference: https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        tokenizer.padding_side = "left"
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.pad_token_id = tokenizer.eod_id

        tokenizer.chat_template = CHAT_TEMPLATE
        return tokenizer
    
    def _load_model(self, model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map=self.device, use_flash_attn=False, fp16=True)
        return LanguageModel(model, tokenizer=tokenizer, dispatch=True, **kwargs)
    
    def _get_system_message(self):
        self.system_role = True
        return SYSTEM_PROMPT
    
    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("transformer.h")(self.model)
    
    def _get_attn_modules(self):
        return [attrgetter("attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
    
    