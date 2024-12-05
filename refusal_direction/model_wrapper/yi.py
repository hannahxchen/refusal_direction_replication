from operator import attrgetter
import torch
from transformers import AutoTokenizer
from .base import ModelBase

SYSTEM_PROMPT = """You are a helpful assistant."""
YI_REFUSAL_TOKS = [59597] # ['I']
# Noting some other top refusal tokens. But really a vast majority of the probability is placed on the first.
YI_REFUSAL_TOKS_EXTRA = [59597, 2301, 4786] # ['I', 'It', 'As']


# Newer version: https://huggingface.co/01-ai/Yi-1.5-6B-Chat
class YiModel(ModelBase):
    def __init__(self, model_name="01-ai/yi-6b-chat"):
        super().__init__(model_name, torch_dtype=torch.float16)
        
    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, clean_up_tokenization_spaces=True, **kwargs)
        tokenizer.padding_side = "left"
        return tokenizer
    
    def _get_system_message(self):
        self.system_role = True
        return SYSTEM_PROMPT
    
    def _get_refusal_toks(self):
        return YI_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("model.layers")(self.model)
    
    def _get_attn_modules(self):
        return [attrgetter("self_attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
    
    