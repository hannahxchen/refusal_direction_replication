from operator import attrgetter
import torch
from transformers import AutoTokenizer
from .base import ModelBase

GEMMA_REFUSAL_TOKS = [235285] # ['I']

class GemmaModel(ModelBase):
    def __init__(self, model_name="google/gemma-2b-it"):
        super().__init__(model_name, torch_dtype=torch.bfloat16)
        assert self.model.config.architectures[0] == 'GemmaForCausalLM'

    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, **kwargs)
        tokenizer.padding_side = "left"
        return tokenizer
    
    def _get_system_message(self):
        self.system_role = False
        return None
    
    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("model.layers")(self.model)
    
    def _get_attn_modules(self):
        return [attrgetter("self_attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
    