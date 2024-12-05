from operator import attrgetter
import torch
from transformers import AutoTokenizer
from .base import ModelBase

LLAMA3_REFUSAL_TOKS = [40] # 'I'

class Llama3Model(ModelBase):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__(model_name, torch_dtype=torch.bfloat16)
        assert self.model.config.architectures[0] == 'LlamaForCausalLM'
        self.norm_module = self.model.model.norm
        
    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, **kwargs)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_system_message(self):
        return "You are a helpful assistant."
    
    def _get_refusal_toks(self):
        return LLAMA3_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("model.layers")(self.model)
    
    def _get_attn_modules(self):
        return [attrgetter("self_attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
