from operator import attrgetter
import torch
from transformers import AutoTokenizer
from .base import ModelBase

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py
LLAMA2_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. \
Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. \
If you don't know the answer to a question, please don't share false information.\
"""

LLAMA2_REFUSAL_TOKS = [306] # 'I'

class Llama2Model(ModelBase):
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__(model_name, torch_dtype=torch.float16)
        assert self.model.config.architectures[0] == 'LlamaForCausalLM'

        self.norm_module = self.model.model.norm
        
    def _load_tokenizer(self, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, **kwargs)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def _get_system_message(self):
        self.system_role = True
        return LLAMA2_DEFAULT_SYSTEM_PROMPT
    
    def _get_refusal_toks(self):
        return LLAMA2_REFUSAL_TOKS
    
    def _get_model_block_modules(self):
        return attrgetter("model.layers")(self.model)
    
    def _get_attn_modules(self):
        return [attrgetter("self_attn")(model_block) for model_block in self.model_block_modules]
    
    def _get_mlp_modules(self):
        return [attrgetter("mlp")(model_block) for model_block in self.model_block_modules]
