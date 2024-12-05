import os, warnings
from tqdm import tqdm
from operator import attrgetter
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torchtyping import TensorType
from transformers import AutoTokenizer, BatchEncoding
import nnsight
from nnsight import LanguageModel
from ..utils import ceildiv, chunks, orthogonal_rejection

# Turn off annoying warning messages
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class ModelBase(ABC):
    def __init__(self, model_name: str, **model_kwargs):
        self.model_name = model_name

        self.system_role = False
        self.tokenizer = self._load_tokenizer(model_name)
        self.system_message = self._get_system_message()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.device = device
        self.model = self._load_model(model_name, self.tokenizer, **model_kwargs)
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        self.model_block_modules = self._get_model_block_modules()
        self.attn_modules = self._get_attn_modules()
        self.mlp_modules = self._get_mlp_modules()
        self.lm_head_module = attrgetter("lm_head")(self.model)
        self.intervene_direction = None
        self.actAdd_layer = None
    
    def _load_model(self, model_name: str, tokenizer: AutoTokenizer, **kwargs) -> LanguageModel:
        return LanguageModel(
            model_name, tokenizer=tokenizer, device_map=device, 
            dispatch=True, trust_remote_code=True, **kwargs)
    
    @abstractmethod
    def _load_tokenizer(self, model_name, **kwargs) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_system_message(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass
    
    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass
    
    def _get_eoi_toks(self) -> str:
        '''Get post instruction tokens'''
        return self.tokenizer.encode(self.apply_chat_template(["{instruction}"])[0].split("{instruction}")[-1], add_special_tokens=False)
    
    @abstractmethod
    def _get_refusal_toks(self):
        pass

    def set_dtype(self, *vars):
        if len(vars) == 1:
            return vars[0].to(self.model.dtype)
        else:
            return (var.to(self.model.dtype) for var in vars)

    def set_intervene_direction(self, direction: TensorType["hidden_size"]):
        '''Set the default direction for intervention'''
        self.intervene_direction = self.set_dtype(direction)

    def set_actAdd_intervene_layer(self, layer: int):
        '''Set the default model layer for activation addition'''
        self.actAdd_layer = layer

    def tokenize(self, prompts: Union[List[str], BatchEncoding]):
        if isinstance(prompts, BatchEncoding):
            return prompts
        else:
            return self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    
    def get_token_ids(self, input: str) -> TensorType[-1]:
        if hasattr(self.tokenizer, "add_prefix_space") and self.tokenizer.add_prefix_space is True:
            return torch.tensor(self.tokenizer(input.lstrip(), add_special_tokens=False).input_ids)
        return torch.tensor(self.tokenizer(input, add_special_tokens=False).input_ids)
    
    def apply_chat_template(self, instructions: Union[str, List[str]], outputs: Optional[List[str]] = None, use_system_prompt: Optional[bool] = False) -> List[str]:
        if isinstance(instructions, str):
            instructions = [instructions]
        
        prompts = []
        for i in range(len(instructions)):
            messages = []
            inputs = instructions[i]

            if self.system_message is not None and use_system_prompt:
                if self.system_role:
                    messages.append({"role": "system", "content": self.system_message})
                else:
                    inputs = self.system_message + " " + instructions[i]

            messages.append({"role": "user", "content": inputs})
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if inputs[-1] not in ["\n", " "]:
                inputs += " "

            if outputs is not None:
                inputs += outputs[i]
            prompts.append(inputs)

        return prompts
    
    def get_activations(
        self, layers: Union[List[int], int], prompts: Union[str, List[str], BatchEncoding],
        positions: Optional[List[int]] = [-1]
    ) -> List[TensorType["n_prompt", "n_pos", "hidden_size"]]:
        """Get output activations of prompts given a specific layer(s) and token position(s)"""
        if isinstance(layers, int):
            layers = [layers]

        all_acts = []
        with self.model.trace(prompts) as tracer:
            for layer in layers:
                if positions is None:
                    acts = self.model_block_modules[layer].input
                else:
                    acts = self.model_block_modules[layer].input[:, positions, :]

                acts = acts.detach().to("cpu").to(torch.float64).unsqueeze(0).save()
                all_acts.append(acts)

            self.model_block_modules[layer].output.stop() # Early stopping
        return torch.vstack(all_acts)
    
    def _prepare_act_add_inputs(
        self, prompts: Union[str, List[str], BatchEncoding],
        steering_vec: TensorType["hidden_size"], layer: int, 
        coeffs: Union[float, List[float], TensorType[-1]],
    ):
        inputs = self.tokenize(prompts)
        coeffs = torch.tensor(coeffs)

        if coeffs.dim() != 0:
            coeffs = coeffs[:, None, None]
        if steering_vec is None:
            steering_vec = self.intervene_direction
        if layer is None:
            layer = self.actAdd_layer

        steering_vec, coeffs = self.set_dtype(steering_vec, coeffs)
        return inputs, steering_vec, layer, coeffs
    
    def activation_addition(
        self, prompts: Union[str, List[str], BatchEncoding], 
        steering_vec: Optional[TensorType["hidden_size"]] = None, 
        layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        
        inputs, steering_vec, layer, coeffs = self._prepare_act_add_inputs(prompts, steering_vec, layer, coeffs)

        with self.model.trace(inputs) as tracer:
            self.model_block_modules[layer].input += (steering_vec * coeffs)
            logits = self.lm_head_module.output.detach().to("cpu").save()
        return logits
    
    def _prepare_ablation_inputs(
        self, prompts: Union[str, List[str], BatchEncoding],
        direction: TensorType["hidden_size"]
    ):
        if direction is None:
            direction = self.intervene_direction

        inputs = self.tokenize(prompts)
        unit_direction = direction / (direction.norm(dim=-1) + 1e-8)
        unit_direction = self.set_dtype(unit_direction)
        return inputs, unit_direction
    
    def directional_ablation(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        
        inputs, unit_direction = self._prepare_ablation_inputs(prompts, direction)

        with self.model.trace(inputs) as tracer:
            for layer in range(self.n_layers):
                acts = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].input, unit_direction))
                self.model_block_modules[layer].input = acts

                act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].output[0], unit_direction))
                self.attn_modules[layer].output = (act_post_attn,) + self.attn_modules[layer].output[1:]

                act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].output, unit_direction))
                self.mlp_modules[layer].output = act_post_mlp

            logits = self.lm_head_module.output.detach().to("cpu").save()
        return logits
    
    def get_logits(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[int] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0
    ) -> TensorType["n_prompt", "seq_len", "vocab_size"]:
        '''Get output logits of all token positions'''

        if intervention_method == "actadd":
            logits = self.activation_addition(prompts, direction, steering_layer, coeffs=coeffs)
        elif intervention_method == "ablation":
            logits = self.directional_ablation(prompts, direction)
        else:
            logits = self.model.trace(prompts, trace=False).logits.detach().to("cpu")
        logits = logits.to(torch.float64)
        return logits
    
    def get_last_position_logits(
        self, instructions: List[str], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        intervention_method: Optional[int] = None, 
        steering_layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0, 
        batch_size: Optional[int] = 16
    ) -> TensorType["n_instructions", "vocab_size"]:
        '''Get the logits of the last token position'''
        total = ceildiv(len(instructions), batch_size)
        if total > 5:
            pbar = tqdm(chunks(instructions, batch_size), total=total, desc="Getting last position logits")
        else:
            pbar = chunks(instructions, batch_size)

        last_pos_logits = []
        for instruction_batch in pbar:
            prompts = self.apply_chat_template(instructions=instruction_batch)
            logits = self.get_logits(prompts, direction, intervention_method, steering_layer, coeffs)[:, -1, :]
            
            if last_pos_logits is None:
                last_pos_logits = logits
            else:
                last_pos_logits.append(logits)

        return torch.vstack(last_pos_logits)
    
    def _generate_act_add(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        layer: Optional[int] = None, 
        coeffs: Optional[Union[float, List[float], TensorType[-1]]] = 1.0,
        max_new_tokens: Optional[int] = 10,
        do_sample: Optional[bool] = False, **kwargs
    ) -> TensorType["n_prompt", "seq_len"]:
        """Text generation with activation addition"""
        inputs, direction, layer, coeffs = self._prepare_act_add_inputs(prompts, direction, layer, coeffs)

        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            self.model_block_modules[layer].input += (direction * coeffs)

            for _ in range(max_new_tokens - 1):
                acts = self.model_block_modules[layer].next().input.t[-1]
                self.model_block_modules[layer].input.t[-1] = acts + (direction * coeffs.squeeze(-1))

            outputs = self.model.generator.output.detach().to("cpu").save()
        return outputs.value
    
    def _generate_abalation(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: Optional[TensorType["hidden_size"]] = None, 
        max_new_tokens: Optional[int] = 10,
        do_sample: Optional[bool] = False, **kwargs
    )-> TensorType["n_prompt", "seq_len"]:
        """Text generation with directional ablation"""
        inputs, unit_direction = self._prepare_ablation_inputs(prompts, direction)

        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            for layer in range(self.n_layers):
                acts = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].input, unit_direction))
                self.model_block_modules[layer].input = acts

                act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].output[0], unit_direction))
                self.attn_modules[layer].output = (act_post_attn,) + self.attn_modules[layer].output[1:]

                act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].output, unit_direction))
                self.mlp_modules[layer].output = act_post_mlp
            
                for _ in range(max_new_tokens - 1):
                    act_pre = nnsight.apply(orthogonal_rejection, *(self.model_block_modules[layer].next().input.t[-1], unit_direction))
                    self.model_block_modules[layer].input.t[-1] = act_pre

                    act_post_attn = nnsight.apply(orthogonal_rejection, *(self.attn_modules[layer].next().output[0].t[-1], unit_direction))
                    self.attn_modules[layer].output[0].t[-1] = act_post_attn

                    act_post_mlp = nnsight.apply(orthogonal_rejection, *(self.mlp_modules[layer].next().output.t[-1], unit_direction))
                    self.mlp_modules[layer].output.t[-1] = act_post_mlp

            outputs = self.model.generator.output.detach().to("cpu").save()
        return outputs.value
    
    def generate(
        self, prompts: Union[str, List[str], BatchEncoding], 
        direction: TensorType["hidden_size"] = None, 
        intervention_method: int = None, 
        steering_layer: int = None, 
        coeffs: Union[float, List[float], TensorType[-1]] = 1.0,
        max_new_tokens: int = 10,
        do_sample: bool = False,
        **kwargs
    ) -> TensorType["n_prompt", "seq_len"]:
        if intervention_method =="actadd":
            return self._generate_act_add(prompts, direction, steering_layer, coeffs, max_new_tokens, do_sample, **kwargs)
        elif intervention_method == "ablation":
            return self._generate_abalation(prompts, direction, max_new_tokens, do_sample, **kwargs)
        
        inputs = self.tokenize(prompts)
        with self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs) as tracer:
            outputs = self.model.generator.output.detach().to("cpu").save()
        return outputs.value
    
    def generate_completions(
        self, instructions: List[str], 
        direction: TensorType["hidden_size"] = None, 
        intervention_method: int = None, 
        steering_layer: int = None, 
        coeffs: Union[float, List[float], TensorType[-1]] = 1.0, 
        batch_size: int = 16,
        max_new_tokens: int = 10, 
        do_sample: bool = False,
        return_prompt: bool = False, 
        **generation_kwargs
    ) -> List[str]:
        '''Run text generation in batch with given intervention method and decode the outputs to strings'''
        completions = []
        total = ceildiv(len(instructions), batch_size)

        for instruction_batch in tqdm(chunks(instructions, batch_size), total=total, desc="Generating completions"):
            formatted_prompts = self.apply_chat_template(instruction_batch)
            inputs = self.tokenize(formatted_prompts)
            outputs = self.generate(
                inputs, direction, intervention_method, steering_layer, coeffs, 
                max_new_tokens, do_sample, **generation_kwargs
            )
            
            if return_prompt:
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                seq_len = inputs.input_ids.shape[1]
                decoded_outputs = self.tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)

            completions.extend(decoded_outputs)

        return completions
    