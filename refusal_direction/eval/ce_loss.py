import os
import json
import itertools
from tqdm import tqdm
from typing import List, Optional, Callable, Iterator

import torch
import torch.nn.functional as F
from torchtyping import TensorType
from datasets import load_dataset
from transformers import AutoTokenizer
from ..model_wrapper import ModelBase
from ..config import Config


def batch_iterator_chat_completions(
    tokenizer: AutoTokenizer, format_instructions_fn: Callable, 
    dataset_instructions, dataset_outputs, 
    eoi_toks: TensorType[int, -1], batch_size: Optional[int] = 16
) -> Iterator :
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        formatted_prompts = format_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)
        inputs = tokenizer(formatted_prompts, padding=True, truncation=False, return_tensors="pt")

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position
        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):
                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask 


def batch_iterator_custom_completions(
    completions_file_path: str, tokenizer, format_instructions_fn: Callable, 
    eoi_toks: TensorType[int, -1], batch_size: Optional[int] = 16
) -> Iterator :
    """Yields batches from the custom completions."""

    custom_completions = json.load(open(completions_file_path, 'r'))

    instructions, completions = [], []

    for i in range(len(custom_completions)):
        instructions.append(custom_completions[i]['prompt'])
        completions.append(custom_completions[i]['response'].lstrip())

    return batch_iterator_chat_completions(tokenizer, format_instructions_fn, instructions, completions, eoi_toks, batch_size)


def batch_iterator_alpaca(
    tokenizer: AutoTokenizer, format_instructions_fn: Callable, 
    eoi_toks: TensorType[int, -1], batch_size: Optional[int] = 16
) -> Iterator :
    """Yields batches from the Alpaca dataset."""

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions = [], []

    for i in range(len(dataset)):
        if dataset[i]['input'].strip() == '': # filter for instructions that do not have inputs
            instructions.append(dataset[i]['instruction'])
            completions.append(dataset[i]['output'])

    return batch_iterator_chat_completions(tokenizer, format_instructions_fn, instructions, completions, eoi_toks, batch_size)


def batch_iterator_pile(tokenizer, max_length: int, batch_size: Optional[int] = 16) -> Iterator:
    """Yields batches from the Pile dataset."""
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        yield inputs, loss_mask


def compute_loss_over_dataset(model, batch_iterator, n_batches=256, intervention_label=None):
    accumulated_loss = torch.tensor(0, dtype=torch.float64)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64)

    batch_idx = 0
    pbar = tqdm(total=n_batches, desc=f"Computing loss with intervention: {intervention_label}")
    for inputs, loss_mask in batch_iterator:
        if n_batches != -1 and batch_idx >= n_batches:
            break

        input_ids = inputs["input_ids"]
        logits = model.get_logits(inputs, intervention_method=intervention_label)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
        log_probs_for_labels = torch.cat(
            [
                log_probs_for_labels,
                torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
            ],
            dim=-1
        )

        # apply loss_mask
        log_probs_for_labels = log_probs_for_labels * loss_mask

        accumulated_loss += -log_probs_for_labels.sum()
        accumulated_n_tokens += loss_mask.sum()

        batch_idx += 1
        pbar.update(1)
    
    ce_loss = accumulated_loss / accumulated_n_tokens
    perplexity = torch.exp(ce_loss)    

    return ce_loss, perplexity, accumulated_n_tokens


def evaluate_loss(
    cfg: Config,
    model: ModelBase,
    max_seq_length: Optional[int] = 256,
    dataset_labels: Optional[List[int]] = ["pile", "alpaca", "alpaca_custom_completions"],
    completions_file_path: Optional[str] =None,
    intervention_label: Optional[str] = None,
):
    batch_size = cfg.ce_loss_batch_size
    n_batches = cfg.ce_loss_n_batches
    os.makedirs(cfg.artifact_path() / 'loss_evals', exist_ok=True)

    if (cfg.artifact_path() / f'loss_evals/{intervention_label}_loss_eval.json').is_file():
        result = json.load(open(cfg.artifact_path() / f'loss_evals/{intervention_label}_loss_eval.json', "r"))
    else:
        result = {}
    
    for label in dataset_labels:
        if label in result:
            continue
        if label == 'pile':
            dataset_iterator = batch_iterator_pile(model.tokenizer, batch_size=batch_size, max_length=max_seq_length)
            n = n_batches
        elif label == 'alpaca':
            dataset_iterator = batch_iterator_alpaca(model.tokenizer, model.apply_chat_template, eoi_toks=torch.tensor(model.eoi_toks), batch_size=batch_size)
            n = n_batches
        elif label == 'alpaca_custom_completions':
            assert completions_file_path is not None, "A file path must be passed to load the completions"

            dataset_iterator = batch_iterator_custom_completions(
                completions_file_path=completions_file_path,
                tokenizer=model.tokenizer,
                format_instructions_fn=model.apply_chat_template,
                eoi_toks=torch.tensor(model.eoi_toks),
                batch_size=batch_size,
            )
            n = -1 # process all completions
        else:
            raise ValueError(f"Unknown dataset label: {label}")

        ce_loss, perplexity, n_tokens = compute_loss_over_dataset(model, dataset_iterator, n_batches=n, intervention_label=intervention_label)
        print(f"{label.upper()} DATASET:")
        print(f"CE loss: {ce_loss.item()}, Perplexity: {perplexity.item()}, N tokens: {n_tokens.item()}")

        result[label] = {
            "ce_loss": ce_loss.item(),
            "perplexity": perplexity.item(),
            "n_tokens": n_tokens.item()
        }

        with open(cfg.artifact_path() / f'loss_evals/{intervention_label}_loss_eval.json', "w") as f:
            json.dump(result, f, indent=4)
