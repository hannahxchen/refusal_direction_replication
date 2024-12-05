import logging
import json
from tqdm import tqdm
from typing import List
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from refusal_direction import ModelBase
from ..eval import get_refusal_scores
from ..utils import ceildiv, chunks, save_to_json_file

intervention_labels = ['baseline', 'actadd', 'ablation']

def projection_ratio(acts: TensorType[..., -1], steering_vec: TensorType[-1]) -> TensorType[-1]:
    """The ratio of the scalar projection of input activation onto the steering vector to the steering vector's length"""
    cosin_sim = F.cosine_similarity(acts, steering_vec, dim=-1)
    proj_len = acts.norm(dim=-1) * cosin_sim
    proj_ratio = proj_len.unsqueeze(-1) / steering_vec.norm()
    return proj_ratio.squeeze(-1)


def get_last_position_activations(model: ModelBase, prompts: List[str], batch_size: int = 32):
    acts_all = []
    total = ceildiv(len(prompts), batch_size)
    for prompt_batch in tqdm(chunks(prompts, batch_size), total=total, desc="Extracting activations"):
        prompt_batch = model.apply_chat_template(instructions=prompt_batch)
        acts = model.get_activations(model.actAdd_layer, prompt_batch, positions=[-1]).squeeze()
        acts_all.append(acts)
    return torch.concat(acts_all, dim=0)


def load_magnitude_results(filepath):
    with open(filepath, "r") as f:
        results = json.load(f)
    
    proj_ratios = np.array([x["projection_ratio"] for x in results])
    refusal_scores = np.array([x["refusal_score"] for x in results])
    return proj_ratios, refusal_scores


def evaluate_magnitude(cfg, model: ModelBase, instructions: List[str], save_dir: Path, dataset_name: str):
    """Compute projection ratios and refusal scores of each instruction with all intervention methods"""
    for intervention in intervention_labels:
        logging.info(f"Evaluating vector magnitude on {dataset_name} test with {intervention} intervention")
        intervention_logits = model.get_last_position_logits(instructions, intervention_method=intervention, batch_size=cfg.batch_size)
        refusal_scores = get_refusal_scores(intervention_logits, refusal_toks=model.refusal_toks).tolist()
        completions = model.generate_completions(
            instructions, intervention_method=intervention, 
            max_new_tokens=cfg.max_new_tokens, batch_size=cfg.generation_batch_size
        )

        if intervention == "baseline":
            activations = get_last_position_activations(model, instructions, cfg.batch_size)
            proj_ratios = projection_ratio(activations, model.intervene_direction).tolist()

        results = []
        for i in range(len(instructions)):
            r = {
                "instruction": instructions[i],
                "response": completions[i],
                "refusal_score": refusal_scores[i]
            }
            if intervention == "baseline":
                r["projection_ratio"] = proj_ratios[i]

            results.append(r)

        save_to_json_file(save_dir / f"{dataset_name}_{intervention}.json", results)


def compute_magnitude_corr(eval_dataset_names: List[str], save_dir: Path):
    """Compute the correlation between projection ratios and refusal scores"""

    all_projection_ratios, all_refusal_scores = [], []
    for dataset_name in eval_dataset_names:
        proj_ratios, refusal_scores = load_magnitude_results(save_dir / f"{dataset_name}_baseline.json")
        all_projection_ratios = np.append(all_projection_ratios, proj_ratios)
        all_refusal_scores = np.append(all_refusal_scores, refusal_scores)

    corr, p_val = pearsonr(all_projection_ratios, all_refusal_scores)
    save_to_json_file(save_dir / "magnitude_corr.json", {"corr": corr, "p_val": p_val})