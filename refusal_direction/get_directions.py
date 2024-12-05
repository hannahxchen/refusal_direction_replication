import os
import json
import math
from tqdm import tqdm
from typing import List, Optional, Tuple
from torchtyping import TensorType
import torch
from refusal_direction import ModelBase
from .eval.refusal import get_refusal_scores, plot_refusal_scores
from .utils import ceildiv, chunks, kl_div_fn


def get_mean_activations(
    model: ModelBase, instructions: List[str], positions: Optional[List[int]] = [-1], batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Compute mean activations across instructions for each token position and layer"""
    layers = list(range(model.n_layers))
    total = ceildiv(len(instructions), batch_size)
    activation_sums = None
    for instruction_batch in tqdm(chunks(instructions, batch_size), total=total, leave=False):
        formatted_prompts = model.apply_chat_template(instructions=instruction_batch)
        inputs = model.tokenizer(formatted_prompts, padding=True, truncation=False, return_tensors="pt")
        activations = model.get_activations(layers, inputs, positions) # (n_layer, n_prompt, n_pos, hidden_size)
        activations = activations.sum(dim=1) # (n_layer, n_pos, hidden_size)
        if activation_sums is None:
            activation_sums = activations
        else:
            activation_sums += activations

    mean_activations = activation_sums / len(instructions)
    mean_activations = mean_activations.permute(1, 0, 2) # (n_pos, n_layer, hidden_size)
    return mean_activations


def get_mean_diff(
    model: ModelBase, harmful_instructions: List[str], harmless_instructions: List[str], 
    positions: Optional[List[int]] = [-1], batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Mean activation difference"""
    harmful_mean_acts = get_mean_activations(model, harmful_instructions, positions=positions, batch_size=batch_size)
    harmless_mean_acts = get_mean_activations(model, harmless_instructions, positions=positions, batch_size=batch_size)
    return harmful_mean_acts - harmless_mean_acts


def generate_directions(
    model: ModelBase, harmful_instructions: List[str], harmless_instructions: List[str], 
    artifact_dir: str, batch_size: Optional[int] = 32
) -> List[TensorType["n_pos", "n_layer", "hidden_size"]]:
    """Generate directions from all layers and post instruction token positions"""
    os.makedirs(artifact_dir, exist_ok=True)

    positions = list(range(-len(model.eoi_toks), 0))
    mean_diffs = get_mean_diff(model, harmful_instructions, harmless_instructions, positions=positions, batch_size=batch_size)

    assert mean_diffs.shape == (len(model.eoi_toks), model.n_layers, model.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")
    return mean_diffs


def filter_fn(
    refusal_score, steering_score, kl_div_score, layer, n_layer, 
    kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20
) -> bool:
    """Function for filtering directions. Returns True if the direction should be filtered out
    """
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def select_direction(
    model: ModelBase, 
    harmful_instructions: List[str], harmless_instructions: List[str], 
    candidate_directions, artifact_dir: str, 
    kl_threshold: Optional[float] = 0.1, # directions larger KL score are filtered out
    induce_refusal_threshold: Optional[float] = 0.0, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage: Optional[float] = 0.2, # discard the directions extracted from the last 20% of the model
    batch_size: Optional[int] = 32
) -> Tuple[int, int, TensorType["hidden_size"]]:
    """Select the best vector from a set of candidate directions
    For each vector r, we compute the following:
    1) kl_div_score < 0.1: average KL divergence of harmless_instructions at the last token position (with VS without directional ablation of r)
    2) bypass_score: average refusal score across harmful_instructions (with directional ablation of r).
    3) induce_score > 0: average refusal scores across harmless_instructions (with activation addition of r).
    Addionally, the layer of r < (1 - prune_layer_percentage) * n_layer. This ensures r is not too close to the unembedding directions.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    n_pos, n_layer, hidden_size = candidate_directions.shape
    refusal_toks = model.refusal_toks

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), dtype=torch.float64)

    baseline_refusal_scores_harmful = get_refusal_scores(model.get_last_position_logits(harmful_instructions, batch_size=batch_size), refusal_toks=refusal_toks)
    baseline_harmless_logits = model.get_last_position_logits(harmless_instructions, batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(baseline_harmless_logits, refusal_toks=refusal_toks)

    for source_pos in tqdm(range(-n_pos, 0), desc="Token position", leave=False):
        for source_layer in (pbar := tqdm(range(n_layer), leave=False)):
            pbar.set_description(f"Evaluating model layer {source_layer}")
            direction = candidate_directions[source_pos, source_layer]
            
            # Compute kl_score
            intervention_logits = model.get_last_position_logits(harmless_instructions, direction=direction, intervention_method="ablation", batch_size=batch_size)
            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

            # Compute bypass_score
            intervention_logits = model.get_last_position_logits(harmful_instructions, direction=direction, intervention_method="ablation", batch_size=batch_size)
            refusal_scores = get_refusal_scores(intervention_logits, refusal_toks=refusal_toks)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

            # Compute induce_score
            intervention_logits = model.get_last_position_logits(
                harmless_instructions, direction=direction, intervention_method="actadd", 
                steering_layer=source_layer, coeffs=1.0, batch_size=batch_size
            )
            refusal_scores = get_refusal_scores(intervention_logits, refusal_toks=refusal_toks)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()


    token_labels = model.tokenizer.convert_ids_to_tokens(model.eoi_toks)
    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmful.mean().item(),
        token_labels=token_labels,
        title='Ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='bypass_scores_ablation'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_refusal_scores_harmless.mean().item(),
        token_labels=token_labels,
        title='Adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='induce_scores_actadd'
    )

    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=token_labels,
        title='KL Divergence when ablating direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores_ablation'
    )

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'bypass_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'induce_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'bypass_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'induce_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, key=lambda x: x['bypass_score'], reverse=False)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    assert len(filtered_scores) > 0, "All scores have been filtered out!"

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")
    
    return pos, layer, candidate_directions[pos, layer]
