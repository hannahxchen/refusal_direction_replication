import os
import json
import random
import argparse
import logging

import torch
from .config import Config
from .model_wrapper.list import get_supported_model_class
from .dataset.load_dataset import load_and_sample_datasets, load_eval_dataset, load_dataset_split
from .get_directions import generate_directions, select_direction
from .eval import get_refusal_scores, evaluate_jailbreak, evaluate_loss, evaluate_coherence, evaluate_magnitude, compute_magnitude_corr
from .utils import save_to_json_file

logging.basicConfig(level=logging.INFO)
torch.set_grad_enabled(False);
intervention_labels = ['baseline', 'actadd', 'ablation']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model')
    parser.add_argument('--config_file', type=str, required=False, default=None, help='Load configuration from file.')
    parser.add_argument('--resume_from_step', type=int, required=False, default=-1, help="Resume from step number")
    parser.add_argument('--run_jailbreak_eval', action="store_true", help="Run jailbreak evaluation")
    parser.add_argument('--run_magnitude_eval', action="store_true", help="Run magnitude evaluation")
    return parser.parse_args()

    
def filter_examples(dataset, scores, threshold, harm_type):
    if harm_type == "harmful":
        comparison = lambda x, y: x > y
    else:
        comparison = lambda x, y: x < y
    return [inst for inst, score in zip(dataset, scores) if comparison(score, threshold)]


def filter_data(model, dataset, harm_type="harmful"):
    """Filter datasets based on refusal scores."""
    refusal_toks = model.refusal_toks
    scores = get_refusal_scores(model.get_last_position_logits(dataset), refusal_toks=refusal_toks).tolist()
    return filter_examples(dataset, scores, 0,  harm_type), scores


def load_cached_data(cfg, harm_type, split, threshold=0):
    try:
        data = json.load(open(cfg.artifact_path() / f"datasplits/{harm_type}_{split}.json", "r"))
        prompts = [x["prompt"] for x in data]
        scores = [x["refusal_score"] for x in data]
        return filter_examples(prompts, scores, threshold, harm_type)
    except:
        raise FileNotFoundError


def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'), exist_ok=True)

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
        batch_size=cfg.batch_size
    )

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions):
    """Select and save the direction."""
    os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'), exist_ok=True)

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
        batch_size=cfg.batch_size
    )

    save_to_json_file(f'{cfg.artifact_path()}/direction_metadata.json', {"pos": pos, "layer": layer})
    torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

    return pos, layer, direction


def generate_and_save_completions_for_dataset(cfg, model, intervention_label, instructions, categories=None, save_path=None):
    """Generate and save completions for a dataset."""

    completions = model.generate_completions(
        instructions, intervention_method=intervention_label, 
        max_new_tokens=cfg.max_new_tokens, batch_size=cfg.generation_batch_size
    )

    if categories is None:
        results = [{"prompt": instructions[i], "response": completions[i]} for i in range(len(instructions))]
    else:
        results = [{"category": categories[i], "prompt": instructions[i], "response": completions[i]} for i in range(len(instructions))]
    
    if save_path is None:
        return results
    else:
        save_to_json_file(save_path, results)


def run_pipeline(args):
    """Run the full pipeline."""
    if args.config_file is not None:
        cfg = Config.load(args.config_file)
    else:
        model_alias = os.path.basename(args.model_path)
        
        cfg = Config(model_path=args.model_path, model_alias=model_alias)
        cfg.save()
    
    print(cfg.model_path)
    model = get_supported_model_class(cfg.model_path)
    
    datasets = None

    # 0. Load and sample train/valid splits
    if args.resume_from_step <= 0:
        datasets = load_and_sample_datasets(cfg)

        save_dir = cfg.artifact_path() / 'datasplits'
        os.makedirs(save_dir, exist_ok=True)
        stats = {"harmful": {}, "harmless": {}}

        for split in ["train", "val"]:
            for harm_type in ["harmful", "harmless"]:
                data = datasets[harm_type][split]
                stats[harm_type][split] = {"original": len(data)}

                # Generate and save completions on train/val splits
                results = generate_and_save_completions_for_dataset(cfg, model, 'baseline', datasets[harm_type][split])
                
                # Filter datasets based on refusal scores
                if getattr(cfg, f"filter_{split}"):
                    filtered, scores = filter_data(model, data, harm_type=harm_type)
                    stats[harm_type][split]["filtered"] = len(filtered)

                    for i, r in enumerate(results):
                        r["refusal_score"] = scores[i]
                    
                    datasets[harm_type][split] = filtered

                save_to_json_file(cfg.artifact_path() / f"datasplits/{harm_type}_{split}.json", results)
        
        if cfg.filter_train or cfg.filter_val:
            save_to_json_file(cfg.artifact_path() / "datasplits/filterd_stats.json", stats)

    # 1. Generate candidate refusal directions
    if args.resume_from_step <= 1:
        logging.info("Generating candidate refusal directions")
        if datasets is None:
            harmful_train = load_cached_data(cfg, "harmful", "train")
            harmless_train = load_cached_data(cfg, "harmless", "train")
        else:
            harmful_train, harmless_train = datasets["harmful"]["train"], datasets["harmless"]["train"]

        generate_and_save_candidate_directions(cfg, model, harmful_train, harmless_train)

    candidate_directions = torch.load(os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))
    
    # 2. Select the most effective refusal direction
    if args.resume_from_step <= 2:
        logging.info("Evaluating candidate directions")
        if datasets is None:
            harmful_val = load_cached_data(cfg, "harmful", "val")
            harmless_val = load_cached_data(cfg, "harmless", "val")
        else:
            harmful_val, harmless_val = datasets["harmful"]["val"], datasets["harmless"]["val"]
        pos, layer, direction = select_and_save_direction(cfg, model, harmful_val, harmless_val, candidate_directions)


    direction_metadata = json.load(open(f'{cfg.artifact_path()}/direction_metadata.json', "r"))
    layer = direction_metadata["layer"]
    direction = torch.load(f'{cfg.artifact_path()}/direction.pt')
    model.set_intervene_direction(direction)
    model.set_actAdd_intervene_layer(layer)

    # 3a. Generate and save completions on harmful/harmless evaluation datasets
    if args.resume_from_step <= 3:
        logging.info("Generating completions on harmful evaluation datasets")
        save_dir = cfg.artifact_path() / 'completions'
        os.makedirs(save_dir, exist_ok=True)

        for dataset_name in cfg.evaluation_datasets:
            dataset = load_eval_dataset(dataset_name)
            instructions = [data["instruction"] for data in dataset]
            categories = [data["category"] for data in dataset]

            for intervention in intervention_labels:
                generate_and_save_completions_for_dataset(cfg, model, intervention, instructions, categories, save_dir / f'{dataset_name}_{intervention}_completions.json')
    
        random.seed(3456)
        harmless_test = random.sample(load_dataset_split(harmtype='harmless', split='test'), cfg.n_test)
        instructions = [data["instruction"] for data in harmless_test]
        categories = [data["category"] for data in harmless_test]

        for intervention in intervention_labels:
            generate_and_save_completions_for_dataset(cfg, model, intervention, instructions, categories, save_dir / f'harmless_{intervention}_completions.json')

    # 4. Evaluate loss on harmless datasets
    if args.resume_from_step <= 5:
        logging.info("Evaluating model CE loss")
        for intervention in intervention_labels:
            on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_{intervention}_completions.json')
            evaluate_loss(cfg, model, completions_file_path=on_distribution_completions_file_path, intervention_label=intervention)

    # 5. Evaluate model coherence on general benchmarks
    if args.resume_from_step <= 6:
        for intervention in intervention_labels:
            evaluate_coherence(cfg, model, intervention_label=intervention)



def run_jailbreak_eval(args):
    """Run jailbreak evaluation on generated completions"""
    cfg = Config.load(args.config_file)
    os.makedirs(cfg.artifact_path() / "evaluation", exist_ok=True)

    logging.info("Running jailbreak evaluation")
    # 3b. Evaluate completions and save results on harmful evaluation datasets
    for dataset_name in cfg.evaluation_datasets:
        evaluate_jailbreak(dataset_name, cfg.artifact_path(), methodologies=cfg.jailbreak_eval_methodologies, batch_size=cfg.batch_size)

    evaluate_jailbreak("harmless", cfg.artifact_path(), methodologies=cfg.jailbreak_eval_methodologies, batch_size=cfg.batch_size)


def run_magnitude_eval(args):
    """Run vector magnitude evaluation"""
    cfg = Config.load(args.config_file)
    save_dir = cfg.artifact_path() / "magnitude_eval"
    os.makedirs(save_dir, exist_ok=True)

    print(cfg.model_path)
    model = get_supported_model_class(cfg.model_path)

    direction_metadata = json.load(open(f'{cfg.artifact_path()}/direction_metadata.json', "r"))
    layer = direction_metadata["layer"]
    direction = torch.load(f'{cfg.artifact_path()}/direction.pt')
    model.set_intervene_direction(direction)
    model.set_actAdd_intervene_layer(layer)

    logging.info("Running magnitude evaluation")

    eval_datasets = {}
    # Load harmful/harmless test split
    eval_datasets["harmful_test"] = load_dataset_split("harmful", split="test", instructions_only=True)
    eval_datasets["harmless_test"] = load_dataset_split("harmless", split="test", instructions_only=True)
    # Load XSTest safe/unsafe instructions
    eval_datasets["xstest_safe"] = load_eval_dataset("xstest_safe")
    eval_datasets["xstest_unsafe"] = load_eval_dataset("xstest_unsafe")

    # Compute projection ratios and refusal scores
    for dataset_name, instructions in eval_datasets.items():
        evaluate_magnitude(cfg, model, instructions, save_dir, dataset_name)

    # Compute the correlation between projection ratios and refusal scores of each example
    compute_magnitude_corr(list(eval_datasets), save_dir)


if __name__ == "__main__":
    args = parse_arguments()

    if args.run_jailbreak_eval:
        run_jailbreak_eval(args)
    elif args.run_magnitude_eval:
        run_magnitude_eval(args)
    else:
        run_pipeline(args)
