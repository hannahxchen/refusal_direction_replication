import os
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
from datasets import load_dataset
import pandas as pd
import torch
import torch.nn.functional as F

from refusal_direction.config import Config
from refusal_direction.model_wrapper import ModelBase
from refusal_direction.utils import ceildiv, chunks, save_to_json_file

QA_prompt = """\
Q: What is human life expectancy in the United States?\n\
A: Human life expectancy in the United States is 78 years.\n\n\
Q: Who was president of the United States in 1955?\n
A: Dwight D. Eisenhower was president of the United States in 1955.\n\n\
Q: Which party did he belong to?\n\
A: He belonged to the Republican Party.\n\n\
Q: What is the square root of banana?\n\
A: I have no comment.\n\n\
Q: How does a telescope work?\n\
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\n\
Q: Where were the 1992 Olympics held?\n\
A: The 1992 Olympics were held in Barcelona, Spain.\
"""
ans_choices = ['A', 'B', 'C', 'D', 'E']

def prepare_task_inputs(model: ModelBase, task_name: str) -> List[Dict]:
    data = []
    if task_name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")

        for _id, x in enumerate(ds):
            prompt = f'Question: {x["question"].strip()}'
            for i, ans in enumerate(x["choices"]):
                prompt += f'\n{ans_choices[i]}. {ans}'

            prompt = model.apply_chat_template(prompt)[0] + "\nAnswer:"
            gold_label_idx = x["answer"]

            for i, ans in enumerate(x["choices"]):
                data.append({
                    'question_id': _id,
                    'prompt': prompt,
                    'target': f' {ans_choices[i]}',
                    'correct': gold_label_idx == i
                })

    elif task_name == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
        for x in ds:
            prompt = f'Question: {x["question"]}'
            for i, ans in enumerate(x["choices"]["text"]):
                prompt += f'\n({ans_choices[i]}) {ans}'

            prompt = model.apply_chat_template(prompt)[0] + "\nAnswer:"
            gold_label_idx = x['choices']["label"].index(x["answerKey"])

            for i, ans in enumerate(x['choices']["label"]):
                data.append({
                    'question_id': x['id'],
                    'prompt': prompt,
                    'target': " " + ans,
                    'correct': gold_label_idx == i
                })

    elif task_name == "truthful_qa":
        ds = load_dataset("EleutherAI/truthful_qa_mc")["validation"]
        for _id, x in enumerate(ds):
            prompt = model.apply_chat_template(f'{QA_prompt}\n\nQ: {x["question"]}')[0] + "\nA:"
            gold_label_idx = x["label"]
            for i, ans in enumerate(x['choices']):
                data.append({
                    'question_id': _id,
                    'prompt': prompt,
                    'target': " " + ans,
                    'correct': gold_label_idx == i
                })
    else:
        raise NotImplementedError
    return data


def get_ans_prob(logits, ans_token_ids) -> float:
    log_probs = F.log_softmax(logits[-len(ans_token_ids):], dim=-1)
    token_probs = torch.gather(log_probs, -1, ans_token_ids.unsqueeze(-1)).squeeze()
    ans_prob = torch.sum(token_probs).item()
    return ans_prob


def compute_task_acc(
    model: ModelBase, 
    task_inputs: List[Dict],
    intervention_label: Optional[str] = "baseline",
    batch_size: Optional[int] = 32
) -> Tuple[float, float]:
    total = ceildiv(len(task_inputs), batch_size)
    outputs = []

    for input_batch in tqdm(chunks(task_inputs, batch_size), total=total):
        prompts = [x["prompt"] + x["target"] for x in input_batch]
        logits = model.get_logits(prompts, intervention_method=intervention_label)[:, :-1, :]

        for i, x in enumerate(input_batch):
            target_token_ids = model.get_token_ids(x["target"])
            x["ans_prob"] = get_ans_prob(logits[i], target_token_ids)

        outputs.extend(input_batch)

    df = pd.DataFrame.from_records(outputs)
    df = df.sort_values('ans_prob', ascending=False).drop_duplicates(['question_id'])
    correct = df[df.correct].shape[0]
    n_questions = len(df["question_id"].unique())
    return correct / n_questions


def evaluate_coherence(
    cfg: Config, model: ModelBase, 
    intervention_label: str,
    tasks: List[str] = ["arc", "truthful_qa", "mmlu"],
):
    save_dir = cfg.artifact_path() / "coherence_evals"
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    for task in tasks:
        task_inputs = prepare_task_inputs(model, task)
        acc = compute_task_acc(model, task_inputs, intervention_label, cfg.batch_size)
        results[task] = acc
        print(f"Task: {task}, Intervention: {intervention_label}, Accuracy={acc:.3f}")
        
        save_to_json_file(save_dir / f'{intervention_label}.json', results)