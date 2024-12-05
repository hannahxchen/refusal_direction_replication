import os
import json
import random

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']
SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = [
    "advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", 
    "jailbreakbench", "strongreject", "alpaca", "xstest_safe", "xstest_unsafe"
]


def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset


def load_eval_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset


def load_and_sample_datasets(cfg):
    """Load datasets and sample them based on the configuration."""
    random.seed(42)
    datasets = {"harmful": {}, "harmless": {}}
    datasets["harmful"]["train"] = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), cfg.n_train)
    datasets["harmless"]["train"] = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), cfg.n_train)
    datasets["harmful"]["val"] = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), cfg.n_val)
    datasets["harmless"]["val"] = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), cfg.n_val)
    datasets["harmful"]["test"] = random.sample(load_dataset_split(harmtype='harmful', split='test', instructions_only=True), cfg.n_test)
    datasets["harmless"]["test"] = random.sample(load_dataset_split(harmtype='harmless', split='test', instructions_only=True), cfg.n_test)
    return datasets
