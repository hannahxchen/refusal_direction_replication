import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard

@dataclass
class Config(YAMLWizard):
    model_path: str
    model_alias: str
    save_dir: str = Path().absolute() / "runs"

    # Per label
    n_train: int = 128
    n_val: int = 32
    n_test: int = 100

    filter_train: bool = True
    filter_val: bool = True

    evaluation_datasets: List[str] = field(default_factory=lambda: ["jailbreakbench", "harmfulbench_test"])
    max_new_tokens: int = 512
    batch_size: int = 32
    generation_batch_size: int = 24
    jailbreak_eval_methodologies: List[str] = field(default_factory=lambda: ["substring_matching", "llamaguard2"])
    refusal_eval_methodologies: List[str] = field(default_factory=lambda: ["substring_matching"])
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    def artifact_path(self) -> str:
        return self.save_dir / self.model_alias
    
    def save(self):
        os.makedirs(self.artifact_path(), exist_ok=True)
        self.to_yaml_file(self.artifact_path() / 'config.yaml')

    def load(filepath: str):
        try:
            return Config.from_yaml_file(filepath)
        
        except FileNotFoundError:
            return None