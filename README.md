# Replication of Refusal in Language Models Is Mediated by a Single Direction

This repository includes code and results for replicating the paper "[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)". The code mostly follows the [original codebase](https://github.com/andyrdt/refusal_direction) with some modifications to build on [NNsight](https://github.com/ndif-team/nnsight) library.

## Setup

All the required dependencies and packages are listed in `environment.yml`. To create a virtual enviornment from it, run the following command:

```bash
conda env create --name envname --file=environment.yml
```

## Running the Code

To reproduce the main results, run the following command:

```bash
python -m refusal_direction.run --model_path MODEL_PATH
```

Alternatively, you can create a `config.yaml` file and use `--config_file path/to/config.yaml` instead.

The main pipeline includes the following steps:

0. Data preprocessing
1. Generate candidate directions
2. Select a direction
3. Run and save completions on evaluation datasets with different intervention
4. Evaluate cross entropy loss on harmless data
5. Evaluate model coherence on general language benchmarks

You can also resume from any of the steps listed above. For example, to resume from Step 4 to evaluate model loss, you can run the command:

```bash
python -m refusal_direction.run --config_file path/to/config.yaml --resume_from_step 4
```


Finally, to compute the refusal scores (substring match) and safety scores (Llama Guard 2) on the generated completions (from Step 3), run the command as follows:

```bash
python -m refusal_direction.run --config_file path/to/config.yaml --run_jailbreak_eval
```


Note: For the model coherence evaluation, the code is not provided in the paper's GitHub repository. We currently have implemented MMLU, ARC, and TruthfulQA tasks. We observe some accuracy differences between our implementation and results reported in the original paper. There may be some implementation differences.

## Additional Evaluation

The evaluation in the original paper mainly focuses on evaluating the refusal *direction*. Additionally, we include evaluation on the *magnitude* of the refusal vector, which assesses how well the scalar projection on the refusal vector can reflect the degree of refusal scores in the model outputs. We perform evaluation on the harmful/harmless test split and also on a more challenging dataset, [XSTest](https://github.com/paul-rottger/exaggerated-safety).

To run this evaluation, you can use the following command:
```bash
python -m refusal_direction.run --config_file path/to/config.yaml --run_magnitude_eval
```
Further analysis and results can be found in `analysis/magnitude_evaluation.ipynb`.


### References
- [Refusal in language models is mediated by a single direction](https://arxiv.org/abs/2406.11717). (Arditi et al., 2024) [[Github repo](https://github.com/andyrdt/refusal_direction)]
- NNsight library: https://nnsight.net/
- [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](https://aclanthology.org/2024.naacl-long.301) (Röttger et al., NAACL 2024) [[GitHub repo](https://github.com/paul-rottger/exaggerated-safety)]

