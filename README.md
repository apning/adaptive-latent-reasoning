# Learning When to Stop: Adaptive Latent Reasoning via Reinforcement Learning

Note: This project is a work-in-progress. Active development is performed in a private repo. Updates may not be reflected immediately on public repo.

### Environment

Install

```bash
conda env create -f environment.yml && conda activate adaptive-latent-reasoning
```

### Replicating results

Please see training_scripts/README.md for more instructions.

### Trained model weights

All weights used for results in the paper are available on Hugging Face.

**From the main results:**

| Model | Hugging Face repo |
| --- | --- |
| CoT SFT | Lapisbird/adaLR-model-cot_sft |
| No-CoT SFT | Lapisbird/adaLR-model-no_cot_sft |
| Latent-6 | Lapisbird/adaLR-model-latent-6 |
| Latent-6 + RL | Lapisbird/adaLR-model-latent-6_rl |
| Latent-6-by-1 | Lapisbird/adaLR-model-latent-6-by-1 |
| Latent-6-by-1 + RL | Lapisbird/adaLR-model-latent-6-by-1_rl |

**From the knowledge distillation for SFT section in Appendix:**

| Model (Appendix) | Hugging Face repo |
| --- | --- |
| codi | Lapisbird/adaLR-appendix-model-codi |
| codi + intermediate | Lapisbird/adaLR-appendix-model-codi_intermediate |
| meaned | Lapisbird/adaLR-appendix-model-meaned |
| meaned + intermediate | Lapisbird/adaLR-appendix-model-meaned_intermediate |
| meaned + codi | Lapisbird/adaLR-appendix-model-meaned_codi |

You can load these models using the function automodelforcausallm_from_pretrained_latent from src.model_creation.

Ex:
```python
from transformers import AutoTokenizer
from src.model_creation import automodelforcausallm_from_pretrained_latent

repo_id = "Lapisbird/adaLR-model-latent-6"

model = automodelforcausallm_from_pretrained_latent(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

### Utilities

notebooks/eval/test_set_eval.ipynb provides model evaluation code.

notebooks/utilities/lora_merge_and_push.ipynb provides code to merge LoRA into a model and push to HF repo.
