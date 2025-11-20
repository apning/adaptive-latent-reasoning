Use the following options with sft.bash or grpo.bash to replicate results of the paper.

As noted in the paper, latent training is subject to random divergence during training. If this happens, you will need to find the latest saved checkpoint at which divergence had not yet occurred, delete all subsequently saved checkpoints, and resume training from there.

Meta's Llama 3.2 1B Instruct is used as the base model for many of the training options below. Access requires agreeing to Meta's terms and then authenticating via a HF token in the terminal (or whatever other method you prefer). You can agree to the terms here:
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Main results SFT:
For use with sft.bash. Latent models take a little less than ~26 hours to train on 2 x A100-80gb with micro batch size 32. CoT and No-CoT SFT models take even less time and can handle a larger micro batch size.


CoT SFT:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS=
LORA_CONFIG_OPTIONS="llama_target_modules/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/slice_proportion_85p/batch_size_128/steps_30k/lr_1e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/no_decay_some_params/val_period_200/save_checkpoint_period_4k/gc_if_necessary"
VAL_CONFIG_OPTIONS="gsm8k-aug-val"
DIR_NAME="cot_sft"
```

No-CoT SFT:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS=
LORA_CONFIG_OPTIONS="llama_target_modules/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/slice_proportion_85p/no_cot/batch_size_128/steps_30k/lr_1e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/no_decay_some_params/val_period_200/save_checkpoint_period_4k/gc_if_necessary"
VAL_CONFIG_OPTIONS="gsm8k-aug-val"
DIR_NAME="no_cot_sft"
```

Latent-6:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/slice_proportion_85p/remove_last_reasoning_step/remove_negative_answers/latent_flat_6/batch_size_128/steps_30k/codi_loss_20/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking"
DIR_NAME="latent-6"
```

Latent-6-by-1:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/slice_proportion_85p/remove_last_reasoning_step/remove_negative_answers/latent_plus_5_by_1_max_10/batch_size_128/steps_30k/codi_loss_20/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking"
DIR_NAME="latent-6-by-1"
```

### Main results RL
For use with grpo.bash. Models take approximately ~60 hours to train on 1 x A100-80gb with micro batch size 64.

The options below use the SFT model weights from the paper. If you would like to use weights from your own SFT, first merge your LoRA weights into the model and save it into a path or push it to a repo along with its tokenizer. The notebook in notebooks/utilities/lora_merge_and_push.ipynb can help with merging and pushing to HF. Then you will need to add an option for your model in options/grpo_training_options.py and replace it in TRAINING_CONFIG_OPTIONS below. (Sorry for the added complexity.)

Latent-6 + RL:
```bash
TRAINING_CONFIG_OPTIONS="model-latent-6/slice_proportion_15p/batch_size_128/rel_len_penalty_1e-4/rel_len_acc_req_ALL/rel_len_reward_1e-1"
GRPO_CONFIG_OPTIONS="standard-1a/cosine_lr/lr_1e-5/weight_decay_1e-3/beta_1e-3/steps_60k/eval_steps_200/save_steps_200/no_save_limit/num_gen_16"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_32/alpha_32/dropout_0p1"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking"
DIR_NAME="latent-6+rl"
```

Latent-6-by-1 + RL:
```bash
TRAINING_CONFIG_OPTIONS="model-latent-6-by-1/slice_proportion_15p/batch_size_128/rel_len_penalty_1e-4/rel_len_acc_req_ALL/rel_len_reward_1e-1"
GRPO_CONFIG_OPTIONS="standard-1a/cosine_lr/lr_1e-5/weight_decay_1e-3/beta_1e-3/steps_60k/eval_steps_200/save_steps_200/no_save_limit/num_gen_16"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_32/alpha_32/dropout_0p1"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking"
DIR_NAME="latent-6-by-1+rl"
```


### Appendix Knowledge Distillation for SFT Results
For use with sft.bash. Models take a little less than ~26 hours to train on 2 x A100-80gb with micro batch size 32.

codi:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP/detach_binary_head_inputs"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/remove_last_reasoning_step/remove_negative_answers/latent_flat_6/batch_size_128/steps_30k/codi_loss_20/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/mask_latent_reasoning_labels/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking/latent_thought_count_override_6"
DIR_NAME="appendix-codi"
```

codi + intermediate:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP/detach_binary_head_inputs"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/remove_last_reasoning_step/remove_negative_answers/latent_flat_6/batch_size_128/steps_30k/codi_loss_20/intermediate_block_llama_3_12/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/mask_latent_reasoning_labels/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking/latent_thought_count_override_6"
DIR_NAME="appendix-codi+intermediate"
```

meaned:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP/detach_binary_head_inputs"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/latent_flat_6/batch_size_128/steps_30k/mean_reasoning_loss_20/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/mask_latent_reasoning_labels/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking/latent_thought_count_override_6"
DIR_NAME="appendix-meaned"
```

meaned + intermediate:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP/detach_binary_head_inputs"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/latent_flat_6/batch_size_128/steps_30k/mean_reasoning_loss_20/intermediate_block_llama_3_12/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/mask_latent_reasoning_labels/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking/latent_thought_count_override_6"
DIR_NAME="appendix-meaned+intermediate"
```

meaned + codi:
```bash
MODEL_PATH_OR_REPO="meta-llama/Llama-3.2-1B-Instruct"
LATENT_SETTINGS_OPTIONS="recurrent_filter_MLP/detach_binary_head_inputs"
LORA_CONFIG_OPTIONS="llama_target_modules/latent_modules_to_save/r_128/alpha_32/dropout_0p1"
TRAINING_CONFIG_OPTIONS="gsm8k-aug/remove_last_reasoning_step/remove_negative_answers/latent_flat_6/batch_size_128/steps_30k/codi_loss_20/mean_reasoning_loss_20/calc_smooth_l1/norm_layerwise_std/lr_8e-4/cosine-warmup_900/max_grad_norm_2/weight_decay_0p1/val_period_200/save_checkpoint_period_4k/mask_latent_reasoning_labels/no_decay_some_params/shift_target_token_one"
VAL_CONFIG_OPTIONS="gsm8k-aug-val/latent_thinking/latent_thought_count_override_6"
DIR_NAME="appendix-meaned+codi"
```
