#### Make sure to run this script FROM PROJECT ROOT

### Begin options specifications
# If you do not wish to specify, just leave them blank or as empty quotes

MODEL_PATH_OR_REPO=
LATENT_SETTINGS_OPTIONS=
LORA_CONFIG_OPTIONS=
TRAINING_CONFIG_OPTIONS=
VAL_CONFIG_OPTIONS=
DIR_NAME=

## Saving
SAVE_DIR_OVERRIDE=
HF_SAVING_USERNAME=
DO_NOT_SAVE=False

## Computational
TIME_LIMIT=26
# If more than 1 device, make sure to specify appropriate micro batch size!
NUM_DEVICES=2
MICRO_BATCH_SIZE=32
MIN_MICRO_BATCH_SIZE=


### No need to edit below here, probably

eval "$(conda shell.bash hook)" # activate conda

# Deactivate until no envs left then reactivate to prevent env stacking
while [ -n "$CONDA_DEFAULT_ENV" ]; do
  conda deactivate
done
conda activate base

conda activate adaptive-latent-thinking


torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_DEVICES train_sft.py \
  --model_path_or_repo="$MODEL_PATH_OR_REPO" \
  --latent_thinking_settings_options="$LATENT_SETTINGS_OPTIONS" \
  --lora_config_options="$LORA_CONFIG_OPTIONS" \
  --training_config_options="$TRAINING_CONFIG_OPTIONS" \
  --val_config_options="$VAL_CONFIG_OPTIONS" \
  --dir_name="$DIR_NAME" \
  --save_dir_override="$SAVE_DIR_OVERRIDE" \
  --hf_saving_username="$HF_SAVING_USERNAME" \
  --do_not_save="$DO_NOT_SAVE" \
  --time_limit="$TIME_LIMIT" \
  --wrap_up_time \
  --micro_batch_size="$MICRO_BATCH_SIZE" \
  --min_micro_batch_size="$MIN_MICRO_BATCH_SIZE"