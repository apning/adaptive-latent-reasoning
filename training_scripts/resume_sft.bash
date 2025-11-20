#### Make sure to run this script FROM PROJECT ROOT

### Begin options specifications
# If you do not wish to specify, just leave them blank or as empty quotes


RESUME_PATH=

## Saving
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
  --resume_path="$RESUME_PATH" \
  --do_not_save="$DO_NOT_SAVE" \
  --time_limit="$TIME_LIMIT" \
  --wrap_up_time \
  --micro_batch_size="$MICRO_BATCH_SIZE" \
  --min_micro_batch_size="$MIN_MICRO_BATCH_SIZE"