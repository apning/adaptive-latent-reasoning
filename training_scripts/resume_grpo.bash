#### Make sure to run this script FROM PROJECT ROOT

### Begin options specifications
# If you do not wish to specify, just leave them blank or as empty quotes


RESUME_PATH=


## Computational
TIME_LIMIT=40
# If more than 1 device, make sure to specify appropriate micro batch size!
NUM_DEVICES=1
MICRO_BATCH_SIZE=64


### No need to edit below here, probably

eval "$(conda shell.bash hook)" # activate conda

# Deactivate until no envs left then reactivate to prevent env stacking
while [ -n "$CONDA_DEFAULT_ENV" ]; do
  conda deactivate
done
conda activate base

conda activate adaptive-latent-thinking

if [ "$NUM_DEVICES" -gt 1 ]; then
  MULTI_GPU="--multi_gpu"
else
  MULTI_GPU=""
fi

accelerate launch $MULTI_GPU --num_processes=$NUM_DEVICES --num_machines=1 --mixed_precision=no train_grpo.py \
  --resume_path="$RESUME_PATH" \
  --time_limit="$TIME_LIMIT" \
  --wrap_up_time \
  --micro_batch_size="$MICRO_BATCH_SIZE"
