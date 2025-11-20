For sft, run sft.bash. For GRPO, run grpo.bash. **ALWAYS RUN FROM THE PROJECT ROOT, not from this directory.**

To resume, provide the path of a single trial in either resume_sft.bash or resume_grpo.bash like so:

```bash
RESUME_PATH="checkpoints/latent-flat-6/20251108-141935"
```

'TIME_LIMIT' specifies the time limit of training in hours. 'MICRO_BATCH_SIZE' is the size of the initial micro batch size. If only a single device is used, then adaptive micro-batching will halve micro batch size upon OOM. 'MIN_MICRO_BATCH_SIZE' denotes the minimum size to which micro batch size will return if it temporarily falls below this value. This is useful if, for example, only a few samples of the training dataset necessitate a smaller micro batch size, as it prevents micro batch size from permanently falling below a minimum value.

Refer to OPTIONS.md for the options to replicate paper results.