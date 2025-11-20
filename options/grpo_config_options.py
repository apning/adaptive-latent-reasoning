GRPO_CONFIG_OPTIONS = {
    "standard-1a": {
        ## Training
        "do_train": True,
        "warmup_steps": 1000,
        ## Computational
        "gradient_checkpointing": False,
        "bf16": True,
        "max_completion_length": 40,
        ## DDP
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
        ## Logging
        "logging_first_step": True,
        "report_to": "tensorboard",
        "disable_tqdm": True,
        ## Eval
        "metric_for_best_model": "custom_eval/score",
        "greater_is_better": True,
        "do_eval": True,
        "eval_on_start": True,
        "load_best_model_at_end": True,
        ## Saving
        "save_total_limit": 3,
    },
    ## Model
    "temp_0p1": {"temperature": 0.1},
    "temp_0p5": {"temperature": 0.5},
    ## Training
    "lr_8e-4": {"learning_rate": 8e-4},
    "lr_1e-4": {"learning_rate": 1e-4},
    "lr_3e-5": {"learning_rate": 3e-5},
    "lr_1e-5": {"learning_rate": 1e-5},
    "lr_1e-6": {"learning_rate": 1e-6},
    "weight_decay_0p1": {"weight_decay": 0.1},
    "weight_decay_1e-3": {"weight_decay": 1e-3},
    "weight_decay_1e-5": {"weight_decay": 1e-5},
    "beta_0p04": {"beta": 0.04},
    "beta_1e-3": {"beta": 1e-3},
    "cosine_lr": {"lr_scheduler_type": "cosine"},  # default is linear
    "constant_lr": {"lr_scheduler_type": "constant"},
    "steps_15k": {"max_steps": 15000},
    "steps_30k": {"max_steps": 30000},
    "steps_60k": {"max_steps": 60000},
    "num_gen_16": {"num_generations": 16},
    ## Evaluation
    "eval_steps_20": {"eval_strategy": "steps", "eval_steps": 20},
    "eval_steps_200": {"eval_strategy": "steps", "eval_steps": 200},
    "eval_steps_500": {"eval_strategy": "steps", "eval_steps": 500},
    ## Saving
    "save_steps_40": {"save_steps": 40},
    "save_steps_200": {"save_steps": 200},  # NEEDS to be a multiple of eval_steps
    "save_steps_500": {"save_steps": 500},  # NEEDS to be a multiple of eval_steps
    "save_steps_1k": {"save_steps": 1000},  # NEEDS to be a multiple of eval_steps
    "no_save_limit": {"save_total_limit": None},
}
"""

Set later by training script:
    * output_dir
    * reward_weights
    * per_device_train_batch_size
    * per_device_eval_batch_size
    * gradient_accumulation_steps

"""
