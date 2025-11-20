LORA_CONFIG_OPTIONS = {
    "default": {},
    ## Model-specific target modules
    "llama_target_modules": {
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    },
    "gpt2_target_modules": {"target_modules": ["c_attn", "c_proj", "c_fc"]},
    ## Modules to save
    "latent_modules_to_save": {
        "modules_to_save": [
            "recurrent_filter",
            "_start_embed",
            "_end_embed",
            "_continue_embed",
            "_stop_embed",
            "_start_embed_in",
            "_end_embed_in",
            "_start_embed_out",
            "_continue_embed_out",
            "_stop_embed_out",
        ]
    },
    ## General LoRA arguments
    "r_8": {"r": 8},
    "r_32": {"r": 32},
    "r_128": {"r": 128},
    "alpha_8": {"lora_alpha": 8},
    "alpha_32": {"lora_alpha": 32},
    "alpha_128": {"lora_alpha": 128},
    "alpha_256": {"lora_alpha": 256},
    "dropout_0p1": {"lora_dropout": 0.1},
}
