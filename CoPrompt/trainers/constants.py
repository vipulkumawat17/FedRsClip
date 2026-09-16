def get_dataset_specified_config(dataset):
    """Get dataset specific."""
    cfg = {
        "ImageNet": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "Caltech101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 4.0,
        },
        "OxfordPets": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 0.1,
        },
        "StanfordCars": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 4.0,
        },
        "OxfordFlowers": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "Food101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 0.5,
            "OPTIM.MAX_EPOCH": 5,
        },
        "FGVCAircraft": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 2.0,
        },
        "SUN397": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "DescribableTextures": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 8.0,
        },
        "EuroSAT": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 0.1,
        },
        "UCF101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 2.0,
        },
        # --- custom datasets (JSONL-prepared) - mirroring the closest ---
        # --- analogous original-benchmark dataset's hyperparameters ---
        "PreparedCaltech101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 4.0,
        },
        "PreparedFood101": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 0.5,
            "OPTIM.MAX_EPOCH": 5,
        },
        "PreparedEuroSAT": {
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 2,
            "TRAINER.W": 0.1,
        },
        "PreparedFlowers102": {
            # mirrors OxfordFlowers - closest analogous flower-classification dataset
            "TRAINER.CoPrompt.PROMPT_DEPTH": 9,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
        "PreparedCifar100": {
            # no direct analog in the original 11 - uses the paper's ImageNet-style
            # generic-object-classification settings as a reasonable default.
            "TRAINER.CoPrompt.PROMPT_DEPTH": 12,
            "TRAINER.CoPrompt.N_CTX": 4,
            "TRAINER.W": 8.0,
        },
    }.get(dataset, {})

    items = [f"{k} {v}" for k, v in cfg.items()]
    return " ".join(items).split(" ") if items else []
