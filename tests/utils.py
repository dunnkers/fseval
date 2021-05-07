from omegaconf import OmegaConf


def get_single_config(config_repo, group, path):
    """Returns a single config, e.g. `ranker/chi2`. Has the structured config
    defined in `fseval/config.py` merged into it."""
    cfg = OmegaConf.create()

    item_config = config_repo.load_config(f"{group}/{path}").config
    cfg[group] = item_config

    structured_config = config_repo.load_config("base_config").config
    structured_config.merge_with(cfg)

    return structured_config[group]
