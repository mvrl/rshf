from huggingface_hub import get_collection, hf_hub_download
import json
import ssl

#HACK: Ignore SSL errors
ssl._create_default_https_context = ssl._create_unverified_context


def list_models(model_name):
    collection = get_collection("MVRL/remote-sensing-foundation-models-664e8fcd67d8ca8c03f42d00")
    models = filter(lambda item: model_name.lower() in item.item_id.lower(), collection.items)
    print(f"Available {model_name} pretrained models:\n")
    for model_info in models:
        print(model_info.item_id)

def help(model):
    print(model.__doc__)

def from_config(model_class, repo_id, revision=None, **kwargs):
    """Load a model with randomly initialized weights using the architecture
    configuration stored in a HuggingFace Hub repository.

    This is useful for training a model from scratch while still using the
    same architecture as a known pretrained checkpoint.

    Args:
        model_class: The model class to instantiate (e.g. ``SatMAE``).
        repo_id (str): HuggingFace Hub repository ID
            (e.g. ``"MVRL/satmae-vitlarge-fmow-pretrain-800"``).
        revision (str, optional): Branch, tag, or commit hash to use.
            Defaults to the latest revision.
        **kwargs: Additional keyword arguments that override values read from
            the repository's ``config.json``. These must be valid parameters
            for ``model_class.__init__``; unknown parameters will raise an
            error when the model is instantiated.

    Returns:
        An instance of ``model_class`` with randomly initialized weights.

    Raises:
        huggingface_hub.utils.EntryNotFoundError: If ``config.json`` is not
            found in the repository.
        huggingface_hub.utils.RepositoryNotFoundError: If ``repo_id`` does
            not exist or is not accessible.

    Example:
        >>> from rshf import from_config
        >>> from rshf.satmae import SatMAE
        >>> model = from_config(SatMAE, "MVRL/satmae-vitlarge-fmow-pretrain-800")
    """
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    except Exception as e:
        raise type(e)(
            f"Could not download config.json from '{repo_id}'. "
            f"Ensure the repository exists and contains a config.json file. "
            f"Original error: {e}"
        ) from e
    with open(config_path) as f:
        config = json.load(f)
    # Remove internal HuggingFace Hub metadata keys (prefixed with "_")
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    config.update(kwargs)
    return model_class(**config)