from huggingface_hub import HfApi
import ssl

#HACK: Ignore SSL errors
ssl._create_default_https_context = ssl._create_unverified_context


def list_models(model_name):
    model_filter = lambda model_info: "mvrl" in model_info.modelId.lower() and model_name.lower() in model_info.modelId.lower()
    models = filter(model_filter, HfApi().list_models())
    print(f"Available {model_name} pretrained models:")
    for model_info in models:
        print(model_info.modelId)