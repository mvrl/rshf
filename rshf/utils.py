from huggingface_hub import get_collection
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