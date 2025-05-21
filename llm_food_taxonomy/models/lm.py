import glob
import os
from pathlib import Path

from llm_food_taxonomy.models.hf_model import FastHFModel
from llm_food_taxonomy.models.openai_model import ConversationGPT3


def get_lm(model: str, quantize: bool = False, local_model_path: str = None):
    """
    Get language model for dspy given the model name
    :param model: str: model name
    :param local_model_path: str: path to local model
    :param quantize: bool: if True, quantize the model
    """
    if "gpt" in model:

        lm = ConversationGPT3(model=model,
                              api_key=os.environ["OPENAI_API_KEY"],
                              top_p=0,
                              temperature=0.0,
                              max_tokens=300)
    else:
        model_path = None
        available_models = glob.glob(local_model_path)
        if model.split("/")[-1] in available_models:
            model_path = Path(local_model_path) / model.split("/")[-1]
        lm = FastHFModel(model=model,
                         checkpoint_path=model_path,
                         do_sample=False,
                         temperature=0.01,
                         max_new_tokens=500,
                         max_len=8192,
                         quantize=quantize)
    return lm
