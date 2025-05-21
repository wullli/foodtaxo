import glob
import itertools
import json
import os
import shutil
from collections import deque
from functools import partial
from pathlib import Path
from re import sub

import numpy as np
import pandas as pd
from chromadb import Settings
from networkx import descendants
from tqdm.auto import tqdm

from llm_food_taxonomy.graph.taxonomy import Taxonomy
import chromadb
import networkx as nx

base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tqdm.pandas()


def load_json(json_path: str) -> dict:
    """
    Load a json file
    :param json_path: path to the json file
    :return: the taxonomy as a dictionary
    """
    with open(json_path) as f:
        taxonomy = json.load(f)
    return taxonomy





def snake_case(s):
    """
    Replace hyphens with spaces, then apply regular expression substitutions for title case conversion
    and add an underscore between words, finally convert the result to lowercase
    :param s: The string to convert
    :return:
    """
    return '_'.join(
        sub('([A-Z][a-z]+)', r' \1',
            sub('([A-Z]+)', r' \1',
                s.replace('-', ' '))).split()).lower()


def dspy_to_chat(prompt: str, assistant_start: str = "Reasoning:"):
    if "---" in prompt:
        parts = prompt.split("---")
        system = "---".join(parts[:3])
        chat = [{"role": "system", "content": system.strip()}]
        for i, msg in enumerate(parts[3:]):
            if assistant_start in msg:
                split_msg = msg.split(assistant_start)
                user = assistant_start.join(split_msg[:-1])
                assistant = split_msg[-1]
                chat.append({"role": "user", "content": f"{user.strip()}".strip()})
                chat.append({"role": "assistant", "content": f"{assistant_start} {assistant.strip()}".strip()})
            else:
                raise ValueError(f"Unrecognized message type: {msg}")
        return chat
    else:
        return [{"role": "user", "content": prompt}]
