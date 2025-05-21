from pathlib import Path
from typing import List

import fasttext
from fasttext.util import download_model
from langchain.embeddings.base import Embeddings

file_dir = Path(__file__).parent
data_path = file_dir.parent.parent / "data"


class FastTextEmbeddings(Embeddings):

    def __init__(self, model_path: str = data_path / "fasttext" / "cc.en.300.bin"):
        if model_path is None or not Path(model_path).exists():
            print(str(Path(model_path).absolute()))
            download_model('en', if_exists='ignore', )  # English
            fasttext.load_model('cc.en.300.bin')
            self.emb_model = fasttext.load_model('cc.en.300.bin')
        else:
            self.emb_model = fasttext.load_model(str(model_path))

    def embed_query(self, text: str) -> List[float]:
        return self.emb_model.get_sentence_vector(text).astype(float).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
