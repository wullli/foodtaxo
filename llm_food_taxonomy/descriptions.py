from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import dspy
from dspy import ChainOfThought, backtrack_handler
from llm_food_taxonomy.data.loader import load_taxonomy
from llm_food_taxonomy.models.lm import get_lm
from llm_food_taxonomy.models.signatures import DescribeConcept

TOPICS = {
    "mesh": "medical subject headings",
    "bettybossi": "food",
    "semeval_food": "food",
    "wikitax": "food",
    "semeval_verb": "verbs",
}


class DescriptionModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cot = ChainOfThought(DescribeConcept)

    def forward(self, concept: str, topic: str):
        pred = self.cot(concept=concept, topic=topic)
        dspy.Suggest(pred.description != "", "The description should not be empty.")
        dspy.Suggest(pred.description != concept, "The description should not be the same as the concept.")
        dspy.Suggest(pred.description != "None", "The description should not be 'None'.")
        dspy.Suggest(len(pred.description) < 150, "The description should be shorter than 150 characters.")
        return pred


def main(data: str, model: str = "gpt-4o-mini"):
    turbo = get_lm(model=model, quantize=False)
    dspy.configure(lm=turbo, trace=[])

    terms, taxo = load_taxonomy(f"data/{data}/")
    id_to_name = dict(zip(terms.node_id.values, terms.node_name.values))
    program = DescriptionModule().activate_assertions(handler=backtrack_handler, max_backtracks=3)

    with open(f"data/{data}/{data}.desc", "w") as f:
        for concept_id, term in tqdm(id_to_name.items(), desc="Generating descriptions..."):
            pred = program(concept=term, topic=TOPICS[data])
            f.write(f"{concept_id}\t{pred.description}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--data", choices=["mesh", "bettybossi", "semeval_food", "wikitax", "semeval_verb"],
                        default="bettybossi")
    args = parser.parse_args()

    api_key_path = Path(".") / ".env"
    if Path(api_key_path).exists():
        load_dotenv(dotenv_path=str(api_key_path))

    main(args.data)
