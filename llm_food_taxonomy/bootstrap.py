import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from llm_food_taxonomy.data.loader import load_taxonomy
from llm_food_taxonomy.graph.taxonomy import Taxonomy
from llm_food_taxonomy.models.completion_model import LlmTaxonomyCompletionModel
from llm_food_taxonomy.models.retrieve import GteEmbedding

logging.basicConfig(level=logging.INFO, force=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_k", type=int, default=10)
    parser.add_argument("--infer_k", type=int, default=100)
    parser.add_argument("--terms_k", type=int, default=10)
    parser.add_argument("--quantize", type=int, default=0)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--max_backtracks", type=int, default=3)
    parser.add_argument("--allow_generation", type=bool, default=True)
    parser.add_argument("--min_ingredient_occurence", type=int, default=5)
    parser.add_argument("--generate_taxonomy_desc", type=bool, default=True)
    parser.add_argument("--subsets", type=int, default=3,
                        help="Bootstrap subsets of nodes to show investigate variability")
    parser.add_argument("--data", choices=["food", "mesh", "bettybossi", "semeval_food"],
                        default="food")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_concepts", type=int, default=None)
    parser.add_argument("--reasoning_model", type=bool, default=False)

    args = parser.parse_args()

    api_key_path = Path(".") / ".env"
    if Path(api_key_path).exists():
        load_dotenv(dotenv_path=str(api_key_path))

    if args.data == "food":
        ingredients = pd.read_csv("./data/recipes/recipe1m_plus_ingredients_cleaned.csv", index_col=None)
        ingredients = ingredients[["recipe_ingredient_names", "ing_count"]]
        ingredients = ingredients[ingredients.ing_count > args.min_ingredient_occurence]
        ingredients = ingredients.sort_values(by="ing_count", ascending=False)
        if args.num_concepts is not None:
            ingredients = ingredients.sample(args.num_concepts, random_state=args.seed)

        id_to_name = {nid: nn for nid, nn in enumerate(ingredients.recipe_ingredient_names)}
        id_to_desc = id_to_name
        ids = list(range(len(ingredients)))
        test_terms = pd.DataFrame({"node_id": ids,
                                   "node_name": ingredients.recipe_ingredient_names,
                                   "desc": [id_to_desc[id] for id in ids]})
    else:
        terms, taxo = load_taxonomy(f"data/{args.data}/")
        id_to_name = dict(zip(terms.node_id.values, terms.node_name.values))
        id_to_desc = dict(zip(terms.node_id.values, terms.desc.values))
        tax = Taxonomy(taxo[["hypernym", "hyponym"]].values.tolist(), id_to_name=id_to_name)
        leaves = [id_to_name[i] for i in tax.leaves()]
        if args.num_concepts is not None:
            leaves = np.random.RandomState(seed=args.seed).choice(leaves, args.num_concepts, replace=False)
        test_terms = pd.DataFrame({"node_id": list(range(len(leaves))),
                                   "node_name": leaves,
                                   "desc": [id_to_desc[id] for id in tax.leaves()]})
        id_to_name = {nid: nn for nid, nn in zip(test_terms.node_id.values, test_terms.node_name.values)}
        id_to_desc = {nid: nn for nid, nn in zip(test_terms.node_id.values, test_terms.desc.values)}

    embedding_fn = GteEmbedding()
    model = LlmTaxonomyCompletionModel(model=args.model,
                                       n_train_samples=30,
                                       n_val_samples=30,
                                       k_relations=args.train_k,
                                       embedding_fn=embedding_fn,
                                       balanced=False,
                                       unsupervised=args.allow_generation,
                                       unsupervised_children=False,
                                       recursive=True,
                                       german=args.data == "bettybossi",
                                       generate_taxonomy_desc=args.generate_taxonomy_desc,
                                       max_backtracks=args.max_backtracks,
                                       reasoning_model=args.reasoning_model)

    if args.trial:
        test_terms = test_terms.iloc[:10]

    model.program.retrieve.k = args.infer_k

    simple_name = args.model if not "/" in args.model else args.model.split("/")[-1]
    output_dir = Path("./output/") / f"unsupervised_{args.name}" / simple_name
    stream_dir = output_dir / "stream"
    if stream_dir.exists():
        shutil.rmtree(str(stream_dir), ignore_errors=False, onerror=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)

    positions = model.complete(test_terms=test_terms.copy(),
                               id_to_name=id_to_name,
                               id_to_desc=id_to_desc,
                               stream_dir=stream_dir,
                               return_output=False)

    if args.trial:
        model.llm.inspect_history(n=100)
    else:
        with open(output_dir / "predictions.pos.json", "w") as f:
            json.dump(positions, f)
        with open(output_dir / "concepts.terms", "w") as f:
            for id, name in id_to_name.items():
                f.write(f"{id}\t{name}\n")
        with open(output_dir / "model_config.json", "w") as f:
            json.dump(model.config, f)


if __name__ == "__main__":
    main()
