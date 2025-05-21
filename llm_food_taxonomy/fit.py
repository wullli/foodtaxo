import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pprint

import fasttext
import pandas as pd
from dotenv import load_dotenv

from llm_food_taxonomy.data.loader import load_subgraph_taxonomy
from llm_food_taxonomy.models.completion_model import LlmTaxonomyCompletionModel, FitMethod
from llm_food_taxonomy.models.retrieve import GteEmbedding

logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--taxonomy_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_k", type=int, default=10)
    parser.add_argument("--infer_k", type=int, default=100)
    parser.add_argument("--terms_k", type=int, default=10)
    parser.add_argument("--quantize", type=int, default=0)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--max_backtracks", type=int, default=3)
    parser.add_argument("--unsupervised", type=bool, default=False)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--method", choices=["none", "zero", "few", "few_basic"],
                        default="none")
    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--reasoning_model", type=bool, default=False)

    args = parser.parse_args()

    api_key_path = Path(".") / ".env"
    if Path(api_key_path).exists():
        load_dotenv(dotenv_path=str(api_key_path))

    data_name = args.taxonomy_dir.split("/")[-1]
    val_terms, val_taxo = load_subgraph_taxonomy(args.taxonomy_dir, mode="val")
    val_id_to_name = {d["node_id"]: d["node_name"] for d in val_terms.to_dict(orient="records")}
    val_id_to_desc = {d["node_id"]: d["desc"] for d in val_terms.to_dict(orient="records")}
    embedding_fn = GteEmbedding()
    optimizer_options = {}
    fit_method = FitMethod.NONE
    if args.method == "zero":
        optimizer_options = dict(depth=2, breadth=5)
        fit_method = FitMethod.ZERO_SHOT
    elif args.method == "few":
        optimizer_options = dict(num_candidates=10,
                                 num_trials=10,
                                 max_bootstrapped_demos=4,
                                 view_data_batch_size=5,
                                 max_labeled_demos=4)
        fit_method = FitMethod.FEW_SHOT
    elif args.method == "few_basic":
        fit_method = FitMethod.FEW_SHOT_BASIC
        optimizer_options = dict(max_bootstrapped_demos=4,
                                 max_labeled_demos=4,
                                 max_rounds=1)

    model = LlmTaxonomyCompletionModel(model=args.model,
                                       n_train_samples=20,
                                       n_val_samples=20,
                                       k_relations=args.train_k,
                                       embedding_fn=embedding_fn,
                                       balanced=False,
                                       optimizer_options=optimizer_options,
                                       method=fit_method,
                                       unsupervised=args.unsupervised,
                                       recursive=False,
                                       quantize=args.quantize,
                                       max_backtracks=args.max_backtracks,
                                       reasoning_model=args.reasoning_model)

    validation_nodes = val_terms[val_terms.split == "val"]
    train_nodes = val_terms[val_terms.split == "train"]

    if args.trial:
        validation_nodes = validation_nodes.sample(10, random_state=123)

    train_taxo = val_taxo[
        val_taxo.apply(lambda r: (str(r.hypernym) in train_nodes.node_id.values)
                                 and (str(r.hyponym) in train_nodes.node_id.values),
                       axis=1)
    ]

    simple_name = args.model if not "/" in args.model else args.model.split("/")[-1]
    simple_name = f"unsupervised_{simple_name}" if args.unsupervised else simple_name
    model_dir = Path(f"./data/models/dspy/{data_name}/{args.mode}")
    model_name = f"{args.method}_{simple_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.method != "none" and args.model_file is None:
        print(f"--- Fitting Prompt in {args.method} shot mode ---")
        pprint(model.config)
        scores = model.fit(train_terms=train_nodes.copy(),
                           train_taxo=train_taxo.copy(),
                           val_terms=validation_nodes.copy(),
                           id_to_name=val_id_to_name,
                           id_to_desc=val_id_to_desc)

        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / f"{model_name}.json"))

        scores_df = pd.DataFrame({v: [k] for k, v in scores.items()})
        print(scores_df)
    elif args.model_file is not None:
        model.load(args.model_file)

    model.program.retrieve.k = args.infer_k

    print(f"--- Completing taxonomy with {args.method} shot mode ---")
    test_terms, test_taxo = load_subgraph_taxonomy(args.taxonomy_dir, mode=args.mode)

    test_nodes = test_terms[test_terms.split == args.mode]
    train_nodes = test_terms[test_terms.split == "train"]

    if args.trial:
        test_nodes = test_nodes.sample(10, random_state=123)

    train_taxo = test_taxo[
        test_taxo.apply(lambda r: (str(r.hypernym) in train_nodes.node_id.values)
                                  and (str(r.hyponym) in train_nodes.node_id.values),
                        axis=1)
    ]

    test_id_to_name = {d["node_id"]: d["node_name"] for d in val_terms.to_dict(orient="records")}
    pprint(model.config)
    positions, outputs = model.complete(train_terms=train_nodes.copy(),
                                        train_taxo=train_taxo.copy(),
                                        test_terms=test_nodes.copy(),
                                        id_to_name=test_id_to_name,
                                        id_to_desc=val_id_to_desc,
                                        return_output=True)

    dir_name = f"{data_name}_{args.name}" if args.name != "" else data_name
    output_dir = Path("./output/") / dir_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.trial:
        model.llm.inspect_history(n=100)
    else:
        with open(output_dir / "predictions.pos.json", "w") as f:
            json.dump(positions, f)
        with open(output_dir / "predictions.outputs.json", "w") as f:
            json.dump(outputs, f)
        with open(output_dir / "concepts.terms", "w") as f:
            for id, name in test_id_to_name.items():
                f.write(f"{id}\t{name}\n")
        with open(output_dir / "model_config.json", "w") as f:
            json.dump(model.config, f)


if __name__ == "__main__":
    main()
