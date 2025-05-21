from llm_food_taxonomy.evaluation.completion.parent_metric import ParentMetric
from llm_food_taxonomy.evaluation.completion.wupalmer_metric import WuPSimilarity
from llm_food_taxonomy.evaluation.completion.position_metric import PositionMetric
from llm_food_taxonomy.evaluation.metric import Metric

METRIC_REGISTRY = {
    "wup": WuPSimilarity,
    "position": PositionMetric,
    "parent": ParentMetric,
}
