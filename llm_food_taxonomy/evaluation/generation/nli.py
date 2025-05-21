from functools import partial
from typing import Literal

import dspy
from dspy import ChainOfThought


class Nli(dspy.Signature):
    """
    Does the premise entail the hypothesis? Only two possible answers: entails, contradicts
    Do not return any other answer.
    """
    premise = dspy.InputField(desc="Concept to describe", format=lambda x: f"'''{x}'''")
    hypothesis = dspy.InputField(desc="Broader topic of the concept", format=lambda x: f"'''{x}'''")
    entailment: Literal["entails", "contradicts"] = dspy.OutputField(desc="Answer to the entailment question, "
                                                                          "must be either entails or contradicts")


class NliModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cot = ChainOfThought(Nli)

    def forward(self, premise: str, hypothesis: str):
        pred = self.cot(premise=premise, hypothesis=hypothesis)
        return pred