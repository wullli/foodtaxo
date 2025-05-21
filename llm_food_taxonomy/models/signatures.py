import dspy


class DescribeConcept(dspy.Signature):
    """
    Describe the given concept in one sentence in the context of the given topic.
    """
    concept = dspy.InputField(desc="Concept to describe")
    topic = dspy.InputField(desc="Broader topic of the concept")
    description = dspy.OutputField(desc="Description of the concept in the context of the topic")


class DescribeTaxonomy(dspy.Signature):
    """
    Describe the topic of the taxonomy given a sample list of existing concepts from the taxonomy.
    Also think about how the rough structure of the taxonomy could look like.
    """
    concepts = dspy.InputField(desc='Sample list of existing concepts',
                               format=lambda x: ("```" + "\n".join(x) + "```") if isinstance(x, list) else x)
    description = dspy.OutputField(desc='Description and rough structure of the taxonomy')


class GenerateParents(dspy.Signature):
    """
    Which are the most specific parent concepts of the given child concept in a taxonomy considering the context?
    In your reasoning, state how and why the child is a kind of the parent.
    Do not add additional comments or information, only return the output in the described format.
    """
    context = dspy.InputField(desc="List of existing parent-child (supertype-subtype) relations in the taxonomy.",
                              format=lambda x: ("```" + "\n".join(x) + "```") if isinstance(x, list) else x)
    child = dspy.InputField(desc="Child concept (subtype) that you need to place in a taxonomy.")
    description = dspy.InputField(desc="Description of the child concept.", format=lambda x: str(x))
    interpretation = dspy.OutputField(desc="Description of the child concept in relation to the context taxonomy. "
                                           "Infer what is meant by the child concept from the context.")
    parents = dspy.OutputField(
        desc="Comma separated list of one or more parents of the child concept. "
             "Valid parents are in the context. "
             "If there are no suitable parents, return None.")


class GenerateParentsUnsupervisedNoDesc(GenerateParents):
    parents = dspy.OutputField(
        desc="Comma separated list of one or more parents (supertypes) of the child concept. "
             "A parent concept must be a more general type of the child concept. "
             "If there are no suitable existing parents, invent them.")


class GenerateParentsUnsupervised(GenerateParentsUnsupervisedNoDesc):
    taxonomy_description = dspy.InputField(desc='Description of the taxonomy')


class GenerateChildren(dspy.Signature):
    """
    Which of the candidates are child concepts (subtypes) of the given parent concept (supertype) in a taxonomy?
    The context shows existing parents and children concepts and whether the children are leaves.
    In your reasoning, state how the parent concept is a supertype of the selected child concepts.
    Do not add additional comments or information, only return the output in the described format.
    """
    context = dspy.InputField(desc="List of existing parent-child (supertype-subtype) relations in the taxonomy.",
                              format=lambda x: ("```" + "\n".join(x) + "```") if isinstance(x, list) else x)
    candidates = dspy.InputField(desc="Candidate children of the concept separated by commas to select from.")
    parent = dspy.InputField(desc="Parent concept that you need to place in a taxonomy.")
    description = dspy.InputField(desc="Description of the parent concept.", format=lambda x: str(x))
    interpretation = dspy.InputField(desc="Description of the child concept in relation to the taxonomy.")
    leaf = dspy.OutputField(
        desc="Whether the parent concept should be added as a leaf (has no children). Answer with Yes or No.")
    children = dspy.OutputField(
        desc="Comma separated list of candidates that are children of the parent concept in a taxonomy."
             "A child concept must be a type of the parent concept."
             "Separate with commas.")


class GermanGenerateParentsUnsupervised(GenerateParentsUnsupervised):
    __doc__ = GenerateParents.__doc__ + "\nAll returned parents should be in German."


class GermanGenerateChildren(GenerateChildren):
    __doc__ = GenerateChildren.__doc__ + "\nAll returned children should be in German."
