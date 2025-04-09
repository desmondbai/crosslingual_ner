
def evaluate(system_spacy_data, gold_spacy_data):
    """
    Evaluates NER system predictions.

    Args:
        system_spacy_data: Predicted data in spaCy format.
        gold_spacy_data: Ground truth data in spaCy format.

    Returns:
        Tuple: (precision, recall, f1_score)
    """
    n_sys = 0
    n_gold = 0
    n_correct = 0
    for sys_sent, gold_sent in zip(system_spacy_data, gold_spacy_data):
        sys_entities = set(sys_sent[1]["entities"])
        gold_entities = set(gold_sent[1]["entities"])
        n_sys += len(sys_entities)
        n_gold += len(gold_entities)
        n_correct += len(sys_entities.intersection(gold_entities))

    precision = n_correct / n_sys * 100 if n_sys else 0
    recall = n_correct / n_gold * 100 if n_gold else 0
    fscore = 0 if n_correct == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, fscore
