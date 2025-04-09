
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from random import shuffle, seed
from copy import deepcopy
import random

def train_spacy_model(train_data, dev_data, iterations=20):
    """
    Train a spaCy NER model on given training data.

    Args:
        train_data: Training data in spaCy format.
        dev_data: Development data in spaCy format.
        iterations: Number of training iterations.

    Returns:
        Trained spaCy model (nlp object).
    """
    nlp = spacy.blank("da")  # Danish language model
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()
    seed(42)
    random.seed(42)

    for itn in range(iterations):
        print(f"Starting iteration {itn}")
        losses = {}
        random.shuffle(train_data)
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Losses at iteration {itn}: {losses}")

    return nlp
