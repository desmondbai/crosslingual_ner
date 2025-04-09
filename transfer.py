
import spacy
import numpy as np
import torch
from copy import deepcopy
import random
from spacy.training import Example
from spacy.util import minibatch

def init_model(spacy_train_data, language):
    """
    Initialize a spaCy model with pretrained embeddings.

    Args:
        spacy_train_data: Training data in spaCy format.
        language: Language code (e.g., "da").

    Returns:
        A tuple (model, optimizer)
    """
    model = spacy.blank(language)
    np.random.seed(0)
    random.seed(0)
    spacy.util.fix_random_seed(0)
    torch.manual_seed(0)

    model.vocab.from_disk("data/vocab")

    ner = model.add_pipe("ner", config={"model": {"tok2vec": {"pretrained_vectors": True}}})

    for _, annotations in spacy_train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]
    model.disable_pipes(*other_pipes)

    optimizer = model.begin_training()
    return model, optimizer


def retrain(spacy_train_data, spacy_dev_data, epochs, model):
    """
    Fine-tune an existing model on new data.

    Args:
        spacy_train_data: Training data.
        spacy_dev_data: Validation data.
        epochs: Number of epochs.
        model: Pre-initialized spaCy model.

    Returns:
        Fine-tuned model.
    """
    spacy_train_data = deepcopy(spacy_train_data)
    model = deepcopy(model)

    for itn in range(epochs):
        losses = {}
        random.shuffle(spacy_train_data)
        batches = minibatch(spacy_train_data, size=5)

        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(model.make_doc(texts[i]), annotations[i]) for i in range(len(texts))]
            model.update(examples, drop=0.1, losses=losses)

        print("Loss for epoch %u: %.4f" % (itn+1, losses.get("ner", 0.0)))
        spacy_dev_sys = annotate(spacy_dev_data, model)
        p, r, f = evaluate(spacy_dev_sys, spacy_dev_data)
        print("  PRECISION: %.2f%%, RECALL: %.2f%%, F-SCORE: %.2f%%" % (p, r, f))
    return model


def annotate(data, model):
    """
    Annotate text using a trained spaCy model.

    Args:
        data: List of (text, {"entities": [...]}) format.
        model: Trained spaCy model.

    Returns:
        List of annotated (text, {"entities": [...]}) outputs.
    """
    output = []
    for text, _ in data:
        doc = model(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        output.append((text, {"entities": entities}))
    return output

from evaluation import evaluate
