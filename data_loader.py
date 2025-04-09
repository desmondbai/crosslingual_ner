
import io

def read_data(file):
    """
    Reads NER dataset in CoNLL format.

    Args:
        file: File object or string buffer.

    Returns:
        A list of sentences, where each sentence is a list of (token, tag) tuples.
    """
    data = []
    sentence = []
    for line in file:
        if line.strip() == "":
            data.append(sentence)
            sentence = []
        else:
            sentence.append(tuple(line.strip().split("\t")))
    if sentence:
        data.append(sentence)
    return data


def get_spacy_ner_data(data):
    """
    Converts token-tag pairs into spaCy NER training format.

    Args:
        data: List of sentences, each as a list of (token, tag) pairs.

    Returns:
        List of tuples: (text, {"entities": [(start, end, label), ...]})
    """
    spacy_ner_data = []
    for sent in data:
        ind = 0
        entities = []
        full_sentence = " ".join([pair[0] for pair in sent])
        for token, tag in sent:
            iob_tag = tag[0]
            if iob_tag == "B":
                entity = (ind, ind + len(token), tag[2:])
                entities.append(entity)
            elif iob_tag == "I" and entities:
                start, end, label = entities.pop()
                entity = (start, end + len(token) + 1, label)
                entities.append(entity)
            ind += (len(token) + 1)
        spacy_ner_data.append((full_sentence, {"entities": entities}))
    return spacy_ner_data
