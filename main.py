
from data_loader import read_data, get_spacy_ner_data
from train import train_spacy_model
from evaluation import evaluate
from transfer import init_model, retrain, annotate

def main():
    # === Load Data ===
    with open("data/danish-train.conll", encoding="utf-8") as f:
        danish_train = read_data(f)
    with open("data/danish-dev.conll", encoding="utf-8") as f:
        danish_dev = read_data(f)
    with open("data/danish-test.conll", encoding="utf-8") as f:
        danish_test = read_data(f)

    # === Convert to spaCy Format ===
    danish_train_spacy = get_spacy_ner_data(danish_train)
    danish_dev_spacy = get_spacy_ner_data(danish_dev)
    danish_test_spacy = get_spacy_ner_data(danish_test)

    # === Baseline Model Training ===
    print("\n=== Training Baseline Danish Model ===")
    danish_model = train_spacy_model(danish_train_spacy, danish_dev_spacy, iterations=20)
    danish_test_sys = annotate(danish_test_spacy, danish_model)
    p, r, f = evaluate(danish_test_sys, danish_test_spacy)
    print("Baseline Model - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%" % (p, r, f))

    # === Transfer Model Training ===
    print("\n=== Transfer Learning with Pretrained Embeddings ===")
    english_model, _ = init_model(danish_train_spacy, "da")
    transfer_model = retrain(danish_train_spacy, danish_dev_spacy, epochs=20, model=english_model)
    transfer_test_sys = annotate(danish_test_spacy, transfer_model)
    p, r, f = evaluate(transfer_test_sys, danish_test_spacy)
    print("Transfer Model - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%" % (p, r, f))

if __name__ == "__main__":
    main()
