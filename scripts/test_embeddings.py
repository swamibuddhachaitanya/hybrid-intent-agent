import json

from src.preprocessing.embedding_preprocessor import EmbeddingPreprocessor


def main():
    with open("data/processed/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    sample_texts = [item["text"] for item in train_data[:5]]

    print("Encoding 5 sample texts:")
    for t in sample_texts:
        print(f"   - {t}")

    preprocessor = EmbeddingPreprocessor()
    vectors = preprocessor.transform(sample_texts)

    print(f"\n output shape: {vectors.shape}")
    print(f"\n vector dimension: {vectors.shape[1]}")
    print(f"\n First vector slice( first 5 values ): {vectors[0][:5].round(4)}")


if __name__ == "__main__":
    main()
