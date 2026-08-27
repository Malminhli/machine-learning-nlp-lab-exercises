# Task 1 & 2: Environment Setup & Sentiment Analysis using BERT/DistilBERT
# Requirements: pip install torch transformers

from transformers import pipeline


def main():
    # Load sentiment-analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Text input
    text = "I love using Hugging Face transformers!"

    # Run inference
    result = sentiment_pipeline(text)

    # Output result
    print("--- Sentiment Analysis Result ---")
    print(f"Input Text: {text}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()