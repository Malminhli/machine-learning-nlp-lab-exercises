# Machine Learning & NLP Lab Exercises

A collection of Python lab exercises covering foundational Natural Language Processing and Machine Learning topics.

## Topics covered

- Text preprocessing with **Gensim**.
- Word embeddings with **Word2Vec**.
- Dimensionality reduction with **PCA**.
- Neural-network exercises with **PyTorch**.
- Natural-language pipelines with **Hugging Face Transformers**.

## Project structure

| Files | Purpose |
|---|---|
| `Lab1-Task*.py`, `Lap1-Task1-2.py` | Text preprocessing and Word2Vec exercises. |
| `Lab2-Task*.py`, `Lap2-Task1.py` | Introductory PyTorch exercises. |
| `Lab3-Task*.py` | Neural-network and NumPy exercises. |
| `Lab4-Task*.py` | Transformer pipeline exercises. |
| `script*.py` | Supporting experiments and scripts. |
| `word2vec.model` | Saved Word2Vec model used by the exercises. |

## Requirements

Install the required Python packages in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

The repository contains the source code and the small saved `word2vec.model` file. The local `.venv` directory and PyCharm workspace files are intentionally excluded because they are machine-specific and can be recreated.

## License

This repository contains personal learning exercises. Reuse is permitted for educational purposes with attribution.
