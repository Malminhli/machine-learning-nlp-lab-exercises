from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

# --- Task 1: تجهيز البيانات وتقطيعها ---
corpus = [
    "Natural language processing is a key area of machine learning.",
    "Word embeddings are used to represent words as vectors.",
    "Gensim provides an efficient implementation of Word2Vec.",
    "Training word embeddings captures the relationship between words.",
    "Word2Vec can be used to perform various NLP tasks.",
    "Exploring word vectors helps to understand semantic relationships."
]

tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus]

# --- Task 2: تدريب النموذج وحفظه ---
# الآن المتغير tokenized_corpus معرف وموجود فوقه مباشرة
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
print("Word2Vec model trained and saved successfully!")

# طباعة متجه كلمة word
word = "word"
print(f"\nEmbedding for '{word}':\n", model.wv[word])