from gensim.utils import simple_preprocess

# 1. النصوص التي سندرّب النموذج عليها
corpus = [
    "Natural language processing is a key area of machine learning.",
    "Word embeddings are used to represent words as vectors.",
    "Gensim provides an efficient implementation of Word2Vec.",
    "Training word embeddings captures the relationship between words.",
    "Word2Vec can be used to perform various NLP tasks.",
    "Exploring word vectors helps to understand semantic relationships."
]

# 2. تقطيع النصوص إلى كلمات (Tokens)
tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus]
print("Tokenized Corpus:\n", tokenized_corpus)