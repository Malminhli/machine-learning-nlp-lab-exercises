import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

# 1. تحميل النموذج الذي حفظناه في Task 2
model = Word2Vec.load("word2vec.model")

# 2. استخراج الكلمات ومتجهاتها
words = list(model.wv.index_to_key)
word_vectors = [model.wv[word] for word in words]

# 3. تقليل الأبعاد إلى 2D باستخدام PCA[cite: 1]
pca = PCA(n_components=2)
word_vecs_2d = pca.fit_transform(word_vectors)

# 4. استكشاف الكلمات الأكثر تشابهاً مع كلمة "word"[cite: 1]
search_word = "word"
similar_words = model.wv.most_similar(search_word)
print(f"Words similar to '{search_word}':\n", similar_words)

# 5. رسم الكلمات بيانياً[cite: 1]
plt.figure(figsize=(10, 10))
plt.scatter(word_vecs_2d[:, 0], word_vecs_2d[:, 1], edgecolors="k", c="r")

for i, word in enumerate(words):
  plt.text(word_vecs_2d[i, 0], word_vecs_2d[i, 1], word)

plt.title("Word Embeddings Visualization")
plt.show()