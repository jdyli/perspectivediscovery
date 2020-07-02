'''
Create random model based on tfidf
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd

def fit_tfidf(data, topic_n=6, topwords_n=60):
    doc_list = []
    for doc in data:
        doc_list.append(' '.join([str(elem) for elem in doc]))
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(doc_list)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors # get the first vector out (for the first document)
    tfidf_df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names())
    tfidf_df['sum'] = tfidf_df.sum(axis=1)
    tokens_frequency = tfidf_df['sum'].sort_values(ascending=False)

    topic_list = []
    for i in range(0, topic_n):
        random_topic = random.choices(tokens_frequency.index[:topwords_n], k=10)
        topic_list.append(random_topic)

    return topic_list, tokens_frequency[:topwords_n]