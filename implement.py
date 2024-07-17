import subprocess
import os
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import pipeline

# Run the web crawler as a subprocess
subprocess.run(['python', 'web scawler.py'])

# Data Chunking
with open("scraped_data.json", "r", encoding='utf-8') as file:
    text = file.read()

sentences = sent_tokenize(text)

# Train Word2Vec model for embeddings 
model = Word2Vec(sentences=[s.split() for s in sentences], vector_size=128, window=5, min_count=1, workers=4)
embeddings = np.array([model.wv[s.split()].mean(axis=0) for s in sentences])
embeddings = normalize(embeddings, axis=1, norm='l2')

# Create a Nearest Neighbors model
neighbors_model = NearestNeighbors(n_neighbors=min(5, len(embeddings)), algorithm='auto', metric='euclidean')
neighbors_model.fit(embeddings)

# Query Data
def query_embeddings(query_vectors, top_k):
    distances, indices = neighbors_model.kneighbors(query_vectors, n_neighbors=min(top_k, len(embeddings)))
    return distances, indices

# Generate random query vectors for demonstration
query_vectors = embeddings[:min(5, len(embeddings))]

# Perform search
top_k = min(5, len(embeddings))
distances, indices = query_embeddings(query_vectors, top_k)

# Display results
for i, (distance_list, index_list) in enumerate(zip(distances, indices)):
    print(f"\nQuery {i + 1}:")
    for distance, index in zip(distance_list, index_list):
        print(f"Index: {index}, Distance: {distance}")

# Re-ranking 
def re_rank_results(distances, indices):
    re_ranked_results = []
    for distance_list, index_list in zip(distances, indices):
        re_ranked_result = sorted(zip(distance_list, index_list), key=lambda x: x[0])
        re_ranked_results.append(re_ranked_result)
    return re_ranked_results

re_ranked_results = re_rank_results(distances, indices)

# Display re-ranked results
for i, result in enumerate(re_ranked_results):
    print(f"\nRe-ranked Query {i + 1}:")
    for distance, index in result:
        print(f"Index: {index}, Distance: {distance}")

# Question Answering using LLM
qa_pipeline = pipeline("question-answering")

# Example query
query = "What is CUDA?"

context = ". ".join(sentences)  # Create a context from the sentences

# Generate answer
result = qa_pipeline(question=query, context=context)
print(f"Answer: {result['answer']}")
