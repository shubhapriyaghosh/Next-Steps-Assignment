# Next-Steps-Assignment
git clone <repository-url>
cd <repository-directory>
pip install scrapy transformers sentence-transformers sklearn beautifulsoup4 nest_asyncio

Web Crawler
The web crawler is implemented using Scrapy, a powerful web crawling framework. The crawler scrapes data from the NVIDIA CUDA documentation site and follows links up to a specified depth.

Key Components:
Spider Class: NvidiaDocsSpider is a subclass of CrawlSpider. It specifies the starting URL, allowed domains, and rules for following links.
Custom Settings: Limits the depth of crawling and the number of pages to scrape for efficiency.
Parse Function: Extracts visible text from each page and yields the URL and content.
Data Chunking and Embedding
After scraping, the data is chunked into sentences and converted into embeddings using a pre-trained model from the sentence-transformers library.

Key Components:
Sentence Embeddings: Train word2vec embeddings.
Clustering: Clusters the sentences using KMeans clustering to group similar sentences together.
Hybrid Retrieval
A hybrid retrieval method is implemented to retrieve relevant data based on a query. It combines semantic similarity using embeddings and clustering results.

Key Components:
Cosine Similarity: Computes the similarity between the query embedding and the sentence embeddings.
Heap Queue: Retrieves the top k most similar sentences based on cosine similarity.
Question Answering
The retrieved and re-ranked data is passed to a question-answering model to generate answers. 
Key Components:
Question Answering Pipeline: Uses the pipeline function from transformers to create a question-answering pipeline.
Context Generation: Joins the retrieved sentences to form a context for the QA model.
Running the Script


Create a new Python file

Copy and paste the script into the file.

Ensure you have the necessary packages installed.

Run the script:
python web scawler.py

