import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import nest_asyncio
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from transformers import pipeline

nest_asyncio.apply()

# Web Crawler Class
class NvidiaDocsSpider(CrawlSpider):
    name = 'nvidia_docs'
    allowed_domains = ['docs.nvidia.com']
    start_urls = ['https://docs.nvidia.com/cuda/']

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 5,  
        'CLOSESPIDER_PAGECOUNT': 50  # Limit the number of pages to scrape
    }

    def parse_item(self, response):
        page_content = self.extract_text(response)
        page_url = response.url
        yield {
            'url': page_url,
            'content': page_content
        }

    def extract_text(self, response):
        soup = BeautifulSoup(response.body, 'html.parser')
        [s.extract() for s in soup(['style', 'script', 'head', 'title', 'meta', '[document]'])]
        visible_text = soup.getText(separator=' ')
        return visible_text.strip()

# Function to run the spider
def run_spider():
    process = CrawlerProcess(settings={
        'FEEDS': {
            'scrapy_data.json': {'format': 'json'},
        },
    })
    
    process.crawl(NvidiaDocsSpider)
    process.start()

run_spider()
