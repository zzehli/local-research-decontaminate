import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

    
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
)
index_name = "gsm8k"
es.indices.delete(index=index_name)
print("Deleted all indices")