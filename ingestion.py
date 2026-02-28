import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_document_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORING")

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"Splitted into {len(batches)}")

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await asyncio.to_thread(vector_store.add_documents, batch)
            log_success(f"Successfully added batch {batch_num}/{len(batches)}")
        except Exception as e:
            log_error(f"Vector Store Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if False in results:
        log_warning("Some of the batches failed")
    else:
        log_success("All batches processed sucessfully")


async def main():
    """Main sync function to orhcestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "Tavily Crawl: Starting to Crawl documentation from https://docs.langchain.com/",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/",
            "max_depth": 5,
            "extract_depth": "advanced",
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(
        f"TavilyCrawl: Succesfully crawled {len(all_docs)} URLs from documentation site"
    )

    log_header("DOCUMENT CHUNKING PHASE")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    await index_document_async(splitted_docs, batch_size=100)

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
