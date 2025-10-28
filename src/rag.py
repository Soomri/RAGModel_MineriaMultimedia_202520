# src/rag.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.retriever import RetrieverFactory

load_dotenv()

class RAGPipeline:
    """Pipeline de Retrieval-Augmented Generation (RAG)."""
    def __init__(self, retriever_type="tfidf", chunks=None, top_k=5):
        self.retriever = RetrieverFactory.create_and_fit(retriever_type, chunks)
        self.top_k = top_k
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint="https://pnl-maestria.openai.azure.com/"
        )

    def generate_answer(self, query):
        """Genera respuesta combinando retrieval + modelo generativo."""
        results = self.retriever.query(query, self.top_k)
        context = "\n".join([r["text"] for r in results])

        prompt = f"""
        You are an expert assistant.
        Use the following context to answer the question concisely.
        
        Context:
        {context}

        Question: {query}
        Answer:
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip(), results
