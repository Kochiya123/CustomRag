import os
import requests

class Rerank:
    def __init__(self):
        self.api_token = os.getenv("JINA_TOKEN")
        if not self.api_token:
            raise ValueError("JINA_TOKEN environment variable is not set")
        self.api_url = "https://api.jina.ai/v1/rerank"
        self.model_name = "jina-reranker-v3-base-multilingual"
        print("Jina Rerank v3 API initialized")

    def rerank_query(self, query, documents):
        """
        Rerank documents based on query relevance using Jina Rerank v3 API.
        
        Args:
            query: The search query string
            documents: List of tuples (product_id, product_text) or list of strings
        
        Returns:
            List of tuples: (product_id, product_text, score) - sorted by score (highest first)
            Note: product_id is always preserved in the first position
        """
        # Handle both list of strings and list of tuples (product_id, product_text)
        if documents and isinstance(documents[0], tuple):
            # Extract product_id from first element, product_text from second element
            doc_ids = [doc[0] for doc in documents]  # product_id is preserved here
            doc_texts = [doc[1] for doc in documents]
        else:
            # If documents are strings, generate sequential IDs
            doc_ids = list(range(len(documents)))
            doc_texts = documents

        # Early return if no documents
        if not doc_texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": len(doc_texts)  # Return all documents, sorted by relevance
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract results from Jina API response
            # Jina returns results sorted by relevance: [{"index": 2, "relevance_score": 0.95}, {"index": 0, "relevance_score": 0.87}, ...]
            # The index corresponds to the original document position
            results = []
            for item in result.get("results", []):
                original_index = item.get("index")
                score = item.get("relevance_score", 0.0)

                # Map back to original document using the index
                if 0 <= original_index < len(doc_ids):
                    doc_id = doc_ids[original_index]
                    doc_text = doc_texts[original_index]
                    results.append((doc_id, doc_text, score))
            # Results are already sorted by relevance score (highest first) from Jina API
            return results
            
        except Exception as e:
            print(f"Error in Jina Rerank API call: {e}")
            # Fallback: return documents in original order with zero scores
            return [(doc_ids[i], doc_texts[i], 0.0) for i in range(len(doc_texts))]

    def combine_and_rerank_together(self, query, products_from_query, products_from_vector):
        """
        Combine products from multiple sources, remove duplicates, and rerank.
        
        Args:
            query: The search query string
            products_from_query: List of tuples (product_id, product_text)
            products_from_vector: List of tuples (product_id, product_text)
        
        Returns:
            List of tuples: (product_id, product_text, score) - top 10 results
            Note: product_id is always preserved in the first position
        """
        # Remove duplicates while preserving both sources
        # Products are tuples: (product_id, product_text)
        seen = set()
        all_products = []

        for product in products_from_query + products_from_vector:
            product_id = product[0]  # Extract product_id from first element
            if product_id not in seen:
                seen.add(product_id)
                all_products.append(product)  # Preserve full tuple (product_id, product_text)

        # Early return if no products
        if not all_products:
            return []

        # Rerank all products together (batched processing is automatic)
        # rerank_query will preserve product_id in the returned tuples
        reranked_all = self.rerank_query(query, all_products)

        # Return top 10: format is (product_id, product_text, score)
        return reranked_all[:10]