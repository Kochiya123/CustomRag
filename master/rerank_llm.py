import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Rerank:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('AITeamVN/Vietnamese_Reranker')
        self.model = AutoModelForSequenceClassification.from_pretrained('AITeamVN/Vietnamese_Reranker')
        self.model.eval()
        self.MAX_LENGTH = 2304
        print('model loaded')

    def rerank_query(self, query, documents):
        """
        Rerank documents based on query relevance.
        
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

        # Create pairs of (query, document) for the reranker
        pairs = [[query, doc] for doc in doc_texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=self.MAX_LENGTH
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        # Combine documents with their scores
        # Format: (product_id, product_text, score) - product_id is always first element
        results = [
            (doc_ids[i], doc_texts[i], scores[i].item())
            for i in range(len(doc_texts))
        ]

        # Sort by score (highest first)
        results.sort(key=lambda x: x[2], reverse=True)

        return results

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
        reranker = Rerank()
        reranked_all = reranker.rerank_query(query, all_products)

        # Return top 10: format is (product_id, product_text, score)
        return reranked_all[:10]