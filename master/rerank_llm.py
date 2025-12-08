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
        # Handle both list of strings and list of tuples (id, text)
        if documents and isinstance(documents[0], tuple):
            doc_ids = [doc[0] for doc in documents]
            doc_texts = [doc[1] for doc in documents]
        else:
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
        results = [
            (doc_ids[i], doc_texts[i], scores[i].item())
            for i in range(len(doc_texts))
        ]

        # Sort by score (highest first)
        results.sort(key=lambda x: x[2], reverse=True)

        return results

    def combine_and_rerank_together(self, query, products_from_query, products_from_vector):
        # Remove duplicates while preserving both sources
        seen = set()
        all_products = []

        for product in products_from_query + products_from_vector:
            product_id = product[0]
            if product_id not in seen:
                seen.add(product_id)
                all_products.append(product)

        # Early return if no products
        if not all_products:
            return []

        # Rerank all products together (batched processing is automatic)
        reranker = Rerank()
        reranked_all = reranker.rerank_query(query, all_products)

        return reranked_all[:10]  # Return top 10