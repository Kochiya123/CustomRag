"""
Hybrid Recommendation System - Python Implementation
Updated to work with your mysql_connect() method
"""

import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict


class HybridRecommender:
    """
    A hybrid recommendation system combining collaborative and content-based filtering
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.collaborative_weight = 0.6
        self.content_weight = 0.4

        # Interaction weights
        self.FAVORITE_WEIGHT = 3
        self.PURCHASE_WEIGHT = 5
        self.QUANTITY_MULTIPLIER = 2
        self.RECENT_MULTIPLIER = 1.5
        self.RECENT_DAYS = 30

    def get_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get personalized recommendations for a user

        Args:
            user_id: Target user ID
            limit: Number of recommendations to return

        Returns:
            List of recommended products with scores
        """
        # Get user's interaction history
        user_interactions = self._get_user_interactions(user_id)

        if not user_interactions:
            # Cold start: return trending products
            return self._get_trending_products(limit)

        # Get collaborative filtering recommendations
        collab_scores = self._collaborative_filtering(user_id, user_interactions)

        # Get content-based recommendations
        content_scores = self._content_based_filtering(user_id, user_interactions)

        # Combine scores
        hybrid_scores = self._combine_scores(collab_scores, content_scores)

        # Filter and rank
        final_recommendations = self._filter_and_rank(
            user_id,
            hybrid_scores,
            user_interactions,
            limit
        )

        return final_recommendations

    def _get_user_interactions(self, user_id: int) -> Dict[int, float]:
        """
        Get all user interactions with products and calculate scores

        Returns:
            Dictionary mapping product_id to interaction score
        """
        query = """
        SELECT 
            p.id as product_id,
            pf.created_at as favorite_date,
            oi.quantity,
            o.created_at as purchase_date
        FROM products p
        LEFT JOIN product_favorites pf ON p.id = pf.product_id AND pf.user_id = %s
        LEFT JOIN order_items oi ON p.id = oi.product_id
        LEFT JOIN orders o ON oi.order_id = o.id AND o.user_id = %s
        WHERE (pf.id IS NOT NULL OR oi.id IS NOT NULL)
            AND p.is_active = true
        """

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, (user_id, user_id))
        results = cursor.fetchall()
        cursor.close()

        interactions = defaultdict(float)
        current_date = datetime.now()

        for row in results:
            product_id = row['product_id']
            score = 0

            # Favorite score
            if row['favorite_date']:
                score += self.FAVORITE_WEIGHT
                days_ago = (current_date - row['favorite_date']).days
                if days_ago < self.RECENT_DAYS:
                    score *= self.RECENT_MULTIPLIER

            # Purchase score
            if row['purchase_date']:
                quantity = row['quantity'] or 1
                score += self.PURCHASE_WEIGHT + (quantity - 1) * self.QUANTITY_MULTIPLIER
                days_ago = (current_date - row['purchase_date']).days
                if days_ago < self.RECENT_DAYS:
                    score *= self.RECENT_MULTIPLIER

            interactions[product_id] = max(interactions[product_id], score)

        return interactions

    def _collaborative_filtering(self, user_id: int, user_interactions: Dict) -> Dict[int, float]:
        """
        Find similar users and recommend products they liked

        Returns:
            Dictionary mapping product_id to collaborative score
        """
        FAVORITE_WEIGHT = 1.0
        ORDER_WEIGHT = 4.0
        # Find similar users
        similar_users = self._find_similar_users(user_id, user_interactions)

        if not similar_users:
            return {}

        # Get products liked by similar users
        product_scores = defaultdict(float)

        user_product_ids = tuple(user_interactions.keys())
        similar_user_ids = tuple([u['user_id'] for u in similar_users])

        # Get favorites from similar users
        query = """
        SELECT 
            pf.product_id,
            pf.user_id,
        FROM product_favorites pf
        WHERE pf.user_id IN ({})
            AND pf.product_id NOT IN ({})
        GROUP BY pf.product_id, pf.user_id
        """.format(
            ','.join(['%s'] * len(similar_user_ids)),
            ','.join(['%s'] * len(user_product_ids)) if user_product_ids else '%s'
        )

        cursor = self.db.cursor(dictionary=True)
        params = similar_user_ids + (user_product_ids if user_product_ids else (0,))
        cursor.execute(query, params)
        fav_results = cursor.fetchall()

        # Get purchases from similar users
        query2 = """
        SELECT 
            oi.product_id,
            o.user_id,
            SUM(oi.quantity) as interaction_count
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.id
        WHERE o.user_id IN ({})
            AND oi.product_id NOT IN ({})
        GROUP BY oi.product_id, o.user_id
        """.format(
            ','.join(['%s'] * len(similar_user_ids)),
            ','.join(['%s'] * len(user_product_ids)) if user_product_ids else '%s'
        )

        cursor.execute(query2, params)
        purchase_results = cursor.fetchall()
        cursor.close()

        # Calculate scores based on similarity
        similarity_map = {u['user_id']: u['similarity'] for u in similar_users}

        for row in fav_results:
            product_id = row['product_id']
            v = row['user_id']

            product_scores[product_id] += similarity_map.get(v, 0) * FAVORITE_WEIGHT

        for row in purchase_results:
            product_id = row['product_id']
            v = row['user_id']
            cnt = row['interaction_count']

            product_scores[product_id] += (
                    similarity_map.get(v, 0) * ORDER_WEIGHT * cnt
            )

        # Normalize scores
        if product_scores:
            max_score = max(product_scores.values())
            return {pid: score / max_score for pid, score in product_scores.items()}
        return {}

    def _find_similar_users(self, user_id: int, user_interactions: Dict) -> List[Dict]:
        """
        Find users with similar interaction patterns

        Returns:
            List of dictionaries with user_id and similarity score
        """
        user_product_ids = tuple(user_interactions.keys())

        if not user_product_ids:
            return []

        query = """
        SELECT 
            u2.id as user_id,
            COUNT(DISTINCT pf2.product_id) as common_favorites,
            (
                SELECT COUNT(*) 
                FROM product_favorites 
                WHERE user_id = u2.id
            ) as total_favorites
        FROM product_favorites pf1
        JOIN product_favorites pf2 ON pf1.product_id = pf2.product_id
        JOIN user u2 ON pf2.user_id = u2.id
        WHERE pf1.user_id = %s 
            AND u2.id != %s
        GROUP BY u2.id
        HAVING COUNT(DISTINCT pf2.product_id) >= 2
        ORDER BY common_favorites DESC
        LIMIT 20
        """

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, (user_id, user_id))
        results = cursor.fetchall()
        cursor.close()

        similar_users = []
        total_user_products = len(user_product_ids)

        for row in results:
            common = row['common_favorites']
            total = row['total_favorites']

            # Jaccard similarity
            similarity = common / (total_user_products + total - common)

            similar_users.append({
                'user_id': row['user_id'],
                'similarity': similarity,
                'common_products': common
            })

        return similar_users

    def _content_based_filtering(self, user_id: int, user_interactions: Dict) -> Dict[int, float]:
        """
        Recommend products similar to user's interaction history

        Returns:
            Dictionary mapping product_id to content-based score
        """
        # Get user's preferred categories
        user_categories = self._get_user_categories(user_interactions)

        if not user_categories:
            return {}

        # Get user's price preferences
        user_price_range = self._get_user_price_range(user_interactions)

        # Find products in similar categories
        category_ids = tuple(user_categories.keys())
        user_product_ids = tuple(user_interactions.keys()) or (0,)

        query = """
        SELECT 
            p.id,
            p.price,
            pc.category_id,
            COUNT(DISTINCT pc.category_id) as category_matches
        FROM products p
        JOIN product_categories pc ON p.id = pc.product_id
        WHERE pc.category_id IN ({})
            AND p.is_active = true
            AND p.stock > 0
            AND p.id NOT IN ({})
            AND p.product_type = 0
            AND p.is_custom = 0
        GROUP BY p.id, p.price, pc.category_id
        """.format(
            ','.join(['%s'] * len(category_ids)),
            ','.join(['%s'] * len(user_product_ids))
        )

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, category_ids + user_product_ids)
        results = cursor.fetchall()
        cursor.close()

        product_scores = {}

        for row in results:
            product_id = row['id']
            price = float(row['price']) if row['price'] else 0
            category_id = row['category_id']

            # Category match score
            category_weight = user_categories.get(category_id, 0)
            category_score = category_weight

            # Price similarity score
            if user_price_range['min'] and user_price_range['max']:
                avg_price = (user_price_range['min'] + user_price_range['max']) / 2
                price_diff = abs(price - avg_price) / avg_price if avg_price > 0 else 0
                price_score = max(0, 1 - price_diff)
            else:
                price_score = 0.5

            # Combined content score
            content_score = 0.7 * category_score + 0.3 * price_score

            if product_id in product_scores:
                product_scores[product_id] = max(product_scores[product_id], content_score)
            else:
                product_scores[product_id] = content_score

        return product_scores

    def _get_user_categories(self, user_interactions: Dict) -> Dict[int, float]:
        """
        Get user's preferred categories with weights
        """
        if not user_interactions:
            return {}

        product_ids = tuple(user_interactions.keys())

        query = """
        SELECT 
            pc.category_id,
            c.name,
            COUNT(*) as frequency
        FROM product_categories pc
        JOIN categories c ON pc.category_id = c.id
        WHERE pc.product_id IN ({})
        GROUP BY pc.category_id, c.name
        """.format(','.join(['%s'] * len(product_ids)))

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, product_ids)
        results = cursor.fetchall()
        cursor.close()

        categories = {}
        total_frequency = sum(row['frequency'] for row in results) if results else 1

        for row in results:
            category_id = row['category_id']
            frequency = row['frequency']
            categories[category_id] = frequency / total_frequency

        return categories

    def _get_user_price_range(self, user_interactions: Dict) -> Dict[str, float]:
        """
        Get user's typical price range
        """
        if not user_interactions:
            return {'min': None, 'max': None, 'avg': None}

        product_ids = tuple(user_interactions.keys())

        query = """
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price
        FROM products
        WHERE id IN ({})
        """.format(','.join(['%s'] * len(product_ids)))

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, product_ids)
        result = cursor.fetchone()
        cursor.close()

        return {
            'min': float(result['min_price']) if result and result['min_price'] else None,
            'max': float(result['max_price']) if result and result['max_price'] else None,
            'avg': float(result['avg_price']) if result and result['avg_price'] else None
        }

    def _combine_scores(self, collab_scores: Dict, content_scores: Dict) -> Dict[int, float]:
        """
        Combine collaborative and content-based scores
        """
        all_products = set(collab_scores.keys()) | set(content_scores.keys())

        hybrid_scores = {}
        for product_id in all_products:
            collab = collab_scores.get(product_id, 0)
            content = content_scores.get(product_id, 0)

            hybrid_score = (
                    self.collaborative_weight * collab +
                    self.content_weight * content
            )
            hybrid_scores[product_id] = hybrid_score

        return hybrid_scores

    def _filter_and_rank(
            self,
            user_id: int,
            scores: Dict,
            user_interactions: Dict,
            limit: int
    ) -> List[Dict]:
        """
        Filter out invalid products and apply ranking boosts
        """
        if not scores:
            return []

        product_ids = tuple(scores.keys())

        query = """
        SELECT 
            p.id,
            p.name,
            p.price,
            p.images,
            p.stock,
            p.created_at,
            COALESCE(fav_count.count, 0) as favorite_count,
            COALESCE(purchase_count.count, 0) as purchase_count
        FROM products p
        LEFT JOIN (
            SELECT product_id, COUNT(*) as count
            FROM product_favorites
            GROUP BY product_id
        ) fav_count ON p.id = fav_count.product_id
        LEFT JOIN (
            SELECT product_id, SUM(quantity) as count
            FROM order_items
            GROUP BY product_id
        ) purchase_count ON p.id = purchase_count.product_id
        WHERE p.id IN ({})
            AND p.is_active = true
            AND p.stock > 0
        """.format(','.join(['%s'] * len(product_ids)))

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, product_ids)
        results = cursor.fetchall()
        cursor.close()

        recommendations = []
        current_date = datetime.now()

        for row in results:
            product_id = row['id']
            base_score = scores[product_id]

            # Apply boost factors
            boosted_score = base_score

            # Recently added boost (within 30 days)
            days_old = (current_date - row['created_at']).days
            if days_old < 30:
                boosted_score *= 1.10

            # Stock availability boost
            if row['stock'] > 10:
                boosted_score *= 1.05

            # Popularity boost
            popularity = row['favorite_count'] + row['purchase_count']
            if popularity > 50:
                boosted_score *= 1.08

            recommendations.append({
                'product_id': product_id,
                'name': row['name'],
                'price': float(row['price']) if row['price'] else 0,
                'images': row['images'],
                'stock': row['stock'],
                'score': boosted_score,
                'favorite_count': row['favorite_count'],
                'purchase_count': row['purchase_count']
            })

        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Ensure diversity - limit products from same category
        diverse_recommendations = self._ensure_diversity(recommendations, limit)

        return diverse_recommendations[:limit]

    def _ensure_diversity(self, recommendations: List[Dict], limit: int) -> List[Dict]:
        """
        Ensure recommendations span multiple categories
        """
        if len(recommendations) <= limit:
            return recommendations

        # Get category for each product
        product_ids = tuple([r['product_id'] for r in recommendations])

        query = """
        SELECT pc.product_id, pc.category_id
        FROM product_categories pc
        WHERE pc.product_id IN ({})
        """.format(','.join(['%s'] * len(product_ids)))

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, product_ids)
        results = cursor.fetchall()
        cursor.close()

        # Map products to categories
        product_categories = defaultdict(list)
        for row in results:
            product_categories[row['product_id']].append(row['category_id'])

        # Select diverse products
        diverse_recs = []
        category_counts = defaultdict(int)
        max_per_category = max(2, limit // 3)

        for rec in recommendations:
            product_id = rec['product_id']
            categories = product_categories.get(product_id, [])

            # Check if we can add this product
            can_add = True
            for cat_id in categories:
                if category_counts[cat_id] >= max_per_category:
                    can_add = False
                    break

            if can_add:
                diverse_recs.append(rec)
                for cat_id in categories:
                    category_counts[cat_id] += 1

            if len(diverse_recs) >= limit:
                break

        # Fill remaining slots with highest scores
        if len(diverse_recs) < limit:
            remaining = [r for r in recommendations if r not in diverse_recs]
            diverse_recs.extend(remaining[:limit - len(diverse_recs)])

        return diverse_recs

    def _get_trending_products(self, limit: int) -> List[Dict]:
        """
        Get trending products for cold start scenarios
        """
        query = """
        SELECT 
            p.id,
            p.name,
            p.price,
            p.images,
            p.stock,
            COALESCE(recent_fav.count, 0) + COALESCE(recent_purchase.count, 0) as trend_score
        FROM products p
        LEFT JOIN (
            SELECT product_id, COUNT(*) as count
            FROM product_favorites
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY product_id
        ) recent_fav ON p.id = recent_fav.product_id
        LEFT JOIN (
            SELECT oi.product_id, SUM(oi.quantity) as count
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY oi.product_id
        ) recent_purchase ON p.id = recent_purchase.product_id
        WHERE p.is_active = true
            AND p.stock > 0
        ORDER BY trend_score DESC
        LIMIT %s
        """

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        cursor.close()

        return [
            {
                'product_id': row['id'],
                'name': row['name'],
                'price': float(row['price']) if row['price'] else 0,
                'images': row['images'],
                'stock': row['stock'],
                'score': row['trend_score'],
                'reason': 'Trending now'
            }
            for row in results
        ]

    def get_similar_products(self, product_id: int, limit: int = 5) -> List[Dict]:
        """
        Get products similar to a given product
        """
        # Get product's categories
        query = """
        SELECT category_id 
        FROM product_categories 
        WHERE product_id = %s
        """

        cursor = self.db.cursor(dictionary=True)
        cursor.execute(query, (product_id,))
        category_results = cursor.fetchall()

        if not category_results:
            cursor.close()
            return []

        category_ids = tuple([r['category_id'] for r in category_results])

        # Find similar products in same categories
        query = """
        SELECT 
            p.id,
            p.name,
            p.price,
            p.images,
            COUNT(DISTINCT pc.category_id) as category_matches
        FROM products p
        JOIN product_categories pc ON p.id = pc.product_id
        WHERE pc.category_id IN ({})
            AND p.id != %s
            AND p.is_active = true
            AND p.stock > 0
        GROUP BY p.id, p.name, p.price, p.images
        ORDER BY category_matches DESC, RAND()
        LIMIT %s
        """.format(','.join(['%s'] * len(category_ids)))

        cursor.execute(query, category_ids + (product_id, limit))
        results = cursor.fetchall()
        cursor.close()

        return [
            {
                'product_id': row['id'],
                'name': row['name'],
                'price': float(row['price']) if row['price'] else 0,
                'images': row['images'],
                'category_matches': row['category_matches']
            }
            for row in results
        ]


# Example usage
if __name__ == "__main__":
    import os
    import mysql.connector


    # Your connection method
    def mysql_connect():
        try:
            print("Connecting to MySQL database...")
            db = mysql.connector.connect(
                user=os.getenv('MYSQL_USERNAME'),
                password=os.getenv('MYSQL_PASSWORD'),
                host=os.getenv('MYSQL_HOST'),
                database=os.getenv('MYSQL_USERNAME')
            )
            return db
        except (Exception, mysql.connector.Error) as error:
            print(error)
            return None


    # Initialize recommender
    db_conn = mysql_connect()
    if db_conn:
        recommender = HybridRecommender(db_conn)

        # Get recommendations for user
        recommendations = recommender.get_recommendations(user_id=1, limit=10)

        print(f"Found {len(recommendations)} recommendations:")
        for rec in recommendations:
            print(f"  - {rec['name']}: Score {rec['score']:.3f}")

        db_conn.close()