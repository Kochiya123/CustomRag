import re
from typing import Dict, Any



def extract_info(cur, user_query: str) -> Dict[str, Any]:
    cur.execute('SELECT product_name FROM product_vector')
    FLOWERS = [row[0].lower() for row in cur.fetchall() if row[0]]
    cur.execute('SELECT category_name FROM category_vector')
    CATEGORIES = [row[0].lower() for row in cur.fetchall() if row[0]]
    preference_map = {
        # Greater than
        "trên": ">=",
        "hơn": ">=",
        "cao hơn": ">=",
        "ít nhất": ">=",
        "tối thiểu": ">=",
        "lớn hơn": ">=",
        "vượt quá": ">=",
        "trở lên": ">=",
        "không dưới": ">=",
        ">=": ">=",
        ">": ">=",

        # Less than
        "dưới": "<=",
        "rẻ hơn": "<=",
        "thấp hơn": "<=",
        "không quá": "<=",
        "tối đa": "<=",
        "nhỏ hơn": "<=",
        "ít hơn": "<=",
        "không vượt quá": "<=",
        "<=": "<=",
        "<": "<=",
        "giá tối đa": "<=",

        # Exact
        "bằng": "=",
        "chính xác": "=",
        "đúng bằng": "=",
        "giá là": "=",
        "giá đúng": "=",
        "=": "=",
        "giá cố định": "=",
        "giá chính xác": "=",
        "giá đúng bằng": "=",

        # Range
        "đến": "=",
        "khoảng": "=",
        "trong khoảng": "=",
        "dao động": "=",
        "nằm trong khoảng": "=",
        "giá từ": "=",
        "tới": "=",
        "giữa": "="
    }
    info = {
        "flower": None,
        "category": None,
        "price": None,
        "preference": None,
        "delivery": None,
        "voucher": None,
    }

    # Normalize query
    query = user_query.lower()

    # 1. Extract price (regex for numbers + "k/ngàn/đ")
    price_match = re.search(r'(\d+)\s?(k|ngàn|đ|vnd|tr)', query)
    if price_match:
        value = int(price_match.group(1)) * (1000 if price_match.group(2) in ["k", "ngàn"] else 1000000 if price_match.group(2) in ["tr"] else 1)
        info["price"] = value

    for word in preference_map:
        if word in query:
            info["preference"] = preference_map[word]

    voucher_match = re.search(r'(?i)\b(voucher|mã\s?giảm\s?giá|khuyến\s?mãi|ưu\s?đãi|discount|promo\s?code|coupon)\b', query)
    bool(voucher_match) if voucher_match else None

    # 2. Match flower names
    for flower in FLOWERS:
        if flower in query:
            info["flower"] = flower
            break  # Stop after first match

    # 3. Match occasions
    for cat in CATEGORIES:
        if cat in query:
            info["category"] = cat
            break  # Stop after first match

    # 4. Delivery intent
    if "giao ngay" in query or "trong ngày" in query:
        info["delivery"] = "same-day"
    elif "giao hàng" in query:
        info["delivery"] = "delivery"

    return info

