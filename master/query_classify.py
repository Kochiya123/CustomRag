import re
from typing import Dict, Any

from master.main import cur

# Example dictionaries for quick keyword matching
FLOWERS = cur.execute('SELECT flowers FROM flowers').fetchall()
CATEGORIES = cur.execute('SELECT occasions FROM occasions').fetchall()
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
    "đến": "BETWEEN",
    "khoảng": "BETWEEN",
    "trong khoảng": "BETWEEN",
    "dao động": "BETWEEN",
    "nằm trong khoảng": "BETWEEN",
    "giá từ": "BETWEEN",
    "tới": "BETWEEN",
    "giữa": "BETWEEN"
}


def extract_info(user_query: str) -> Dict[str, Any]:
    info = {
        "intent": None,
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
            break

    # 3. Match occasions
    for cat in CATEGORIES:
        if cat in query:
            info["category"] = cat
            break

    # 4. Delivery intent
    if "giao ngay" in query or "trong ngày" in query:
        info["delivery"] = "same-day"
    elif "giao hàng" in query:
        info["delivery"] = "delivery"

    # 6. Intent classification (very simple)
    if "giữ được bao lâu" in query or "chăm sóc" in query:
        info["intent"] = "care_tips"
    elif "giá" in query or "bao nhiêu" in query:
        info["intent"] = "price_query"
    elif "giao hàng" in query:
        info["intent"] = "delivery_query"
    else:
        info["intent"] = "product_search"

    return info

