import re
from typing import Dict, Any



def extract_info(cur, user_query: str) -> Dict[str, Any]:
    intent_map = {
        # General flower inquiries
        "hoa nào": "all_flowers",
        "loại hoa": "all_flowers",
        "các loại hoa": "all_flowers",
        "danh sách hoa": "all_flowers",
        "shop có hoa gì": "all_flowers",
        "những loại hoa nào": "all_flowers",
        "có những hoa gì": "all_flowers",
        "hoa gì": "all_flowers",
        "hoa": "all_flowers",
        "sản phẩm": "all_flowers",

        # General category/occasion inquiries
        "dịp nào": "all_categories",
        "loại nào": "all_categories",
        "các dịp": "all_categories",
        "danh sách dịp": "all_categories",
        "có hoa cho dịp nào": "all_categories",
        "loại hoa nào" : "all_categories",

        # Delivery inquiries
        "có giao hàng": "delivery_info",
        "ship không": "delivery_info",
        "có giao không": "delivery_info",
        "giao tận nơi": "delivery_info",
        "có giao ngay": "delivery_info",

        # Price inquiries (general, not specific numbers)
        "giá bao nhiêu": "price_info",
        "giá thế nào": "price_info",
        "giá cả": "price_info",
        "bảng giá": "price_info",
    }

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
        "đến": "between",
        "khoảng": "between",
        "trong khoảng": "between",
        "dao động": "between",
        "nằm trong khoảng": "between",
        "giá từ": "between",
        "tới": "between",
        "giữa": "between",
        "tầm": "between",
    }
    info = {
        "flower": None,
        "category": None,
        "price": None,
        "preference": None,
        "delivery": None,
        "voucher": None,
        "intent": None,  # New field for intent detection
    }

    price_patterns = [
        # Match "1tr5" or "1.5tr" (1.5 million)
        (r'(\d+)(?:tr|triệu)(\d+)', lambda m: int(m.group(1)) * 1000000 + int(m.group(2)) * 100000),

        # Match "1.5tr" or "1,5tr" (1.5 million with decimal)
        (r'(\d+)[.,](\d+)\s?(?:tr|triệu)', lambda m: int(m.group(1)) * 1000000 + int(m.group(2)) * 100000),

        # Match "10tr" or "10 triệu" (millions)
        (r'(\d+)\s?(?:tr|triệu)', lambda m: int(m.group(1)) * 1000000),

        # Match "500k" or "500 ngàn" (thousands)
        (r'(\d+)\s?(?:k|ngàn)', lambda m: int(m.group(1)) * 1000),

        # Match plain numbers with VND
        (r'(\d+)\s?(?:đ|vnd|dong)', lambda m: int(m.group(1))),
    ]

    # Normalize query
    query = user_query.lower()

    # 0. Check for general intents first (before specific extraction)
    for pattern, intent in intent_map.items():
        if pattern in query:
            info["intent"] = intent
            break  # Stop after first match

    # 1. Extract price (regex for numbers + "k/ngàn/đ")
    for pattern, converter in price_patterns:
        match = re.search(pattern, query)
        if match:
            info["price"] = converter(match)

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

