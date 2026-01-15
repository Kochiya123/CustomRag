

def text_split(text):
    parts = text.split('. ')
    for part in parts:
        if part.startswith('Mô tả:'):
            mo_ta = part.replace('Mô tả:', '').strip()
            print(mo_ta)
            return mo_ta
    return None