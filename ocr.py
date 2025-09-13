import pytesseract
from PIL import Image
import re
import os
import asyncio
from googletrans import Translator


# -------------------- Configuration --------------------
# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Map states to Tesseract language codes
state_lang_map = {
    'mp': 'hin',        # Hindi
    'telangana': 'tel', # Telugu
    'tripura': 'ben',   # Bengali
    'odisha': 'ori'     # Odia
}

# -------------------- OCR Extraction --------------------
def extract_text(file, state_key=None):
    """
    Extract text from an uploaded image (path or FileStorage)
    """
    lang = state_lang_map.get(state_key, 'hin')  # default Hindi

    # Open image
    if hasattr(file, 'read'):
        img = Image.open(file).convert('RGB')
    else:
        img = Image.open(file).convert('RGB')

    # Run OCR
    text = pytesseract.image_to_string(img, lang=lang)
    return text

# -------------------- Patta Parser --------------------
def parse_patta(text, state_key):
    """
    Extract name, father_name, village, khata_no using strict regex
    """
    result = {
        "name": "Unknown",
        "father_name": "Unknown",
        "village": "Unknown",
        "khata_no": "Unknown",
        "state": state_key
    }

    # Normalize text
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Regex patterns for different languages
    # Hindi (MP)
    if state_key == 'mp':
        for line in lines:
            name_match = re.search(r'नाम\s*[:\-]?\s*([\w\s\u0900-\u097F]+)', line)
            father_match = re.search(r'पिता का नाम\s*[:\-]?\s*([\w\s\u0900-\u097F]+)', line)
            village_match = re.search(r'गांव\s*[:\-]?\s*([\w\s\u0900-\u097F]+)', line)
            khata_match = re.search(r'खत संख्या\s*[:\-]?\s*(\d+)', line)

            if father_match:
                result['father_name'] = father_match.group(1).strip()
            elif name_match:
                result['name'] = name_match.group(1).strip()
            if village_match:
                result['village'] = village_match.group(1).strip()
            if khata_match:
                result['khata_no'] = khata_match.group(1).strip()

    # Telugu (Telangana)
    elif state_key == 'telangana':
        for line in lines:
            name_match = re.search(r'పేరు\s*[:\-]?\s*([\w\s\u0C00-\u0C7F]+)', line)
            father_match = re.search(r'తండ్రి పేరు\s*[:\-]?\s*([\w\s\u0C00-\u0C7F]+)', line)
            village_match = re.search(r'గ్రామం\s*[:\-]?\s*([\w\s\u0C00-\u0C7F]+)', line)
            khata_match = re.search(r'ఖాతా సంఖ్య\s*[:\-]?\s*(\d+)', line)

            if father_match:
                result['father_name'] = father_match.group(1).strip()
            elif name_match:
                result['name'] = name_match.group(1).strip()
            if village_match:
                result['village'] = village_match.group(1).strip()
            if khata_match:
                result['khata_no'] = khata_match.group(1).strip()

    # Bengali (Tripura)
    elif state_key == 'tripura':
        for line in lines:
            name_match = re.search(r'নাম\s*[:\-]?\s*([\w\s\u0980-\u09FF]+)', line)
            father_match = re.search(r'পিতার নাম\s*[:\-]?\s*([\w\s\u0980-\u09FF]+)', line)
            village_match = re.search(r'গ্রাম\s*[:\-]?\s*([\w\s\u0980-\u09FF]+)', line)
            khata_match = re.search(r'খত সংখ্যা\s*[:\-]?\s*(\d+)', line)

            if father_match:
                result['father_name'] = father_match.group(1).strip()
            elif name_match:
                result['name'] = name_match.group(1).strip()
            if village_match:
                result['village'] = village_match.group(1).strip()
            if khata_match:
                result['khata_no'] = khata_match.group(1).strip()

    # Odia (Odisha) - fallback to English pattern if OCR struggles
    elif state_key == 'odisha':
        for line in lines:
            name_match = re.search(r'Name\s*[:\-]?\s*([\w\s]+)', line)
            father_match = re.search(r'Father\s*[:\-]?\s*([\w\s]+)', line)
            village_match = re.search(r'Village\s*[:\-]?\s*([\w\s]+)', line)
            khata_match = re.search(r'Khata\s*[:\-]?\s*(\d+)', line)

            if father_match:
                result['father_name'] = father_match.group(1).strip()
            elif name_match:
                result['name'] = name_match.group(1).strip()
            if village_match:
                result['village'] = village_match.group(1).strip()
            if khata_match:
                result['khata_no'] = khata_match.group(1).strip()

    return result

# translation


translator = Translator()

async def translate_text(text, dest='en'):
    return (await translator.translate(text, dest=dest)).text

async def translate_fields_async(patta_dict):
    translated = {}
    for key, value in patta_dict.items():
        if key != 'state' and value != "Unknown":
            try:
                translated_text = await translate_text(value)
            except Exception:
                translated_text = value
            translated[key] = translated_text
        else:
            translated[key] = value
    return translated

# Example usage
# -------------------- Quick Test --------------------
if __name__ == "__main__":
    state_files = {
        "mp": "static/mp_patta.png",
        # Add Telangana, Tripura, Odisha examples here
        # "telangana": "static/telangana_patta.png",
        # "tripura": "static/tripura_patta.png",
        # "odisha": "static/odisha_patta.png"
    }

    for state_key, example_file in state_files.items():
        print(f"\n--- {state_key.upper()} Patta ---")
        text = extract_text(example_file, state_key)
        print("Extracted Text:\n", text)
        parsed = parse_patta(text, state_key)
        print("Parsed Patta Info:", parsed)
        translated = asyncio.run(translate_fields_async(parsed))
        print("translated: ", translated)
