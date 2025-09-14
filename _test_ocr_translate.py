import json
import patta_ocr

res = patta_ocr.run_ocr_on_file_internal('static/mp_patta.png', 'mp', use_enhance=True, translate=True)
print(json.dumps(res, ensure_ascii=False, indent=2))
