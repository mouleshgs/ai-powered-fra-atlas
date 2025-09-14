import requests, json
url = 'http://127.0.0.1:5000/extract_patta'
files = {'image': open('static/mp_patta.png', 'rb')}
data = {'state':'mp'}
print('Posting to', url)
r = requests.post(url, files=files, data=data)
print('Status', r.status_code)
try:
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
except Exception as e:
    print('Failed to parse JSON:', e)
    print(r.text)
