from flask import Flask, request, jsonify
from dss import recommend_schemes, recommend_with_priority, normalize_applicant

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({'error': 'empty payload'}), 400
    # Accept either raw applicant or wrapper {"applicant": {...}}
    applicant = payload.get('applicant') if isinstance(payload, dict) and 'applicant' in payload else payload
    a = normalize_applicant(applicant)
    schemes = recommend_schemes(a)
    scored = recommend_with_priority(a)
    return jsonify({'applicant': a, 'eligible': schemes, 'scored': scored})

@app.route('/recommend/batch', methods=['POST'])
def recommend_batch():
    payload = request.get_json(force=True)
    if not payload or not isinstance(payload, list):
        return jsonify({'error': 'expected a list of applicants'}), 400
    out = []
    for raw in payload:
        a = normalize_applicant(raw)
        out.append({'applicant': a, 'eligible': recommend_schemes(a), 'scored': recommend_with_priority(a)})
    return jsonify(out)


# Self-test using Flask test client when run directly
if __name__ == '__main__':
    sample = {
        'name': 'Self Test',
        'land_size': 1.0,
        'tribe': False,
        'water_access': 'none',
        'income': 25000,
        'age': 40,
        'household_size': 4,
        'agriculture': True,
        'labourer': False
    }
    with app.test_client() as c:
        resp = c.post('/recommend', json=sample)
        print('Status:', resp.status_code)
        print('JSON:', resp.get_json())
    print('If you want to run a server: python dss_api.py and POST to http://127.0.0.1:5000/recommend')
