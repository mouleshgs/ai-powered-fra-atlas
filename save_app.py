from flask import Flask, request, jsonify, send_from_directory, render_template
import os, json, tempfile
import asyncio
import requests
from ocr import extract_text, parse_patta, translate_fields_async


app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
    _use_cors = True
except Exception:
    _use_cors = False

STATE_FILE_MAP = {
    'mp': 'mp.geojson',
    'odisha': 'odisha.geojson',
    'tripura': 'tripura.geojson',
    'telangana': 'telangana.geojson'
}

GEOJSON_DIR = os.path.join(app.root_path, "static", "geojson")

@app.route('/')
def home():
    return render_template('user/apply.html')

@app.route('/admin')
def admin():
    return render_template('index.html')

def json_response(data, status=200):
    resp = jsonify(data)
    resp.status_code = status
    if not _use_cors:
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

# Endpoint to get geojson dynamically
@app.route('/geojson/<state>')
def get_geojson(state):
    if state not in STATE_FILE_MAP:
        return json_response({'success': False, 'message': 'Invalid stateKey'}, 400)
    return send_from_directory(GEOJSON_DIR, STATE_FILE_MAP[state])

@app.route('/save_app', methods=['POST', 'OPTIONS'])
def save_app():
    if request.method == 'OPTIONS':
        return json_response({'success': True, 'message': 'ok'})

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return json_response({'success': False, 'message': 'Invalid JSON body: ' + str(e)}, 400)

    stateKey = data.get('stateKey')
    featureId = data.get('featureId')
    appId = data.get('appId')
    updatedApp = data.get('updatedApp')

    if not all([stateKey, featureId, appId, updatedApp]):
        return json_response({'success': False, 'message': 'Missing parameters'}, 400)

    if stateKey not in STATE_FILE_MAP:
        return json_response({'success': False, 'message': 'Invalid stateKey'}, 400)

    file_path = os.path.join(GEOJSON_DIR, STATE_FILE_MAP[stateKey])
    if not os.path.isfile(file_path):
        return json_response({'success': False, 'message': f'GeoJSON file not found: {STATE_FILE_MAP[stateKey]}'}, 404)

    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            geo = json.load(fh)
    except Exception as e:
        return json_response({'success': False, 'message': 'Failed to parse geojson: ' + str(e)}, 500)

    features = geo.get('features')
    if not isinstance(features, list):
        return json_response({'success': False, 'message': 'No features array in geojson'}, 500)

    found_feature = None
    for feat in features:
        if str(feat.get('id')) == str(featureId):
            found_feature = feat
            break

    if not found_feature:
        return json_response({'success': False, 'message': 'Feature not found'}, 404)

    props = found_feature.setdefault('properties', {})
    apps = props.setdefault('applications', [])
    if not isinstance(apps, list):
        apps = []
        props['applications'] = apps

    found_app_index = None
    for i, a in enumerate(apps):
        if str(a.get('id')) == str(appId):
            found_app_index = i
            break

    if not isinstance(updatedApp, dict):
        return json_response({'success': False, 'message': 'updatedApp must be an object'}, 400)

    if found_app_index is not None:
        apps[found_app_index].update(updatedApp)
    else:
        apps.append(updatedApp)

    try:
        fd, tmp_path = tempfile.mkstemp(prefix='geojson_', suffix='.tmp', dir=GEOJSON_DIR)
        with os.fdopen(fd, 'w', encoding='utf-8') as tmpf:
            json.dump(geo, tmpf, ensure_ascii=False, indent=2)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmp_path, file_path)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return json_response({'success': False, 'message': 'Failed to write file: ' + str(e)}, 500)

    return json_response({'success': True, 'message': 'Saved'})

@app.route("/geocode")
def geocode():
    q = request.args.get("q")
    if not q:
        return json_response({"success": False, "message": "Missing query"}, 400)

    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "json", "limit": 1, "q": q}
    headers = {"User-Agent": "FRA-App/1.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return json_response({"success": False, "message": str(e)}, 500)

@app.route("/extract_patta", methods=["POST"])
def extract_patta_endpoint():
    state_key = request.form.get("stateKey")
    file = request.files.get("pattaFile")
    if not file or not state_key:
        return jsonify({"error": "Missing file or state"}), 400
    
    text = extract_text(file)
    data = parse_patta(text, state_key)
    data = asyncio.run(translate_fields_async(data))

    data["state"] = state_key
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
