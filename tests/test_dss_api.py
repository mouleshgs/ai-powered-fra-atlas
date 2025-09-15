import json
from save_app import app


def test_recommend_endpoint():
    client = app.test_client()
    sample = {
        'name': 'API Test',
        'land_size': 1.0,
        'tribe': False,
        'water_access': 'none',
        'income': 25000,
        'age': 40,
        'household_size': 4,
        'agriculture': True
    }
    resp = client.post('/recommend', json=sample)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'dss_recommendations' in data or 'eligible' in data or 'scored' in data


def test_recommend_batch_endpoint():
    client = app.test_client()
    batch = [
        {'name': 'A', 'land_size': 1.0, 'agriculture': True, 'income': 20000},
        {'name': 'B', 'land_size': 0.0, 'labourer': True}
    ]
    resp = client.post('/recommend/batch', json=batch)
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 2
