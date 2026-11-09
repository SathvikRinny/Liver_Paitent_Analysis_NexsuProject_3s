import pytest

from app import app

HEALTHY = {
    'Gender': 'female', 'Age': 30, 'Total_Bilirubin': 0.7, 'Direct_Bilirubin': 0.2,
    'Alkaline_Phosphotase': 120, 'Alamine_Aminotransferase': 20, 'Aspartate_Aminotransferase': 22,
    'Total_Protiens': 7.0, 'Albumin': 4.0, 'Albumin_and_Globulin_Ratio': 1.3,
}
UNHEALTHY = {
    'Gender': 'male', 'Age': 60, 'Total_Bilirubin': 7.5, 'Direct_Bilirubin': 3.9,
    'Alkaline_Phosphotase': 500, 'Alamine_Aminotransferase': 120, 'Aspartate_Aminotransferase': 200,
    'Total_Protiens': 6.0, 'Albumin': 2.5, 'Albumin_and_Globulin_Ratio': 0.6,
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()


def test_pages_load(client):
    assert client.get('/').status_code == 200
    assert client.get('/predict').status_code == 200
    assert client.get('/missing').status_code == 404


def test_risk_is_ordered(client):
    low = client.post('/api/predict', json=HEALTHY).get_json()
    high = client.post('/api/predict', json=UNHEALTHY).get_json()
    assert low['probability'] < high['probability']
    assert high['risk'] == 'high'
    assert low['out_of_range'] == []
    assert len(high['out_of_range']) >= 5


def test_form_post_renders_result(client):
    res = client.post('/predict', data=UNHEALTHY)
    assert res.status_code == 200
    assert b'High risk' in res.data


def test_validation_errors(client):
    bad = dict(HEALTHY, Age='abc', Albumin=99)
    bad.pop('Gender')
    res = client.post('/api/predict', json=bad)
    assert res.status_code == 400
    assert set(res.get_json()['errors']) == {'Age', 'Albumin', 'Gender'}

    res = client.post('/predict', data=bad)
    assert res.status_code == 400
    assert b'Please fix the highlighted fields' in res.data
