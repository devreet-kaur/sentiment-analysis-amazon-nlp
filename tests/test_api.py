"""
tests/test_api.py - MAI203 Amazon Fine Food Reviews
Pytest test suite for the FastAPI sentiment prediction endpoint.
Run: pytest tests/test_api.py -v
Note: requires a trained model at results/models/best_model.pkl
"""
from fastapi.testclient import TestClient

from src.app import app, load_model

# Load model once at module level for all tests
load_model()
client = TestClient(app)

# ── Health endpoint ──────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get('/health')
    assert response.status_code == 200

def test_health_contains_status_ok():
    response = client.get('/health')
    assert response.json()['status'] == 'ok'

def test_health_contains_model_field():
    response = client.get('/health')
    assert 'model' in response.json()

# ── Classes endpoint ─────────────────────────────────────────────────────────

def test_classes_returns_200():
    response = client.get('/classes')
    assert response.status_code == 200

def test_classes_contains_expected_labels():
    response = client.get('/classes')
    classes = response.json()['classes']
    assert 'positive' in classes
    assert 'negative' in classes

# ── Predict endpoint: valid inputs ───────────────────────────────────────────

def test_predict_positive_review():
    response = client.post('/predict', json={'text': 'This coffee is absolutely amazing!'})
    assert response.status_code == 200
    data = response.json()
    assert data['label'] == 'positive'

def test_predict_negative_review():
    response = client.post('/predict', json={'text': 'Terrible product, completely inedible.'})
    assert response.status_code == 200
    data = response.json()
    assert data['label'] == 'negative'

def test_predict_response_has_required_fields():
    response = client.post('/predict', json={'text': 'Great taste and fast delivery!'})
    data = response.json()
    assert 'label' in data
    assert 'confidence' in data
    assert 'all_scores' in data

def test_predict_confidence_is_float_between_0_and_1():
    response = client.post('/predict', json={'text': 'Really good quality product.'})
    confidence = response.json()['confidence']
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0

def test_predict_all_scores_sum_to_one():
    response = client.post('/predict', json={'text': 'Average product, nothing special.'})
    all_scores = response.json()['all_scores']
    total = sum(all_scores.values())
    assert abs(total - 1.0) < 1e-4

def test_predict_label_is_in_known_classes():
    response = client.post('/predict', json={'text': 'Decent dog food, my pet likes it.'})
    label = response.json()['label']
    assert label in ['positive', 'negative']

def test_predict_optional_review_id_is_returned():
    response = client.post('/predict', json={'text': 'Good product.', 'review_id': 'R001'})
    assert response.json()['review_id'] == 'R001'

def test_predict_empty_text_returns_422():
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 422

def test_predict_missing_text_field_returns_422():
    response = client.post('/predict', json={})
    assert response.status_code == 422

def test_predict_html_text_is_handled():
    """HTML tags in review text should not crash the API."""
    response = client.post('/predict', json={'text': '<br/>Great product!<br/>'})
    assert response.status_code == 200
    assert response.json()['label'] in ['positive', 'negative']
