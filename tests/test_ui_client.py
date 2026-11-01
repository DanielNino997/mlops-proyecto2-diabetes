"""Pruebas del cliente de la interfaz (sin Streamlit), con la API simulada."""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import api_client  # noqa: E402


def _resp(status, body):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = body
    r.text = str(body)
    return r


def test_ok():
    body = {"prediction": 1, "probability": 0.7}
    with mock.patch("api_client.requests.post", return_value=_resp(200, body)) as post:
        res = api_client.post_predict("http://x:8000/", {"a": 1})
    assert res.ok and res.data == body
    assert post.call_args.kwargs["headers"] == {"X-Client": "streamlit"}
    assert post.call_args.args[0] == "http://x:8000/predict"


def test_422_friendly():
    detail = [
        {"type": "literal_error", "loc": ["body", "age"], "input": "70", "ctx": {"expected": "'[0-10)', '[10-20)'"}},
        {"type": "missing", "loc": ["body", "race"], "input": {}},
        {"type": "less_than_equal", "loc": ["body", "time_in_hospital"], "input": 99, "ctx": {"le": 14}},
    ]
    with mock.patch("api_client.requests.post", return_value=_resp(422, {"detail": detail})):
        res = api_client.post_predict("http://x", {})
    assert not res.ok and res.status_code == 422
    by_field = {e["campo"]: e for e in res.errors}
    assert "Opciones" in by_field["age"]["problema"] and by_field["age"]["valor enviado"] == "70"
    assert by_field["race"]["problema"] == "Falta este campo"
    assert "14" in by_field["time_in_hospital"]["problema"]


def test_connection_and_503():
    import requests
    with mock.patch("api_client.requests.post", side_effect=requests.exceptions.ConnectionError()):
        res = api_client.post_predict("http://x", {})
    assert not res.ok and "conectar" in res.message
    with mock.patch("api_client.requests.post", return_value=_resp(503, {"detail": "sin modelo"})):
        res = api_client.post_predict("http://x", {})
    assert res.status_code == 503 and res.message == "sin modelo"


def test_model_info_down():
    import requests
    with mock.patch("api_client.requests.get", side_effect=requests.exceptions.ConnectionError()):
        assert api_client.get_model_info("http://x") is None
