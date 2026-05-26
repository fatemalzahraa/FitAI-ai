from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_analiz_recommended_size_var():
    r = client.post("/analiz", json={"urunId": 1, "vucutTipi": "Elma", "urunKesim": "Regular", "kumasEsnek": 1, "mevcutBeden": "M", "yorumOzeti": "beden küçük"})
    assert r.status_code == 200
    data = r.json()
    assert "recommendedSize" in data
    assert "iadeRiskPuani" in data


def test_yorum_analiz_toplu():
    r = client.post("/yorum-analiz/toplu", json={"yorumlar": [{"urunId": 1, "magazaId": 1, "yorumMetni": "Beden küçük geldi"}]})
    assert r.status_code == 200
    assert r.json()["toplamYorum"] == 1


def test_talimat():
    r = client.post("/talimat", json={"talimat": "Oversize skorunu %20 artır"})
    assert r.status_code == 200
    assert "sonuclar" in r.json()
