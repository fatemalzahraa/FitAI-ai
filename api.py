import os
import re
import logging
from typing import Optional, List, Dict, Any

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from sentiment import duygu_skoru_hesapla, tema_bul, tekrar_kontrol
from talimat_engine import talimat_isle, talimat_kaydet, geri_al

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("FITAI_BACKEND_URL", "https://localhost:44399")
SEND_TO_BACKEND = os.getenv("FITAI_SEND_TO_BACKEND", "false").lower() == "true"

app = FastAPI(title="FitAI AI Servisi", version="3.0")

model = joblib.load("model.pkl")
le_vucut = joblib.load("le_vucut.pkl")
le_kesim = joblib.load("le_kesim.pkl")

class AnalizIstegi(BaseModel):
    urunId: Optional[int] = None
    vucutTipi: str
    urunKesim: str
    kumasEsnek: int = Field(ge=0, le=1)
    mevcutBeden: Optional[str] = None
    yorumOzeti: Optional[str] = None

class Yorum(BaseModel):
    yorumMetni: str
    urunId: int
    magazaId: Optional[int] = 1

class TopluYorumIstegi(BaseModel):
    yorumlar: List[Yorum]

class TalimatIstegi(BaseModel):
    talimat: str


def _risk_puani_hesapla(skor: float, kumas_esnek: int, yorum_ozeti: Optional[str]) -> int:
    risk = max(0, min(100, int(round(100 - skor))))
    if kumas_esnek == 0:
        risk += 8
    if yorum_ozeti:
        y = yorum_ozeti.lower()
        if any(k in y for k in ["dar", "küçük", "kucuk", "beden", "iade", "kalıp"]):
            risk += 12
        if any(k in y for k in ["bol", "büyük", "buyuk"]):
            risk += 6
    return max(0, min(100, risk))


def _risk_etiketi(risk_puani: int) -> str:
    if risk_puani <= 30:
        return "Düşük"
    if risk_puani <= 60:
        return "Orta"
    return "Yüksek"


def _beden_oner(mevcut_beden: Optional[str], risk_puani: int, yorum_ozeti: Optional[str]) -> str:
    bedenler = ["XS", "S", "M", "L", "XL", "XXL"]
    beden = (mevcut_beden or "M").upper()
    if beden not in bedenler:
        beden = "M"
    idx = bedenler.index(beden)
    y = (yorum_ozeti or "").lower()
    if any(k in y for k in ["dar", "küçük", "kucuk", "kalıp küçük"]):
        idx = min(len(bedenler) - 1, idx + 1)
    elif any(k in y for k in ["bol", "büyük", "buyuk", "kalıp büyük"]):
        idx = max(0, idx - 1)
    elif risk_puani > 65:
        idx = min(len(bedenler) - 1, idx + 1)
    return bedenler[idx]


def _tavsiye(vucut_tipi: str, urun_kesim: str, skor: float, recommended_size: str, risk: str) -> str:
    if skor >= 75:
        uyum = "yüksek uyum bekleniyor"
    elif skor >= 55:
        uyum = "orta seviye uyum bekleniyor"
    else:
        uyum = "uyum riski var"
    return f"{vucut_tipi} vücut tipi için {urun_kesim} kesimde {uyum}. Önerilen beden: {recommended_size}. İade riski: {risk}."


def backend_gonder(path: str, payload: Dict[str, Any]) -> bool:
    if not SEND_TO_BACKEND:
        return False
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=10, verify=False)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Backend ulaşılamadı veya endpoint hazır değil: %s", e)
        return False

@app.get("/")
def home():
    return {"message": "FitAI AI Servisi çalışıyor", "backendAktif": SEND_TO_BACKEND}

@app.post("/analiz")
def analiz(istek: AnalizIstegi, background_tasks: BackgroundTasks):
    try:
        v = le_vucut.transform([istek.vucutTipi])[0]
        k = le_kesim.transform([istek.urunKesim])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz vucutTipi veya urunKesim")

    girdi = pd.DataFrame([[v, k, istek.kumasEsnek]], columns=["vucut_encoded", "kesim_encoded", "kumas_esnek"])
    skor = round(float(model.predict(girdi)[0]), 1)
    risk_puani = _risk_puani_hesapla(skor, istek.kumasEsnek, istek.yorumOzeti)
    risk = _risk_etiketi(risk_puani)
    beden = _beden_oner(istek.mevcutBeden, risk_puani, istek.yorumOzeti)

    sonuc = {
        "urunId": istek.urunId,
        "vucutTipi": istek.vucutTipi,
        "urunKesim": istek.urunKesim,
        "uyumSkoru": skor,
        "iadeRiski": risk,
        "iadeRiskPuani": risk_puani,
        "recommendedSize": beden,
        "tavsiye": _tavsiye(istek.vucutTipi, istek.urunKesim, skor, beden, risk)
    }
    background_tasks.add_task(backend_gonder, "/api/vucut-uyum-skoru", sonuc)
    return sonuc

@app.post("/yorum-analiz/toplu")
def toplu_yorum_analiz(veri: TopluYorumIstegi, background_tasks: BackgroundTasks):
    bulgular = []
    for yorum in veri.yorumlar:
        duygu = duygu_skoru_hesapla(yorum.yorumMetni)
        tema = tema_bul(yorum.yorumMetni)
        oneri_map = {
            "beden": "Beden sorunu tekrar ediyor, beden tablosu güncellenmeli.",
            "kumas": "Kumaş kalitesi ve açıklaması kontrol edilmeli.",
            "iade": "İade sebebi ürün açıklamasıyla karşılaştırılmalı.",
            "kalip": "Kalıp bilgisi ürün detayına eklenmeli.",
        }
        bulgu = {
            "urunId": yorum.urunId,
            "magazaId": yorum.magazaId,
            "yorum": yorum.yorumMetni,
            "tema": tema,
            "duyguSkoru": duygu["skor"],
            "duyguEtiketi": duygu["etiket"],
            "tekrarSayisi": 1,
            "oneriMetni": oneri_map.get(tema, "Genel müşteri geri bildirimi incelenmeli."),
            "durum": "Tamamlandı"
        }
        bulgular.append(bulgu)
        background_tasks.add_task(backend_gonder, "/api/yorum-analiz", bulgu)
    return {"analizSonuclari": bulgular, "uyarilar": tekrar_kontrol(bulgular), "toplamYorum": len(bulgular)}

@app.post("/talimat")
def talimat(veri: TalimatIstegi, background_tasks: BackgroundTasks):
    sonuc = talimat_isle(veri.talimat)
    talimat_kaydet(sonuc)
    background_tasks.add_task(backend_gonder, "/api/ai-talimat", sonuc)
    return sonuc

@app.post("/talimat/undo")
def talimat_undo():
    sonuc = geri_al()
    if not sonuc:
        return {"durum": "Boş", "mesaj": "Geri alınacak talimat yok"}
    return {"durum": "GeriAlındı", "geriAlinanTalimat": sonuc}
