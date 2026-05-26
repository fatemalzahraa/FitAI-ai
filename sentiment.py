from collections import Counter
import os

_duygu_modeli = None

NEGATIF = ["kötü", "kotü", "dar", "küçük", "kucuk", "iade", "beğenmedim", "kalitesiz", "ince", "sorun", "yanlış"]
POZITIF = ["güzel", "guzel", "iyi", "harika", "kaliteli", "beğendim", "tam", "rahat", "mükemmel"]

def modeli_getir():
    global _duygu_modeli
    if _duygu_modeli is None:
        from transformers import pipeline
        _duygu_modeli = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")
    return _duygu_modeli

def duygu_skoru_hesapla(metin):
    if os.getenv("FITAI_USE_TRANSFORMERS", "false").lower() == "true":
        try:
            sonuc = modeli_getir()(metin)[0]
            if sonuc["label"].lower() == "positive":
                return {"etiket": "Olumlu", "skor": round(float(sonuc["score"]), 2)}
            return {"etiket": "Olumsuz", "skor": round(-float(sonuc["score"]), 2)}
        except Exception:
            pass
    m = metin.lower()
    p = sum(k in m for k in POZITIF)
    n = sum(k in m for k in NEGATIF)
    if p >= n and p > 0:
        return {"etiket": "Olumlu", "skor": min(1, round(0.55 + p * 0.15, 2))}
    if n > 0:
        return {"etiket": "Olumsuz", "skor": max(-1, round(-0.55 - n * 0.15, 2))}
    return {"etiket": "Nötr", "skor": 0.0}

temalar = {
    "beden": ["beden", "numara", "küçük", "kucuk", "büyük", "buyuk", "dar", "bol"],
    "kumas": ["kumaş", "kumas", "kalite", "ince", "kalın", "saglam", "sağlam"],
    "kalip": ["kalıp", "kalip", "kesim", "model", "şekil"],
    "iade": ["iade", "geri gönderdim", "geri gonderdim", "iade ettim"]
}

def tema_bul(metin):
    m = metin.lower()
    for tema, kelimeler in temalar.items():
        if any(k in m for k in kelimeler):
            return tema
    return "genel"

def tekrar_kontrol(bulgular, esik=3):
    sayac = Counter([b["tema"] for b in bulgular])
    return [f"UYARI: '{tema}' teması {sayi} kez tekrarlandı!" for tema, sayi in sayac.items() if sayi >= esik]
