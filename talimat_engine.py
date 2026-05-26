import re
import joblib
import pandas as pd

model = joblib.load("model.pkl")
le_vucut = joblib.load("le_vucut.pkl")
le_kesim = joblib.load("le_kesim.pkl")
gecmis = []

def talimat_isle(talimat_metni):
    t = talimat_metni.lower()
    kesim = next((k for k in le_kesim.classes_ if k.lower() in t), None)
    vucut = next((v for v in le_vucut.classes_ if v.lower() in t), None)
    m = re.search(r"%\s*(\d+)", t)
    yuzde = int(m.group(1)) if m else 10
    artir = "artır" in t or "artir" in t or "yükselt" in t
    azalt = "azalt" in t or "düşür" in t or "dusur" in t
    if not (artir or azalt):
        return {"talimat": talimat_metni, "durum": "Anlaşılamadı", "hata": "Artır/azalt yönü bulunamadı", "sonuclar": []}
    durumlar = []
    for v in le_vucut.classes_:
        for k in le_kesim.classes_:
            if kesim and k != kesim: continue
            if vucut and v != vucut: continue
            durumlar.append((v, k, 1))
    sonuclar = []
    for v, k, esnek in durumlar:
        girdi = pd.DataFrame([[le_vucut.transform([v])[0], le_kesim.transform([k])[0], esnek]], columns=["vucut_encoded", "kesim_encoded", "kumas_esnek"])
        eski = float(model.predict(girdi)[0])
        yeni = eski * (1 + yuzde / 100) if artir else eski * (1 - yuzde / 100)
        yeni = max(0, min(100, round(yeni, 1)))
        sonuclar.append({"vucutTipi": v, "urunKesim": k, "eskiSkor": round(eski, 1), "yeniSkor": yeni, "degisim": round(yeni - eski, 1)})
    return {"talimat": talimat_metni, "durum": "Tamamlandı", "anlasilan": f"{'Artır' if artir else 'Azalt'} %{yuzde}", "etkilenen": len(sonuclar), "sonuclar": sonuclar}

def talimat_kaydet(sonuc):
    gecmis.append(sonuc)
    return len(gecmis)

def geri_al():
    if not gecmis:
        return None
    return gecmis.pop()
