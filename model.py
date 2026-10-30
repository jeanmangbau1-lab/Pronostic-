import io
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
from math import factorial
from datetime import date, timedelta
import cv2
import webbrowser

st.set_page_config(page_title="Pronostics Football", layout="centered")

D1, D2, D3 = "1re division", "2e division", "3e division"
PAYS = {
    "Angleterre": {D1: "E0", D2: "E1", D3: "E2"},
    "Écosse": {D1: "SC0", D2: "SC1", D3: "SC2"},
    "Allemagne": {D1: "D1", D2: "D2"},
    "Italie": {D1: "I1", D2: "I2"},
    "Espagne": {D1: "SP1", D2: "SP2"},
    "France": {D1: "F1", D2: "F2"},
    "Pays-Bas": {D1: "N1"}, "Belgique": {D1: "B1"},
    "Portugal": {D1: "P1"}, "Turquie": {D1: "T1"}, "Grèce": {D1: "G1"},
    "Argentine": {D1: "ARG"}, "Autriche": {D1: "AUT"},
    "Brésil": {D1: "BRA"}, "Chine": {D1: "CHN"},
    "Danemark": {D1: "DNK"}, "Finlande": {D1: "FIN"},
    "Irlande": {D1: "IRL"}, "Japon": {D1: "JPN"},
    "Mexique": {D1: "MEX"}, "Norvège": {D1: "NOR"},
    "Pologne": {D1: "POL"}, "Roumanie": {D1: "ROU"},
    "Russie": {D1: "RUS"}, "Suède": {D1: "SWE"},
    "Suisse": {D1: "SWZ"}, "États-Unis": {D1: "USA"},
}
TOUS_CODES = {c for divs in PAYS.values() for c in divs.values()}
SAISONS = ["2223", "2324", "2425", "2526", "2627"]
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
PAYS_NEW = {
    "ARG": "argentina", "AUT": "austria", "BRA": "brazil", "CHN": "china",
    "DNK": "denmark", "FIN": "finland", "IRL": "ireland", "JPN": "japan",
    "MEX": "mexico", "NOR": "norway", "POL": "poland", "ROU": "romania",
    "RUS": "russia", "SWE": "sweden", "SWZ": "switzerland", "USA": "usa",
}

COMPETITIONS_INTER = [
    "Coupe d'Afrique des Nations (CAN)",
    "Coupe du monde FIFA",
    "Coupe d'Asie des Nations",
    "Copa América",
    "Euro (UEFA)",
    "Gold Cup (CONCACAF)",
    "Ligue des nations UEFA",
    "Éliminatoires Coupe du monde",
    "Éliminatoires CAN",
    "Ligue des champions CAF",
    "Match amical international",
    "Autre compétition",
]

URL_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
URL_NEW_FIXTURES = "https://www.football-data.co.uk/new_league_fixtures.csv"
NAVIGATEUR = {"User-Agent": "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36"}

LOG = "predictions_log.csv"
KS = np.arange(10)
FACT = np.array([factorial(k) for k in range(10)], dtype=float)
MASQUE_OVER = (KS[:, None] + KS[None, :]) > 2
COLS_REC = ["Date", "HomeTeam", "AwayTeam", "lh", "la", "hg", "ag"]
COLS_CAL = ["code", "Date", "Time", "HomeTeam", "AwayTeam"]

def rec_vide():
    return pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "HomeTeam": pd.Series(dtype=object),
        "AwayTeam": pd.Series(dtype=object),
        "lh": pd.Series(dtype=float),
        "la": pd.Series(dtype=float),
        "hg": pd.Series(dtype=float),
        "ag": pd.Series(dtype=float)})

def cal_vide():
    return pd.DataFrame({
        "code": pd.Series(dtype=object),
        "Date": pd.Series(dtype="datetime64[ns]"),
        "Time": pd.Series(dtype=object),
        "HomeTeam": pd.Series(dtype=object),
        "AwayTeam": pd.Series(dtype=object)})

def lire_dates(s):
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

def nettoyer(d):
    d = d[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    d["Date"] = lire_dates(d["Date"])
    d = d.dropna()
    d["FTHG"] = d["FTHG"].astype(int)
    d["FTAG"] = d["FTAG"].astype(int)
    return d.sort_values("Date", kind="stable").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def charger(code):
    if len(code) == 3 and code.isalpha():
        try:
            b = pd.read_csv(f"https://www.football-data.co.uk/new/{code}.csv")
            b = b.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                                  "HG": "FTHG", "AG": "FTAG"})
            b = b[b["Season"].astype(str).str[:4].astype(int) >= 2022]
            return nettoyer(b)
        except Exception:
            return pd.DataFrame()
    morceaux = []
    for s in SAISONS:
        try:
            morceaux.append(pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv",
                encoding="latin-1", on_bad_lines="skip"))
        except Exception:
            pass
    if not morceaux:
        return pd.DataFrame()
    try:
        return nettoyer(pd.concat(morceaux, ignore_index=True))
    except Exception:
        return pd.DataFrame()

def telecharger_csv(url):
    r = requests.get(url, headers=NAVIGATEUR, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("latin-1")),
                       on_bad_lines="skip")

@st.cache_data(show_spinner=False, ttl=1800)
def calendrier_cache():
    morceaux, erreurs = [], []
    try:
        f = telecharger_csv(URL_FIXTURES)
        f["code"] = f["Div"]
        morceaux.append(f.reindex(columns=COLS_CAL))
    except Exception as e:
        erreurs.append(f"fixtures.csv : {type(e).__name__} {e}")
    try:
        n = telecharger_csv(URL_NEW_FIXTURES)
        n = n.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        inverse = {v: k for k, v in PAYS_NEW.items()}
        n["code"] = n["Country"].astype(str).str.strip().str.lower().map(inverse)
        morceaux.append(n.reindex(columns=COLS_CAL))
    except Exception as e:
        erreurs.append(f"new_league_fixtures.csv : {type(e).__name__} {e}")
    if not morceaux:
        raise RuntimeError(" | ".join(erreurs))
    c = pd.concat(morceaux, ignore_index=True)
    c["Date"] = lire_dates(c["Date"])
    c = c.dropna(subset=["code", "Date", "HomeTeam", "AwayTeam"])
    c = c[c["code"].isin(TOUS_CODES)]
    if c.empty:
        return cal_vide()
    return c.sort_values(["Date", "Time"]).reset_index(drop=True)

def calendrier():
    try:
        return calendrier_cache(), ""
    except Exception as e:
        return cal_vide(), f"{type(e).__name__} : {e}"

@st.cache_data(show_spinner=False)
def forces(d):
    d = d.copy()
    d["w"] = np.linspace(0.5, 1.0, len(d))
    mh = np.average(d["FTHG"], weights=d["w"])
    ma = np.average(d["FTAG"], weights=d["w"])
    moy = (mh + ma) / 2
    att, dfn = {}, {}
    for eq in set(d["HomeTeam"]) | set(d["AwayTeam"]):
        h = d[d["HomeTeam"] == eq]
        a = d[d["AwayTeam"] == eq]
        wtot = h["w"].sum() + a["w"].sum()
        marq = (np.sum(h["FTHG"] * h["w"]) + np.sum(a["FTAG"] * a["w"])) / wtot
        enc = (np.sum(h["FTAG"] * h["w"]) + np.sum(a["FTHG"] * a["w"])) / wtot
        att[eq] = marq / moy
        dfn[eq] = enc / moy
    return att, dfn, mh, ma

def lambdas_brutes(dom, ext, att, dfn, mh, ma):
    return att[dom] * dfn[ext] * mh, att[ext] * dfn[dom] * ma

def lambdas_manuelles(att_a, def_a, att_b, def_b, neutre):
    lh = (att_a + def_b) / 2
    la = (att_b + def_a) / 2
    if not neutre:
        lh *= 1.10
        la *= 0.95
    return max(lh, 0.05), max(la, 0.05)

def matrice(lh, la, nul=1.0):
    ph = np.exp(-lh) * lh ** KS / FACT
    pa = np.exp(-la) * la ** KS / FACT
    m = np.outer(ph, pa)
    m = m / m.sum()
    if nul != 1.0:
        m = m.copy()
        m[KS, KS] = m[KS, KS] * nul
        m = m / m.sum()
    return m

def issues(m):
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()

SOMME = KS[:, None] + KS[None, :]
REUSSITE = {
    "1": lambda h, a: h > a,
    "N": lambda h, a: h == a,
    "2": lambda h, a: h < a,
    "1N": lambda h, a: h >= a,
    "N2": lambda h, a: h <= a,
    "12": lambda h, a: h != a,
    "+1,5 buts": lambda h, a: h + a >= 2,
    "+2,5 buts": lambda h, a: h + a >= 3,
    "-2,5 buts": lambda h, a: h + a <= 2,
    "-3,5 buts": lambda h, a: h + a <= 3,
    "Les 2 marquent": lambda h, a: h > 0 and a > 0,
    "Une équipe ne marque pas": lambda h, a: h == 0 or a == 0,
}
POOL = ["1", "2", "1N", "N2", "12"]

def probas_marches(m):
    p1, pn, p2 = issues(m)
    deux = m[1:, 1:].sum()
    return {
        "1": p1, "N": pn, "2": p2,
        "1N": p1 + pn, "N2": pn + p2, "12": p1 + p2,
        "+1,5 buts": m[SOMME >= 2].sum(), "+2,5 buts": m[SOMME >= 3].sum(),
        "-2,5 buts": m[SOMME <= 2].sum(), "-3,5 buts": m[SOMME <= 3].sum(),
        "Les 2 marquent": deux, "Une équipe ne marque pas": 1 - deux,
    }

def meilleur_pari(pm):
    nom = max(POOL, key=lambda k: pm[k])
    return nom, pm[nom]

def prevoir(lh, la, corr):
    m = matrice(lh * corr["kh"], la * corr["ka"], corr["nul"])
    pm = probas_marches(m)
    pari, pari_p = meilleur_pari(pm)
    idx = np.argsort(m, axis=None)[::-1][:3]
    top3 = [(int(i // 10), int(i % 10), float(m[i // 10, i % 10])) for i in idx]
    return {"pm": pm, "top3": top3, "pari": pari, "pari_p": float(pari_p)}

CORR_NEUTRE = {"kh": 1.0, "ka": 1.0, "nul": 1.0, "n": 0}

@st.cache_data(show_spinner=False)
def backtest(d, n_max=300):
    n = len(d)
    k = min(n_max, int(n * 0.25))
    if k < 30:
        return rec_vide()
    train, test = d.iloc[: n - k], d.iloc[n - k:]
    att, dfn, mh, ma = forces(train)
    lignes = []
    for r in test.itertuples():
        if r.HomeTeam in att and r.AwayTeam in att:
            lh, la = lambdas_brutes(r.HomeTeam, r.AwayTeam, att, dfn, mh, ma)
            lignes.append((r.Date, r.HomeTeam, r.AwayTeam, lh, la, r.FTHG, r.FTAG))
    if not lignes:
        return rec_vide()
    return pd.DataFrame(lignes, columns=COLS_REC)

def resultats_journal(code, d):
    if not os.path.exists(LOG):
        return rec_vide()
    try:
        j = pd.read_csv(LOG)
        j = j[j["code"] == code].copy()
        if j.empty:
            return rec_vide()
        j["Date"] = pd.to_datetime(j["Date"], errors="coerce").astype("datetime64[ns]")
        res = d[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        res["Date"] = res["Date"].astype("datetime64[ns]")
        m = j.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
        if m.empty:
            return rec_vide()
        m = m.rename(columns={"FTHG": "hg", "FTAG": "ag"})
        return m[COLS_REC]
    except Exception:
        return rec_vide()

def corrections(rec):
    n = len(rec)
    if n < 30:
        return dict(CORR_NEUTRE, n=n)
    w = n / (n + 50)
    nul_pred = np.mean([np.trace(matrice(r.lh, r.la)) for r in rec.itertuples()])
    nul_obs = float((rec["hg"] == rec["ag"]).mean())
    kh = 1 + w * (rec["hg"].mean() / rec["lh"].mean() - 1)
    ka = 1 + w * (rec["ag"].mean() / rec["la"].mean() - 1)
    nul = 1 + w * (nul_obs / nul_pred - 1)
    return {"kh": float(np.clip(kh, 0.8, 1.25)),
            "ka": float(np.clip(ka, 0.8, 1.25)),
            "nul": float(np.clip(nul, 0.75, 1.35)), "n": n}

def carte(dom, ext, prev):
    pm = prev["pm"]
    st.markdown(f"**{dom} vs {ext}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("1", f"{pm['1'] * 100:.0f}%")
    c2.metric("N", f"{pm['N'] * 100:.0f}%")
    c3.metric("2", f"{pm['2'] * 100:.0f}%")
    d1, d2, d3 = st.columns(3)
    d1.metric("+1,5 buts", f"{pm['+1,5 buts'] * 100:.0f}%")
    d2.metric("+2,5 buts", f"{pm['+2,5 buts'] * 100:.0f}%")
    d3.metric("Les 2 marquent", f"{pm['Les 2 marquent'] * 100:.0f}%")
    st.success(f"Pari conseillé : **{prev['pari']}** ({prev['pari_p'] * 100:.0f}%)")
    st.caption("Scores probables : " + " | ".join(
        f"{i}-{j} ({p * 100:.0f}%)" for i, j, p in prev["top3"]))
    st.divider()

def lire_qr_code(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Impossible de décoder l'image."
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img)
        if data:
            return data, None
        else:
            return None, "Aucun QR code détecté sur l'image."
    except Exception as e:
        return None, f"Erreur lors de la lecture : {e}"

def saisie_manuelle(att, dfn, mh, ma, corr, nom_champ, code):
    if "matchs_manuels" not in st.session_state:
        st.session_state["matchs_manuels"] = []

    with st.expander("📸 Saisie manuelle, Scan QR Code Elephant Bet & Photo", expanded=True):
        
        st.markdown("### Scanner un ticket / Fiche Elephant Bet ou importer une image")
        photo_prise = st.camera_input("Prendre en photo un bulletin ou QR code")
        fichier_charge = st.file_uploader("Ou importer un fichier (Image, PDF...)", type=["jpg", "jpeg", "png", "pdf"])
        
        image_a_analyser = photo_prise if photo_prise else fichier_charge
        
        if image_a_analyser is not None:
            bytes_data = image_a_analyser.getvalue()
            data_qr, erreur_qr = lire_qr_code(bytes_data)
            
            if data_qr:
                st.success("🔍 **QR Code détecté avec succès !**")
                st.code(data_qr, language="text")
                if data_qr.startswith("http"):
                    st.info("Ce QR code contient un lien vers Elephant Bet.")
                    if st.button("🌐 Ouvrir la vérification sur Elephant Bet"):
                        webbrowser.open(data_qr)
                else:
                    st.info("Code de réservation ou identifiant détecté.")
            else:
                st.warning(f"⚠️ {erreur_qr} (L'analyse manuelle reste disponible ci-dessous).")

        st.markdown("---")
        options = list(COMPETITIONS_INTER)
        if att:
            options = [f"Championnat : {nom_champ}"] + options
        comp = st.selectbox("Compétition", options, key="man_comp")
        nouveau = None

        if att and comp.startswith("Championnat"):
            equipes = sorted(att.keys())
            dom = st.selectbox("Équipe à domicile", equipes, key=f"man_dom_{code}")
            ext = st.selectbox("Équipe à l'extérieur", equipes,
                               index=1 if len(equipes) > 1 else 0,
                               key=f"man_ext_{code}")
            if st.button("Calculer le pronostic", key="man_btn_champ"):
                if dom == ext:
                    st.warning("Choisis deux équipes différentes.")
                else:
                    lh, la = lambdas_brutes(dom, ext, att, dfn, mh, ma)
                    nouveau = (comp, dom, ext, prevoir(lh, la, corr))
        else:
            nom_a = st.text_input("Équipe 1", key="man_nom_a")
            nom_b = st.text_input("Équipe 2", key="man_nom_b")
            neutre = st.checkbox("Terrain neutre", value=True, key="man_neutre")
            c1, c2 = st.columns(2)
            with c1:
                att_a = st.number_input("Buts marqués (éq. 1)", 0.0, 6.0, 1.4, 0.1, key="man_att_a")
                def_a = st.number_input("Buts encaissés (éq. 1)", 0.0, 6.0, 1.0, 0.1, key="man_def_a")
            with c2:
                att_b = st.number_input("Buts marqués (éq. 2)", 0.0, 6.0, 1.2, 0.1, key="man_att_b")
                def_b = st.number_input("Buts encaissés (éq. 2)", 0.0, 6.0, 1.1, 0.1, key="man_def_b")
            if st.button("Calculer le pronostic", key="man_btn_inter"):
                if not nom_a.strip() or not nom_b.strip():
                    st.warning("Écris le nom des deux équipes.")
                elif nom_a.strip().lower() == nom_b.strip().lower():
                    st.warning("Écris deux équipes différentes.")
                else:
                    lh, la = lambdas_manuelles(att_a, def_a, att_b, def_b, neutre)
                    nouveau = (comp, nom_a.strip(), nom_b.strip(), prevoir(lh, la, CORR_NEUTRE))

        if nouveau:
            st.session_state["matchs_manuels"].insert(0, nouveau)

        for titre, d, e, prev in st.session_state["matchs_manuels"]:
            st.caption(titre)
            carte(d, e, prev)

        if st.session_state["matchs_manuels"]:
            if st.button("Vider la liste", key="man_clear"):
                st.session_state["matchs_manuels"] = []
                st.rerun()

st.title("⚽ Pronostics Football")

pays = st.selectbox("Pays", list(PAYS.keys()))
divisions = PAYS[pays]
div_nom = st.selectbox("Division", list(divisions.keys()))
code = divisions[div_nom]
nom_champ = f"{pays} - {div_nom}"

ref = st.date_input("Choisis une date de la semaine", value=date.today())
lundi = ref - timedelta(days=ref.weekday())
dimanche = lundi + timedelta(days=6)
st.caption(f"Semaine du {lundi:%d/%m/%Y} au {dimanche:%d/%m/%Y}")
seuil = st.slider("Confiance minimale du pari conseillé (%)", 50, 90, 70, 5) / 100

zone_manuelle = st.container()

with st.spinner("Chargement..."):
    donnees = charger(code)
    cal, erreur_cal = calendrier()

if donnees.empty:
    st.warning("Impossible de charger les données automatiques de ce championnat. Utilisez la saisie manuelle, la caméra ou les fichiers ci-dessous.")
    with zone_manuelle:
        saisie_manuelle(None, None, None, None, CORR_NEUTRE, nom_champ, code)
    st.stop()

att, dfn, mh, ma = forces(donnees)
rec_log = resultats_journal(code, donnees)
rec_bt = backtest(donnees)
morceaux_rec = [x for x in (rec_log, rec_bt) if not x.empty]
rec = pd.concat(morceaux_rec, ignore_index=True) if morceaux_rec else rec_vide()
corr = corrections(rec)

with zone_manuelle:
    saisie_manuelle(att, dfn, mh, ma, corr, nom_champ, code)

cal = cal.copy()
cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce")
cal = cal.dropna(subset=["Date"])
cal = cal[(cal["code"] == code) & (cal["Date"] >= pd.Timestamp(lundi)) & (cal["Date"] <= pd.Timestamp(dimanche))]

onglet1, onglet2 = st.tabs(["Pronostics", "Apprentissage"])

with onglet1:
    st.subheader("Programme des matchs")
    if erreur_cal:
        st.info("Calendrier automatique indisponible. Vous pouvez utiliser la prise de photo ou la saisie manuelle.")
    elif cal.empty:
        st.info("Aucun match programmé pour cette période dans ce championnat.")
    else:
        for _, m in cal.iterrows():
            dom, ext = m["HomeTeam"], m["AwayTeam"]
            if dom in att and ext in att:
                lh, la = lambdas_brutes(dom, ext, att, dfn, mh, ma)
                prev = prevoir(lh, la, corr)
                if prev["pari_p"] >= seuil:
                    carte(dom, ext, prev)
