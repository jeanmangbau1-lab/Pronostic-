import io
import os
import re
import difflib
from datetime import date, datetime
import requests
import streamlit as st
import pandas as pd
import numpy as np
from math import factorial

try:
    import cv2
except Exception:
    cv2 = None

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
SAISONS = ["2223", "2324", "2425", "2526", "2627"]

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

LOG = "predictions_log.csv"
KS = np.arange(10)
FACT = np.array([factorial(k) for k in range(10)], dtype=float)
SOMME = KS[:, None] + KS[None, :]
COLS_REC = ["Date", "HomeTeam", "AwayTeam", "lh", "la", "hg", "ag"]
MAX_AFFICHE = 20


def rec_vide():
    return pd.DataFrame({
        "Date": pd.Series(dtype="datetime64[ns]"),
        "HomeTeam": pd.Series(dtype=object),
        "AwayTeam": pd.Series(dtype=object),
        "lh": pd.Series(dtype=float),
        "la": pd.Series(dtype=float),
        "hg": pd.Series(dtype=float),
        "ag": pd.Series(dtype=float)})


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


# ---------------------------------------------------------------
# API EN DIRECT (résultats récents + calendrier)
# ---------------------------------------------------------------
FD_CODES = {"E0": "PL", "E1": "ELC", "D1": "BL1", "I1": "SA",
            "SP1": "PD", "F1": "FL1", "P1": "PPL", "N1": "DED"}
TSDB_IDS = {"E0": 4328, "E1": 4329, "D1": 4331, "I1": 4332, "SP1": 4335,
            "F1": 4334, "P1": 4344, "N1": 4337, "B1": 4338, "T1": 4339,
            "SC0": 4330}
ALIAS = {
    "manchester united": "Man United", "manchester city": "Man City",
    "newcastle united": "Newcastle", "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves", "tottenham hotspur": "Tottenham",
    "west ham united": "West Ham", "brighton hove albion": "Brighton",
    "leeds united": "Leeds", "leicester city": "Leicester",
    "paris saint germain": "Paris SG", "atletico madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao", "real betis": "Betis",
    "real sociedad": "Sociedad", "bayer 04 leverkusen": "Leverkusen",
    "borussia monchengladbach": "M'gladbach",
}


def _norm_nom(n):
    n = re.sub(r"[^\w\s']", " ", str(n).lower())
    n = re.sub(r"\b(fc|cf|afc|ac|as|sc|ssc|rcd|cd|sv|fsv|vfb|vfl|1)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def nom_vers_csv(nom, equipes):
    n = _norm_nom(nom)
    a = ALIAS.get(n)
    if a and a in equipes:
        return a
    return trouver_equipe(n, equipes) or trouver_equipe(nom, equipes)


def _recents_fd(code):
    c = FD_CODES.get(code)
    try:
        cle = st.secrets["FOOTBALL_DATA_KEY"]
    except Exception:
        cle = None
    if not c or not cle:
        return []
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{c}/matches",
            params={"status": "FINISHED"},
            headers={"X-Auth-Token": cle}, timeout=8)
        r.raise_for_status()
        out = []
        for m in r.json().get("matches", []):
            ft = (m.get("score") or {}).get("fullTime") or {}
            if ft.get("home") is None or ft.get("away") is None:
                continue
            out.append((m["utcDate"][:10], m["homeTeam"]["name"],
                        m["awayTeam"]["name"], ft["home"], ft["away"]))
        return out
    except Exception:
        return []


def _recents_tsdb(code):
    i = TSDB_IDS.get(code)
    if not i:
        return []
    try:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/123/eventspastleague.php",
            params={"id": i}, timeout=8)
        r.raise_for_status()
        out = []
        for e in (r.json().get("events") or []):
            if e.get("intHomeScore") is None or e.get("intAwayScore") is None:
                continue
            out.append((e["dateEvent"], e["strHomeTeam"], e["strAwayTeam"],
                        int(e["intHomeScore"]), int(e["intAwayScore"])))
        return out
    except Exception:
        return []


def ajouter_recents(d, code):
    """Ajoute les matchs récents des API. Si tout échoue, renvoie d inchangé."""
    try:
        if d.empty:
            return d
        equipes = set(d["HomeTeam"]) | set(d["AwayTeam"])
        limite = d["Date"].max() - pd.Timedelta(days=14)
        connus = {}
        for r in d[d["Date"] >= limite].itertuples():
            connus.setdefault((r.HomeTeam, r.AwayTeam), []).append(r.Date)
        new = []
        for dt, h, a, hg, ag in _recents_fd(code) + _recents_tsdb(code):
            dt = pd.to_datetime(dt, errors="coerce")
            if pd.isna(dt) or dt < limite:
                continue
            h2, a2 = nom_vers_csv(h, equipes), nom_vers_csv(a, equipes)
            if not h2 or not a2 or h2 == a2:
                continue
            if any(abs((dt - x).days) <= 2 for x in connus.get((h2, a2), [])):
                continue
            connus.setdefault((h2, a2), []).append(dt)
            new.append((dt, h2, a2, int(hg), int(ag)))
        if not new:
            return d
        add = pd.DataFrame(new, columns=["Date", "HomeTeam", "AwayTeam",
                                         "FTHG", "FTAG"])
        return (pd.concat([d, add], ignore_index=True)
                .sort_values("Date", kind="stable").reset_index(drop=True))
    except Exception:
        return d


@st.cache_data(show_spinner=False, ttl=600)
def prochains_tsdb(code):
    i = TSDB_IDS.get(code)
    if not i:
        return []
    try:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/123/eventsnextleague.php",
            params={"id": i}, timeout=8)
        r.raise_for_status()
        return [(e["strHomeTeam"], e["strAwayTeam"], e.get("dateEvent", ""))
                for e in (r.json().get("events") or [])]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=600)
def matchs_du_jour_tsdb(code, jour_iso):
    i = TSDB_IDS.get(code)
    if not i:
        return []
    try:
        r = requests.get(
            "https://www.thesportsdb.com/api/v1/json/123/eventsday.php",
            params={"d": jour_iso, "s": "Soccer"}, timeout=8)
        r.raise_for_status()
        return [(e["strHomeTeam"], e["strAwayTeam"], (e.get("strTime") or "")[:5])
                for e in (r.json().get("events") or [])
                if str(e.get("idLeague")) == str(i)]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
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
        d = nettoyer(pd.concat(morceaux, ignore_index=True))
    except Exception:
        return pd.DataFrame()
    return ajouter_recents(d, code)


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
    return {"pm": pm, "top3": top3, "pari": pari, "pari_p": float(pari_p), "m": m}


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


def carte(titre, ext, prev):
    pm = prev["pm"]
    st.markdown(f"**{titre} vs {ext}**")
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


# ---------------------------------------------------------------
# OUTILS : OCR, QR, noms d'équipes
# ---------------------------------------------------------------
PROG_CACHE = "programme_cache.csv"
TICKETS = "tickets_log.csv"
COLS_PROG = ["cat", "date", "hora", "id", "home", "away", "o1", "ox", "o2", "source"]

# Codes de catégories du programme Elephant Bet -> championnats avec données
CAT_CODES = {
    "ESP1": "SP1", "ESP2": "SP2", "ALE1": "D1", "ALE2": "D2", "POR1": "P1",
    "FRA1": "F1", "FRA2": "F2", "BÉL1": "B1", "TUR1": "T1", "INGLA1": "E0",
    "INGLA2": "E1", "INGLA3": "E2", "ESCÓ1": "SC0", "ESCÓ2": "SC1",
    "ITA1": "I1", "ITA2": "I2", "HOL1": "N1", "GRÉ1": "G1", "NOR1": "NOR",
    "DINA1": "DNK", "POL1": "POL", "ROM1": "ROU", "IRL1": "IRL",
    "ARG1": "ARG", "ÁUSTRI1": "AUT", "BRA1": "BRA", "MEX1A": "MEX",
    "SUÉ1": "SWE", "FIN1": "FIN", "USA1": "USA", "JAP1": "JPN",
    "RÚS1": "RUS", "CHI1": "CHN", "SUÍÇA1": "SWZ",
}

CHOIX_LISTE = ["1", "N", "2", "1N", "N2", "12", "+1,5 buts", "+2,5 buts",
               "-2,5 buts", "-3,5 buts", "Les 2 marquent"]
CHOIX_MAP = {"1": "1", "N": "N", "2": "2", "1N": "1N", "N2": "N2", "12": "12",
             "+1,5 buts": ("O", 1.5), "+2,5 buts": ("O", 2.5),
             "-2,5 buts": ("U", 2.5), "-3,5 buts": ("U", 3.5),
             "Les 2 marquent": "BTTS"}

SEP = re.compile(r"\s+(?:-|–|—|vs\.?|v|contre|x)\s+", re.I)
RE_DATE_S = re.compile(r"^\d{2}/\d{2}/\d{4}$")
RE_DATE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})\D{0,3}(\d{1,2}[:h.]\d{2})")
RE_COTE = re.compile(r"=>\s*@?\s*(\d+[.,]\s?\d+)")
RE_ID = re.compile(r"\[\s*(\d{2,5})\s*\]")


def num(x):
    try:
        return float(str(x).replace(",", ".").strip())
    except Exception:
        return None


def prog_vide():
    return pd.DataFrame(columns=COLS_PROG)


def sim(a, b):
    a, b = str(a).lower().strip(), str(b).lower().strip()
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if len(a) >= 4 and len(b) >= 4 and (a in b or b in a):
        r = max(r, 0.85)
    return r


def trouver_equipe(nom, equipes):
    bas = {e.lower(): e for e in equipes}
    n = str(nom).lower().strip()
    if n in bas:
        return bas[n]
    for k, v in bas.items():
        if len(n) >= 4 and len(k) >= 4 and (n in k or k in n):
            return v
    r = difflib.get_close_matches(n, list(bas), n=1, cutoff=0.65)
    return bas[r[0]] if r else None


def lire_qr(octets):
    if cv2 is None:
        return ""
    try:
        img = cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        det = cv2.QRCodeDetector()
        for rot in (None, cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            x = img if rot is None else cv2.rotate(img, rot)
            d, _, _ = det.detectAndDecode(x)
            if d:
                return d.strip()
    except Exception:
        pass
    return ""


def _score_ocr(t):
    return (3 * len(re.findall(r"\d{1,2}/\d{1,2}/\d{4}", t))
            + 2 * t.count("=>")
            + len(re.findall(r"\d{1,2}[:.]\d{2}\D{0,4}\d{3,4}", t))
            + 3 * ("alor" in t))


@st.cache_data(show_spinner=False)
def ocr_image(octets, cote_final=3000, angles=(-10, 0, 10)):
    """Lit le texte d'une photo (essaie plusieurs angles, garde le meilleur)."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(octets)).convert("L")
    except Exception:
        return ""

    def echelle(im, cote):
        f = cote / max(im.size)
        return im.resize((max(1, int(im.size[0] * f)), max(1, int(im.size[1] * f))),
                         Image.LANCZOS)

    def pivote(im, a):
        return im if a == 0 else im.rotate(a, expand=True, fillcolor=255,
                                           resample=Image.BICUBIC)
    try:
        petit = echelle(img, 1000)
        meilleur, angle = -1, 0
        if len(angles) > 1:
            for a in angles:
                t = pytesseract.image_to_string(pivote(petit, a), config="--psm 6")
                s = _score_ocr(t)
                if s > meilleur:
                    meilleur, angle = s, a
        else:
            angle = angles[0]
        return pytesseract.image_to_string(pivote(echelle(img, cote_final), angle),
                                           config="--psm 6")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def extraire_texte_pdf(octets):
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(octets)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return ""


# ---------------------------------------------------------------
# PROGRAMME : PDF (précis) et photo (approximatif)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def lire_programme_pdf(octets):
    try:
        import pdfplumber
    except Exception:
        return prog_vide()
    lignes, date_cour = [], None
    try:
        with pdfplumber.open(io.BytesIO(octets)) as pdf:
            for page in pdf.pages:
                dates = sorted((w["top"], w["text"]) for w in page.extract_words()
                               if RE_DATE_S.match(w["text"]))
                for tb in page.find_tables():
                    data = tb.extract()
                    for i, row in enumerate(tb.rows):
                        top = row.bbox[1]
                        for y, d in dates:
                            if y <= top + 1:
                                date_cour = d
                        v = [(c or "").replace("\n", " ").strip() for c in data[i][:8]]
                        if len(v) < 8:
                            continue
                        if not re.match(r"^\d{1,2}:\d{2}$", v[1]) or not v[2].isdigit():
                            continue
                        lignes.append([v[0], date_cour, v[1], v[2],
                                       re.sub(r"\s+", " ", v[3]),
                                       re.sub(r"\s+", " ", v[4]),
                                       num(v[5]), num(v[6]), num(v[7]), "PDF"])
    except Exception:
        return prog_vide()
    if not lignes:
        return prog_vide()
    df = pd.DataFrame(lignes, columns=COLS_PROG)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
    return df


def nettoyer_nom(s):
    mots = [w for w in re.split(r"\s+", s) if re.search(r"[A-ZÀ-Ý]", w)
            and not re.fullmatch(r"[\d.,]+", w)]
    return " ".join(mots).strip(" -|[]")


def lire_programme_photo(texte, jour):
    lignes = []
    for l in texte.splitlines():
        m = re.search(r"(\d{1,2})[:.](\d{2})\D{0,6}?(\d{3,4})\D{0,3}(.+)$", l)
        if not m:
            continue
        reste = m.group(4)
        parts = [nettoyer_nom(p) for p in re.split(r"\s*[|\[\]]+\s*", reste)]
        parts = [p for p in parts if len(p) >= 3]
        if len(parts) < 2:
            continue
        h, mi = int(m.group(1)), m.group(2)
        hora = f"{h:02d}:{mi}" if h <= 23 else ""
        cat = re.sub(r"[^A-Za-zÀ-ÿ0-9]", "", l[:m.start()])[-8:].upper()
        lignes.append([cat, jour, hora, m.group(3), parts[0], parts[1],
                       None, None, None, "Photo"])
    return pd.DataFrame(lignes, columns=COLS_PROG) if lignes else prog_vide()


# ---------------------------------------------------------------
# ANALYSE
# ---------------------------------------------------------------
def ligue(code):
    d = charger(code)
    if d.empty:
        return None
    return forces(d)


@st.cache_data(show_spinner=False)
def corr_pour(code):
    d = charger(code)
    if d.empty:
        return dict(CORR_NEUTRE)
    parts = [x for x in (resultats_journal(code, d), backtest(d)) if not x.empty]
    rec = pd.concat(parts, ignore_index=True) if parts else rec_vide()
    return corrections(rec)


def analyse_match(home, away, cat):
    """Modèle statistique si les deux équipes sont reconnues, sinon None."""
    code = CAT_CODES.get(str(cat).strip().upper()) or CAT_CODES.get(str(cat).strip())
    if not code:
        return None
    L = ligue(code)
    if not L:
        return None
    att, dfn, mh, ma = L
    eq = list(att)
    x, y = trouver_equipe(home, eq), trouver_equipe(away, eq)
    if not x or not y or x == y:
        return None
    lh, la = lambdas_brutes(x, y, att, dfn, mh, ma)
    return {"dom": x, "ext": y, "code": code, "prev": prevoir(lh, la, corr_pour(code))}


def cotes_valides(o1, ox, o2):
    try:
        return all(o is not None and not pd.isna(o) and o > 1 for o in (o1, ox, o2))
    except Exception:
        return False


def probas_cotes(o1, ox, o2):
    if not cotes_valides(o1, ox, o2):
        return None
    inv = np.array([1 / o1, 1 / ox, 1 / o2])
    return inv / inv.sum()


def probas_finales(o1, ox, o2, prev):
    imp = probas_cotes(o1, ox, o2)
    mod = None
    if prev is not None:
        mod = np.array([prev["pm"]["1"], prev["pm"]["N"], prev["pm"]["2"]])
    if imp is not None and mod is not None:
        return (imp + mod) / 2, imp, mod, "cotes + modèle"
    if mod is not None:
        return mod, None, mod, "modèle seul"
    if imp is not None:
        return imp, imp, None, "cotes seules"
    return None, None, None, "aucune donnée"


def proba_choix(m, ch):
    if ch == "1":
        return float(issues(m)[0])
    if ch == "N":
        return float(issues(m)[1])
    if ch == "2":
        return float(issues(m)[2])
    if ch == "1N":
        return float(issues(m)[0] + issues(m)[1])
    if ch == "N2":
        return float(issues(m)[1] + issues(m)[2])
    if ch == "12":
        return float(issues(m)[0] + issues(m)[2])
    if ch == "BTTS":
        return float(m[1:, 1:].sum())
    if isinstance(ch, tuple):
        k, x = ch
        return float(m[SOMME > x].sum() if k == "O" else m[SOMME < x].sum())
    return None


def libelle_choix(ch, a, b):
    if ch == "1":
        return f"Victoire {a}"
    if ch == "2":
        return f"Victoire {b}"
    if ch == "N":
        return "Match nul"
    if ch == "1N":
        return f"{a} ou nul"
    if ch == "N2":
        return f"Nul ou {b}"
    if ch == "12":
        return "Pas de nul"
    if ch == "BTTS":
        return "Les 2 marquent"
    if isinstance(ch, tuple):
        return f"{'+' if ch[0] == 'O' else '-'}{ch[1]:g} buts"
    return "?"


def interpreter_choix(sel, a, b):
    s = sel.lower().strip()
    mn = re.search(r"(\d+(?:[.,]\d+)?)", s)
    x = float(mn.group(1).replace(",", ".")) if mn else None
    if ("mais" in s or "over" in s or "acima" in s) and x is not None:
        return ("O", x)
    if ("menos" in s or "under" in s or "abaixo" in s) and x is not None:
        return ("U", x)
    if "ambas" in s or "both" in s or "btts" in s:
        return "BTTS"
    if "empate" in s or s in ("x", "draw", "n"):
        return "N"
    if not s:
        return None
    ra, rb = sim(s, a), sim(s, b)
    if max(ra, rb) < 0.4:
        return None
    return "1" if ra >= rb else "2"


def verdict(v):
    if v >= 1.05:
        return "✅ Cote intéressante : le modèle estime la probabilité plus haute que la cote"
    if v >= 0.90:
        return "➖ Cote correcte"
    return "⚠️ Cote basse pour le risque"


# ---------------------------------------------------------------
# TICKET : lecture
# ---------------------------------------------------------------
def nettoyer_equipe(s):
    s = re.sub(r"^\W*\w{0,3}[)\]|]\s*", "", s.strip())
    return s.strip(" -|")


def lire_ticket(texte):
    lignes = [re.sub(r"\s+", " ", l).strip() for l in texte.splitlines()]
    legs, cur = [], None
    for l in lignes:
        if not l:
            continue
        md = RE_DATE.search(l)
        if md:
            mid = RE_ID.search(l)
            try:
                dt = datetime.strptime(md.group(1), "%d/%m/%Y")
                ok_date = 2020 <= dt.year <= 2035
            except ValueError:
                ok_date = False
            cur = {"date": md.group(1) if ok_date else "",
                   "heure": re.sub(r"[h.]", ":", md.group(2)),
                   "id": mid.group(1) if mid else "", "a": None, "b": None,
                   "marche": "", "sel": "", "cote": None}
            legs.append(cur)
            continue
        mc = RE_COTE.search(l)
        p = None if mc else SEP.split(l, maxsplit=1)
        est_eq = (p is not None and len(p) == 2
                  and 2 <= len(p[0].strip()) <= 35 and 2 <= len(p[1].strip()) <= 35)
        if est_eq:
            if cur is None or cur["a"] is not None:
                cur = {"date": "", "heure": "", "id": "", "a": None, "b": None,
                       "marche": "", "sel": "", "cote": None}
                legs.append(cur)
            cur["a"], cur["b"] = nettoyer_equipe(p[0]), nettoyer_equipe(p[1])
        elif mc and cur is not None:
            cur["sel"] = l[:mc.start()].strip()
            cur["cote"] = num(re.sub(r"\s", "", mc.group(1)))
        elif cur is not None and cur["a"] and not cur["marche"]:
            cur["marche"] = l
    out = []
    for g in legs:
        if g["a"] and g["cote"]:
            g["choix"] = interpreter_choix(g["sel"], g["a"], g["b"])
            out.append(g)
    return out


def lire_mise(texte):
    m = re.search(r"valor\s+total\D{0,10}?(\d[\d\s]*)\s*kz", texte, re.I)
    if m:
        return num(re.sub(r"\s", "", m.group(1)))
    return None


def trouver_dans_programme(leg, prog):
    if prog is None or prog.empty:
        return None
    if leg.get("id"):
        r = prog[prog["id"].astype(str) == str(leg["id"])]
        if len(r):
            return r.iloc[0]
    best, meilleur = None, 0.0
    for r in prog.itertuples():
        s = (sim(leg["a"], r.home) + sim(leg["b"], r.away)) / 2
        if s > meilleur:
            best, meilleur = r, s
    if best is not None and meilleur >= 0.72:
        return prog[(prog["id"].astype(str) == str(best.id))
                    & (prog["home"] == best.home)].iloc[0]
    return None


def analyser_leg(leg, prog):
    row = trouver_dans_programme(leg, prog)
    res = {"row": row, "p_mod": None, "p_imp": None, "prev": None}
    ch = leg.get("choix")
    if row is None or ch is None:
        return res
    a = analyse_match(row["home"], row["away"], row["cat"])
    if a:
        res["prev"] = a["prev"]
        res["p_mod"] = proba_choix(a["prev"]["m"], ch)
    if ch in ("1", "N", "2"):
        pi = probas_cotes(row["o1"], row["ox"], row["o2"])
        if pi is not None:
            res["p_imp"] = float(pi[{"1": 0, "N": 1, "2": 2}[ch]])
    return res


def proba_utilisee(res, cote):
    if res["p_mod"] is not None and res["p_imp"] is not None:
        return (res["p_mod"] + res["p_imp"]) / 2, "cotes + modèle"
    if res["p_mod"] is not None:
        return res["p_mod"], "modèle"
    if res["p_imp"] is not None:
        return res["p_imp"], "cotes du programme"
    return 1.0 / cote, "approximation (pas d'analyse)"


# ---------------------------------------------------------------
# HISTORIQUE DES TICKETS
# ---------------------------------------------------------------
COLS_TK = ["code", "enregistre_le", "mise", "cote_totale", "nb_matchs",
           "proba_totale", "statut", "matchs"]


def charger_tickets():
    if not os.path.exists(TICKETS):
        return pd.DataFrame(columns=COLS_TK)
    try:
        df = pd.read_csv(TICKETS)
        for c in COLS_TK:
            if c not in df.columns:
                df[c] = ""
        return df[COLS_TK]
    except Exception:
        return pd.DataFrame(columns=COLS_TK)


def sauver_tickets(df):
    try:
        df.to_csv(TICKETS, index=False)
        return True
    except Exception:
        return False


def miniatures(fichiers):
    if not fichiers:
        return
    cols = st.columns(min(len(fichiers), 5))
    for i, f in enumerate(fichiers):
        with cols[i % len(cols)]:
            try:
                st.image(f.getvalue(), width=64)
            except Exception:
                st.caption("🖼️")


# ---------------------------------------------------------------
# ONGLET 1 : PROGRAMME
# ---------------------------------------------------------------
def carte_programme(n, r, an):
    o1, ox, o2 = r["o1"], r["ox"], r["o2"]
    prev = an["prev"] if an else None
    fin, imp, mod, src = probas_finales(o1, ox, o2, prev)
    dom = an["dom"] if an else r["home"]
    ext = an["ext"] if an else r["away"]
    st.markdown(f"**{n}. {r['home']} vs {r['away']}**")
    st.caption(f"{r['cat'] or 'Autre'} · {r['date']:%d/%m} {r['hora']} · ID {r['id']}")
    if fin is None:
        st.warning("Pas de cotes ni de statistiques pour ces équipes : pas d'analyse.")
        st.divider()
        return
    cotes = [o1, ox, o2]
    c1, c2, c3 = st.columns(3)
    for col, nom, p, ct in zip((c1, c2, c3), ("1", "N", "2"), fin, cotes):
        lab = f"{nom} · {ct:.2f}" if ct and not pd.isna(ct) else nom
        col.metric(lab, f"{p * 100:.0f}%")
    k = int(np.argmax(fin))
    nom_k = ["1", "N", "2"][k]
    ct_k = f" — cote {cotes[k]:.2f}" if cotes[k] and not pd.isna(cotes[k]) else ""
    st.success(f"Favori : **{nom_k}** ({fin[k] * 100:.0f}%){ct_k}")
    if mod is not None and cotes_valides(o1, ox, o2):
        for j, nm in enumerate(("1", "N", "2")):
            if mod[j] * cotes[j] >= 1.08:
                st.info(f"⭐ Cote intéressante : **{nm}** (cote {cotes[j]:.2f}, "
                        f"modèle {mod[j] * 100:.0f}%)")
    if prev is not None:
        pm = prev["pm"]
        st.caption(f"+1,5 buts {pm['+1,5 buts'] * 100:.0f}% · "
                   f"+2,5 buts {pm['+2,5 buts'] * 100:.0f}% · "
                   f"Les 2 marquent {pm['Les 2 marquent'] * 100:.0f}%")
        st.caption("Scores probables : " + " | ".join(
            f"{i}-{j} ({p * 100:.0f}%)" for i, j, p in prev["top3"]))
    st.caption(f"Source de l'analyse : {src}")
    st.divider()


def construire_programme():
    st.markdown("##### 📄 Importer le programme")
    fichiers = st.file_uploader(
        "PDF (le plus précis) ou photos du programme",
        type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True, key="prog_up")
    photos = [f for f in (fichiers or []) if not f.name.lower().endswith(".pdf")]
    pdfs = [f for f in (fichiers or []) if f.name.lower().endswith(".pdf")]
    miniatures(photos)
    jour_photo = date.today()
    if photos:
        jour_photo = st.date_input("📆 Date des matchs de la photo", value=date.today(),
                                   key="prog_jour_photo")
    morceaux = []
    for f in pdfs:
        with st.spinner("Lecture du PDF..."):
            d = lire_programme_pdf(f.getvalue())
        if d.empty:
            st.warning(f"Impossible de lire le tableau de {f.name}.")
        morceaux.append(d)
    for f in photos:
        with st.spinner("Lecture de la photo (peut prendre quelques secondes)..."):
            texte = ocr_image(f.getvalue(), 4400, (0,))
        d = lire_programme_photo(texte, jour_photo)
        if d.empty:
            st.warning("Photo peu lisible : aucun match reconnu. "
                       "Prends-la bien à plat, ou utilise le PDF.")
        else:
            st.warning(f"📷 {len(d)} match(s) seulement reconnu(s) sur la photo, "
                       "avec des noms parfois incomplets. Pour tout le programme, "
                       "utilise le PDF : lecture exacte avec les cotes.")
        morceaux.append(d)
    morceaux = [m for m in morceaux if not m.empty]
    if morceaux:
        prog = pd.concat(morceaux, ignore_index=True).drop_duplicates(
            subset=["id", "home", "date"])
        try:
            prog.to_csv(PROG_CACHE, index=False)
        except Exception:
            pass
        return prog
    if os.path.exists(PROG_CACHE):
        try:
            prog = pd.read_csv(PROG_CACHE, dtype={"id": str})
            prog["date"] = pd.to_datetime(prog["date"], errors="coerce").dt.date
            prog["cat"] = prog["cat"].fillna("")
            st.caption("Dernier programme importé (gardé en mémoire).")
            return prog.dropna(subset=["date"])
        except Exception:
            pass
    return prog_vide()


def onglet_programme(prog):
    st.subheader("📅 Programme des matchs")
    if prog.empty:
        st.info("Importe le PDF du programme (ou une photo) ci-dessus pour voir les matchs.")
        return None
    prog = prog.copy()
    prog["id"] = prog["id"].astype(str)
    prog["cat"] = prog["cat"].fillna("").astype(str)
    dates = sorted(prog["date"].dropna().unique())
    defaut = date.today() if date.today() in dates else next(
        (d for d in dates if d >= date.today()), dates[0])
    st.caption("Dates dans le programme : " + " · ".join(
        f"{d:%d/%m} ({int((prog['date'] == d).sum())} matchs)" for d in dates))
    jour = st.date_input("📆 Choisis une date", value=defaut, key="prog_jour")
    jj = prog[prog["date"] == jour].sort_values(["hora", "id"])
    if jj.empty:
        st.warning("Aucun match ce jour-là. Dates disponibles : "
                   + ", ".join(f"{d:%d/%m}" for d in dates))
        return None

    with st.expander("🔎 Filtres", expanded=False):
        cats = sorted({c if c else "Autre" for c in jj["cat"]})
        choix = st.multiselect("Championnats / compétitions (tous par défaut)", cats,
                               default=[], key="prog_cats")
        q = st.text_input("Chercher une équipe", key="prog_q")
        seuil = st.slider("Confiance minimale du favori (%)", 0, 90, 0, 5,
                          key="prog_seuil") / 100
    if choix:
        jj = jj[jj["cat"].replace("", "Autre").isin(choix)]
    if q.strip():
        qq = q.strip().lower()
        jj = jj[jj["home"].str.lower().str.contains(qq, regex=False)
                | jj["away"].str.lower().str.contains(qq, regex=False)]
    if seuil > 0 and not jj.empty:
        def conf(r):
            p = probas_cotes(r["o1"], r["ox"], r["o2"])
            return 1.0 if p is None else float(p.max())
        jj = jj[jj.apply(conf, axis=1) >= seuil]

    nb = st.selectbox("Nombre de matchs affichés", [20, 50, 100, "Tous"], index=0,
                      key="prog_nb")
    total = len(jj)
    affiche = jj if nb == "Tous" else jj.head(int(nb))
    st.success(f"{total} match(s) le {jour:%d/%m/%Y} — affichage de {len(affiche)}")

    liste = []
    with st.spinner("Analyse des matchs..."):
        for n, (_, r) in enumerate(affiche.iterrows(), start=1):
            an = analyse_match(r["home"], r["away"], r["cat"])
            carte_programme(n, r, an)
            liste.append((r, an))
    return liste


def ticket_conseille(liste):
    st.subheader("🎯 Ticket conseillé")
    if not liste:
        st.caption("Affiche d'abord des matchs dans le programme.")
        return
    c1, c2 = st.columns(2)
    seuil = c1.slider("Confiance minimale (%)", 50, 90, 70, 5, key="tc_seuil") / 100
    nmax = c2.slider("Nombre de matchs", 1, 6, 3, key="tc_n")
    extra = st.checkbox("Inclure doubles chances et buts (si statistiques)",
                        value=False, key="tc_extra")
    cands = []
    for r, an in liste:
        prev = an["prev"] if an else None
        fin, imp, mod, src = probas_finales(r["o1"], r["ox"], r["o2"], prev)
        if fin is None:
            continue
        opts = [("1", fin[0], r["o1"]), ("N", fin[1], r["ox"]), ("2", fin[2], r["o2"])]
        if extra and prev is not None:
            pm = prev["pm"]
            for k in ("1N", "N2", "12", "+1,5 buts", "-3,5 buts"):
                opts.append((k, float(pm[k]), None))
        nom, p, ct = max(opts, key=lambda t: t[1])
        if p >= seuil:
            cands.append((f"{r['home']} vs {r['away']}", nom, float(p), ct))
    cands.sort(key=lambda t: -t[2])
    cands = cands[:nmax]
    if not cands:
        st.warning("Aucun match n'atteint cette confiance. Baisse le seuil.")
        return
    ptot, cotetot, reel = 1.0, 1.0, True
    for i, (nom_match, pari, p, ct) in enumerate(cands, start=1):
        if ct is None or pd.isna(ct):
            reel = False
            ct_aff = f"cote juste ≈ {1 / p:.2f}"
            cotetot *= 1 / p
        else:
            ct_aff = f"cote {ct:.2f}"
            cotetot *= ct
        ptot *= p
        st.markdown(f"**{i}. {nom_match}** → **{pari}** ({p * 100:.0f}%) · {ct_aff}")
    st.metric("Probabilité que tout passe", f"{ptot * 100:.0f}%")
    st.caption(("Cote combinée : " if reel else "Cote combinée indicative : ")
               + f"{cotetot:.2f}")
    st.caption("⚠️ Ce sont des probabilités, pas des certitudes : "
               "plus tu combines de matchs, plus le risque de perdre augmente.")


# ---------------------------------------------------------------
# ONGLET 2 : TICKET
# ---------------------------------------------------------------
def onglet_ticket(prog):
    st.subheader("🎫 Mon ticket")
    fichiers = st.file_uploader("📷 Photo(s) du ticket (à plat, sans doigt dessus)",
                                type=["jpg", "jpeg", "png"],
                                accept_multiple_files=True, key="tk_up")
    miniatures(fichiers)
    codes, textes = [], []
    for f in fichiers or []:
        octets = f.getvalue()
        q = lire_qr(octets)
        if q:
            codes.append(q)
        with st.spinner("Lecture du ticket (quelques secondes)..."):
            textes.append(ocr_image(octets))
    sig = "-".join(f"{f.name}{f.size}" for f in (fichiers or [])) or "vide"
    texte = st.text_area("Texte lu sur le ticket (modifiable)", value="\n".join(textes),
                         height=150, key=f"tk_txt_{sig}")
    legs = list(lire_ticket(texte))

    st.markdown("##### ➕ Ajouter des matchs depuis le programme")
    if prog is not None and not prog.empty:
        etiq = {f"{r['id']} · {r['date']:%d/%m} {r['hora']} · {r['home']} - {r['away']}": r
                for _, r in prog.sort_values(["date", "hora"]).iterrows()}
        sel = st.multiselect("Choisis les matchs de ton ticket", list(etiq), key="tk_sel")
        for lab in sel:
            r = etiq[lab]
            st.markdown(f"**{r['home']} - {r['away']}**")
            c1, c2 = st.columns(2)
            pari = c1.selectbox("Pari", CHOIX_LISTE, key=f"tk_p_{r['id']}")
            if pari in ("1", "N", "2"):
                defaut = {"1": r["o1"], "N": r["ox"], "2": r["o2"]}[pari]
                defaut = float(defaut) if defaut and not pd.isna(defaut) else 1.5
            else:
                defaut = 1.5
            cote = c2.number_input("Cote", 1.01, 1000.0, defaut, 0.01,
                                   key=f"tk_c_{r['id']}_{pari}")
            legs.append({"a": r["home"], "b": r["away"], "choix": CHOIX_MAP[pari],
                         "cote": float(cote), "id": str(r["id"]),
                         "date": f"{r['date']:%d/%m/%Y}", "heure": r["hora"],
                         "sel": pari, "marche": "", "src": "programme"})
    else:
        st.caption("Importe d'abord le programme (onglet Programme) pour choisir les matchs.")

    legs = [g for g in legs if g.get("choix") is not None]
    if not legs:
        st.info("Aucun match de ticket reconnu. Photographie le ticket bien à plat, "
                "corrige le texte, ou ajoute les matchs depuis le programme.")
        return
    st.success(f"{len(legs)} match(s) dans le ticket")
    mise_lue = lire_mise(texte)
    mise = st.number_input("Mise (Kz)", 0.0, 100000000.0, float(mise_lue or 200.0), 50.0,
                           key=f"tk_mise_{sig}")
    cote_tot, p_tot, resumes = 1.0, 1.0, []
    for i, g in enumerate(legs, start=1):
        res = analyser_leg(g, prog)
        p, src = proba_utilisee(res, g["cote"])
        cote_tot *= g["cote"]
        p_tot *= p
        lib = libelle_choix(g["choix"], g["a"], g["b"])
        resumes.append(f"{g['a']}-{g['b']} ({lib})")
        st.markdown(f"**{i}. {g['a']} vs {g['b']}**")
        st.caption(f"{g.get('date', '')} {g.get('heure', '')} — {lib} @ {g['cote']:.2f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité", f"{p * 100:.0f}%")
        c2.metric("Cote juste", f"{1 / p:.2f}")
        if res["p_mod"] is not None:
            v = res["p_mod"] * g["cote"]
            c3.metric("Valeur", f"{v:.2f}")
            st.caption(verdict(v))
        else:
            c3.metric("Valeur", "—")
            st.caption("Pas de statistiques pour ces équipes.")
        st.caption(f"Source : {src}")
        st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Cote totale", f"{cote_tot:.2f}")
    c2.metric("Gain possible", f"{mise * cote_tot:,.0f} Kz".replace(",", " "))
    c3.metric("Tout passe", f"{p_tot * 100:.1f}%")
    st.caption("Le gain réel peut inclure un bonus ou des taxes : vérifie le ticket. "
               "Vérifie aussi que tous les matchs du ticket sont dans la liste.")

    st.markdown("##### 🔢 Code du ticket")
    defaut_code = codes[0] if codes else ""
    if not defaut_code:
        m = re.search(r"\b[0-9A-F]{16}\b", texte)
        defaut_code = m.group(0) if m else ""
    code_tk = st.text_input("Code (QR ou écrit sous le QR)", value=defaut_code,
                            key=f"tk_code_{sig}")
    if st.button("💾 Enregistrer ce ticket", key="tk_save"):
        df = charger_tickets()
        ligne = {"code": code_tk, "enregistre_le": date.today().isoformat(),
                 "mise": mise, "cote_totale": round(cote_tot, 2),
                 "nb_matchs": len(legs), "proba_totale": round(p_tot, 4),
                 "statut": "En attente", "matchs": " ; ".join(resumes)}
        df = pd.concat([df, pd.DataFrame([ligne])], ignore_index=True)
        if sauver_tickets(df):
            st.success("Ticket enregistré.")
        else:
            st.error("Impossible d'enregistrer sur cet appareil.")


def historique():
    st.subheader("🗂️ Historique de mes tickets")
    df = charger_tickets()
    if df.empty:
        st.caption("Aucun ticket enregistré pour le moment.")
        return
    edite = st.data_editor(
        df, key="tk_hist", use_container_width=True, hide_index=True,
        disabled=[c for c in COLS_TK if c != "statut"],
        column_config={"statut": st.column_config.SelectboxColumn(
            "statut", options=["En attente", "Gagné", "Perdu"])})
    if st.button("💾 Sauver les statuts", key="tk_hist_save"):
        if sauver_tickets(edite):
            st.success("Statuts enregistrés.")
    g = edite[edite["statut"] == "Gagné"]
    p = edite[edite["statut"] == "Perdu"]
    net = float((g["mise"] * g["cote_totale"] - g["mise"]).sum() - p["mise"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gagnés", len(g))
    c2.metric("Perdus", len(p))
    c3.metric("En attente", int((edite["statut"] == "En attente").sum()))
    c4.metric("Bilan (Kz)", f"{net:,.0f}".replace(",", " "))


# ---------------------------------------------------------------
# ONGLET 3 : MANUEL / CHAMPIONNATS / INTERNATIONAL
# ---------------------------------------------------------------
LIBELLES_CHAMP = {f"🏆 {p} – {dv}": c for p, dvs in PAYS.items() for dv, c in dvs.items()}
LIBELLES_INTER = [f"🌍 {c}" for c in COMPETITIONS_INTER]


def saisir_cotes(cle):
    st.caption("Cotes (optionnel, 0 = pas de cote) — ex. Arsenal 1.90 · Liverpool 3.90")
    c1, c2, c3 = st.columns(3)
    o1 = c1.number_input("Cote 1", 0.0, 1000.0, 0.0, 0.05, key=f"cote1_{cle}")
    ox = c2.number_input("Cote N", 0.0, 1000.0, 0.0, 0.05, key=f"coteN_{cle}")
    o2 = c3.number_input("Cote 2", 0.0, 1000.0, 0.0, 0.05, key=f"cote2_{cle}")
    return (float(o1), float(ox), float(o2))


def probas_manuelles(prev, cotes):
    o1, ox, o2 = cotes
    mod = np.array([prev["pm"]["1"], prev["pm"]["N"], prev["pm"]["2"]])
    if cotes_valides(o1, ox, o2):
        return probas_cotes(o1, ox, o2), mod, "cotes 1, N et 2"
    if o1 > 1 and o2 > 1:
        a, b = 1 / o1, 1 / o2
        pn = mod[1]
        imp = np.array([(1 - pn) * a / (a + b), pn, (1 - pn) * b / (a + b)])
        return imp, mod, "cotes 1 et 2 (le nul est estimé par le modèle)"
    return None


def afficher_cotes_manuelles(prev, cotes):
    r = probas_manuelles(prev, cotes)
    if r is None:
        return
    imp, mod, note = r
    fin = (imp + mod) / 2
    st.markdown("**Avec tes cotes**")
    c1, c2, c3 = st.columns(3)
    for col, nom, p, ct in zip((c1, c2, c3), ("1", "N", "2"), fin, cotes):
        col.metric(f"{nom} · {ct:.2f}" if ct > 1 else nom, f"{p * 100:.0f}%")
    k = int(np.argmax(fin))
    st.success(f"Favori : **{['1', 'N', '2'][k]}** ({fin[k] * 100:.0f}%) "
               f"— cote juste {1 / fin[k]:.2f}")
    for j, nm in enumerate(("1", "N", "2")):
        if cotes[j] > 1 and mod[j] * cotes[j] >= 1.08:
            st.info(f"⭐ Cote intéressante : **{nm}** (cote {cotes[j]:.2f}, "
                    f"modèle {mod[j] * 100:.0f}%)")
    st.caption(f"Analyse faite avec les {note} et le modèle statistique.")


def onglet_manuel():
    st.subheader("✍️ Match manuel")
    if "matchs_manuels" not in st.session_state:
        st.session_state["matchs_manuels"] = []
    options = list(LIBELLES_CHAMP) + LIBELLES_INTER
    comp = st.selectbox("Compétition (tous les championnats et les matchs "
                        "internationaux)", options, key="man_comp")
    nouveau = None
    st.session_state["comp_code"] = None

    if comp in LIBELLES_CHAMP:
        code = LIBELLES_CHAMP[comp]
        st.session_state["comp_code"] = code
        with st.spinner("Chargement du championnat..."):
            L = ligue(code)
        if L:
            att, dfn, mh, ma = L
            equipes = sorted(att)
            idx_d, idx_e = 0, (1 if len(equipes) > 1 else 0)
            jour = st.date_input("📆 Calendrier des matchs", value=date.today(),
                                 key=f"man_jour_{code}")
            du_jour = matchs_du_jour_tsdb(code, jour.isoformat())
            ch = 0
            if du_jour:
                labels = ["— choisir un match —"] + [
                    f"{t} · {a} vs {b}" for a, b, t in du_jour]
                ch = labels.index(st.selectbox(
                    f"Matchs du {jour:%d/%m/%Y}", labels,
                    key=f"man_next_{code}_{jour}"))
                if ch:
                    x = nom_vers_csv(du_jour[ch - 1][0], equipes)
                    y = nom_vers_csv(du_jour[ch - 1][1], equipes)
                    if x:
                        idx_d = equipes.index(x)
                    if y:
                        idx_e = equipes.index(y)
            else:
                st.caption("Aucun match trouvé ce jour-là (ou API indisponible) : "
                           "choisis les équipes à la main.")
            dom = st.selectbox("Équipe à domicile", equipes, index=idx_d,
                               key=f"man_dom_{code}_{jour}_{ch}")
            ext = st.selectbox("Équipe à l'extérieur", equipes, index=idx_e,
                               key=f"man_ext_{code}_{ch}_{jour}")
            cotes = saisir_cotes("champ")
            if st.button("Calculer le pronostic", key="man_btn_champ"):
                if dom == ext:
                    st.warning("Choisis deux équipes différentes.")
                else:
                    lh, la = lambdas_brutes(dom, ext, att, dfn, mh, ma)
                    nouveau = (comp, dom, ext, prevoir(lh, la, corr_pour(code)), cotes)
        else:
            st.warning("Données indisponibles pour ce championnat. "
                       "Utilise une compétition internationale (saisie libre).")
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
        cotes = saisir_cotes("inter")
        if st.button("Calculer le pronostic", key="man_btn_inter"):
            if not nom_a.strip() or not nom_b.strip():
                st.warning("Écris le nom des deux équipes.")
            elif nom_a.strip().lower() == nom_b.strip().lower():
                st.warning("Écris deux équipes différentes.")
            else:
                lh, la = lambdas_manuelles(att_a, def_a, att_b, def_b, neutre)
                nouveau = (comp, nom_a.strip(), nom_b.strip(),
                           prevoir(lh, la, CORR_NEUTRE), cotes)

    if nouveau:
        st.session_state["matchs_manuels"].insert(0, nouveau)
    for item in st.session_state["matchs_manuels"]:
        titre, d, e, prev = item[:4]
        st.caption(titre)
        carte(d, e, prev)
        if len(item) > 4:
            afficher_cotes_manuelles(prev, item[4])
    if st.session_state["matchs_manuels"]:
        if st.button("Vider la liste", key="man_clear"):
            st.session_state["matchs_manuels"] = []
            st.rerun()


def onglet_apprentissage():
    st.subheader("📈 Apprentissage")
    code = st.session_state.get("comp_code")
    if not code:
        st.info("Choisis un championnat dans l'onglet « Manuel » pour voir "
                "les corrections du modèle.")
        return
    corr = corr_pour(code)
    st.write(f"Matchs utilisés pour corriger le modèle : **{corr['n']}**")
    if corr["n"] < 30:
        st.info("Pas assez de matchs : le modèle reste neutre.")
    else:
        st.write(f"Correction domicile : x{corr['kh']:.2f}")
        st.write(f"Correction extérieur : x{corr['ka']:.2f}")
        st.write(f"Correction matchs nuls : x{corr['nul']:.2f}")


# ---------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------
st.title("⚽ Pronostics Football")

prog = construire_programme()
t1, t2, t3, t4 = st.tabs(["📅 Programme", "🎫 Ticket", "✍️ Manuel", "📈 Apprentissage"])

with t1:
    liste = onglet_programme(prog)
    ticket_conseille(liste)

with t2:
    onglet_ticket(prog)
    historique()

with t3:
    onglet_manuel()

with t4:
    onglet_apprentissage()
