import io
import os
import re
import difflib
import zlib
from datetime import date, datetime
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


def ligne_total_buts(prev):
    m = prev.get("m")
    if m is None:
        return ""
    t = [m[SOMME <= 1].sum(), m[(SOMME >= 2) & (SOMME <= 3)].sum(),
         m[(SOMME >= 4) & (SOMME <= 5)].sum(), m[SOMME >= 6].sum()]
    return ("Total de buts : " + " · ".join(
        f"{n} : {p * 100:.0f}%" for n, p in zip(("0-1", "2-3", "4-5", "6+"), t)))


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
    st.caption(ligne_total_buts(prev))
    st.caption("Scores les plus probables (chacun reste faible) : " + " | ".join(
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


def outils_dispo():
    ok = {}
    ok["pdf"] = False
    for mod in ("pypdf", "pdfplumber"):
        try:
            __import__(mod)
            ok["pdf"] = True
            break
        except Exception:
            pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ok["ocr"] = True
    except Exception:
        ok["ocr"] = False
    ok["qr"] = cv2 is not None
    return ok


MSG_PDF = ("❌ La lecture des PDF n'est pas installée sur cet appareil "
           "(module « pypdf » manquant : ajoute-le dans requirements.txt).")
MSG_OCR = ("❌ La lecture automatique des photos n'est pas installée "
           "(programme « tesseract-ocr » manquant). Sur un hébergement, ajoute "
           "tesseract-ocr dans packages.txt. Dans un APK, la lecture de photo "
           "n'est pas possible : utilise le PDF, ou ajoute les matchs depuis "
           "le programme / à la main.")


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
def _textes_pdf(octets):
    from pypdf import PdfReader
    rd = PdfReader(io.BytesIO(octets))
    out = []
    for page in rd.pages:
        buf = []
        page.extract_text(visitor_text=lambda t, cm, tm, fd, fs:
                          buf.append(t.strip()) if t.strip() else None)
        out.extend(buf)
    return out


def lire_programme_pdf_pypdf(octets):
    """Lecture sans outil lourd : chaque case du tableau est un morceau de texte."""
    try:
        ch = _textes_pdf(octets)
    except Exception:
        return prog_vide()
    lignes, date_cour, n = [], None, len(ch)
    for i, t in enumerate(ch):
        if RE_DATE_S.match(t):
            date_cour = t
            continue
        if not re.match(r"^\d{1,2}:\d{2}$", t) or i + 7 > n:
            continue
        if not (ch[i + 1].isdigit() and 2 <= len(ch[i + 1]) <= 5):
            continue
        home, away = ch[i + 2], ch[i + 3]
        if num(home) is not None or num(away) is not None:
            continue
        cat = ch[i - 1] if i > 0 and re.match(r"^[A-ZÀ-Ý][A-ZÀ-Ý0-9]{1,9}$", ch[i - 1]) else ""
        lignes.append([cat, date_cour, t, ch[i + 1], re.sub(r"\s+", " ", home),
                       re.sub(r"\s+", " ", away), num(ch[i + 4]), num(ch[i + 5]),
                       num(ch[i + 6]), "PDF"])
    if not lignes:
        return prog_vide()
    df = pd.DataFrame(lignes, columns=COLS_PROG)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.date
    return df


def lire_programme_pdf_plumber(octets):
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


@st.cache_data(show_spinner=False)
def lire_programme_pdf(octets):
    df = lire_programme_pdf_pypdf(octets)
    if df.empty:
        df = lire_programme_pdf_plumber(octets)
    return df


def nettoyer_nom(s):
    mots = [w for w in re.split(r"\s+", s) if re.search(r"[A-ZÀ-Ý]", w)
            and not re.fullmatch(r"[\d.,]+", w)]
    return " ".join(mots).strip(" -|[]")


MESES = {"JANEIRO": 1, "FEVEREIRO": 2, "MARCO": 3, "MARÇO": 3, "ABRIL": 4, "MAIO": 5,
         "JUNHO": 6, "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9, "OUTUBRO": 10,
         "NOVEMBRO": 11, "DEZEMBRO": 12}
RE_DATE_PT = re.compile(r"(\d{1,2})\s+DE\s+([A-ZÇ]+)\s+DE\s+(\d{4})\D{0,4}(\d{1,2}[:.]\d{2})", re.I)
RE_COTE3 = re.compile(r"\d{1,3}[.,]\d{2}")


def _nettoyer_titre(txt, fin_de_nom=False):
    txt = re.sub(r"[+\-]\d{2,4}\b", " ", txt)
    mots = txt.split()
    if fin_de_nom:
        while mots and (not re.search(r"[A-Za-zÀ-ÿ]", mots[-1]) or mots[-1][0].islower()):
            mots.pop()
    return " ".join(mots).strip(" .-|_")


CORRECTIONS_OCR = {"Campedes": "Campeões", "Campeoes": "Campeões", "Milado": "Milão",
                   "Milao": "Milão", "Atenas": "Atenas"}


def _corriger_ocr(txt):
    for faux, vrai in CORRECTIONS_OCR.items():
        txt = txt.replace(faux, vrai)
    return txt


def lire_capture_elephant(texte, jour):
    """Capture d'écran de l'appli Elephant Bet : 'Équipe A vs. Équipe B' + cotes 1 N 2."""
    L = [re.sub(r"\s+", " ", l).strip() for l in texte.splitlines()]
    L = [l for l in L if l]
    triples = [i for i, l in enumerate(L)
               if len(RE_COTE3.findall(l)) == 3 and not re.search(r"[A-Za-z]{3,}", l)]
    lignes, debut = [], 0
    for t in triples:
        bloc = L[debut:t]
        debut = t + 1
        cotes = [num(x) for x in RE_COTE3.findall(L[t])]
        d_idx, dt, hora = None, None, ""
        for k in range(len(bloc) - 1, -1, -1):
            m = RE_DATE_PT.search(bloc[k])
            if m:
                d_idx, hora = k, re.sub(r"[.]", ":", m.group(4))
                mois = MESES.get(m.group(2).upper())
                try:
                    dt = date(int(m.group(3)), mois, int(m.group(1))) if mois else None
                except ValueError:
                    dt = None
                break
        lab = next((k for k in range(len(bloc) - 1, -1, -1)
                    if re.search(r"EMPATE|DRAW", bloc[k], re.I)), len(bloc))
        comp_idx = lab - 1 if lab >= 1 else None
        comp = bloc[comp_idx] if comp_idx is not None else ""
        fin_titre = comp_idx if comp_idx is not None else len(bloc)
        if d_idx is not None:
            ini = d_idx + 1
        else:
            vs = [k for k in range(fin_titre) if re.search(r"\bvs\.?(\s|$)", bloc[k], re.I)]
            if not vs:
                continue
            ini = vs[-1]
        titre = " ".join(bloc[ini:fin_titre])
        parts = re.split(r"\s+vs\.?\s+", titre, maxsplit=1, flags=re.I)
        if len(parts) != 2:
            continue
        a, b = _nettoyer_titre(parts[0]), _nettoyer_titre(parts[1], True)
        if len(a) < 2 or len(b) < 2:
            continue
        lignes.append([_corriger_ocr(comp), dt, hora, "", _corriger_ocr(a), _corriger_ocr(b),
                       cotes[0], cotes[1], cotes[2], "Capture"])
    if not lignes:
        return prog_vide()
    n = len(lignes)
    for idx, lg in enumerate(lignes):     # date coupée : celle du match d'avant, sinon la date choisie
        if lg[1] is None:
            prec = next((lignes[k][1] for k in range(idx - 1, -1, -1)
                         if lignes[k][1] is not None), None)
            suiv = next((lignes[k][1] for k in range(idx + 1, n)
                         if lignes[k][1] is not None), None)
            cand = prec or jour
            if suiv is not None and cand > suiv:
                cand = suiv
            lg[1] = cand
    for lg in lignes:
        lg[3] = "C" + str(zlib.crc32(f"{lg[4]}|{lg[5]}|{lg[1]}".encode()) % 100000)
    return pd.DataFrame(lignes, columns=COLS_PROG)


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
    return analyse_match_code(home, away, code)


def analyse_match_code(home, away, code):
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


def cote_ok(o):
    try:
        return o is not None and not pd.isna(o) and o > 1
    except Exception:
        return False


def cotes_doubles(o1, ox, o2):
    """Estimation des cotes 1N, N2, 12 à partir des cotes 1, N, 2 (pas besoin de les taper)."""
    if not cotes_valides(o1, ox, o2):
        return None
    k = 0.92   # les vraies cotes sont ~8 % plus basses que la formule (mesuré sur Elephant Bet)
    return {"1N": max(1.01, k / (1 / o1 + 1 / ox)), "N2": max(1.01, k / (1 / ox + 1 / o2)),
            "12": max(1.01, k / (1 / o1 + 1 / o2))}


def probas_cotes(o1, ox, o2, pn=None):
    if cotes_valides(o1, ox, o2):
        inv = np.array([1 / o1, 1 / ox, 1 / o2])
        return inv / inv.sum()
    if pn is not None and cote_ok(o1) and cote_ok(o2):
        a, b = 1 / o1, 1 / o2
        return np.array([(1 - pn) * a / (a + b), pn, (1 - pn) * b / (a + b)])
    return None


def probas_finales(o1, ox, o2, prev):
    mod = None
    if prev is not None:
        mod = np.array([prev["pm"]["1"], prev["pm"]["N"], prev["pm"]["2"]])
    imp = probas_cotes(o1, ox, o2, pn=(mod[1] if mod is not None else 0.26))
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
    infos = [r["cat"] or "Autre", f"{r['date']:%d/%m} {r['hora']}".strip()]
    if str(r["id"]).isdigit():
        infos.append(f"ID {r['id']}")
    st.caption(" · ".join(infos))
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
    dd = cotes_doubles(o1, ox, o2)
    if dd:
        st.caption("Doubles chances (cotes estimées) : "
                   + " · ".join(f"{k} {v:.2f}" for k, v in dd.items()))
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
        st.caption(ligne_total_buts(prev))
        st.caption("Scores les plus probables (chacun reste faible) : " + " | ".join(
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
    if pdfs and not OUTILS["pdf"]:
        st.error(MSG_PDF)
    if photos and not OUTILS["ocr"]:
        st.error(MSG_OCR)
    for f in pdfs if OUTILS["pdf"] else []:
        with st.spinner("Lecture du PDF..."):
            d = lire_programme_pdf(f.getvalue())
        if d.empty:
            st.warning(f"Impossible de lire le tableau de {f.name}.")
        morceaux.append(d)
    for f in photos if OUTILS["ocr"] else []:
        with st.spinner("Lecture de la photo (peut prendre quelques secondes)..."):
            texte = ocr_image(f.getvalue(), 3000, (0,))
            d = lire_capture_elephant(texte, jour_photo)
            if d.empty:
                texte = ocr_image(f.getvalue(), 4400, (0,))
                d = lire_programme_photo(texte, jour_photo)
        if d.empty:
            st.warning("Photo peu lisible : aucun match reconnu. "
                       "Prends-la bien à plat, ou utilise le PDF.")
        else:
            if (d["source"] == "Capture").all():
                st.success(f"📱 {len(d)} match(s) lu(s) sur la capture d'écran, avec les cotes.")
                coupes = len(re.findall(r"\+\d{3}\b", texte)) - len(d)
                if coupes > 0:
                    st.caption(f"{coupes} match(s) coupé(s) en bas de l'écran (sans cotes) "
                               "ne sont pas lus : fais défiler et refais une capture.")
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


def ticket_conseille(liste, cle="tc"):
    st.subheader("🎯 Ticket conseillé")
    if not liste:
        st.caption("Affiche d'abord des matchs dans le programme.")
        return
    c1, c2 = st.columns(2)
    seuil = c1.slider("Confiance minimale (%)", 50, 90, 70, 5, key=f"{cle}_seuil") / 100
    nmax = c2.slider("Nombre de matchs", 1, 10, 3, key=f"{cle}_n")
    extra = st.checkbox("Inclure doubles chances et buts (si statistiques)",
                        value=False, key=f"{cle}_extra")
    cands = []
    for r, an in liste:
        prev = an["prev"] if an else None
        fin, imp, mod, src = probas_finales(r["o1"], r["ox"], r["o2"], prev)
        if fin is None:
            continue
        opts = [("1", fin[0], r["o1"]), ("N", fin[1], r["ox"]), ("2", fin[2], r["o2"])]
        if extra:
            dd = cotes_doubles(r["o1"], r["ox"], r["o2"])
            if dd:
                opts += [("1N", fin[0] + fin[1], dd["1N"]), ("N2", fin[1] + fin[2], dd["N2"]),
                         ("12", fin[0] + fin[2], dd["12"])]
            if prev is not None:
                pm = prev["pm"]
                for k in ("+1,5 buts", "-3,5 buts"):
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
               + f"{cotetot:.2f} (les cotes de doubles chances sont estimées)")
    if len(cands) >= 5 or ptot < 0.25:
        st.warning(f"Avec {len(cands)} matchs, il n'y a que {ptot * 100:.0f}% de chances que "
                   "tout passe : il suffit d'un seul match raté pour perdre le ticket.")
    st.caption(f"Valeur attendue : {ptot * cotetot:.2f} "
               "(sous 1,00, ce ticket rapporte en moyenne moins que la mise).")
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
    if fichiers and not OUTILS["ocr"]:
        st.error(MSG_OCR)
    if fichiers and not OUTILS["qr"]:
        st.warning("Lecture du QR code indisponible (module « opencv » manquant) : "
                   "écris le code à la main.")
    for f in fichiers or []:
        octets = f.getvalue()
        q = lire_qr(octets)
        if q:
            codes.append(q)
        if OUTILS["ocr"]:
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


_GRILLE = np.arange(0.2, 3.65, 0.05)
_PMF = np.array([np.exp(-l) * l ** KS / FACT for l in _GRILLE])
_CDF_AVANT = np.hstack([np.zeros((len(_GRILLE), 1)), np.cumsum(_PMF, axis=1)[:, :-1]])


def lambdas_depuis_cotes(o1, ox, o2):
    """Cherche les buts attendus (domicile, extérieur) qui redonnent les probabilités des cotes."""
    imp = probas_cotes(o1, ox, o2, pn=0.26)
    if imp is None:
        return None
    p1 = _PMF @ _CDF_AVANT.T          # P(domicile gagne) pour chaque couple
    pn = _PMF @ _PMF.T                # P(nul)
    p2 = 1 - p1 - pn
    err = (p1 - imp[0]) ** 2 + (pn - imp[1]) ** 2 + (p2 - imp[2]) ** 2
    i, j = np.unravel_index(np.argmin(err), err.shape)
    return float(_GRILLE[i]), float(_GRILLE[j])


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
            dom = st.selectbox("Équipe à domicile", equipes, key=f"man_dom_{code}")
            ext = st.selectbox("Équipe à l'extérieur", equipes,
                               index=1 if len(equipes) > 1 else 0, key=f"man_ext_{code}")
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
        cotes = saisir_cotes("inter")
        deduit = lambdas_depuis_cotes(*(c if c > 1 else None for c in cotes))
        if deduit:
            st.success(f"Buts attendus calculés automatiquement avec tes cotes : "
                       f"{deduit[0]:.2f} - {deduit[1]:.2f}")
        else:
            st.caption("Sans cotes, l'application ne connaît pas ces équipes : "
                       "règle leurs buts moyens ci-dessous.")
        with st.expander("Buts moyens (utilisés seulement si tu n'as pas de cotes)",
                         expanded=not deduit):
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
                if deduit:
                    lh, la = deduit
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
    liste_rapide()


RE_FLOAT = re.compile(r"(?<![\d.,])\d{1,3}[.,]\d{1,2}(?![\d])")


def lire_lignes_cotes(texte):
    """« Arsenal 1.90 - Liverpool 3.90 » ou « Arsenal - Liverpool 1.90 3.60 3.90 »."""
    out = []
    for l in texte.splitlines():
        l = l.strip()
        if not l:
            continue
        cotes = [num(x) for x in RE_FLOAT.findall(l)]
        reste = re.sub(r"[@=>]+", " ", RE_FLOAT.sub(" ", l))
        reste = re.sub(r"\s+", " ", reste).strip()
        p = SEP.split(reste, maxsplit=1)
        if len(p) != 2 or not p[0].strip() or not p[1].strip():
            continue
        if len(cotes) >= 3:
            o1, ox, o2 = cotes[:3]
        elif len(cotes) == 2:
            o1, ox, o2 = cotes[0], None, cotes[1]
        else:
            o1 = ox = o2 = None
        out.append((p[0].strip(), p[1].strip(), o1, ox, o2))
    return out


def liste_rapide():
    st.markdown("##### 📝 Plusieurs matchs d'un coup")
    texte = st.text_area(
        "Un match par ligne, avec les cotes (1 N 2, ou seulement 1 et 2)",
        value="", height=140, key="rapide_txt",
        placeholder="Arsenal 1.90 - Liverpool 3.90\nReal Madrid - Barcelone 2.10 3.40 3.20")
    lignes = lire_lignes_cotes(texte)
    if texte.strip() and not lignes:
        st.warning("Aucun match compris. Écris par exemple : Arsenal 1.90 - Liverpool 3.90")
        return
    code = st.session_state.get("comp_code")
    liste = []
    for n, (a, b, o1, ox, o2) in enumerate(lignes, start=1):
        r = {"cat": "Manuel", "date": date.today(), "hora": "", "id": "",
             "home": a, "away": b, "o1": o1, "ox": ox, "o2": o2}
        an = analyse_match_code(a, b, code)
        carte_programme(n, r, an)
        liste.append((r, an))
    if liste:
        ticket_conseille(liste, "rap")


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
# MATCH DÉTAILLÉ : lecture des tables de marchés (captures d'écran)
# ---------------------------------------------------------------
RE_ODD = re.compile(r"^\d{1,3}\.\d{1,2}$")


def _lignes(mots, tol=20):
    """Regroupe des mots en lignes (même hauteur), triés de gauche à droite."""
    out = []
    for m in sorted(mots, key=lambda m: m["y"]):
        if out and abs(out[-1][0] - m["y"]) <= tol:
            out[-1][1].append(m)
        else:
            out.append([m["y"], [m]])
    return [(y, " ".join(w["t"] for w in sorted(ms, key=lambda w: w["x"]))) for y, ms in out]


@st.cache_data(show_spinner=False)
def lire_page_marches(octets):
    """Lit une capture de la page d'un match : [{titre, label, cote}]."""
    try:
        import pytesseract
        from PIL import Image
        im = Image.open(io.BytesIO(octets)).convert("L")
    except Exception:
        return [], ""
    W, H = im.size
    a = np.array(im)
    try:
        d = pytesseract.image_to_data(im, config="--psm 11",
                                      output_type=pytesseract.Output.DICT)
        haut = Image.fromarray(255 - a[: int(0.13 * H)])
        entete = pytesseract.image_to_string(haut, config="--psm 6")
    except Exception:
        return [], ""
    mots = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t or float(d["conf"][i]) < 30:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        if not (0.31 * H < y < 0.865 * H):
            continue
        noir = np.percentile(a[y:y + h, x:x + w], 5) < 60
        mots.append({"t": t, "x": x, "y": y, "xc": x + w / 2, "noir": noir})
    def relire(m):
        try:
            x0, y0 = max(m["x"] - 8, 0), max(m["y"] - 6, 0)
            case = Image.fromarray(a[y0:y0 + 44, x0:x0 + 140]).resize((420, 132))
            t = pytesseract.image_to_string(
                case, config="--psm 7 -c tessedit_char_whitelist=0123456789.").strip()
            return t if RE_ODD.match(t) else None
        except Exception:
            return None
    for m in mots:
        if m["noir"] and re.fullmatch(r"\d{2,4}", m["t"]):
            t = relire(m)
            if not t:
                n = len(m["t"])
                t = m["t"][0] + "." + m["t"][1:] if n <= 3 else m["t"][:2] + "." + m["t"][2:]
            m["t"] = t
    cotes = [m for m in mots if m["noir"] and RE_ODD.match(m["t"])]
    titres = _lignes([m for m in mots if m["noir"] and not RE_ODD.match(m["t"])
                      and not re.fullmatch(r"[\d.]+", m["t"])], 25)
    labels = [m for m in mots if not m["noir"]]
    col = lambda m: 0 if m["xc"] < 247 else (1 if m["xc"] < 473 else 2)
    sorties = []
    for c in cotes:
        cc = col(c)
        prec = [o["y"] for o in cotes if col(o) == cc and o["y"] < c["y"] - 5]
        borne = max(prec) + 40 if prec else 0
        tit = [(y, t) for y, t in titres if y < c["y"]]
        if tit:
            borne = max(borne, tit[-1][0] + 40)
        bloc = [m for m in labels if col(m) == cc and borne <= m["y"] < c["y"]]
        label = " ".join(t for _, t in _lignes(bloc, 20))
        sorties.append({"titre": tit[-1][1] if tit else "", "label": label,
                        "cote": float(c["t"]), "y": c["y"], "cc": cc})
    return sorties, entete


def _norm_nombres(txt):
    def f(m):
        t = m.group(2).replace("O", "0").replace("o", "0").replace("S", "5").replace("s", "5")
        if "." not in t and len(t) == 2:
            t = t[0] + "." + t[1]
        return f"{m.group(1)} {t}"
    return re.sub(r"\b(de)\s*([0-9OoSs.]{2,4})\b", f, txt)


def _sq(s):
    s = s.lower()
    for a_, b_ in (("é", "e"), ("ã", "a"), ("ç", "c"), ("ê", "e"), ("á", "a"), ("ó", "o")):
        s = s.replace(a_, b_)
    return re.sub(r"[^a-z0-9]", "", s)


def _issue(part, home, away):
    p = _sq(part)
    if not p:
        return None
    cand = {"1": _sq(home), "2": _sq(away), "N": "empate"}
    best = max(cand, key=lambda k: difflib.SequenceMatcher(None, p, cand[k]).ratio())
    r = difflib.SequenceMatcher(None, p, cand[best]).ratio()
    return best if r >= 0.5 else None


def interpreter_selection(it, home, away):
    """Transforme (titre, label) en sélection normalisée, ou None si illisible."""
    label = _norm_nombres(it["label"].replace("€", "e"))
    titre = _sq(it["titre"])
    moitie = "parte" in titre or "1aparte" in titre
    res_txt, cond = label, None
    if "goloeres" in titre or "goloeres" in _sq(it["label"]):
        return None
    m = re.search(r"\b(?:e\s+)?(mais|menos)\s*(?:de)?\s*(\d\.\d)?", label, re.I)
    if m:
        x = float(m.group(2)) if m.group(2) else it.get("x_defaut")
        if x is None:
            return None
        cond = ("O" if m.group(1).lower() == "mais" else "U", x)
        res_txt = label[:m.start()]
    else:
        m = re.search(r"(?:e\s*)?(sim|n[aãdi]o|ndo)\s*$", label, re.I)
        if m:
            cond = "BTTS_S" if m.group(1).lower() == "sim" else "BTTS_N"
            res_txt = label[:m.start()]
    res_txt = re.sub(r"\s+e\s*$", "", res_txt.strip(), flags=re.I)
    if not res_txt.strip():                       # « mais de 0.5 » : total seul
        return {"scope": "1P" if moitie else "FT", "res": None, "cond": cond} if cond else None
    if "ganhaqualq" in titre or "ganhaqual" in titre:
        quel = "1" if re.search(r"(^|\D)1(\D|$)", it["titre"]) else "2"
        oui = "sim" in _sq(label)
        return {"scope": "ANY", "res": None, "cond": ("GANHA", quel, oui)}
    if "golo" in _sq(label):
        return None
    parts = [p for p in re.split(r"\s+ou\s+|/", res_txt, flags=re.I) if p.strip()]
    issues_ = {_issue(p, home, away) for p in parts}
    if None in issues_ or not issues_:
        return None
    code = "".join(k for k in ("1", "N", "2") if k in issues_)
    code = {"1N": "1N", "N2": "N2", "12": "12"}.get(code, code)
    return {"scope": "1P" if moitie else "FT", "res": code, "cond": cond}


def analyser_marches(items, home, away):
    """Regroupe les sélections, calcule les marges et les probabilités sans marge."""
    from collections import Counter
    for img in {i.get("img") for i in items}:
        xs = [float(m.group(1)) for i in items if i.get("img") == img
              for m in [re.search(r"(?:mais|menos)\s*(?:de)?\s*(\d\.\d)", _norm_nombres(i["label"]), re.I)] if m]
        if xs:
            xd = Counter(xs).most_common(1)[0][0]
            for i in items:
                if i.get("img") == img:
                    i["x_defaut"] = xd
    sel = []
    for it in items:
        s = interpreter_selection(it, home, away)
        if s:
            s["cote"] = it["cote"]
            sel.append(s)
    seen, uniq = set(), []
    for s in sel:                                 # doublons entre captures qui se chevauchent
        cle = (s["scope"], s["res"], str(s["cond"]), s["cote"])
        if cle not in seen:
            seen.add(cle)
            uniq.append(s)
    sel = uniq
    groupes = {}
    for s in sel:
        c = s["cond"]
        if s["scope"] == "ANY":
            cle = ("Gagne une mi-temps", f"équipe {c[1]}", 1)
        elif s["res"] in ("1", "N", "2") and c in ("BTTS_S", "BTTS_N"):
            cle = ("1x2 + les 2 marquent", "", 1)
        elif s["res"] in ("1", "N", "2") and isinstance(c, tuple):
            cle = (f"1x2 + total {c[1]:g}", "", 1)
        elif s["res"] in ("1N", "N2", "12") and isinstance(c, tuple):
            cle = (f"Double chance + total {c[1]:g}", "", 2)
        elif s["res"] in ("1", "N", "2") and c is None:
            cle = ("1x2 (mi-temps)" if s["scope"] == "1P" else "1x2", "", 1)
        elif s["res"] in ("1N", "N2", "12") and c is None:
            cle = ("Double chance (mi-temps)" if s["scope"] == "1P" else "Double chance", "", 2)
        elif s["res"] is None and isinstance(c, tuple):
            cle = ("Total buts (mi-temps)" if s["scope"] == "1P" else "Total buts", f"{c[1]:g}", 1)
        else:
            continue
        groupes.setdefault(cle, []).append(s)
    attendu = {"1x2 + les 2 marquent": 6, "1x2 (mi-temps)": 3, "1x2": 3,
               "Double chance (mi-temps)": 3, "Double chance": 3, "Total buts": 2,
               "Total buts (mi-temps)": 2, "Gagne une mi-temps": 2}
    marches = []
    for cle, lst in groupes.items():
        nom = cle[0] + (f" {cle[1]}" if cle[1] else "")
        att = 6 if (cle[0].startswith("1x2 + total") or cle[0].startswith("Double chance + total")) \
            else attendu.get(cle[0], len(lst))
        marches.append({"nom": nom, "n": len(lst), "attendu": att, "facteur": cle[2],
                        "complet": len(lst) == att,
                        "somme": sum(1 / s["cote"] for s in lst), "items": lst})
    for g in marches:
        g["marge"] = g["somme"] / g["facteur"] - 1 if g["complet"] else None
    return sel, marches


def probabilites_match(marches):
    """Probabilités sans marge reconstruites à partir des marchés complets."""
    out = {}
    par_nom = {g["nom"]: g for g in marches if g["complet"]}
    g = par_nom.get("1x2 + les 2 marquent")
    if g:
        tot = g["somme"]
        for r in ("1", "N", "2"):
            out[r] = sum(1 / s["cote"] for s in g["items"] if s["res"] == r) / tot
        out["BTTS"] = sum(1 / s["cote"] for s in g["items"] if s["cond"] == "BTTS_S") / tot
    for nom, g in par_nom.items():
        m = re.match(r"(?:1x2|Double chance) \+ total ([\d.]+)$", nom)
        if m:
            x = float(m.group(1))
            over = sum(1 / s["cote"] for s in g["items"] if s["cond"][0] == "O")
            out[f"+{x:g}"] = over / g["somme"]
            if "1" not in out and nom.startswith("1x2"):
                for r in ("1", "N", "2"):
                    out[r] = sum(1 / s["cote"] for s in g["items"] if s["res"] == r) / g["somme"]
    return out


def deviner_equipes(items):
    ligne = [i for i in items if "parte" in _sq(i["titre"]) and "1x2" in _sq(i["titre"])]
    dom = next((i["label"] for i in ligne if i.get("cc") == 0), "")
    ext = next((i["label"] for i in ligne if i.get("cc") == 2), "")
    return dom.strip(), ext.strip()


def nom_selection(s, home, away):
    res = {"1": "1", "N": "N", "2": "2", "1N": "1N", "N2": "N2", "12": "12", None: ""}[s["res"]]
    c = s["cond"]
    if isinstance(c, tuple) and c[0] in ("O", "U"):
        cond = f"{'+' if c[0] == 'O' else '-'}{c[1]:g} buts"
    elif c == "BTTS_S":
        cond = "les 2 marquent"
    elif c == "BTTS_N":
        cond = "une équipe ne marque pas"
    elif isinstance(c, tuple) and c[0] == "GANHA":
        return f"Équipe {c[1]} gagne une mi-temps : {'oui' if c[2] else 'non'}"
    else:
        cond = ""
    mt = " (mi-temps)" if s["scope"] == "1P" else ""
    return (" + ".join(x for x in (res, cond) if x) + mt).strip()


def onglet_detail():
    st.subheader("🔎 Match détaillé")
    st.info("Importe les captures de la page d'un match (onglet « Todos apostas »). "
            "L'application lit toutes les cotes, calcule les marges du bookmaker et "
            "reconstruit les probabilités du match. Tu n'as rien à taper.")
    if not OUTILS["ocr"]:
        st.error(MSG_OCR)
        return
    fichiers = st.file_uploader("Captures du match (plusieurs possibles)",
                                type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                key="det_up")
    miniatures(fichiers)
    if not fichiers:
        return
    items = []
    for k, f in enumerate(fichiers):
        with st.spinner(f"Lecture de la capture {k + 1}/{len(fichiers)}..."):
            sel_, _ent = lire_page_marches(f.getvalue())
        for x in sel_:
            x = dict(x)
            x["img"] = k
            items.append(x)
    if not items:
        st.warning("Aucune cote lue. Utilise des captures de la page des marchés d'un match.")
        return
    sig = "-".join(f"{f.name}{f.size}" for f in fichiers)
    dom0, ext0 = deviner_equipes(items)
    c1, c2 = st.columns(2)
    home = c1.text_input("Équipe 1 (domicile)", value=dom0, key=f"det_h_{sig}")
    away = c2.text_input("Équipe 2 (extérieur)", value=ext0, key=f"det_a_{sig}")
    sel, marches = analyser_marches(items, home, away)
    st.success(f"{len(items)} cotes lues, {len(sel)} comprises, {len(marches)} marchés trouvés.")

    P = probabilites_match(marches)
    if P:
        st.markdown("**Probabilités du match (marge du bookmaker retirée)**")
        if all(k in P for k in ("1", "N", "2")):
            a, b, c = st.columns(3)
            a.metric(f"1 · {home[:12]}", f"{P['1'] * 100:.0f}%")
            b.metric("N", f"{P['N'] * 100:.0f}%")
            c.metric(f"2 · {away[:12]}", f"{P['2'] * 100:.0f}%")
        tot = [(k, P[k]) for k in ("+1.5", "+2.5", "+3.5", "+4.5") if k in P]
        if tot:
            cols = st.columns(len(tot) + (1 if "BTTS" in P else 0))
            for col, (k, v) in zip(cols, tot):
                col.metric(k.replace(".", ",") + " buts", f"{v * 100:.0f}%")
            if "BTTS" in P:
                cols[-1].metric("2 marquent", f"{P['BTTS'] * 100:.0f}%")
    else:
        st.warning("Pas assez de marchés complets pour reconstruire les probabilités : "
                   "fais défiler la page du match et ajoute les captures manquantes.")

    st.markdown("**Marge du bookmaker par marché** (plus elle est basse, mieux c'est pour toi)")
    complets = sorted([g for g in marches if g["complet"]], key=lambda g: g["marge"])
    for g in complets:
        st.write(f"• {g['nom']} : **{g['marge'] * 100:.0f}%**")
    if complets:
        st.caption(f"Le moins cher : {complets[0]['nom']} ({complets[0]['marge'] * 100:.0f}%). "
                   f"Le plus cher : {complets[-1]['nom']} ({complets[-1]['marge'] * 100:.0f}%).")
    inc = [g for g in marches if not g["complet"]]
    if inc:
        st.caption("Marchés incomplets (lignes coupées, ajoute une capture) : "
                   + ", ".join(f"{g['nom']} {g['n']}/{g['attendu']}" for g in inc))

    with st.expander("Voir toutes les cotes lues"):
        st.caption(f"1 = {home or 'équipe 1'} · 2 = {away or 'équipe 2'} · N = nul")
        lignes = []
        for g in marches:
            for s in g["items"]:
                p = (1 / s["cote"]) / (g["somme"] / g["facteur"]) if g["complet"] else None
                lignes.append({"Choix": nom_selection(s, home, away), "Cote": s["cote"],
                               "Proba": f"{p * 100:.0f}%" if p else "—"})
        if lignes:
            st.dataframe(pd.DataFrame(lignes), hide_index=True)
    bloc_combinaisons(P, sel, marches, home, away)


# ---------------------------------------------------------------
# PLUSIEURS CHOIX SUR LE MÊME MATCH
# ---------------------------------------------------------------
_G2 = np.arange(0.2, 4.45, 0.05)
_PMF2 = np.array([np.exp(-l) * l ** KS / FACT for l in _G2])
_CDF2 = np.hstack([np.zeros((len(_G2), 1)), np.cumsum(_PMF2, axis=1)[:, :-1]])


def _cdf_poisson(k, S):
    """P(total <= k) pour une somme de deux Poisson de moyenne S."""
    return np.exp(-S) * sum(S ** t / factorial(t) for t in range(int(k) + 1))


def ajuster_lambdas(P):
    """Cherche les buts attendus (domicile, extérieur) qui collent aux probabilités lues."""
    li, lj = _G2[:, None], _G2[None, :]
    S = li + lj
    pred = {"1": _PMF2 @ _CDF2.T, "N": _PMF2 @ _PMF2.T}
    pred["2"] = 1 - pred["1"] - pred["N"]
    pred["BTTS"] = (1 - np.exp(-li)) * (1 - np.exp(-lj))
    for x in (0.5, 1.5, 2.5, 3.5, 4.5):
        pred[f"+{x:g}"] = 1 - _cdf_poisson(int(x), S)
    cles = [k for k in pred if k in P]
    if len(cles) < 4 or not all(k in P for k in ("1", "N", "2")):
        return None
    err = sum((pred[k] - P[k]) ** 2 for k in cles)
    i, j = np.unravel_index(np.argmin(err), err.shape)
    ecart = max(abs(pred[k][i, j] - P[k]) for k in cles)
    return float(_G2[i]), float(_G2[j]), float(ecart)


def proba_combinaison(m, res, buts, btts):
    """Probabilité que TOUTES les conditions soient vraies (même match)."""
    I, J = KS[:, None], KS[None, :]
    masque = np.ones_like(m, dtype=bool)
    if res:
        masque &= {"1": I > J, "N": I == J, "2": I < J, "1N": I >= J,
                   "N2": I <= J, "12": I != J}[res]
    if buts:
        masque &= (SOMME > buts[1]) if buts[0] == "O" else (SOMME < buts[1])
    if btts is not None:
        masque &= ((I > 0) & (J > 0)) if btts else ((I == 0) | (J == 0))
    return float(m[masque].sum())


def cote_reelle(sel, marches, res, buts, btts):
    """Cherche la cote officielle lue sur les captures pour cette combinaison exacte."""
    cond = "BTTS_S" if btts is True else ("BTTS_N" if btts is False else buts)
    if (buts is not None) and (btts is not None):
        return None
    for g in marches:
        for s in g["items"]:
            if s["scope"] == "FT" and s["res"] == res and s["cond"] == cond and cond is not None:
                return s["cote"], g.get("marge")
    return None


def bloc_combinaisons(P, sel, marches, home, away):
    st.markdown("##### 🧩 Plusieurs choix sur ce match")
    fit = ajuster_lambdas(P)
    if not fit:
        st.caption("Il faut au moins le 1x2 et un marché de buts complets pour calculer "
                   "les combinaisons.")
        return
    lh, la, ecart = fit
    m = matrice(lh, la)
    c1, c2, c3 = st.columns(3)
    res = c1.selectbox("Résultat", ["—", "1", "N", "2", "1N", "N2", "12"], key="cb_res")
    but = c2.selectbox("Buts", ["—", "+0,5", "+1,5", "+2,5", "+3,5", "+4,5",
                                "-1,5", "-2,5", "-3,5", "-4,5"], key="cb_but")
    bt = c3.selectbox("Les 2 marquent", ["—", "Oui", "Non"], key="cb_bt")
    st.caption(f"1 = {home or 'équipe 1'} · 2 = {away or 'équipe 2'} · N = nul")
    r = None if res == "—" else res
    b = None if but == "—" else (("O" if but[0] == "+" else "U"), float(but[1:].replace(",", ".")))
    t = None if bt == "—" else (bt == "Oui")
    if r is None and b is None and t is None:
        st.caption("Choisis au moins deux conditions pour voir la probabilité du combiné.")
        return
    p = proba_combinaison(m, r, b, t)
    a1, a2 = st.columns(2)
    a1.metric("Probabilité", f"{p * 100:.1f}%")
    a2.metric("Cote juste", f"{1 / p:.2f}" if p > 0 else "—")
    reel = cote_reelle(sel, marches, r, b, t)
    if reel:
        cote, marge = reel
        txt = f"Cote lue sur Elephant Bet : **{cote:.2f}**"
        if marge is not None:
            txt += f" → en moyenne, tu récupères {100 / (1 + marge):.0f}% de ta mise."
        st.info(txt)
    else:
        st.caption("Cette combinaison exacte n'est pas dans les cotes lues : "
                   "regarde sa cote sur Elephant Bet avant de jouer.")
    st.caption(f"Calcul approximatif (écart moyen du modèle : {ecart * 100:.0f} points). "
               "Les probabilités du combiné ne se multiplient pas : elles sont calculées "
               "sur le score complet du match.")


# ---------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------
st.title("⚽ Pronostics Football")

OUTILS = outils_dispo()
with st.expander("🔧 Diagnostic de la lecture automatique",
                 expanded=not all(OUTILS.values())):
    st.write(("✅" if OUTILS["pdf"] else "❌") + " Lecture des PDF")
    st.write(("✅" if OUTILS["ocr"] else "❌") + " Lecture des photos (texte)")
    st.write(("✅" if OUTILS["qr"] else "❌") + " Lecture des QR codes")
    if not all(OUTILS.values()):
        st.caption("Un ❌ veut dire que l'outil n'est pas installé : vérifie "
                   "requirements.txt et packages.txt.")

prog = construire_programme()
t1, t2, t5, t3, t4 = st.tabs(["📅 Programme", "🎫 Ticket", "🔎 Match", "✍️ Manuel",
                              "📈 Apprentissage"])

with t1:
    liste = onglet_programme(prog)
    ticket_conseille(liste)

with t2:
    onglet_ticket(prog)
    historique()

with t5:
    onglet_detail()

with t3:
    onglet_manuel()

with t4:
    onglet_apprentissage()
