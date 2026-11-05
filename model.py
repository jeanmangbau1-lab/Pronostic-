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
        
