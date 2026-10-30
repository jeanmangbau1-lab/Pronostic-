import io
import os
import re
import difflib
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
# LECTURE DU PROGRAMME (photo / PDF / texte)
# ---------------------------------------------------------------
SEP = re.compile(r"\s+(?:-|–|—|vs\.?|v|contre|x)\s+", re.I)


def extraire_texte(octets, nom):
    try:
        if nom.lower().endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(octets)) as pdf:
                return "\n".join((p.extract_text() or "") for p in pdf.pages)
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(octets)))
    except Exception:
        return ""


def extraire_matchs(texte):
    matchs = []
    for l in texte.splitlines():
        l = re.sub(r"\d{1,2}[:h]\d{2}", " ", l)
        l = re.sub(r"\d{1,2}/\d{1,2}(/\d{2,4})?", " ", l)
        l = re.sub(r"^\s*\d+\s*[.)]\s*", "", l)
        l = re.sub(r"\s+", " ", l).strip()
        p = SEP.split(l, maxsplit=1)
        if len(p) == 2:
            a, b = p[0].strip(), p[1].strip()
            if 2 <= len(a) <= 30 and 2 <= len(b) <= 30:
                matchs.append((a, b))
    return matchs


def trouver_equipe(nom, equipes):
    r = difflib.get_close_matches(nom, equipes, n=1, cutoff=0.6)
    return r[0] if r else None


def lire_qr_code(image_bytes):
    if cv2 is None:
        return None, "Lecture QR indisponible."
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Impossible de décoder l'image."
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if data:
            return data, None
        return None, "Aucun QR code détecté."
    except Exception as e:
        return None, f"Erreur lors de la lecture : {e}"


def saisie_manuelle(att, dfn, mh, ma, corr, nom_champ, code):
    if "matchs_manuels" not in st.session_state:
        st.session_state["matchs_manuels"] = []

    with st.expander("✍️ Saisie manuelle d'un match", expanded=False):
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
                    nouveau = (comp, nom_a.strip(), nom_b.strip(),
                               prevoir(lh, la, CORR_NEUTRE))

        if nouveau:
            st.session_state["matchs_manuels"].insert(0, nouveau)

        for titre, d, e, prev in st.session_state["matchs_manuels"]:
            st.caption(titre)
            carte(d, e, prev)

        if st.session_state["matchs_manuels"]:
            if st.button("Vider la liste", key="man_clear"):
                st.session_state["matchs_manuels"] = []
                st.rerun()


def zone_programme(att, dfn, mh, ma, corr):
    st.subheader("📄 Programme des matchs")
    st.info("Photographie ou importe le programme (photo ou PDF). "
            "Tu peux aussi écrire ou coller les matchs, un par ligne : "
            "Équipe A - Équipe B")

    photo = st.camera_input("📷 Photographier le programme", key="prog_photo")
    fichier = st.file_uploader("📁 Importer une photo ou un PDF",
                               type=["jpg", "jpeg", "png", "pdf"], key="prog_file")
    source = photo if photo else fichier

    texte = ""
    if source is not None:
        octets = source.getvalue()
        texte = extraire_texte(octets, getattr(source, "name", "photo.jpg"))
        if not texte.strip():
            st.warning("Lecture automatique impossible. "
                       "Écris ou colle les matchs dans la case ci-dessous.")
        qr, _ = lire_qr_code(octets)
        if qr:
            st.caption(f"QR code détecté : {qr}")

    texte = st.text_area("Matchs (modifiable)", value=texte, height=150,
                         key="prog_texte")
    maxi = st.slider("Nombre maximum de matchs affichés", 5, 100, MAX_AFFICHE, 5)

    liste = extraire_matchs(texte)
    if not liste:
        st.caption("Aucun match détecté pour le moment.")
        return 0

    st.success(f"{len(liste)} match(s) détecté(s) — "
               f"affichage des {min(len(liste), maxi)} premiers")
    equipes = sorted(att.keys()) if att else []
    for i, (a, b) in enumerate(liste[:maxi], start=1):
        dom = trouver_equipe(a, equipes) if equipes else None
        ext = trouver_equipe(b, equipes) if equipes else None
        if dom and ext and dom != ext:
            lh, la = lambdas_brutes(dom, ext, att, dfn, mh, ma)
            carte(f"{i}. {dom}", ext, prevoir(lh, la, corr))
        else:
            lh, la = lambdas_manuelles(1.4, 1.0, 1.2, 1.1, False)
            carte(f"{i}. {a}", b, prevoir(lh, la, CORR_NEUTRE))
            st.caption("⚠️ Équipes inconnues dans ce championnat : estimation générique.")
    return len(liste)


# ---------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------
st.title("⚽ Pronostics Football")

pays = st.selectbox("Pays", list(PAYS.keys()))
divisions = PAYS[pays]
div_nom = st.selectbox("Division", list(divisions.keys()))
code = divisions[div_nom]
nom_champ = f"{pays} - {div_nom}"

with st.spinner("Chargement..."):
    donnees = charger(code)

if donnees.empty:
    st.warning("Données automatiques indisponibles pour ce championnat. "
               "Les matchs seront calculés avec une estimation générique.")
    zone_programme(None, None, None, None, CORR_NEUTRE)
    saisie_manuelle(None, None, None, None, CORR_NEUTRE, nom_champ, code)
    st.stop()

att, dfn, mh, ma = forces(donnees)
rec_log = resultats_journal(code, donnees)
rec_bt = backtest(donnees)
morceaux_rec = [x for x in (rec_log, rec_bt) if not x.empty]
rec = pd.concat(morceaux_rec, ignore_index=True) if morceaux_rec else rec_vide()
corr = corrections(rec)

onglet1, onglet2 = st.tabs(["Pronostics", "Apprentissage"])

with onglet1:
    zone_programme(att, dfn, mh, ma, corr)
    saisie_manuelle(att, dfn, mh, ma, corr, nom_champ, code)

with onglet2:
    st.subheader("Apprentissage")
    st.write(f"Matchs utilisés pour corriger le modèle : **{corr['n']}**")
    if corr["n"] < 30:
        st.info("Pas assez de matchs : le modèle reste neutre.")
    else:
        st.write(f"Correction domicile : x{corr['kh']:.2f}")
        st.write(f"Correction extérieur : x{corr['ka']:.2f}")
        st.write(f"Correction matchs nuls : x{corr['nul']:.2f}")
