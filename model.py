import io, os, re, difflib
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
    "Coupe d'Afrique des Nations (CAN)", "Coupe du monde FIFA",
    "Coupe d'Asie des Nations", "Copa América", "Euro (UEFA)",
    "Gold Cup (CONCACAF)", "Ligue des nations UEFA",
    "Éliminatoires Coupe du monde", "Éliminatoires CAN",
    "Ligue des champions CAF", "Match amical international",
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
        "lh": pd.Series(dtype=float), "la": pd.Series(dtype=float),
        "hg": pd.Series(dtype=float), "ag": pd.Series(dtype=float)})


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
        h = d[d["HomeTeam"] == eq]; a = d[d["AwayTeam"] == eq]
        wtot = h["w"].sum() + a["w"].sum()
        marq = (np.sum(h["FTHG"] * h["w"]) + np.sum(a["FTAG"] * a["w"])) / wtot
        enc = (np.sum(h["FTAG"] * h["w"]) + np.sum(a["FTHG"] * a["w"])) / wtot
        att[eq] = marq / moy
        dfn[eq] = enc / moy
    return att, dfn, mh, ma


def lambdas_manuelles(att_a, def_a, att_b, def_b, neutre):
    lh = (att_a + def_b) / 2
    la = (att_b + def_a) / 2
    if not neutre:
        lh *= 1.10; la *= 0.95
    return max(lh, 0.05), max(la, 0.05)


def matrice(lh, la, nul=1.0):
    ph = np.exp(-lh) * lh ** KS / FACT
    pa = np.exp(-la) * la ** KS / FACT
    m = np.outer(ph, pa); m = m / m.sum()
    if nul != 1.0:
        m = m.copy(); m[KS, KS] = m[KS, KS] * nul; m = m / m.sum()
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
    n = len(d); k = min(n_max, int(n * 0.25))
    if k < 30:
        return rec_vide()
    train, test = d.iloc[: n - k], d.iloc[n - k:]
    att, dfn, mh, ma = forces(train)
    lignes = []
    for r in test.itertuples():
        if r.HomeTeam in att and r.AwayTeam in att:
            lh = att[r.HomeTeam] * dfn[r.AwayTeam] * mh
            la = att[r.AwayTeam] * dfn[r.HomeTeam] * ma
            lignes.append((r.Date, r.HomeTeam, r.AwayTeam, lh, la, r.FTHG, r.FTAG))
    return pd.DataFrame(lignes, columns=COLS_REC) if lignes else rec_vide()


def resultats_journal(code, d):
    if not os.path.exists(LOG):
        return rec_vide()
    try:
        j = pd.read_csv(LOG); j = j[j["code"] == code].copy()
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
    kh = 1 + w * (rec["hg"].mean() / max(rec["lh"].mean(), 1e-6) - 1)
    ka = 1 + w * (rec["ag"].mean() / max(rec["la"].mean(), 1e-6) - 1)
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


# ============ OCR (inchangé) ============
SEP = re.compile(r"\s+(?:-|–|—|vs\.?|v|contre|x)\s+", re.I)


def verifier_ocr():
    try:
        import pytesseract
    except ImportError:
        return False, "module python 'pytesseract' non installé"
    try:
        v = pytesseract.get_tesseract_version()
        return True, f"Tesseract {v}"
    except Exception as e:
        return False, f"binaire 'tesseract' introuvable ({e})"


def pretraiter_image(im):
    from PIL import ImageOps
    im = ImageOps.exif_transpose(im).convert("L")
    w, h = im.size
    if max(w, h) < 1600:
        f = 1600 / max(w, h)
        im = im.resize((int(w * f), int(h * f)))
    return ImageOps.autocontrast(im)


def ocr_image(im):
    try:
        import pytesseract
    except ImportError:
        return ""
    im = pretraiter_image(im)
    try:
        return pytesseract.image_to_string(im, lang="fra")
    except Exception:
        try:
            return pytesseract.image_to_string(im)
        except Exception:
            return ""


def extraire_texte(octets, nom):
    nom = (nom or "").lower()
    if nom.endswith(".pdf"):
        erreurs = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(octets)) as pdf:
                txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
            if txt.strip():
                return txt, None
        except Exception as e:
            erreurs.append(f"pdfplumber: {e}")
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(octets)
            txt = "\n".join(ocr_image(p.render(scale=2).to_pil()) for p in doc)
            if txt.strip():
                return txt, None
            erreurs.append("OCR PDF vide")
        except Exception as e:
            erreurs.append(f"OCR PDF: {e}")
        return "", " | ".join(erreurs) or "PDF illisible"
    try:
        from PIL import Image
        txt = ocr_image(Image.open(io.BytesIO(octets)))
        if txt.strip():
            return txt, None
        return "", "Aucun texte détecté dans l'image"
    except Exception as e:
        return "", f"OCR image: {e}"


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
        return (data, None) if data else (None, "Aucun QR code détecté.")
    except Exception as e:
        return None, f"Erreur : {e}"


# ============ MULTI-LIGUE ============
def lister_ligues():
    return [(f"{pays} - {div}", code)
            for pays, divs in PAYS.items() for div, code in divs.items()]


def construire_index(donnees_multi):
    """{nom_ligue: (code, df)} -> index équipe + conflits éventuels."""
    index, conflits = {}, {}
    for nom_ligue, (code, d) in donnees_multi.items():
        if d.empty:
            continue
        att, dfn, mh, ma = forces(d)
        for eq in att:
            if eq in index:
                conflits.setdefault(eq, [index[eq]["ligue"]]).append(nom_ligue)
                continue
            index[eq] = {"ligue": nom_ligue, "code": code,
                         "att": att[eq], "dfn": dfn[eq], "mh": mh, "ma": ma}
    return index, conflits


def lambdas_multi(eq_dom, eq_ext, index):
    a, b = index[eq_dom], index[eq_ext]
    if a["ligue"] == b["ligue"]:
        return max(a["att"] * b["dfn"] * a["mh"], 0.05), \
               max(b["att"] * a["dfn"] * b["ma"], 0.05)
    # cross-ligue : reconstruire les forces absolues puis normaliser
    moy_a = (a["mh"] + a["ma"]) / 2
    moy_b = (b["mh"] + b["ma"]) / 2
    moy_g = (moy_a + moy_b) / 2
    lh = (a["att"] * moy_a) * (b["dfn"] * moy_b) / moy_g
    la = (b["att"] * moy_b) * (a["dfn"] * moy_a) / moy_g
    return max(lh, 0.05), max(la, 0.05)


def saisie_manuelle(index, corrections_par_ligue):
    if "matchs_manuels" not in st.session_state:
        st.session_state["matchs_manuels"] = []

    with st.expander("✍️ Saisie manuelle d'un match", expanded=False):
        ligues = sorted({v["ligue"] for v in index.values()})
        choix = st.selectbox(
            "Championnat", ["— Inter-championnats (libre) —"] + ligues,
            key="man_ligue_sel")
        nouveau = None

        if choix.startswith("—"):
            comp = st.selectbox("Compétition", COMPETITIONS_INTER, key="man_comp")
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
        else:
            equipes = sorted(eq for eq, v in index.items() if v["ligue"] == choix)
            dom = st.selectbox("Équipe à domicile", equipes, key="man_dom_multi")
            ext = st.selectbox("Équipe à l'extérieur", equipes,
                               index=1 if len(equipes) > 1 else 0, key="man_ext_multi")
            if st.button("Calculer le pronostic", key="man_btn_multi"):
                if dom == ext:
                    st.warning("Choisis deux équipes différentes.")
                else:
                    lh, la = lambdas_multi(dom, ext, index)
                    corr = corrections_par_ligue.get(choix, CORR_NEUTRE)
                    nouveau = (choix, dom, ext, prevoir(lh, la, corr))

        if nouveau:
            st.session_state["matchs_manuels"].insert(0, nouveau)
        for titre, d, e, prev in st.session_state["matchs_manuels"]:
            st.caption(titre); carte(d, e, prev)
        if st.session_state["matchs_manuels"]:
            if st.button("Vider la liste", key="man_clear"):
                st.session_state["matchs_manuels"] = []
                st.rerun()


def zone_programme(index, conflits, corrections_par_ligue):
    st.subheader("📄 Programme des matchs")
    st.info("Photographie, importe (photo/PDF) ou colle ton programme. "
            "Un match par ligne : `Équipe A - Équipe B`. "
            "Les matchs peuvent venir de plusieurs championnats.")

    ok_ocr, msg_ocr = verifier_ocr()
    if not ok_ocr:
        st.error(f"⚠️ OCR indisponible : {msg_ocr}. La saisie manuelle reste utilisable.")
    else:
        st.caption(f"OCR prêt : {msg_ocr}")

    photo = st.camera_input("📷 Photographier le programme", key="prog_photo")
    fichier = st.file_uploader("📁 Importer une photo ou un PDF",
                               type=["jpg", "jpeg", "png", "pdf"], key="prog_file")
    source = photo if photo else fichier

    texte = ""
    if source is not None:
        octets = source.getvalue()
        with st.spinner("Lecture du document..."):
            texte, erreur = extraire_texte(octets, getattr(source, "name", "photo.jpg"))
        if erreur:
            st.warning(f"Lecture automatique : {erreur}")
        if not texte.strip():
            st.info("Écris ou colle les matchs dans la case ci-dessous.")

    texte = st.text_area("Matchs (modifiable)", value=texte, height=150, key="prog_texte")
    maxi = st.slider("Nombre maximum de matchs affichés", 5, 100, MAX_AFFICHE, 5)

    liste = extraire_matchs(texte)
    if not liste:
        st.caption("Aucun match détecté pour le moment.")
        return 0

    st.success(f"{len(liste)} match(s) détecté(s) — affichage des "
               f"{min(len(liste), maxi)} premiers")
    equipes = list(index.keys())
    nb_ok = 0
    for i, (a, b) in enumerate(liste[:maxi], start=1):
        dom = trouver_equipe(a, equipes)
        ext = trouver_equipe(b, equipes)
        if dom and ext and dom != ext:
            lh, la = lambdas_multi(dom, ext, index)
            ligue = index[dom]["ligue"]
            corr = corrections_par_ligue.get(ligue, CORR_NEUTRE)
            carte(f"{i}. {dom}", ext, prevoir(lh, la, corr))
            st.caption(f"🏆 {ligue}")
            nb_ok += 1
        else:
            lh, la = lambdas_manuelles(1.4, 1.0, 1.2, 1.1, False)
            carte(f"{i}. {a}", b, prevoir(lh, la, CORR_NEUTRE))
            st.caption("⚠️ Équipes non reconnues : estimation générique.")
    if nb_ok == 0 and equipes:
        st.info("Aucune équipe reconnue — vérifie l'orthographe ou "
                "ajoute le championnat concerné dans la sélection en haut.")
    return len(liste)


# ============ INTERFACE ============
st.title("⚽ Pronostics Football")

ligues = lister_ligues()
labels = [nom for nom, _ in ligues]
labels_to_code = dict(ligues)

choix = st.multiselect(
    "Championnats (sélectionne-en plusieurs pour mixer les matchs)",
    labels, default=[labels[0]],
    help="Chaque match du programme sera routé vers son championnat d'origine.",
)
if not choix:
    st.warning("Sélectionne au moins un championnat.")
    st.stop()

with st.spinner(f"Chargement de {len(choix)} championnat(s)..."):
    donnees_multi = {}
    for nom in choix:
        code = labels_to_code[nom]
        d = charger(code)
        if not d.empty:
            donnees_multi[nom] = (code, d)
        else:
            st.warning(f"Données indisponibles pour {nom}.")

if not donnees_multi:
    st.warning("Aucun championnat disponible — les matchs seront estimés génériquement.")
    zone_programme({}, {}, {})
    saisie_manuelle({}, {})
    st.stop()

index, conflits = construire_index(donnees_multi)

with st.spinner("Calcul des corrections d'apprentissage..."):
    corrections_par_ligue = {}
    for nom, (code, d) in donnees_multi.items():
        rec_log = resultats_journal(code, d)
        rec_bt = backtest(d)
        morceaux = [x for x in (rec_log, rec_bt) if not x.empty]
        rec = pd.concat(morceaux, ignore_index=True) if morceaux else rec_vide()
        corrections_par_ligue[nom] = corrections(rec)

if conflits:
    with st.expander(f"⚠️ {len(conflits)} équipe(s) présente(s) dans plusieurs championnats"):
        for eq, lg in conflits.items():
            st.write(f"- **{eq}** : {', '.join(lg)} (la 1ʳᵉ ligue est utilisée)")

onglet1, onglet2 = st.tabs(["Pronostics", "Appre
