import streamlit as st
import pandas as pd
import numpy as np
from math import exp, factorial
from datetime import date, timedelta

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


# ---------- DONNEES ----------
@st.cache_data(show_spinner=False)
def charger(code):
    if len(code) == 3 and code.isalpha():
        try:
            b = pd.read_csv(f"https://www.football-data.co.uk/new/{code}.csv")
            b = b.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                                  "HG": "FTHG", "AG": "FTAG"})
            b = b[b["Season"].astype(str).str[:4].astype(int) >= 2022]
            d = b[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
            d["FTHG"] = d["FTHG"].astype(int)
            d["FTAG"] = d["FTAG"].astype(int)
            return d.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
    morceaux = []
    for s in SAISONS:
        try:
            morceaux.append(pd.read_csv(
                f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv"))
        except Exception:
            pass
    if not morceaux:
        return pd.DataFrame()
    d = pd.concat(morceaux, ignore_index=True)
    d = d[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    d["FTHG"] = d["FTHG"].astype(int)
    d["FTAG"] = d["FTAG"].astype(int)
    return d.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=3600)
def calendrier():
    colonnes = ["code", "Date", "Time", "HomeTeam", "AwayTeam"]
    morceaux = []
    try:
        f = pd.read_csv("https://www.football-data.co.uk/fixtures.csv")
        f["code"] = f["Div"]
        morceaux.append(f.reindex(columns=colonnes))
    except Exception:
        pass
    try:
        n = pd.read_csv("https://www.football-data.co.uk/new_league_fixtures.csv")
        n = n.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
        inverse = {v: k for k, v in PAYS_NEW.items()}
        n["code"] = n["Country"].astype(str).str.strip().str.lower().map(inverse)
        morceaux.append(n.reindex(columns=colonnes))
    except Exception:
        pass
    if not morceaux:
        return pd.DataFrame(columns=colonnes)
    c = pd.concat(morceaux, ignore_index=True)
    c["Date"] = pd.to_datetime(c["Date"], dayfirst=True, errors="coerce")
    c = c.dropna(subset=["code", "Date", "HomeTeam", "AwayTeam"])
    c = c[c["code"].isin(TOUS_CODES)]
    return c.sort_values(["Date", "Time"]).reset_index(drop=True)


# ---------- MODELE DE POISSON ----------
def poisson(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


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


def calcul(dom, ext, att, dfn, mh, ma):
    lh = att[dom] * dfn[ext] * mh
    la = att[ext] * dfn[dom] * ma
    p1 = pn = p2 = over = btts = 0.0
    scores = []
    for i in range(10):
        for j in range(10):
            p = poisson(i, lh) * poisson(j, la)
            scores.append((p, i, j))
            if i > j:
                p1 += p
            elif i == j:
                pn += p
            else:
                p2 += p
            if i + j > 2:
                over += p
            if i > 0 and j > 0:
                btts += p
    t = p1 + pn + p2
    scores.sort(reverse=True)
    top3 = [(i, j, p / t) for p, i, j in scores[:3]]
    return p1 / t, pn / t, p2 / t, over / t, btts / t, top3


def carte(dom, ext, att, dfn, mh, ma):
    p1, pn, p2, over, btts, top3 = calcul(dom, ext, att, dfn, mh, ma)
    st.markdown(f"**{dom} vs {ext}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("1", f"{p1 * 100:.0f}%")
    c2.metric("N", f"{pn * 100:.0f}%")
    c3.metric("2", f"{p2 * 100:.0f}%")
    d1, d2 = st.columns(2)
    d1.metric("+2,5 buts", f"{over * 100:.0f}%")
    d2.metric("Les 2 marquent", f"{btts * 100:.0f}%")
    st.caption("Scores probables : " + " | ".join(
        f"{i}-{j} ({p * 100:.0f}%)" for i, j, p in top3))
    st.divider()


# ---------- INTERFACE ----------
st.title("⚽ Pronostics Football")

pays = st.selectbox("Pays", list(PAYS.keys()))
divisions = PAYS[pays]
div_nom = st.selectbox("Division", list(divisions.keys()))
code = divisions[div_nom]

ref = st.date_input("Choisis une date de la semaine", value=date.today())
lundi = ref - timedelta(days=ref.weekday())
dimanche = lundi + timedelta(days=6)
st.caption(f"Semaine du {lundi:%d/%m/%Y} au {dimanche:%d/%m/%Y}")
choix = st.selectbox("Jour", ["Toute la semaine"] + JOURS)

with st.spinner("Chargement..."):
    donnees = charger(code)
    cal = calendrier()

if donnees.empty:
    st.error("Impossible de charger les données de ce championnat.")
    st.stop()

att, dfn, mh, ma = forces(donnees)

cal = cal[(cal["code"] == code)
          & (cal["Date"] >= pd.Timestamp(lundi))
          & (cal["Date"] <= pd.Timestamp(dimanche))]
if choix != "Toute la semaine":
    cal = cal[cal["Date"].dt.weekday == JOURS.index(choix)]

st.subheader("Programme des matchs")
total
