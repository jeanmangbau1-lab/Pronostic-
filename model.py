
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import re
import unicodedata
from math import exp, factorial
from PIL import Image

st.set_page_config(page_title="Pronostics Football", layout="centered")

CHAMPIONNATS = {
    "Angleterre": "E0",
    "Écosse": "SC0",
    "Allemagne": "D1",
    "Italie": "I1",
    "Espagne": "SP1",
    "France": "F1",
    "Pays-Bas": "N1",
    "Belgique": "B1",
    "Portugal": "P1",
    "Turquie": "T1",
    "Grèce": "G1",
    "Argentine": "ARG",
    "Autriche": "AUT",
    "Brésil": "BRA",
    "Chine": "CHN",
    "Danemark": "DNK",
    "Finlande": "FIN",
    "Irlande": "IRL",
    "Japon": "JPN",
    "Mexique": "MEX",
    "Norvège": "NOR",
    "Pologne": "POL",
    "Roumanie": "ROU",
    "Russie": "RUS",
    "Suède": "SWE",
    "Suisse": "SWZ",
    "États-Unis": "USA",
}
SAISONS = ["2223", "2324", "2425", "2526", "2627"]

ALIAS = {
    "manchester city": "Man City", "man city": "Man City",
    "manchester united": "Man United", "man utd": "Man United",
    "man united": "Man United",
    "nottingham forest": "Nott'm Forest", "nottingham": "Nott'm Forest",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "newcastle united": "Newcastle",
    "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "west ham united": "West Ham",
    "leeds united": "Leeds", "leicester city": "Leicester",
    "paris saint-germain": "Paris SG", "paris saint germain": "Paris SG",
    "psg": "Paris SG", "paris": "Paris SG",
    "olympique de marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "saint-etienne": "St Etienne", "saint etienne": "St Etienne",
    "sporting": "Sp Lisbon", "sporting cp": "Sp Lisbon",
    "sporting lisbonne": "Sp Lisbon", "sporting lisbon": "Sp Lisbon",
    "sporting braga": "Sp Braga", "braga": "Sp Braga",
    "fc porto": "Porto", "sl benfica": "Benfica",
    "vitoria guimaraes": "Guimaraes",
}


def norm(t):
    t = unicodedata.normalize("NFD", str(t))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.lower()


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
            morceaux.append(
                pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{s}/{code}.csv")
            )
        except Exception:
            pass
    if not morceaux:
        return pd.DataFrame()
    d = pd.concat(morceaux, ignore_index=True)
    d = d[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    d["FTHG"] = d["FTHG"].astype(int)
    d["FTAG"] = d["FTAG"].astype(int)
    return d.reset_index(drop=True)


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


def predire(dom, ext, att, dfn, mh, ma):
    lh = att[dom] * dfn[ext] * mh
    la = att[ext] * dfn[dom] * ma
    p1 = pn = p2 = 0.0
    meilleur, pmax = (0, 0), 0
    for i in range(8):
        for j in range(8):
            p = poisson(i, lh) * poisson(j, la)
            if i > j:
                p1 += p
            elif i == j:
                pn += p
            else:
                p2 += p
            if p > pmax:
                pmax, meilleur = p, (i, j)
    t = p1 + pn + p2
    return p1 / t, pn / t, p2 / t, meilleur


def marches(dom, ext, att, dfn, mh, ma):
    lh = att[dom] * dfn[ext] * mh
    la = att[ext] * dfn[dom] * ma
    scores, over, btts = [], 0.0, 0.0
    for i in range(10):
        for j in range(10):
            p = poisson(i, lh) * poisson(j, la)
            scores.append((p, i, j))
            if i + j > 2:
                over += p
            if i > 0 and j > 0:
                btts += p
    t = sum(s[0] for s in scores)
    scores.sort(reverse=True)
    top3 = [(i, j, p / t) for p, i, j in scores[:3]]
    return over / t, btts / t, top3


def carte(dom, ext, res):
    p1, pn, p2, sc = res
    over, btts, top3 = marches(dom, ext, att, dfn, mh, ma)
    st.markdown(f"**{dom} vs {ext}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("1", f"{p1 * 100:.0f}%")
    c2.metric("N", f"{pn * 100:.0f}%")
    c3.metric("2", f"{p2 * 100:.0f}%")
    d1, d2 = st.columns(2)
    d1.metric("+2,5 buts", f"{over * 100:.0f}%")
    d2.metric("Les 2 marquent", f"{btts * 100:.0f}%")
    st.caption(
        "Scores probables : "
        + " | ".join(f"{i}-{j} ({p * 100:.0f}%)" for i, j, p in top3)
    )
    st.divider()
