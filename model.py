if st.button("🔍 Analyser le match"):
    if not equipe_dom.strip() or not equipe_ext.strip():
        st.error("Entre les deux noms des équipes.")
    else:
        pred = pronostiquer(
            historique,
            equipe_dom.strip(),
            equipe_ext.strip(),
            neutre
        )
        st.session_state["prediction_manuelle"] = {
            "prediction": pred,
            "domicile": equipe_dom.strip(),
            "exterieur": equipe_ext.strip(),
            "cote1": cote1,
            "coten": coten,
            "cote2": cote2
        }

if "prediction_manuelle" in st.session_state:
    resultat = st.session_state["prediction_manuelle"]
    pred = resultat["prediction"]

    st.markdown("### Résultat de l'analyse")
    a, b, c = st.columns(3)
    a.metric("1", f"{pred['1']*100:.1f}%")
    b.metric("N", f"{pred['N']*100:.1f}%")
    c.metric("2", f"{pred['2']*100:.1f}%")

    st.write(f"Over 2,5 : {pred['Over 2.5']*100:.1f}%")
    st.write(f"BTTS : {pred['BTTS']*100:.1f}%")

    for (x, y), p in pred["scores"]:
        st.write(
            f"{resultat['domicile']} {x} - {y} "
            f"{resultat['exterieur']} : {p*100:.1f}%"
        )

    if st.button("💾 Sauvegarder le match"):
        enregistrer_prediction({
            "date": str(date.today()),
            "domicile": resultat["domicile"],
            "exterieur": resultat["exterieur"],
            "cote1": resultat["cote1"],
            "coten": resultat["coten"],
            "cote2": resultat["cote2"],
            "p1": pred["1"],
            "pn": pred["N"],
            "p2": pred["2"],
            "over25": pred["Over 2.5"],
            "score1": pred["scores"][0][0][0],
            "score2": pred["scores"][0][0][1],
            "resultat_reel": ""
        })
        st.success("Match enregistré !")
        del st.session_state["prediction_manuelle"]
