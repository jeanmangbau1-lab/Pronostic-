pandas
streamlit
import pandas as pd

saisons = ["2122", "2223", "2324", "2425"]
liens = [f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv" for s in saisons]

morceaux = []
for lien in liens:
    try:
        morceaux.append(pd.read_csv(lien))
    except Exception as e:
        print("Saison ignorée :", lien, e)

df = pd.concat(morceaux, ignore_index=True)
df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
print(len(df), "matchs chargés")
print(df.head())
