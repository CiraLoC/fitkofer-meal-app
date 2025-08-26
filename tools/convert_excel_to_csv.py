# -*- coding: utf-8 -*-
"""
Mali konverter: uzmi Excel sa listom namirnica i prebaci u data/foods.csv.
Pretpostavlja kolone: 'Namirnica' / 'Kcal na 100g' / 'Protein' / 'Mast' / 'UH' / 'Izvor'
Ako su nazivi drugačiji, prilagodi mapiranje ispod.
Pokretanje:
    python tools/convert_excel_to_csv.py path/to/your.xlsx
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "foods.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def main(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    # Pokušaj automatskog mapiranja kolona
    mapping = {
        "Namirnica": "namirnica",
        "Kcal na 100g": "kcal_100g",
        "Kcal na 100 g": "kcal_100g",
        "Protein": "protein_100g",
        "Mast": "mast_100g",
        "UH": "uh_100g",
        "Izvor": "izvor",
    }
    # Normalizuj nazive
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)

    needed = ["namirnica", "kcal_100g", "protein_100g", "mast_100g", "uh_100g"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"Nedostaje kolona: {c}")

    if "izvor" not in df.columns:
        df["izvor"] = ""

    # Normalizuj decimale i tipove
    for c in ["kcal_100g", "protein_100g", "mast_100g", "uh_100g"]:
        df[c] = (df[c].astype(str).str.replace(",", ".", regex=False))
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    out_df = df[["namirnica", "kcal_100g", "protein_100g", "mast_100g", "uh_100g", "izvor"]]
    out_df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Snimljeno u: {OUT}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Upotreba: python tools/convert_excel_to_csv.py putanja/do/fajla.xlsx")
    main(sys.argv[1])
