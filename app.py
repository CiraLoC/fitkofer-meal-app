# -*- coding: utf-8 -*-
"""
Fitkofer Kalkulator Obroka — Streamlit mini-app
Autor: ChatGPT (za Fitkofer)
Python 3.10+
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
READ_ONLY = True  # postavi False za lokalni rad ako želiš da dodaješ svoje namirnice

# ----------------------------
# Osnovna podešavanja stranice
# ----------------------------
st.set_page_config(
    page_title="Fitkofer Kalkulator Obroka",
    page_icon="🍽️",
    layout="wide",
)

# ----------------------------
# Putanje i konstante
# ----------------------------
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
FOODS_CSV = DATA_DIR / "foods.csv"
MY_FOODS_CSV = DATA_DIR / "my_foods.csv"
LOGS_DIR = DATA_DIR / "logs"

REQUIRED_COLUMNS = ["namirnica", "kcal_100g", "protein_100g", "mast_100g", "uh_100g", "izvor"]

# ----------------------------
# Inicijalizacija foldera/fajlova
# ----------------------------
def ensure_directories_and_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Ako my_foods.csv ne postoji, napravi ga sa header-om
    if not MY_FOODS_CSV.exists():
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(MY_FOODS_CSV, index=False, encoding="utf-8")

    # Ako foods.csv ne postoji, napravi prazan (korisnik može da ubaci kasnije)
    if not FOODS_CSV.exists():
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(FOODS_CSV, index=False, encoding="utf-8")

ensure_directories_and_files()

# ----------------------------
# Pomoćne funkcije
# ----------------------------
def normalize_float(x: Any) -> float:
    """
    Prihvata string sa tačkom ili zarezom i vraća float.
    Prazno ili neispravno -> None (koristi se za validaciju).
    """
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def load_food_db() -> pd.DataFrame:
    """Učitaj glavnu bazu i 'moje' namirnice; spoji i očisti kolone."""
    def _read_csv(path: Path) -> pd.DataFrame:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path, encoding="utf-8")
        else:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        # Uskladi nazive kolona
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        return df[REQUIRED_COLUMNS]

    base = _read_csv(FOODS_CSV)
    mine = _read_csv(MY_FOODS_CSV)
    all_foods = pd.concat([base, mine], ignore_index=True).dropna(subset=["namirnica"])
    # Tipovi kolona
    for c in ["kcal_100g", "protein_100g", "mast_100g", "uh_100g"]:
        all_foods[c] = pd.to_numeric(all_foods[c], errors="coerce").fillna(0.0)
    all_foods["namirnica"] = all_foods["namirnica"].astype(str)
    all_foods["izvor"] = all_foods["izvor"].fillna("").astype(str)
    all_foods = all_foods.drop_duplicates(subset=["namirnica"], keep="last").sort_values("namirnica")
    return all_foods.reset_index(drop=True)

def add_my_food(row: Dict[str, Any]) -> None:
    """Dodaj stavku u my_foods.csv"""
    df = pd.DataFrame([row], columns=REQUIRED_COLUMNS)
    if not MY_FOODS_CSV.exists() or MY_FOODS_CSV.stat().st_size == 0:
        df.to_csv(MY_FOODS_CSV, index=False, encoding="utf-8")
    else:
        df.to_csv(MY_FOODS_CSV, mode="a", header=False, index=False, encoding="utf-8")

def calc_for_grams(food_row: pd.Series, grams: float) -> Dict[str, float]:
    """Izračunaj Kcal/Protein/Mast/UH za zadate grame."""
    factor = grams / 100.0
    return {
        "kcal": round(food_row["kcal_100g"] * factor, 2),
        "protein": round(food_row["protein_100g"] * factor, 2),
        "mast": round(food_row["mast_100g"] * factor, 2),
        "uh": round(food_row["uh_100g"] * factor, 2),
    }

def init_session_state():
    if "log" not in st.session_state:
        st.session_state["log"] = []  # lista dict-ova: {namirnica, grami, kcal, protein, mast, uh}
    if "foods_reload_key" not in st.session_state:
        st.session_state["foods_reload_key"] = 0

init_session_state()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🍎 Fitkofer — 30 Dana Discipline")
st.sidebar.markdown(
    "Prati unos kalorija i makroa po obroku/danu. "
    "Unos radiš u gramima, a vrednosti su **na 100 g**. "
    "Proteini iz biljnih izvora su ostavljeni kao 0 u bazi (po tvom pravilu)."
)

daily_target = st.sidebar.text_input("Dnevni kalorijski cilj (kcal) — opcionalno", value="")
daily_target_val = normalize_float(daily_target)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 **Napomena:** Ova aplikacija pruža opšte informativne podatke, "
    "nije zamena za medicinski savet. Za specifične zdravstvene probleme konsultuj lekara."
)

# ----------------------------
# Glavni sadržaj
# ----------------------------
st.title("Fitkofer Kalkulator Obroka")
st.write("Unesi obrok po obrok, beleži grame, i sačuvaj dnevnik kao CSV.")

# Učitaj aktuelnu bazu
foods = load_food_db()

col_left, col_right = st.columns([1.2, 1])

# ---- Leva kolona: Dodavanje u dnevnik ----
with col_left:
    st.subheader("➕ Dodaj namirnicu u dnevnik")
    search = st.text_input("Pretraga namirnica (piši deo naziva)", value="")
    if search:
        filtered = foods[foods["namirnica"].str.contains(search, case=False, na=False)]
    else:
        filtered = foods

    options = filtered["namirnica"].tolist()
    selected = st.selectbox("Izaberi namirnicu", options, index=0 if options else None, placeholder="npr. Pileća prsa")
    grams_str = st.text_input("Količina (g)", value="100")

    add_btn = st.button("Dodaj u dnevnik", type="primary", use_container_width=True)

    if add_btn:
        grams = normalize_float(grams_str)
        if not selected:
            st.warning("⚠️ Izaberi namirnicu iz liste.")
        elif grams is None or grams <= 0:
            st.warning("⚠️ Unesi količinu u gramima (pozitivan broj).")
        else:
            row = foods[foods["namirnica"] == selected].iloc[0]
            vals = calc_for_grams(row, grams)
            st.session_state["log"].append({
                "namirnica": selected,
                "grami": round(grams, 2),
                "kcal": vals["kcal"],
                "protein": vals["protein"],
                "mast": vals["mast"],
                "uh": vals["uh"],
            })
            st.success(f"✅ Dodato: {selected} — {grams:.0f} g")

    # Brze akcije
    c1, c2, c3 = st.columns(3)
    if c1.button("Obriši poslednji", use_container_width=True):
        if st.session_state["log"]:
            removed = st.session_state["log"].pop()
            st.info(f"Uklonjeno: {removed['namirnica']} ({removed['grami']} g)")
        else:
            st.info("Dnevnik je prazan.")
    if c2.button("Isprazni dnevnik", use_container_width=True):
        st.session_state["log"].clear()
        st.info("Dnevnik je prazan.")
    if c3.button("Osveži bazu", use_container_width=True):
        st.session_state["foods_reload_key"] += 1
        foods = load_food_db()
        st.success("Baza osvežena.")

    st.markdown("---")
if not READ_ONLY:
    st.subheader("🆕 Dodaj *moju* namirnicu (na 100 g)")
    mf_col1, mf_col2 = st.columns(2)
    with mf_col1:
        mf_name = st.text_input("Naziv")
        mf_kcal = st.text_input("Kcal / 100 g", value="")
        mf_prot = st.text_input("Proteini / 100 g (g)", value="")
    with mf_col2:
        mf_fat = st.text_input("Masti / 100 g (g)", value="")
        mf_carb = st.text_input("UH / 100 g (g)", value="")
        mf_src = st.text_input("Izvor (opciono)", value="Korisnik")

    save_my_food = st.button("Sačuvaj moju namirnicu", use_container_width=True)
    if save_my_food:
        # Validacija i normalizacija
        kcal = normalize_float(mf_kcal)
        prot = normalize_float(mf_prot)
        fat = normalize_float(mf_fat)
        carb = normalize_float(mf_carb)
        if not mf_name.strip():
            st.warning("⚠️ Unesi naziv.")
        elif any(v is None or v < 0 for v in [kcal, prot, fat, carb]):
            st.warning("⚠️ Kcal/proteini/masti/UH moraju biti nenegativni brojevi.")
        else:
            add_my_food({
                "namirnica": mf_name.strip(),
                "kcal_100g": kcal,
                "protein_100g": prot,
                "mast_100g": fat,
                "uh_100g": carb,
                "izvor": (mf_src or "Korisnik").strip(),
            })
            st.success(f"✅ Dodato u 'Moje namirnice': {mf_name.strip()}")
            # Osveži lokalnu kopiju
            foods = load_food_db()

# ---- Desna kolona: Pregled dnevnika ----
with col_right:
    st.subheader("📒 Dnevnik dana")
    if st.session_state["log"]:
        df_log = pd.DataFrame(st.session_state["log"])
        st.dataframe(df_log, use_container_width=True, height=360)

        totals = df_log[["kcal", "protein", "mast", "uh"]].sum().to_dict()
        st.markdown(
            f"**Ukupno:** 🔥 {totals['kcal']:.0f} kcal | 🧬 Protein {totals['protein']:.1f} g | 🧈 Masti {totals['mast']:.1f} g | 🍞 UH {totals['uh']:.1f} g"
        )

        if isinstance(daily_target_val, float) and daily_target_val > 0:
            diff = totals["kcal"] - daily_target_val
            st.caption(f"Cilj: {daily_target_val:.0f} kcal → Razlika: **{diff:+.0f} kcal**")

        # Bar grafikon (protein, masti, UH)
        fig, ax = plt.subplots()
        ax.bar(["Protein (g)", "Masti (g)", "UH (g)"], [totals["protein"], totals["mast"], totals["uh"]])
        ax.set_title("Makroi (ukupno)")
        ax.set_ylabel("grami")
        st.pyplot(fig, use_container_width=True)

        # Izvoz
        default_name = f"dnevnik_{datetime.now().date().isoformat()}.csv"
        export_name = st.text_input("Naziv fajla za izvoz (CSV)", value=default_name)
        # 1) Preuzimanje direktno (za online deljenje najbitnije)
        csv_bytes = df_log.to_csv(index=False, encoding="utf-8").encode("utf-8")
        st.download_button("⬇️ Preuzmi dnevnik (CSV)", data=csv_bytes, file_name=export_name, mime="text/csv", use_container_width=True)
        
        if st.button("Sačuvaj dnevnik u data/logs", use_container_width=True):
            out_path = LOGS_DIR / export_name
            df_log.to_csv(out_path, index=False, encoding="utf-8")
            st.success(f"💾 Sačuvano: {out_path}")
    else:
        st.info("Dnevnik je prazan. Dodaj prvu namirnicu sa leve strane.")

# Donji deo: Info o bazi
st.markdown("---")
with st.expander("📚 Pregled baze namirnica (spoj foods.csv + my_foods.csv)"):
    st.dataframe(foods, use_container_width=True, height=300)
    st.caption("Saveti: koristi pretragu gore. Vrednosti su na 100 g. Biljni proteini su 0 po tvom pravilu.")
