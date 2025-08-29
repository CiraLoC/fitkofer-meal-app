# -*- coding: utf-8 -*-
"""
Fitkofer Kalkulator Obroka — Streamlit mini-app
Python 3.12+
Verzija: obroci + datum + CSV izvoz (bez JSON sesije i bez snimanja u data/logs)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Podešavanje stranice
# ----------------------------
st.set_page_config(
    page_title="Fitkofer Kalkulator Obroka",
    page_icon="🍽️",
    layout="wide",
)

# ----------------------------
# Konstante i putanje
# ----------------------------
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
FOODS_CSV = DATA_DIR / "foods.csv"
MY_FOODS_CSV = DATA_DIR / "my_foods.csv"

REQUIRED_COLUMNS = ["namirnica", "kcal_100g", "protein_100g", "mast_100g", "uh_100g", "izvor"]

# Lista obroka (slobodno menjaj)
MEAL_OPTIONS = ["Doručak", "Užina 1", "Ručak", "Užina 2", "Večera"]

# READ_ONLY: True = sakrivena sekcija "Moja namirnica" (za cloud),
# False = vidiš je lokalno i možeš da dodaješ.
READ_ONLY = True

# (Opcionalno) Jednostavna lozinka preko Streamlit "secrets"
if "APP_PASSWORD" in st.secrets:
    pwd = st.text_input("Lozinka za pristup", type="password")
    if pwd != st.secrets["APP_PASSWORD"]:
        st.stop()

# ----------------------------
# Inicijalizacija foldera/fajlova
# ----------------------------
def ensure_directories_and_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MY_FOODS_CSV.exists():
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(MY_FOODS_CSV, index=False, encoding="utf-8")
    if not FOODS_CSV.exists():
        pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(FOODS_CSV, index=False, encoding="utf-8")

ensure_directories_and_files()

# ----------------------------
# Helper funkcije
# ----------------------------
def normalize_float(x: Any) -> float | None:
    """Prihvata tačku ili zarez i vrati float; nevažeće -> None."""
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
    """Učitaj foods.csv + my_foods.csv i spoji u jednu bazu."""
    def _read_csv(path: Path) -> pd.DataFrame:
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path, encoding="utf-8")
        else:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        for c in REQUIRED_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[REQUIRED_COLUMNS]

    base = _read_csv(FOODS_CSV)
    mine = _read_csv(MY_FOODS_CSV)
    all_foods = pd.concat([base, mine], ignore_index=True).dropna(subset=["namirnica"])
    for c in ["kcal_100g", "protein_100g", "mast_100g", "uh_100g"]:
        all_foods[c] = pd.to_numeric(all_foods[c], errors="coerce").fillna(0.0)
    all_foods["namirnica"] = all_foods["namirnica"].astype(str)
    all_foods["izvor"] = all_foods["izvor"].fillna("").astype(str)
    all_foods = all_foods.drop_duplicates(subset=["namirnica"], keep="last").sort_values("namirnica")
    return all_foods.reset_index(drop=True)

def add_my_food(row: Dict[str, Any]) -> None:
    """Dodaj u my_foods.csv (koristi se kad READ_ONLY=False)."""
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
        # svaki unos je dict: {datum, obrok, namirnica, grami, kcal, protein, mast, uh}
        st.session_state["log"] = []
    if "foods_reload_key" not in st.session_state:
        st.session_state["foods_reload_key"] = 0
    if "current_date" not in st.session_state:
        st.session_state["current_date"] = datetime.now().date()
    if "current_meal" not in st.session_state:
        st.session_state["current_meal"] = MEAL_OPTIONS[0]

init_session_state()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🍎 Fitkofer — 30 Dana Discipline")
st.sidebar.markdown(
    "Prati unos kalorija i makroa po obroku **i po danu**. "
    "Unosiš grame; vrednosti su na **100 g**. "
    "Biljni proteini su 0 u bazi (po pravilu programa)."
)
daily_target = st.sidebar.text_input("Dnevni kalorijski cilj (kcal) — opcionalno", value="")
daily_target_val = normalize_float(daily_target)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 **Napomena:** Ova aplikacija pruža opšte informativne podatke i nije zamena za medicinski savet. "
    "Za specifične zdravstvene probleme konsultuj lekara."
)

# ----------------------------
# Glavni sadržaj
# ----------------------------
st.title("Fitkofer Kalkulator Obroka")
st.write("Unesi obrok po obrok, beleži grame, i preuzmi dnevnik kao CSV za izabrani datum.")

foods = load_food_db()

col_left, col_right = st.columns([1.2, 1])

# ---- Leva kolona: Unos ----
with col_left:
    st.subheader("➕ Dodaj namirnicu u dnevnik")

    # Datum + Obrok
    cdt, cml = st.columns([1, 1])
    with cdt:
        st.session_state["current_date"] = st.date_input("Datum", value=st.session_state["current_date"])
    with cml:
        st.session_state["current_meal"] = st.selectbox(
            "Obrok", MEAL_OPTIONS, index=MEAL_OPTIONS.index(st.session_state["current_meal"])
        )

    # Pretraga + izbor + grami
    search = st.text_input("Pretraga namirnica (piši deo naziva)", value="")
    filtered = foods[foods["namirnica"].str.contains(search, case=False, na=False)] if search else foods
    options = filtered["namirnica"].tolist()

    if not options:
        st.warning("⚠️ Baza je prazna ili nema poklapanja za pretragu.")
        selected = None
    else:
        selected = st.selectbox("Izaberi namirnicu", options, index=0)

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
                "datum": st.session_state["current_date"].isoformat(),
                "obrok": st.session_state["current_meal"],
                "namirnica": selected,
                "grami": round(grams, 2),
                "kcal": vals["kcal"],
                "protein": vals["protein"],
                "mast": vals["mast"],
                "uh": vals["uh"],
            })
            st.success(f"✅ Dodato: {selected} — {grams:.0f} g ({st.session_state['current_meal']}, {st.session_state['current_date']})")

    # Brze akcije
    c1, c2, c3 = st.columns(3)
    if c1.button("Obriši poslednji", use_container_width=True):
        if st.session_state["log"]:
            removed = st.session_state["log"].pop()
            st.info(f"Uklonjeno: {removed['namirnica']} ({removed['grami']} g, {removed['obrok']})")
        else:
            st.info("Dnevnik je prazan.")
    if c2.button("Isprazni dnevnik", use_container_width=True):
        st.session_state["log"].clear()
        st.info("Dnevnik je prazan.")
    if c3.button("Osveži bazu", use_container_width=True):
        foods = load_food_db()
        st.success("Baza osvežena.")

    st.markdown("---")

    # Sekcija "Moja namirnica" (isključena za cloud)
    if not READ_ONLY:
        st.subheader("🆕 Dodaj *moju* namirnicu (na 100 g)")
        mf_col1, mf_col2 = st.columns(2)
        with mf_col1:
            mf_name = st.text_input("Naziv", key="mf_name")
            mf_kcal = st.text_input("Kcal / 100 g", value="", key="mf_kcal")
            mf_prot = st.text_input("Proteini / 100 g (g)", value="", key="mf_prot")
        with mf_col2:
            mf_fat = st.text_input("Masti / 100 g (g)", value="", key="mf_fat")
            mf_carb = st.text_input("UH / 100 g (g)", value="", key="mf_carb")
            mf_src = st.text_input("Izvor (opciono)", value="Korisnik", key="mf_src")

        save_my_food = st.button("Sačuvaj moju namirnicu", use_container_width=True)
        if save_my_food:
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
                foods = load_food_db()
    else:
        st.caption("ℹ️ Sekcija **„Moja namirnica“** je isključena na ovom linku. Ako treba dodavanje, radi lokalno (READ_ONLY=False).")

# ---- Desna kolona: Pregled / izvoz ----
with col_right:
    st.subheader("📒 Dnevnik dana")

    if st.session_state["log"]:
        df_log = pd.DataFrame(st.session_state["log"])
        today_str = st.session_state["current_date"].isoformat()
        today_mask = df_log["datum"] == today_str
        df_today = df_log.loc[today_mask, ["datum", "obrok", "namirnica", "grami", "kcal", "protein", "mast", "uh"]].copy()

        if df_today.empty:
            st.info("Za izabrani datum nema unosa. Dodaj stavke sa leve strane.")
        else:
            st.dataframe(df_today, use_container_width=True, height=300)

            # Zbir po obroku (za današnji dan)
            by_meal = df_today.groupby("obrok")[["kcal", "protein", "mast", "uh"]].sum().round(2)
            st.markdown("**Zbir po obroku (današnji dan):**")
            st.dataframe(by_meal, use_container_width=True, height=180)

            # Ukupno za dan (grafikon)
            totals = df_today[["kcal", "protein", "mast", "uh"]].sum().to_dict()
            st.markdown(
                f"**Ukupno za {today_str}:** 🔥 {totals['kcal']:.0f} kcal | 🧬 Protein {totals['protein']:.1f} g | "
                f"🧈 Masti {totals['mast']:.1f} g | 🍞 UH {totals['uh']:.1f} g"
            )
            if isinstance(daily_target_val, float) and daily_target_val > 0:
                diff = totals["kcal"] - daily_target_val
                st.caption(f"Cilj: {daily_target_val:.0f} kcal → Razlika: **{diff:+.0f} kcal**")

            fig, ax = plt.subplots()
            ax.bar(["Protein (g)", "Masti (g)", "UH (g)"], [totals["protein"], totals["mast"], totals["uh"]])
            ax.set_title("Makroi (ukupno)")
            ax.set_ylabel("grami")
            st.pyplot(fig, use_container_width=True)

            # Izvoz (CSV za današnji dan) — JEDINO dugme
            default_name = f"dnevnik_{today_str}.csv"
            export_name = st.text_input("Naziv fajla za izvoz (CSV)", value=default_name)
            csv_bytes = df_today.to_csv(index=False, encoding="utf-8").encode("utf-8")
            st.download_button("⬇️ Preuzmi dnevnik (CSV)", data=csv_bytes, file_name=export_name, mime="text/csv", use_container_width=True)

            st.caption("💡 Savet: čuvaj CSV na kraju dana ili posle svakog obroka po želji.")
    else:
        st.info("Dnevnik je prazan. Dodaj prvu namirnicu sa leve strane.")

# Donji deo: Baza
st.markdown("---")
with st.expander("📚 Pregled baze namirnica (spoj foods.csv + my_foods.csv)"):
    st.dataframe(load_food_db(), use_container_width=True, height=300)
    st.caption("Vrednosti su na 100 g. Biljni proteini su 0 po pravilu programa.")
