"""
Aplikacija za Numeričku Aproksimaciju
=====================================

Glavna Streamlit aplikacija koja demonstrira metode numeričke aproksimacije
sa detaljnim step-by-step objašnjenjima i interaktivnim vizualizacijama.

Pokretanje: streamlit run app.py
"""

import streamlit as st

# Konfiguracija stranice
st.set_page_config(
    page_title="Numerička Aproksimacija",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Naslov
st.title("📊 Aplikacija za Numeričku Aproksimaciju")

st.markdown("""
---

## Dobrodošli!

Ova aplikacija demonstrira različite **metode numeričke aproksimacije** sa detaljnim matematičkim
objašnjenjima, step-by-step prikazom rješavanja i interaktivnim grafovima.

### 📚 Implementirane Metode

Koristite **bočnu navigaciju** za pristup pojedinim stranicama:

""")

# Kartice sa metodama aproksimacije
st.markdown("""
#### 📊 Aproksimacija Funkcija — Metoda Najmanjih Kvadrata

Centralna funkcionalnost aplikacije. Implementirano je **6 metoda aproksimacije** podataka:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Linearna aproksimacija:**
    - Linearna regresija (y = ax + b)

    **Nelinearne aproksimacije:**
    - Stepena (y = Ax^B)
    - Eksponencijalna (y = Ae^Bx)
    - Logaritamska (y = a + b·ln(x))
    """)

with col2:
    st.markdown("""
    **Nelinearne aproksimacije (nastavak):**
    - Racionalna (y = P(x)/Q(x))
    - Polinomijalna (stepen n)

    **Automatsko poređenje:**
    - Rangiranje svih modela po R²
    - Preporuka najboljeg modela
    """)

st.markdown("---")

# Teorija metode najmanjih kvadrata
st.markdown("""
### 📐 Metoda Najmanjih Kvadrata

**Princip:** Pronaći funkciju koja najbolje opisuje date podatke minimiziranjem sume kvadrata odstupanja.

$$S = \\sum_{i=1}^{n} [y_i - f(x_i)]^2 \\rightarrow \\min$$

""")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("""
    #### Linearizacija Nelinearnih Modela

    Mnogi nelinearni modeli se mogu transformisati u linearne:

    | Model | Transformacija |
    |-------|----------------|
    | $y = Ax^B$ | $\\ln(y) = \\ln(A) + B\\ln(x)$ |
    | $y = Ae^{Bx}$ | $\\ln(y) = \\ln(A) + Bx$ |
    | $y = \\frac{x}{a+bx}$ | $\\frac{x}{y} = a + bx$ |
    """)

with col_t2:
    st.markdown("""
    #### Koeficijent Determinacije R²

    Mjeri koliko dobro model opisuje podatke:

    $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$

    | R² vrijednost | Interpretacija |
    |---------------|----------------|
    | ≥ 0.9 | Odličan fit |
    | 0.7 - 0.9 | Dobar fit |
    | 0.5 - 0.7 | Umjeren fit |
    | < 0.5 | Slab fit |
    """)

st.markdown("---")

# Primjeri primjene
st.markdown("""
### 🌍 Primjeri Primjene

Numerička aproksimacija se koristi u mnogim oblastima:
""")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown("""
    **Fizika i Inženjerstvo**
    - Analiza eksperimentalnih podataka
    - Kalibracija mjernih instrumenata
    - Predviđanje ponašanja sistema
    """)

with col_p2:
    st.markdown("""
    **Biologija i Medicina**
    - Rast populacije
    - Farmakokinetika
    - Epidemiološki modeli
    """)

with col_p3:
    st.markdown("""
    **Ekonomija i Finansije**
    - Trendovi tržišta
    - Prognoziranje prodaje
    - Analiza vremenskih serija
    """)

st.markdown("---")

# Informacije o projektu
st.markdown("""
### ∫∂ Primjena Aproksimacije u Drugim Numeričkim Metodama

Pored same aproksimacije, aplikacija demonstrira kako se metode aproksimacije mogu
koristiti kao temelj za rješavanje drugih numeričkih problema. Kada nemamo eksplicitnu
funkciju f(x), već samo tablicu izmjerenih vrijednosti (x, y), aproksimacija nam omogućava
da rekonstruišemo funkciju i primijenimo je dalje.
""")

col_id1, col_id2 = st.columns(2)

with col_id1:
    st.markdown("""
    #### ∫ Integracija iz Tablice

    Izračunavanje integrala iz diskretnih podataka
    koristeći aproksimiranu funkciju:

    1. Aproksimiraj podatke odabranom metodom
    2. Integriraj dobijenu funkciju numerički

    *Primjer: Iz tablice brzina vozila
    izračunaj ukupan pređeni put.*
    """)

with col_id2:
    st.markdown("""
    #### ∂ Derivacija iz Tablice

    Izračunavanje derivacije iz diskretnih podataka
    koristeći aproksimiranu funkciju:

    1. Aproksimiraj podatke odabranom metodom
    2. Deriviraj dobijenu funkciju analitički

    *Primjer: Iz tablice temperature tokom
    vremena odredi brzinu hlađenja.*
    """)

st.markdown("---")

st.markdown("""
### ℹ️ O Aplikaciji

Ova aplikacija je razvijena kao projektni zadatak iz predmeta
**Primjena Numeričkih Metoda u Softverskom Inženjerstvu**.

#### Karakteristike:
- 📐 **Metoda najmanjih kvadrata** sa linearizacijom nelinearnih modela
- 📊 **6 metoda aproksimacije** — linearna i nelinearne
- 📈 **Interaktivni grafovi** za vizualizaciju (Plotly)
- 📝 **Step-by-step prikaz** svakog koraka rješavanja
- 🏆 **Automatsko poređenje** i rangiranje modela po R²
- ∫∂ **Integracija i derivacija** kao primjena aproksimacije

---

### 🚀 Kako Koristiti

1. **Odaberite stranicu** iz bočne navigacije (lijeva strana)
2. **Unesite podatke** (vlastite ili predefinisane primjere)
3. **Odaberite metodu** aproksimacije
4. **Pokrenite izračun** i pratite korake
5. **Analizirajte rezultate** i grafove

""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Aplikacija za Numeričku Aproksimaciju | 2024</p>
    <p>Streamlit + Python + NumPy + Plotly</p>
</div>
""", unsafe_allow_html=True)
