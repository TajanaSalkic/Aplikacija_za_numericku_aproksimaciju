"""
Aplikacija za Numeričku Aproksimaciju
=====================================

Glavna Streamlit aplikacija koja demonstrira različite numeričke metode
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

Ova aplikacija demonstrira različite **numeričke metode** sa detaljnim matematičkim
objašnjenjima, step-by-step prikazom rješavanja i interaktivnim grafovima.

### 📚 Implementirane Metode

Koristite **bočnu navigaciju** za pristup pojedinim metodama:

""")

# Kartice sa metodama
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🔍 Traženje Nula Funkcije
    - **Metoda Bisekcije** ✓
    - **Newton-Raphson** ✓
    - **Metoda Sekante** ✓

    *Pronalaženje korijena jednačine f(x) = 0*
    """)

with col2:
    st.markdown("""
    #### ∫ Numerička Integracija
    - **Trapezna metoda** ✓
    - **Simpsonova metoda** ✓
    - **Romberg integracija** ✓
    - **Gaussova kvadratura** ✓

    *Aproksimacija određenog integrala*
    """)

with col3:
    st.markdown("""
    #### ∂ Numerička Derivacija
    - **Forward Difference** ✓
    - **Backward Difference** ✓
    - **Central Difference** ✓
    - **Poređenje grešaka** ✓

    *Aproksimacija derivacija*
    """)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    #### 🔢 Sistemi Jednačina
    - **Jacobijeva metoda** ✓
    - **Gauss-Seidelova metoda** ✓

    *Iterativne metode za Ax = b*
    """)

with col5:
    st.markdown("""
    #### 📈 Regresija i Aproksimacija
    - **Linearna regresija** ✓
    - **Eksponencijalna** ✓
    - **Polinomijalna** ✓

    *Fitovanje podataka*
    """)

with col6:
    st.markdown("""
    #### 🌍 Primjeri iz Stvarnog Života
    - **Fizika/Inženjerstvo**
    - **Biologija/Medicina**
    - **Ekonomija**

    *Praktična primjena metoda*
    """)

st.markdown("---")

# Informacije o projektu
st.markdown("""
### ℹ️ O Aplikaciji

Ova aplikacija je razvijena kao projektni zadatak iz predmeta
**Primjena Numeričkih Metoda u Softverskom Inženjerstvu**.

#### Karakteristike:
- 🔬 **Detaljno matematičko objašnjenje** svake metode sa LaTeX formulama
- 📊 **Interaktivni grafovi** za vizualizaciju (Plotly)
- 📝 **Step-by-step prikaz** svakog koraka rješavanja
- 🌍 **Primjeri iz stvarnog života** iz fizike, biologije i ekonomije
- 🎯 **Poređenje metoda** - brzina konvergencije, preciznost

#### Tehnologije:
- **Python** - programski jezik
- **Streamlit** - web framework
- **NumPy/SciPy** - numeričke kalkulacije
- **Plotly** - interaktivni grafovi
- **SymPy** - simboličko računanje

---

### 🚀 Kako Koristiti

1. **Odaberite kategoriju** iz bočne navigacije (lijeva strana)
2. **Unesite parametre** (funkciju, interval, toleranciju...)
3. **Pokrenite izračun** i pratite korake
4. **Analizirajte rezultate** i grafove

""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Aplikacija za Numeričku Aproksimaciju | 2024</p>
    <p>Streamlit + Python + Plotly</p>
</div>
""", unsafe_allow_html=True)
