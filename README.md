# 📊 Aplikacija za Numeričku Aproksimaciju

**Interaktivna edukativna aplikacija za učenje i primjenu metoda numeričke aproksimacije, integracije i diferencijacije tabličnih podataka — sa step-by-step objašnjenjima i vizualizacijama.**

---

## 🚀 Pokretanje i Korištenje

### Preduslovi

- **Python 3.8+** instaliran na sistemu
- **pip** (Python package manager)

### Instalacija

1. **Instalirati zavisnosti** iz `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   Ovo će instalirati sve potrebne biblioteke:
   - `streamlit` - web framework za interaktivnu aplikaciju
   - `numpy` - numeričke operacije
   - `scipy` - naučne i numeričke metode
   - `sympy` - simbolička matematika
   - `plotly` - interaktivni grafovi
   - `matplotlib` - dodatna vizualizacija
   - `pandas` - rad sa podacima

### Pokretanje Aplikacije

1. **Otvorite terminal/komandnu liniju** u direktorijumu projekta

2. **Pokrenite Streamlit aplikaciju**:
   ```bash
   streamlit run Pocetna.py
   ```

3. **Aplikacija će se automatski otvoriti** u vašem web pretraživaču na adresi `http://localhost:8501`

   > **Napomena:** Ako se aplikacija ne otvori automatski, kopirajte URL koji se prikaže u terminalu i otvorite ga ručno.

### Kako Koristiti Aplikaciju

1. **Navigacija**
   - Koristite **bočnu navigaciju** (lijevi sidebar) za odabir stranice:
     - 📊 **Aproksimacija** - glavna funkcionalnost za aproksimaciju podataka
     - ∫ **Integracija Tablice** - računanje integrala iz tabličnih podataka
     - ∂ **Derivacija Tablice** - računanje derivacija iz tabličnih podataka

2. **Unos Podataka**
   - **Predefinisani primjer** - odaberite jedan od ugrađenih primjera
   - **Vlastiti podaci** - unesite podatke ručno (zarezom odvojene vrijednosti)
   - **Učitaj iz datoteke** - učitajte CSV, Excel ili TXT datoteku

3. **Odabir Metode**
   - Na stranici **Aproksimacija**, odaberite metodu aproksimacije iz padajućeg menija
   - Za **Integraciju** i **Derivaciju**, odaberite metodu aproksimacije i parametre metode

4. **Pokretanje Izračuna**
   - Kliknite na dugme **"🚀 Izračunaj"**
   - Rezultati će se prikazati sa detaljnim koracima rješavanja

5. **Analiza Rezultata**
   - Pregledajte **grafove** sa aproksimiranom funkcijom
   - Pročitajte **step-by-step prikaz** svakog koraka
   - Analizirajte **metrike kvalitete** (R², SSE, MSE, RMSE)
   - Pregledajte **tabele** sa predviđenim vrijednostima i greškama

---

## 📁 Struktura Projekta

```
Aplikacija_za_numericku_aproksimaciju/
│
├── Pocetna.py                    # Glavna Streamlit aplikacija (entry point)
├── requirements.txt               # Python zavisnosti
├── README.md                      # Dokumentacija (ovaj fajl)
│
├── pages/                         # Streamlit stranice aplikacije
│   ├── 1_📊_Aproksimacija.py      # Stranica za aproksimaciju funkcija
│   ├── 2_∫_Integracija_Tablice.py # Stranica za numeričku integraciju
│   └── 3_∂_Derivacija_Tablice.py  # Stranica za numeričku derivaciju
│
├── methods/                       # Moduli sa implementiranim metodama
│   ├── __init__.py
│   ├── regression.py              # Metode aproksimacije (6 metoda)
│   ├── integration.py             # Metode numeričke integracije
│   └── differentiation.py         # Metode numeričke derivacije
│
└── utils/                         # Pomoćni moduli
    ├── __init__.py
    └── plotting.py                 # Funkcije za vizualizaciju (Plotly)
```

---

## 🔧 Funkcionalnosti

### 1. 📊 Numerička Aproksimacija Funkcija

**Glavna funkcionalnost aplikacije** - implementira **6 metoda aproksimacije** podataka metodom najmanjih kvadrata:

#### Implementirane Metode:

1. **Linearna Aproksimacija** (`y = ax + b`)
   - Direktna primjena metode najmanjih kvadrata
   - Najjednostavnija metoda, dobra za linearne trendove

2. **Stepena Aproksimacija** (`y = A·x^B`)
   - Linearizacija: `ln(y) = ln(A) + B·ln(x)`
   - Zahtijeva: `x > 0, y > 0`
   - Koristi se za zakone proporcionalnosti

3. **Eksponencijalna Aproksimacija** (`y = A·e^(Bx)`)
   - Linearizacija: `ln(y) = ln(A) + Bx`
   - Zahtijeva: `y > 0`
   - Za eksponencijalni rast/opadanje

4. **Logaritamska Aproksimacija** (`y = a + b·ln(x)`)
   - Već linearan oblik: `y = a + b·X`, gdje je `X = ln(x)`
   - Zahtijeva: `x > 0`
   - Za logaritamske veze i saturaciju

5. **Racionalna Aproksimacija** (`y = P_r(x)/Q_s(x)`)
   - Opšti model: `(b₀ + b₁x + ... + bᵣxʳ) / (1 + c₁x + ... + cₛxˢ)`
   - Linearizacija kroz least squares sistem
   - Za kompleksnije nelinearne veze

6. **Polinomijalna Aproksimacija** (`y = a₀ + a₁x + a₂x² + ... + aₙxⁿ`)
   - Proizvoljni stepen polinoma (1-10)
   - Koristi sistem normalnih jednačina
   - Najfleksibilnija metoda

#### Dodatne Funkcionalnosti:

- **Automatsko poređenje modela** - rangiranje svih metoda po R² koeficijentu
- **Preporuka najboljeg modela** - automatska preporuka na osnovu R² vrijednosti
- **Step-by-step prikaz** - detaljni prikaz svakog koraka rješavanja
- **Interaktivni grafovi** - vizualizacija podataka i aproksimirane funkcije (Plotly)
- **Statistike kvalitete** - R², SSE, MSE, RMSE, Adjusted R²
- **Tabele rezultata** - predviđene vrijednosti, reziduali, greške u procentima

### 2. ∫ Numerička Integracija iz Tablice

**Primjena aproksimacije za računanje integrala** kada imamo samo tablične podatke:

#### Proces:

1. **Aproksimacija podataka** - odabrana metoda aproksimacije rekonstruiše funkciju iz tablice
2. **Integracija aproksimirane funkcije** - primjena numeričke metode integracije

#### Implementirane Metode Integracije:

- **Trapezna metoda** - aproksimacija linearnom funkcijom, greška O(h²)
- **Simpsonova metoda (1/3)** - aproksimacija parabolom, greška O(h⁴), preciznija

#### Funkcionalnosti:

- Odabir metode aproksimacije (linearna, kvadratna, kubna, eksponencijalna, stepena, logaritamska)
- Automatski odabir najbolje metode (najbolji R²)
- Podešavanje broja podintervala (n)
- Vizualizacija aproksimirane funkcije i područja ispod krivulje
- Detaljni koraci integracije

### 3. ∂ Numerička Derivacija iz Tablice

**Primjena aproksimacije za računanje derivacija** kada imamo samo tablične podatke:

#### Proces:

1. **Aproksimacija podataka** - odabrana metoda aproksimacije rekonstruiše funkciju
2. **Derivacija aproksimirane funkcije** - analitička derivacija dobijene funkcije

#### Implementirane Metode Derivacije:

- **Forward Difference** - unaprijedna diferencija, O(h), za lijevi rub
- **Backward Difference** - unazadna diferencija, O(h), za desni rub
- **Central Difference** - centralna diferencija, O(h²), najpreciznija, za unutrašnjost
- **Automatska detekcija** - automatski bira najbolju metodu na osnovu položaja tačke

#### Funkcionalnosti:

- Odabir metode aproksimacije
- Automatski odabir najbolje metode
- Podešavanje koraka h
- Automatska detekcija metode derivacije (forward/backward/central)
- Vizualizacija funkcije i njenih derivacija
- Tabela derivacija u svim tačkama
- Interpretacija trenda (rastuća/opadajuća funkcija)

---

## 🛠️ Tehnički Detalji

### Arhitektura

- **Frontend:** Streamlit (Python web framework)
- **Backend:** Python 3.8+ sa NumPy, SciPy, SymPy
- **Vizualizacija:** Plotly (interaktivni grafovi)
- **Struktura:** Modularna organizacija koda

### Moduli

#### `methods/regression.py`
- Implementira sve metode aproksimacije
- Svaka metoda vraća detaljne korake za step-by-step prikaz
- Funkcije: `linear_regression()`, `exponential_regression()`, `polynomial_regression()`, `power_regression()`, `logarithmic_regression()`, `rational_regression()`, `compare_regression_models()`

#### `methods/integration.py`
- Implementira trapeznu i Simpsonovu metodu
- Funkcije: `trapezoidal()`, `simpson()`

#### `methods/differentiation.py`
- Implementira forward, backward i central difference metode
- Automatska detekcija metode
- Funkcije: `auto_differentiate()`

#### `utils/plotting.py`
- Funkcije za kreiranje interaktivnih grafova
- Funkcija: `plot_regression()`

---

## 📝 Napomene

- Aplikacija je razvijena kao **edukativni alat** za učenje numeričkih metoda
- Sve metode su implementirane sa **detaljnim step-by-step objašnjenjima**
- Podržani su različiti formati datoteka za unos podataka (CSV, Excel, TXT)
- Aplikacija automatski detektuje najbolju metodu kada je to moguće

---

**Napomena:** Ako naiđete na probleme pri pokretanju ili korištenju aplikacije, provjerite da li su sve zavisnosti ispravno instalirane i da li koristite kompatibilnu verziju Pythona (3.8+).
