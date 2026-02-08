"""
Stranica: Numerička Aproksimacija
=================================

Glavna stranica za aproksimaciju funkcija metodom najmanjih kvadrata.
Implementira linearne i nelinearne metode aproksimacije.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.regression import (
    linear_regression,
    exponential_regression,
    polynomial_regression,
    power_regression,
    logarithmic_regression,
    rational_regression,
    compare_regression_models,
    get_all_approximation_methods
)
from utils.plotting import plot_regression

st.set_page_config(page_title="Aproksimacija", page_icon="📊", layout="wide")

st.title("📊 Numerička Aproksimacija Funkcija")
st.markdown("*Aproksimacija podataka metodom najmanjih kvadrata*")

# Sidebar
st.sidebar.header("⚙️ Postavke")

# Izbor metode
approximation_methods = {
    "Linearna regresija": {
        "function": linear_regression,
        "model": "y = ax + b",
        "description": "Metoda najmanjih kvadrata za linearni model",
        "requirements": "Nema posebnih zahtjeva",
        "linearization": "Direktna primjena (već linearan oblik)"
    },
    "Stepena aproksimacija": {
        "function": power_regression,
        "model": "y = A·x^B",
        "description": "Za podatke koji slijede stepeni zakon",
        "requirements": "x > 0, y > 0",
        "linearization": "ln(y) = ln(A) + B·ln(x)"
    },
    "Eksponencijalna aproksimacija": {
        "function": exponential_regression,
        "model": "y = A·e^(Bx)",
        "description": "Eksponencijalni rast ili opadanje",
        "requirements": "y > 0",
        "linearization": "ln(y) = ln(A) + Bx"
    },
    "Logaritamska aproksimacija": {
        "function": logarithmic_regression,
        "model": "y = a + b·ln(x)",
        "description": "Logaritamska veza između varijabli",
        "requirements": "x > 0",
        "linearization": "y = a + b·X, gdje je X = ln(x)"
    },
    "Racionalna aproksimacija": {
        "function": rational_regression,
        "model": "y = (b0 + b1x + ... + brx^r) / (1 + c1x + ... + csx^s)",
        "description": "Opšti racionalni model P_r(x)/Q_s(x) (least squares, uz c0=1)",
        "requirements": "Preporuka: N > (r + 1 + s); paziti da Q(x) ≠ 0",
        "linearization": "y(1 + c1x + ... + csx^s) ≈ b0 + b1x + ... + brx^r (linearni LS)"
    },
    "Polinomijalna aproksimacija": {
        "function": polynomial_regression,
        "model": "y = a₀ + a₁x + a₂x² + ... + aₙxⁿ",
        "description": "Polinomijalna aproksimacija proizvoljnog stepena",
        "requirements": "stepen < broj tačaka - 1",
        "linearization": "Vandermondova matrica"
    },
    "Poređenje svih modela": {
        "function": compare_regression_models,
        "model": "Svi dostupni modeli",
        "description": "Automatsko poređenje i rangiranje modela po R²",
        "requirements": "-",
        "linearization": "-"
    }
}

method = st.sidebar.selectbox(
    "Odaberite metodu aproksimacije:",
    list(approximation_methods.keys())
)

# Parametar za polinomijalnu
if method == "Polinomijalna aproksimacija":
    degree = st.sidebar.slider("Stepen polinoma:", 1, 10, 2)

# Parametri za opštu racionalnu aproksimaciju
if method == "Racionalna aproksimacija (opšta)":
    r_num = st.sidebar.slider("Stepen brojnika r (P_r):", 0, 5, 1)
    s_den = st.sidebar.slider("Stepen nazivnika s (Q_s):", 0, 5, 1)

# Unos podataka
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Unos Podataka")

data_input_method = st.sidebar.radio(
    "Način unosa:",
    ["Predefinisani primjer", "Vlastiti podaci", "Učitaj iz datoteke"]
)

# Predefinisani dataseti za različite tipove aproksimacije
predefined_datasets = {
    "Relativna gustina zraka ρ": {
        "x": [0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150],
        "y": [1.0, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741],
        "description": "Relativna gustina zraka ρ u funkciji visine h.",
        "task_text": (
            "*Relativna gustina zraka po visini*\n\n"
            "**ZADATAK (postavka):** Relativna gustina zraka ρ je mjerena na različitim visinama h.\n\n"
            "**Podaci:**\n"
            "- h (km): 0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150\n"
            "- ρ: 1.0000, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741\n\n"
            "**Zadatak:** Uraditi kvadratnu aproksimaciju metodom najmanjih kvadrata (polinom stepena 2) i odrediti ρ na **h = 10.5 km**."
        ),
        "suggested": "Polinomijalna aproksimacija",
        "solution_link": "https://www.scribd.com/document/968215224/NM-Chapter-4"
    },
     "Potrošnja goriva": {
        "x": [1310, 1810, 1175, 2360, 1960, 2020, 1755, 1595, 1470, 1430, 1110, 1785],
        "y": [10.2, 8.1, 11.9, 5.5, 6.8, 6.8, 7.7, 8.9, 9.8, 10.2, 13.2, 7.7],
        "description": "Potrošnja goriva φ (km/litar) u funkciji mase vozila M (kg) za Ford i Honda vozila iz 1999. godine.",
        "task_text": (
            "*Relativna gustina zraka po visini*\n\n"
            "**ZADATAK (postavka):** "
            "Tabela prikazuje masu vozila M (1310, 1810, 1175, 2360, 1960, 2020, 1755, 1595, 1470, 1430, 1110, 1785) i prosječnu potrošnju goriva φ (km/litar):10.2, 8.1, 11.9, 5.5, 6.8, 6.8, 7.7, 8.9, 9.8, 10.2, 13.2, 7.7, za motorna vozila marke Ford i Honda proizvedena 1999. godine. Potrebno je fitovati (aproksimirati) podatke pravom:  = a + bM i izračunati standardnu devijaciju (standardnu grešku aproksimacije)**."
        ),
        "suggested": "Linearna regresija (metoda najmanjih kvadrata)",
        "solution_link": ""
    },
    "Kinematička viskoznost vode μk(T)": {
        "x": [0, 21.1, 37.8, 54.4, 71.1, 87.8, 100],
        "y": [1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296],
        "description": "Kinematička viskoznost vode μk (10^-3 m^2/s) u funkciji temperature T (°C).",
        "task_text": (
            "*4.3. Kinematička viskoznost vode u funkciji temperature*\n\n"
            "**ZADATAK (postavka):** Kinematička viskoznost vode μk mijenja se s temperaturom T kako je dato u tabeli.\n\n"
            "**Podaci:**\n"
            "- T (°C): 0, 21.1, 37.8, 54.4, 71.1, 87.8, 100\n"
            "- μk (10^-3 m²/s): 1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296\n\n"
            "**Zadatak:** Odrediti **kubni polinom** (polinom 3. stepena) koji najbolje aproksimira podatke metodom najmanjih kvadrata,\n"
            "i pomoću njega izračunati μk za **T = 10°C, 30°C, 60°C, 90°C**."
    ),
    "suggested": "Polinomijalna aproksimacija (stepen 3)",
    "solution_link": ""
    },
    "Eksponencijalni rast": {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [1.0, 2.7, 7.4, 20.1, 54.6, 148.4],
        'description': 'Eksponencijalni rast (y ≈ e^x)',
        'suggested': 'Eksponencijalna aproksimacija'
    },
    "Stepena funkcija": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8],
        'y': [1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0],
        'description': 'Stepena funkcija (y ≈ x³)',
        'suggested': 'Stepena aproksimacija'
    },
    "Logaritamska aproksimacija": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [0.0, 0.7, 1.1, 1.4, 1.6, 1.8, 1.9, 2.1, 2.2, 2.3],
        'description': 'Logaritamski trend (y ≈ ln(x))',
        'suggested': 'Logaritamska aproksimacija'
    },
    "Kvadratni trend": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8],
        'y': [1.2, 4.1, 9.0, 16.2, 24.8, 36.1, 49.0, 64.2],
        'description': 'Kvadratni trend (y ≈ x²)',
        'suggested': 'Polinomijalna aproksimacija (stepen 2)'
    },
    "Saturacijska kriva": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [0.5, 0.67, 0.75, 0.8, 0.83, 0.86, 0.88, 0.89, 0.9, 0.91],
        'description': 'Saturacija (Michaelis-Menten tip)',
        'suggested': 'Racionalna aproksimacija'
    },
    "Rast populacije": {
        'x': [0, 1, 2, 3, 4, 5, 6],
        'y': [100, 122, 149, 182, 222, 271, 331],
        'description': 'Simulacija rasta populacije (~22% godišnje)',
        'suggested': 'Eksponencijalna aproksimacija'
    },
    "Hlađenje tijela": {
        'x': [0, 5, 10, 15, 20, 25, 30],
        'y': [80, 64.7, 52.5, 42.6, 34.6, 28.2, 23.0],
        'description': 'Newtonov zakon hlađenja',
        'suggested': 'Eksponencijalna aproksimacija'
    },
    "Specifična toplota zraka (niske T)": {
        'x': [300, 400, 500, 600, 700, 800, 900, 1000],
        'y': [1.0045, 1.0134, 1.0296, 1.0507, 1.0743, 1.0984, 1.1212, 1.1410],
        'description': 'Cp [J/mK·10³] za zrak na niskim temperaturama',
        'suggested': 'Linearna aproksimacija (Cp = a + bT)'
    },
    "Specifična toplota zraka (visoke T)": {
        'x': [1000, 1500, 2000, 2500, 3000],
        'y': [1.1410, 1.2095, 1.2520, 1.2782, 1.2955],
        'description': 'Cp [J/mK·10³] za zrak na visokim temperaturama',
        'suggested': 'Polinomijalna aproksimacija (Cp = a + bT + cT²)'
    }
}

if data_input_method == "Predefinisani primjer":
    selected_dataset = st.sidebar.selectbox("Odaberite dataset:", list(predefined_datasets.keys()))
    dataset = predefined_datasets[selected_dataset]
    task_text = dataset.get("task_text", None)
    x_data = np.array(dataset['x'])
    y_data = np.array(dataset['y'])
    st.sidebar.info(f"**Opis:** {dataset['description']}")
    st.sidebar.success(f"**Preporuka:** {dataset['suggested']}")

elif data_input_method == "Vlastiti podaci":
    st.sidebar.markdown("Unesite podatke (zarezom odvojeni):")
    x_input = st.sidebar.text_area("X vrijednosti:", value="1, 2, 3, 4, 5, 6, 7, 8", height=80)
    y_input = st.sidebar.text_area("Y vrijednosti:", value="2.1, 4.0, 5.9, 8.1, 9.8, 12.1, 14.0, 15.9", height=80)

    try:
        x_data = np.array([float(x.strip()) for x in x_input.split(',')])
        y_data = np.array([float(y.strip()) for y in y_input.split(',')])

        if len(x_data) != len(y_data):
            st.error("Broj X i Y vrijednosti mora biti jednak!")
            st.stop()
    except Exception as e:
        st.error(f"Greška pri parsiranju podataka: {e}")
        st.stop()

else:  # Učitaj iz datoteke
    st.sidebar.info("""
    **Podržani formati:**
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Text (.txt)
    - Data (.dat)

    Datoteka treba imati dvije kolone (x i y).
    """)

    uploaded_file = st.sidebar.file_uploader(
        "Odaberite datoteku:",
        type=['csv', 'xlsx', 'xls', 'txt', 'dat']
    )

    if uploaded_file is not None:
        try:
            # Određivanje tipa datoteke
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                # Pokušaj različite separatore
                try:
                    df = pd.read_csv(uploaded_file, sep=',')
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=';')
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep='\t')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')

            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)

            elif file_extension in ['txt', 'dat']:
                # Pokušaj različite separatore za txt i dat
                try:
                    df = pd.read_csv(uploaded_file, sep='\t')
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=' ', skipinitialspace=True)
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=',')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, delim_whitespace=True)

            # Prikaz učitanih podataka
            st.sidebar.success(f"Učitano {len(df)} redova")

            # Izbor kolona
            columns = df.columns.tolist()

            if len(columns) >= 2:
                col_x = st.sidebar.selectbox("Kolona za X:", columns, index=0)
                col_y = st.sidebar.selectbox("Kolona za Y:", columns, index=1 if len(columns) > 1 else 0)

                # Konverzija u numeričke vrijednosti
                x_data = pd.to_numeric(df[col_x], errors='coerce').dropna().values
                y_data = pd.to_numeric(df[col_y], errors='coerce').dropna().values

                # Provjera da su iste dužine
                min_len = min(len(x_data), len(y_data))
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]

                if len(x_data) < 2:
                    st.error("Potrebne su barem 2 numeričke vrijednosti!")
                    st.stop()
            else:
                st.error("Datoteka mora imati barem dvije kolone!")
                st.stop()

        except Exception as e:
            st.error(f"Greška pri učitavanju datoteke: {e}")
            st.stop()
    else:
        # Postavi default vrijednosti dok datoteka nije učitana
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([2, 4, 6, 8, 10])
        st.sidebar.warning("Učitajte datoteku za analizu.")

run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

# Glavni sadržaj
st.markdown("---")
if data_input_method == "Predefinisani primjer" and task_text:
    st.subheader("🧾 Postavka zadatka")
    st.markdown(task_text)
    solution_link = dataset.get("solution_link", "").strip()
    if solution_link:
        st.markdown(f"🔗 [Rješenje zadatka (link)]({solution_link})")
    st.markdown("---")

# Prikaz informacija o metodi
method_info = approximation_methods[method]
col_info1, col_info2 = st.columns([1, 2])

with col_info1:
    st.markdown(f"### Model")
    st.latex(method_info['model'].replace('·', '\\cdot '))

with col_info2:
    st.markdown(f"**Opis:** {method_info['description']}")
    st.markdown(f"**Zahtjevi:** {method_info['requirements']}")
    if method_info['linearization'] != "-":
        st.markdown(f"**Linearizacija:** {method_info['linearization']}")

st.markdown("---")

# Prikaz podataka
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Podaci")
    data_df = pd.DataFrame({'x': x_data, 'y': y_data})
    st.dataframe(data_df, use_container_width=True, height=300)
    st.markdown(f"**Broj tačaka:** {len(x_data)}")

# Teorija za svaku metodu
with st.expander("📖 Teorija - Metoda Najmanjih Kvadrata", expanded=False):
    st.markdown("""
    ### Metoda Najmanjih Kvadrata

    **Cilj:** Pronaći parametre funkcije $f(x)$ koja najbolje opisuje date podatke $(x_i, y_i)$.

    **Princip:** Minimizirati sumu kvadrata odstupanja (reziduala):
    $$S = \sum_{i=1}^{n} [y_i - f(x_i)]^2$$

    ### Linearizacija Nelinearnih Modela

    Mnogi nelinearni modeli se mogu transformisati u linearne:

    | Model | Linearizacija | Novi oblik |
    |-------|---------------|------------|
    | $y = Ax^B$ | $\ln(y) = \ln(A) + B\ln(x)$ | $Y = b + aX$ |
    | $y = Ae^{Bx}$ | $\ln(y) = \ln(A) + Bx$ | $Y = b + ax$ |
    | $y = a + b\ln(x)$ | Već linearan u $\ln(x)$ | $y = a + bX$ |
    | $y = 1/(a+bx)$ | $1/y = a + bx$ | $Y = a + bx$ |
    | $y = \\frac{P_r(x)}{Q_s(x)}$, $Q_s(x)=1+c_1x+...+c_sx^s$ | $y(1+c_1x+...+c_sx^s) \\approx P_r(x)$ | linearni LS sistem |
    | $y = a + b\sqrt{x}$ | Već linearan u $\sqrt{x}$ | $y = a + bX$ |

    ### R² - Koeficijent Determinacije

    $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \\bar{y})^2}$$

    - $R^2 = 1$: Savršen fit
    - $R^2 = 0$: Model nije bolji od srednje vrijednosti
    - $R^2 \\geq 0.9$: Odličan fit
    - $R^2 \\geq 0.7$: Dobar fit
    """)

# Rezultati
if run_button:
    st.markdown("---")

    if method == "Poređenje svih modela":
        st.header("🏆 Poređenje Svih Modela Aproksimacije")

        comparison = compare_regression_models(x_data, y_data)

        # Rangiranje po R²
        st.subheader("📊 Rangiranje Modela")

        ranking_df = pd.DataFrame(comparison['summary'])
        ranking_df['Rang'] = range(1, len(ranking_df) + 1)
        ranking_df['R²'] = ranking_df['r_squared'].apply(lambda x: f"{x:.6f}")
        ranking_df = ranking_df[['Rang', 'model', 'R²', 'equation']]
        ranking_df.columns = ['Rang', 'Model', 'R²', 'Jednačina']
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        # Preporuka
        if 'recommendation' in comparison:
            rec = comparison['recommendation']
            st.success(f"""
            **🎯 Preporuka:** {rec['best_model']}

            **Jednačina:** {rec['equation']}

            **R² = {rec['r_squared']:.6f}** - {rec['interpretation']}
            """)

        # Grafovi najboljih modela
        st.subheader("📈 Vizualizacija Najboljih Modela")

        # Uzmi top 4 modela
        top_models = comparison['summary'][:4]
        cols = st.columns(2)

        for i, model_info in enumerate(top_models):
            model_name = model_info['model']
            if model_name in comparison and 'y_predicted' in comparison[model_name]:
                with cols[i % 2]:
                    res = comparison[model_name]
                    title = f"{model_name}\nR² = {res['r_squared']:.4f}"
                    fig = plot_regression(x_data, y_data, res, title)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        # Pojedinačna metoda
        st.header(f"Rezultati: {method}")

        # Pozovi odgovarajuću funkciju
        if method == "Polinomijalna aproksimacija":
            result = approximation_methods[method]["function"](x_data, y_data, degree)

        elif method == "Racionalna aproksimacija (opšta)":
            result = approximation_methods[method]["function"](x_data, y_data, r=r_num, s=s_den)

        else:
            result = approximation_methods[method]["function"](x_data, y_data)

        # Provjera greške
        if 'error_message' in result:
            st.error(result['error_message'])
            st.stop()

        # Grafikon
        with col2:
            st.subheader("📈 Grafikon")
            title = f"{method}\nR² = {result['r_squared']:.4f}"
            fig = plot_regression(x_data, y_data, result, title)
            st.plotly_chart(fig, use_container_width=True)

        # Metrike
        st.subheader("📊 Rezultati")

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric("R² (Koeficijent determinacije)", f"{result['r_squared']:.6f}")
            if 'r_squared_adjusted' in result:
                st.metric("Adjusted R²", f"{result['r_squared_adjusted']:.6f}")
            if 'r_squared_linear' in result:
                st.caption(f"R² linearizirane regresije: {result['r_squared_linear']:.6f}")

        with col_m2:
            st.markdown("**Jednačina:**")
            equation_latex = result['equation'].replace('·', '\\cdot ')
            st.latex(equation_latex)

        with col_m3:
            # Prikaz parametara ovisno o metodi
            if 'a' in result and 'b' in result and 'A' not in result:
                st.markdown(f"**a** = {result['a']:.6f}")
                st.markdown(f"**b** = {result['b']:.6f}")
            elif 'A' in result and 'B' in result:
                st.markdown(f"**A** = {result['A']:.6f}")
                st.markdown(f"**B** = {result['B']:.6f}")
            elif 'coefficients' in result:
                st.markdown("**Koeficijenti:**")
                for i, c in enumerate(result['coefficients']):
                    st.markdown(f"$a_{i}$ = {c:.6f}")
            elif 'c0' in result:
                st.markdown(f"**c0** = {result['c0']:.6f}")
                for i, val in enumerate(result['c_rest'], start=1):
                    st.markdown(f"**c{i}** = {val:.6f}")

        # Prikaz SSE, MSE, RMSE
        if 'sse' in result:
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                st.metric("SSE (Sum of Squared Errors)", f"{result['sse']:.6f}")
            with col_e2:
                st.metric("MSE (Mean Squared Error)", f"{result['mse']:.6f}")
            with col_e3:
                st.metric("RMSE (Root Mean Squared Error)", f"{result['rmse']:.6f}")

        # Interpretacija R²
        r2 = result['r_squared']
        if r2 >= 0.9:
            st.success(f"Odličan fit! Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        elif r2 >= 0.7:
            st.info(f"Dobar fit. Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        elif r2 >= 0.5:
            st.warning(f"Umjeren fit. Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        else:
            st.error(f"Slab fit. Model objašnjava samo {r2*100:.1f}% varijabilnosti podataka. Pokušajte drugu metodu.")

        # Step-by-step prikaz
        if 'steps' in result and result['steps']:
            st.subheader("📝 Step-by-Step Prikaz")

            for step in result['steps']:
                with st.expander(f"Korak {step.get('step', '?')}: {step['title']}", expanded=(step.get('step', 0) <= 2)):

                    if 'description' in step:
                        st.markdown(step['description'])

                    if 'model_original' in step:
                        st.markdown(f"**Originalni model:** ${step['model_original']}$")
                    if 'linearization' in step:
                        st.markdown(f"**Linearizacija:** ${step['linearization']}$")
                    if 'model_linearized' in step:
                        st.markdown(f"**Linearizirani model:** ${step['model_linearized']}$")

                    # Prikaz substitucije
                    if 'substitution' in step:
                        st.markdown("**Supstitucija:**")
                        if isinstance(step['substitution'], dict):
                            for var, expr in step['substitution'].items():
                                st.markdown(f"- ${var} = {expr}$")
                        else:
                            st.markdown(f"- {step['substitution']}")

                    # Prikaz tabele transformacije
                    if 'transform_table' in step:
                        st.markdown("**Tabela transformacije:**")
                        transform_df = pd.DataFrame(step['transform_table'])
                        st.dataframe(transform_df, use_container_width=True, hide_index=True)

                    # Prikaz suma (N, Σx, Σy, itd.)
                    if 'sums' in step:
                        st.markdown("**Izračunate sume:**")
                        sums_text = " | ".join([f"**{k}** = {v}" for k, v in step['sums'].items()])
                        st.markdown(sums_text)

                    if 'formulas' in step:
                        st.markdown("**Formule:**")
                        for name, val in step['formulas'].items():
                            if isinstance(val, str):
                                st.markdown(f"- {name}: {val}")
                            else:
                                st.markdown(f"- ${name} = {val}$")

                    # Prikaz sistema jednačina
                    if 'system' in step:
                        st.markdown("**Sistem jednačina:**")
                        for eq in step['system']:
                            st.latex(eq.replace('·', r' \cdot '))

                    # Matrični oblik sistema
                    if 'matrix_form' in step:
                        st.markdown("**Matrični oblik:**")
                        mf = step['matrix_form']
                        st.markdown(f"${mf.get('equation', '')}$")

                    # Rješenje sistema
                    if 'solution' in step:
                        st.markdown("**Rješenje:**")
                        for var, val in step['solution'].items():
                            st.markdown(f"- **{var}** = {val:.6f}")

                    if 'calculations' in step:
                        st.markdown("**Računanje:**")
                        for name, calc in step['calculations'].items():
                            st.markdown(f"- {calc}")

                    # Formule za b i a (ako postoje)
                    if 'formula_b' in step:
                        st.markdown(f"**Formula za b:** ${step['formula_b']}$")
                        if 'calculation_b' in step:
                            st.markdown(f"- {step['calculation_b']}")
                        if 'b' in step:
                            st.markdown(f"- **b = {step['b']:.6f}**")

                    if 'formula_a' in step:
                        st.markdown(f"**Formula za a:** ${step['formula_a']}$")
                        if 'calculation_a' in step:
                            st.markdown(f"- {step['calculation_a']}")
                        if 'a' in step:
                            st.markdown(f"- **a = {step['a']:.6f}**")

                    if 'equation' in step:
                        st.success(f"**Jednačina:** {step['equation']}")

                    if 'final_equation' in step:
                        st.success(f"**Konačna jednačina:** {step['final_equation']}")

                    if 'linear_equation' in step:
                        st.info(f"**Linearna jednačina:** {step['linear_equation']}")

                    # Prikaz tabele poređenja (predviđene vs stvarne vrijednosti)
                    if 'comparison_table' in step:
                        st.markdown("**Tabela poređenja:**")
                        comp_df = pd.DataFrame(step['comparison_table'])
                        # Formatiranje brojeva
                        for col in comp_df.columns:
                            if col != 'x':
                                comp_df[col] = comp_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    # R² i statistike
                    if 'r_squared' in step and 'ss_res' not in step:
                        st.metric("R²", f"{step['r_squared']:.6f}")
                    if 'ss_res' in step:
                        st.markdown(f"**SS_res** (suma kvadrata reziduala) = {step['ss_res']:.6f}")
                    if 'ss_tot' in step:
                        st.markdown(f"**SS_tot** (ukupna suma kvadrata) = {step['ss_tot']:.6f}")
                    if 'r_squared_formula' in step:
                        st.markdown(f"**Formula:** ${step['r_squared_formula']}$")
                    if 'r_squared_calculation' in step:
                        st.markdown(f"**Računanje:** {step['r_squared_calculation']}")

                    if 'sse' in step:
                        st.markdown(f"**SSE** (Sum of Squared Errors) = {step['sse']:.6f}")
                    if 'mse' in step:
                        st.markdown(f"**MSE** (Mean Squared Error) = {step['mse']:.6f}")
                    if 'rmse' in step:
                        st.markdown(f"**RMSE** (Root Mean Squared Error) = {step['rmse']:.6f}")

                    if 'interpretation' in step:
                        st.info(step['interpretation'])

        # Tabela predviđenih vrijednosti, reziduala i greške
        st.subheader("📋 Predviđene Vrijednosti i Reziduali")

        if 'residuals' in result and 'errors_percent' in result:
            pred_df = pd.DataFrame({
                'x': x_data,
                'y (stvarno)': y_data,
                'ŷ (predviđeno)': result['y_predicted'],
                'Rezidual (y - ŷ)': result['residuals'],
                'Greška (%)': result['errors_percent']
            })
            pred_df['Rezidual (y - ŷ)'] = pred_df['Rezidual (y - ŷ)'].apply(lambda x: f"{x:.6f}")
            pred_df['Greška (%)'] = pred_df['Greška (%)'].apply(lambda x: f"{x:.2f}")
        else:
            pred_df = pd.DataFrame({
                'x': x_data,
                'y (stvarno)': y_data,
                'ŷ (predviđeno)': result['y_predicted']
            })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# Info o svim metodama
with st.expander("📚 Pregled Svih Metoda Aproksimacije", expanded=False):
    methods_info = get_all_approximation_methods()

    for m in methods_info:
        st.markdown(f"""
        #### {m['name']}
        - **Model:** ${m['model']}$
        - **Linearizacija:** {m['linearization']}
        - **Zahtjevi:** {m['requirements']}
        - *{m['description']}*
        ---
        """)
