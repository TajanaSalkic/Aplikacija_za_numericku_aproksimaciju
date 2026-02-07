"""
Stranica: Numerička Integracija iz Tablice
==========================================

Numerička integracija kada je ulaz tablica podataka (x, y).
Koristi aproksimaciju podataka za računanje integrala.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.integration import (trapezoidal, simpson, integrate_from_table,
                                  integrate_with_interpolation)
from methods.regression import (linear_regression, polynomial_regression,
                                 exponential_regression, power_regression,
                                 logarithmic_regression, compare_regression_models)

st.set_page_config(page_title="Integracija iz Tablice", page_icon="∫", layout="wide")

st.title("∫ Numerička Integracija iz Tablice")
st.markdown("*Izračunavanje integrala kada je ulaz tablica podataka*")

st.info("""
**Princip:** Kada nemamo eksplicitnu funkciju f(x), već samo tablicu vrijednosti (x, y),
možemo izračunati integral koristeći numeričke metode poput trapezne ili Simpsonove,
ili prvo aproksimirati podatke funkcijom pa integrirati tu funkciju.
""")

# Sidebar
st.sidebar.header("⚙️ Postavke")

# Način integracije
integration_approach = st.sidebar.radio(
    "Pristup integraciji:",
    ["Direktna numerička integracija", "Integracija preko aproksimacije"]
)

if integration_approach == "Direktna numerička integracija":
    method = st.sidebar.selectbox(
        "Metoda:",
        ["Automatski odabir", "Trapezna metoda", "Simpsonova metoda"]
    )
    use_interpolation = st.sidebar.checkbox("Koristi kubnu interpolaciju za veću preciznost")
else:
    approx_method = st.sidebar.selectbox(
        "Metoda aproksimacije:",
        ["Linearna", "Kvadratna (polinom 2. stepena)", "Kubna (polinom 3. stepena)",
         "Eksponencijalna", "Stepena", "Logaritamska", "Automatski (najbolji R²)"]
    )

# Unos podataka
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Unos Podataka")

data_source = st.sidebar.radio("Način unosa:", ["Predefinisani primjer", "Vlastiti unos", "Učitaj iz datoteke"])

predefined_examples = {
    "Brzina vozila (m/s) → Put (m)": {
        'x': [0, 1, 2, 3, 4, 5, 6],
        'y': [0, 10, 18, 24, 28, 30, 30],
        'description': 'Brzina vozila tokom vremena. Integral = pređeni put.',
        'unit_x': 's',
        'unit_y': 'm/s',
        'unit_integral': 'm'
    },
    "Snaga (W) → Energija (J)": {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 100, 150, 180, 190, 200],
        'description': 'Snaga uređaja tokom vremena. Integral = utrošena energija.',
        'unit_x': 's',
        'unit_y': 'W',
        'unit_integral': 'J'
    },
    "Kvadratna funkcija (x²)": {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 1, 4, 9, 16, 25],
        'description': 'y = x². Tačna vrijednost ∫x²dx od 0 do 5 = 125/3 ≈ 41.67',
        'unit_x': '',
        'unit_y': '',
        'unit_integral': ''
    },
    "Sinusna funkcija": {
        'x': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.14159],
        'y': [0, 0.479, 0.841, 0.997, 0.909, 0.598, 0.141, 0],
        'description': 'y = sin(x). Tačna vrijednost ∫sin(x)dx od 0 do π = 2',
        'unit_x': 'rad',
        'unit_y': '',
        'unit_integral': ''
    },
    "Eksperimentalni podaci": {
        'x': [0, 2, 4, 6, 8, 10],
        'y': [0, 5.2, 18.1, 37.5, 62.8, 95.0],
        'description': 'Eksperimentalno izmjereni podaci',
        'unit_x': '',
        'unit_y': '',
        'unit_integral': ''
    }
}

if data_source == "Predefinisani primjer":
    selected_example = st.sidebar.selectbox("Primjer:", list(predefined_examples.keys()))
    example = predefined_examples[selected_example]
    x_data = np.array(example['x'])
    y_data = np.array(example['y'])
    st.sidebar.info(example['description'])

elif data_source == "Vlastiti unos":
    x_input = st.sidebar.text_area("X vrijednosti (zarezom odvojene):",
                                   value="0, 1, 2, 3, 4, 5")
    y_input = st.sidebar.text_area("Y vrijednosti:",
                                   value="0, 1, 4, 9, 16, 25")
    try:
        x_data = np.array([float(x.strip()) for x in x_input.split(',')])
        y_data = np.array([float(y.strip()) for y in y_input.split(',')])
    except:
        st.error("Greška pri parsiranju podataka!")
        st.stop()

else:  # Učitaj iz datoteke
    st.sidebar.info("Podržani formati: CSV, Excel, TXT")

    uploaded_file = st.sidebar.file_uploader(
        "Odaberite datoteku:",
        type=['csv', 'xlsx', 'xls', 'txt'],
        key='integration_file'
    )

    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()

            if file_ext == 'csv':
                try:
                    df = pd.read_csv(uploaded_file, sep=',')
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=';')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=';')
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
                if len(df.columns) < 2:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, delim_whitespace=True)

            st.sidebar.success(f"Učitano {len(df)} redova")

            columns = df.columns.tolist()
            if len(columns) >= 2:
                col_x = st.sidebar.selectbox("Kolona za X:", columns, index=0, key='int_col_x')
                col_y = st.sidebar.selectbox("Kolona za Y:", columns, index=1, key='int_col_y')

                x_data = pd.to_numeric(df[col_x], errors='coerce').dropna().values
                y_data = pd.to_numeric(df[col_y], errors='coerce').dropna().values

                min_len = min(len(x_data), len(y_data))
                x_data = x_data[:min_len]
                y_data = y_data[:min_len]
            else:
                st.error("Datoteka mora imati barem dvije kolone!")
                st.stop()
        except Exception as e:
            st.error(f"Greška: {e}")
            st.stop()
    else:
        x_data = np.array([0, 1, 2, 3, 4, 5])
        y_data = np.array([0, 1, 4, 9, 16, 25])
        st.sidebar.warning("Učitajte datoteku za analizu.")

run_button = st.sidebar.button("🚀 Izračunaj Integral", type="primary", use_container_width=True)

# Glavni sadržaj
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Podaci")
    df = pd.DataFrame({'x': x_data, 'y': y_data})
    st.dataframe(df, use_container_width=True, height=300)
    st.markdown(f"**Broj tačaka:** {len(x_data)}")
    st.markdown(f"**Interval:** [{x_data[0]:.4f}, {x_data[-1]:.4f}]")

# Teorija
with st.expander("📖 Teorija - Integracija iz Tablice", expanded=False):
    st.markdown("""
    ### Numerička Integracija iz Diskretnih Podataka

    Kada imamo tablicu vrijednosti $(x_i, y_i)$ umjesto eksplicitne funkcije,
    koristimo numeričke metode za aproksimaciju integrala.

    #### 1. Trapezna Metoda
    $$I \\approx \\sum_{i=0}^{n-1} \\frac{y_i + y_{i+1}}{2} \\cdot (x_{i+1} - x_i)$$

    Za jednako udaljene tačke ($h = x_{i+1} - x_i$):
    $$I \\approx \\frac{h}{2}[y_0 + 2y_1 + 2y_2 + ... + 2y_{n-1} + y_n]$$

    #### 2. Simpsonova Metoda (zahtijeva paran broj intervala)
    $$I \\approx \\frac{h}{3}[y_0 + 4y_1 + 2y_2 + 4y_3 + ... + y_n]$$

    #### 3. Integracija preko Aproksimacije
    1. Aproksimiraj podatke funkcijom $\\hat{f}(x)$ (npr. polinom)
    2. Integriraj tu funkciju analitički ili numerički

    Prednost: Može dati glatku krivulju i omogućiti ekstrapolaciju.
    """)

if run_button:
    st.markdown("---")

    if len(x_data) != len(y_data):
        st.error("Broj X i Y vrijednosti mora biti jednak!")
        st.stop()

    if len(x_data) < 2:
        st.error("Potrebne su barem 2 tačke!")
        st.stop()

    if integration_approach == "Direktna numerička integracija":
        st.header("Direktna Numerička Integracija")

        # Određivanje metode
        if method == "Automatski odabir":
            n_intervals = len(x_data) - 1
            if n_intervals % 2 == 0:
                chosen_method = "simpson"
                method_name = "Simpsonova metoda (automatski odabrana - paran broj intervala)"
            else:
                chosen_method = "trapezoid"
                method_name = "Trapezna metoda (automatski odabrana - neparan broj intervala)"
        elif method == "Trapezna metoda":
            chosen_method = "trapezoid"
            method_name = "Trapezna metoda"
        else:
            chosen_method = "simpson"
            method_name = "Simpsonova metoda"

        # Računanje
        if use_interpolation:
            result = integrate_with_interpolation(x_data, y_data, method=chosen_method, n_interp=100)
            st.success(f"**Integral (sa kubnom interpolacijom): I ≈ {result['integral']:.6f}**")
            if 'direct_integral' in result:
                st.info(f"Direktna integracija bez interpolacije: I ≈ {result['direct_integral']:.6f}")
        else:
            result = integrate_from_table(x_data, y_data, method=chosen_method)
            st.success(f"**Integral: I ≈ {result['integral']:.6f}**")

        st.markdown(f"**Korištena metoda:** {method_name}")

        # Vizualizacija
        with col2:
            st.subheader("📈 Vizualizacija")

            fig = go.Figure()

            # Podaci
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers+lines',
                                    name='Podaci', line=dict(color='blue'),
                                    marker=dict(size=10)))

            # Ispuna - trapezi
            for i in range(len(x_data) - 1):
                fig.add_trace(go.Scatter(
                    x=[x_data[i], x_data[i], x_data[i+1], x_data[i+1], x_data[i]],
                    y=[0, y_data[i], y_data[i+1], 0, 0],
                    fill='toself',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(0,100,255,0.5)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title=f"Integracija: I ≈ {result['integral']:.4f}",
                xaxis_title='x',
                yaxis_title='y',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Koraci
        if 'steps' in result:
            st.subheader("📝 Koraci Računanja")
            for step in result['steps']:
                with st.expander(step['title']):
                    for key, val in step.items():
                        if key != 'title':
                            st.write(f"**{key}:** {val}")

    else:  # Integracija preko aproksimacije
        st.header("Integracija preko Aproksimacije")

        # Aproksimacija
        st.subheader("1. Aproksimacija Podataka")

        if approx_method == "Automatski (najbolji R²)":
            comparison = compare_regression_models(x_data, y_data)
            best_model = comparison['recommendation']['best_model']
            best_result = None

            for name, res in comparison.items():
                if name == best_model:
                    best_result = res
                    break

            if best_result is None:
                best_result = comparison[list(comparison.keys())[0]]

            st.success(f"**Najbolji model:** {best_model} (R² = {comparison['recommendation']['r_squared']:.6f})")
            approx_result = best_result

        else:
            method_map = {
                "Linearna": lambda: linear_regression(x_data, y_data),
                "Kvadratna (polinom 2. stepena)": lambda: polynomial_regression(x_data, y_data, 2),
                "Kubna (polinom 3. stepena)": lambda: polynomial_regression(x_data, y_data, 3),
                "Eksponencijalna": lambda: exponential_regression(x_data, y_data),
                "Stepena": lambda: power_regression(x_data, y_data),
                "Logaritamska": lambda: logarithmic_regression(x_data, y_data)
            }
            approx_result = method_map[approx_method]()

        if 'error_message' in approx_result:
            st.error(approx_result['error_message'])
            st.stop()

        st.info(f"**Jednačina:** {approx_result['equation']}")
        st.info(f"**R²:** {approx_result['r_squared']:.6f}")

        # Integracija aproksimirane funkcije
        st.subheader("2. Integracija Aproksimirane Funkcije")

        # Kreiraj finu mrežu za integraciju
        x_fine = np.linspace(x_data[0], x_data[-1], 1000)
        y_approx = np.array(approx_result['y_predicted'])

        # Interpoliraj predviđene vrijednosti na finu mrežu
        from scipy.interpolate import interp1d
        interp_func = interp1d(x_data, y_approx, kind='linear', fill_value='extrapolate')
        y_fine = interp_func(x_fine)

        # Trapezna integracija na finoj mreži
        integral_value = np.trapz(y_fine, x_fine)

        st.success(f"**Integral aproksimirane funkcije: I ≈ {integral_value:.6f}**")

        # Za poređenje - direktna integracija originalnih podataka
        direct_result = integrate_from_table(x_data, y_data, method='trapezoid')
        st.info(f"Direktna integracija originalnih podataka: I ≈ {direct_result['integral']:.6f}")

        # Vizualizacija
        with col2:
            st.subheader("📈 Vizualizacija")

            fig = go.Figure()

            # Originalni podaci
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                                    name='Originalni podaci',
                                    marker=dict(size=12, color='blue')))

            # Aproksimirana funkcija
            fig.add_trace(go.Scatter(x=x_fine, y=y_fine, mode='lines',
                                    name='Aproksimacija',
                                    line=dict(color='red', width=2)))

            # Ispuna ispod aproksimirane krivulje
            fig.add_trace(go.Scatter(
                x=list(x_fine) + [x_fine[-1], x_fine[0]],
                y=list(y_fine) + [0, 0],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Integral'
            ))

            fig.update_layout(
                title=f"Integral aproksimirane funkcije: I ≈ {integral_value:.4f}",
                xaxis_title='x',
                yaxis_title='y',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tabela vrijednosti
        st.subheader("📋 Poređenje Stvarnih i Aproksimiranih Vrijednosti")

        comp_df = pd.DataFrame({
            'x': x_data,
            'y (stvarno)': y_data,
            'y (aproksimacija)': approx_result['y_predicted'],
            'Razlika': [f"{y - yp:.4f}" for y, yp in zip(y_data, approx_result['y_predicted'])]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
