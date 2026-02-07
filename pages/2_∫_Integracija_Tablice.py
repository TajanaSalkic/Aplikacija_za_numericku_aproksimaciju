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

from methods.regression import (linear_regression, polynomial_regression,
                                 exponential_regression, power_regression,
                                 logarithmic_regression, compare_regression_models)

st.set_page_config(page_title="Integracija iz Tablice", page_icon="∫", layout="wide")

st.title("∫ Numerička Integracija iz Tablice")
st.markdown("*Izračunavanje integrala kada je ulaz tablica podataka*")

st.info("""
**Princip:** Kada nemamo eksplicitnu funkciju f(x), već samo tablicu vrijednosti (x, y),
možemo prvo aproksimirati podatke funkcijom pa integrirati tu funkciju.
""")

# Sidebar
st.sidebar.header("⚙️ Postavke")

approx_method = st.sidebar.selectbox(
    "Metoda aproksimacije:",
    ["Linearna", "Kvadratna (polinom 2. stepena)", "Kubna (polinom 3. stepena)",
     "Eksponencijalna", "Stepena", "Logaritamska", "Automatski (najbolji R²)"]
)

# Unos podataka
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Unos Podataka")

data_source = st.sidebar.radio("Način unosa:", ["Vlastiti unos", "Učitaj iz datoteke"])

if data_source == "Vlastiti unos":
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
    ### Integracija preko Aproksimacije

    Kada imamo tablicu vrijednosti $(x_i, y_i)$ umjesto eksplicitne funkcije:

    1. Aproksimiraj podatke funkcijom $\\hat{f}(x)$ (npr. polinom)
    2. Integriraj tu funkciju analitički ili numerički

    $$\\int_a^b f(x)\\,dx \\approx \\int_a^b \\hat{f}(x)\\,dx$$

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
    integral_value = np.trapezoid(y_fine, x_fine)

    st.success(f"**Integral aproksimirane funkcije: I ≈ {integral_value:.6f}**")

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
