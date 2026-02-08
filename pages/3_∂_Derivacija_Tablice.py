"""
Stranica: Numerička Derivacija iz Tablice
=========================================

Numerička derivacija kada je ulaz tablica podataka (x, y).
Koristi aproksimaciju podataka za računanje derivacije.
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
from methods.differentiation import auto_differentiate


def build_approx_function(result):
    """Kreira callable funkciju iz rezultata regresije."""
    equation = result.get('equation', '')
    if 'c_rest' in result:
        b_coeffs = result['b']
        c_rest = result['c_rest']
        def f(x):
            P = sum(b_coeffs[k] * x**k for k in range(len(b_coeffs)))
            Q = 1.0 + sum(c_rest[k] * x**(k+1) for k in range(len(c_rest)))
            return P / Q if abs(Q) > 1e-12 else P / 1e-12
        return f
    if 'coefficients' in result and 'ln(x)' not in equation:
        coeffs = result['coefficients']
        def f(x):
            return sum(c * x**i for i, c in enumerate(coeffs))
        return f
    if 'A' in result and 'B' in result:
        A, B = result['A'], result['B']
        if 'e^' in equation:
            return lambda x: A * np.exp(B * x)
        else:
            return lambda x: A * (x ** B)
    if 'a' in result and 'b' in result:
        a, b = result['a'], result['b']
        if 'ln(x)' in equation:
            return lambda x: a + b * np.log(x)
        else:
            return lambda x: a * x + b
    return None


st.set_page_config(page_title="Derivacija iz Tablice", page_icon="∂", layout="wide")

st.title("∂ Numerička Derivacija iz Tablice")
st.markdown("*Izračunavanje derivacija kada je ulaz tablica podataka*")

st.info("""
**Princip:** Kada nemamo eksplicitnu funkciju f(x), već samo tablicu vrijednosti (x, y),
možemo prvo aproksimirati podatke funkcijom pa diferencirati tu funkciju.
""")

# Sidebar
st.sidebar.header("⚙️ Postavke")

approx_method = st.sidebar.selectbox(
    "Metoda aproksimacije:",
    ["Linearna", "Kvadratna (polinom 2. stepena)", "Kubna (polinom 3. stepena)",
     "Eksponencijalna", "Stepena", "Logaritamska", "Automatski (najbolji R²)"]
)

h_value = st.sidebar.number_input(
    "Korak h:", min_value=0.0001, max_value=10.0, value=0.001, format="%.4f"
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
        key='diff_file'
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
                col_x = st.sidebar.selectbox("Kolona za X:", columns, index=0, key='diff_col_x')
                col_y = st.sidebar.selectbox("Kolona za Y:", columns, index=1, key='diff_col_y')

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

run_button = st.sidebar.button("🚀 Izračunaj Derivaciju", type="primary", use_container_width=True)

# Glavni sadržaj
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Podaci")
    df = pd.DataFrame({'x': x_data, 'y': y_data})
    st.dataframe(df, use_container_width=True, height=300)
    st.markdown(f"**Broj tačaka:** {len(x_data)}")

# Teorija
with st.expander("📖 Teorija - Numerička Derivacija", expanded=False):
    st.markdown("""
    ### Derivacija preko Aproksimacije

    Kada imamo tablicu vrijednosti $(x_i, y_i)$ umjesto eksplicitne funkcije:

    1. Aproksimiraj podatke funkcijom $\\hat{f}(x)$
    2. Deriviraj tu funkciju analitički

    **Primjer za polinom:**
    - Ako je $\\hat{f}(x) = a_0 + a_1 x + a_2 x^2$
    - Onda je $\\hat{f}'(x) = a_1 + 2a_2 x$
    """)

if run_button:
    st.markdown("---")

    if len(x_data) != len(y_data):
        st.error("Broj X i Y vrijednosti mora biti jednak!")
        st.stop()

    if len(x_data) < 2:
        st.error("Potrebne su barem 2 tačke!")
        st.stop()

    st.header("Derivacija preko Aproksimacije")

    # Aproksimacija
    st.subheader("1. Aproksimacija Podataka")

    if approx_method == "Automatski (najbolji R²)":
        comparison = compare_regression_models(x_data, y_data)
        best_model = comparison['recommendation']['best_model']

        for name, res in comparison.items():
            if name == best_model:
                approx_result = res
                break

        st.success(f"**Najbolji model:** {best_model} (R² = {comparison['recommendation']['r_squared']:.6f})")

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

    # Derivacija aproksimirane funkcije
    st.subheader("2. Derivacija Aproksimirane Funkcije")

    approx_func = build_approx_function(approx_result)
    if approx_func is None:
        st.error("Nije moguće kreirati funkciju iz rezultata aproksimacije!")
        st.stop()

    # Automatska derivacija - bira Forward/Backward/Central ovisno o položaju tačke
    domain = (float(x_data[0]), float(x_data[-1]))
    deriv_at_points = []
    deriv_details = []
    for xi in x_data:
        auto_res = auto_differentiate(f=approx_func, x=float(xi), h=h_value, domain=domain)
        d_info = auto_res['derivatives'][0]
        deriv_at_points.append(d_info['derivative'])
        deriv_details.append(d_info)

    st.success(f"**Automatska detekcija metode** | **h = {h_value}**")

    # Prikaz detalja derivacije po tačkama
    with st.expander("📝 Detalji derivacije po tačkama", expanded=False):
        for d_info in deriv_details:
            st.markdown(f"**x = {d_info['x']:.4f}** → {d_info['method']}")
            st.markdown(f"Razlog: _{d_info['reason']}_")
            st.markdown(f"Formula: `{d_info['formula']}`")
            st.markdown(f"**f'({d_info['x']:.4f}) = {d_info['derivative']:.10f}**")
            st.markdown("---")

    # Fina mreža za vizualizaciju
    x_fine = np.linspace(float(x_data[0]), float(x_data[-1]), 200)
    y_approx_fine = np.array([approx_func(xi) for xi in x_fine])
    dy_approx_fine = np.array([
        auto_differentiate(f=approx_func, x=float(xi), h=h_value, domain=domain)['derivatives'][0]['derivative']
        for xi in x_fine
    ])

    # Tabela rezultata
    st.subheader("📋 Derivacija u Tačkama")

    result_df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        "y' (derivacija)": [f"{d:.6f}" for d in deriv_at_points]
    })
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Vizualizacija
    with col2:
        st.subheader("📈 Vizualizacija")

        fig = go.Figure()

        # Originalni podaci
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='markers',
            name='Originalni podaci',
            marker=dict(size=12, color='blue')
        ))

        # Aproksimirana funkcija
        fig.add_trace(go.Scatter(
            x=x_fine, y=y_approx_fine,
            mode='lines',
            name='Aproksimacija f(x)',
            line=dict(color='blue', width=2)
        ))

        # Derivacija
        fig.add_trace(go.Scatter(
            x=x_fine, y=dy_approx_fine,
            mode='lines',
            name="Derivacija f'(x)",
            line=dict(color='red', width=2, dash='dash')
        ))

        # Derivacija u tačkama
        fig.add_trace(go.Scatter(
            x=x_data, y=deriv_at_points,
            mode='markers',
            name="f'(x) u tačkama",
            marker=dict(size=10, color='red', symbol='diamond')
        ))

        fig.update_layout(
            title="Funkcija, Aproksimacija i Derivacija",
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Interpretacija
    st.subheader("🎯 Interpretacija")

    avg_deriv = np.mean(deriv_at_points)
    if avg_deriv > 0:
        trend = "rastuća"
        icon = "📈"
    elif avg_deriv < 0:
        trend = "opadajuća"
        icon = "📉"
    else:
        trend = "konstantna"
        icon = "➡️"

    st.info(f"""
    {icon} **Prosječna derivacija:** {avg_deriv:.4f}

    Funkcija je pretežno **{trend}** na posmatranom intervalu.

    - Maksimalna derivacija: {max(deriv_at_points):.4f} (u x ≈ {x_data[deriv_at_points.index(max(deriv_at_points))]:.2f})
    - Minimalna derivacija: {min(deriv_at_points):.4f} (u x ≈ {x_data[deriv_at_points.index(min(deriv_at_points))]:.2f})
    """)
