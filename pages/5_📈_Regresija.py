"""
Stranica: Regresija i Aproksimacija
===================================

Implementira:
- Linearna regresija (metoda najmanjih kvadrata) - rađeno na nastavi
- Eksponencijalna aproksimacija - rađeno na nastavi
- Polinomijalna regresija - bonus
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.regression import (linear_regression, exponential_regression,
                                polynomial_regression, compare_regression_models)
from utils.plotting import plot_regression
from utils.latex_helpers import format_linear_regression_formulas, format_exponential_regression_formulas

st.set_page_config(page_title="Regresija", page_icon="📈", layout="wide")

st.title("📈 Regresija i Aproksimacija")
st.markdown("*Fitovanje podataka pomoću različitih modela*")

# Sidebar
st.sidebar.header("⚙️ Postavke")
method = st.sidebar.selectbox(
    "Odaberite metodu:",
    ["Linearna regresija", "Eksponencijalna aproksimacija",
     "Polinomijalna regresija", "Poređenje modela"]
)

# Unos podataka
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Podaci")

data_input_method = st.sidebar.radio(
    "Način unosa:",
    ["Predefinisani primjer", "Vlastiti podaci"]
)

predefined_datasets = {
    "Linearni trend": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [2.1, 4.0, 5.9, 8.1, 9.8, 12.1, 14.0, 15.9, 18.2, 19.9],
        'description': 'Podaci sa približno linearnim trendom (y ≈ 2x)'
    },
    "Eksponencijalni rast": {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [1.0, 2.7, 7.4, 20.1, 54.6, 148.4],
        'description': 'Eksponencijalni rast (y ≈ e^x)'
    },
    "Kvadratni trend": {
        'x': [1, 2, 3, 4, 5, 6, 7, 8],
        'y': [1.2, 4.1, 9.0, 16.2, 24.8, 36.1, 49.0, 64.2],
        'description': 'Kvadratni trend (y ≈ x²)'
    },
    "Rast populacije": {
        'x': [0, 1, 2, 3, 4, 5, 6],
        'y': [100, 122, 149, 182, 222, 271, 331],
        'description': 'Simulacija rasta populacije (~22% godišnje)'
    },
    "Hlađenje tijela": {
        'x': [0, 5, 10, 15, 20, 25, 30],
        'y': [80, 64.7, 52.5, 42.6, 34.6, 28.2, 23.0],
        'description': 'Newtonov zakon hlađenja'
    }
}

if data_input_method == "Predefinisani primjer":
    selected_dataset = st.sidebar.selectbox("Odaberite dataset:", list(predefined_datasets.keys()))
    dataset = predefined_datasets[selected_dataset]
    x_data = np.array(dataset['x'])
    y_data = np.array(dataset['y'])
    st.sidebar.info(dataset['description'])

else:
    st.sidebar.markdown("Unesite podatke (zarezom odvojeni):")
    x_input = st.sidebar.text_input("X vrijednosti:", value="1, 2, 3, 4, 5")
    y_input = st.sidebar.text_input("Y vrijednosti:", value="2.1, 4.0, 5.9, 8.1, 9.8")

    try:
        x_data = np.array([float(x.strip()) for x in x_input.split(',')])
        y_data = np.array([float(y.strip()) for y in y_input.split(',')])
    except:
        st.error("Greška pri parsiranju podataka. Provjerite format.")
        st.stop()

# Parametri za polinomijalnu
if method == "Polinomijalna regresija":
    degree = st.sidebar.slider("Stepen polinoma:", 1, min(10, len(x_data)-1), 2)

run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

# Glavni sadržaj
st.markdown("---")

# Prikaz podataka
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Podaci")
    import pandas as pd
    data_df = pd.DataFrame({'x': x_data, 'y': y_data})
    st.dataframe(data_df, use_container_width=True)

    st.markdown(f"**Broj tačaka:** {len(x_data)}")

# Teorija
if method == "Linearna regresija":
    with st.expander("📖 Teorija", expanded=False):
        st.markdown(format_linear_regression_formulas())

elif method == "Eksponencijalna aproksimacija":
    with st.expander("📖 Teorija", expanded=False):
        st.markdown(format_exponential_regression_formulas())

elif method == "Polinomijalna regresija":
    with st.expander("📖 Teorija (Bonus metoda)", expanded=False):
        st.markdown("""
        ### Polinomijalna Regresija

        **Model:**
        $$y = a_n x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0$$

        **Metoda najmanjih kvadrata:**

        Cilj: Minimizirati $S = \\sum(y_i - p(x_i))^2$

        U matričnom obliku:
        $$X^T X \\cdot a = X^T y$$

        gdje je $X$ Vandermondova matrica.

        **⚠️ Upozorenje - Overfitting:**
        - Visoki stepen može dovesti do "overfittinga"
        - Pravilo: stepen << broj tačaka
        - Koristi Adjusted R² za ocjenu

        **Adjusted R²:**
        $$R^2_{adj} = 1 - (1-R^2)\\frac{n-1}{n-p-1}$$

        gdje je $p$ broj parametara (stepen + 1).
        """)

# Rezultati
if run_button:
    if method == "Linearna regresija":
        st.header("Linearna Regresija")
        result = linear_regression(x_data, y_data)

    elif method == "Eksponencijalna aproksimacija":
        st.header("Eksponencijalna Aproksimacija")

        if np.any(y_data <= 0):
            st.error("Sve y vrijednosti moraju biti pozitivne za eksponencijalnu aproksimaciju!")
            st.stop()

        result = exponential_regression(x_data, y_data)

    elif method == "Polinomijalna regresija":
        st.header(f"Polinomijalna Regresija (stepen {degree})")
        result = polynomial_regression(x_data, y_data, degree)

    elif method == "Poređenje modela":
        st.header("Poređenje Modela")
        comparison = compare_regression_models(x_data, y_data)

        # Rangiranje po R²
        st.subheader("🏆 Rangiranje Modela")
        import pandas as pd

        ranking_df = pd.DataFrame(comparison['summary'])
        ranking_df = ranking_df.sort_values('r_squared', ascending=False)
        ranking_df['r_squared'] = ranking_df['r_squared'].apply(lambda x: f"{x:.6f}")
        ranking_df.columns = ['Model', 'R²', 'Jednačina']
        st.dataframe(ranking_df, use_container_width=True)

        # Grafovi svih modela
        st.subheader("📊 Vizualizacija")

        cols = st.columns(2)
        model_names = list(comparison.keys())
        model_names = [m for m in model_names if m != 'summary' and 'r_squared' in comparison.get(m, {})]

        for i, model_name in enumerate(model_names[:4]):
            with cols[i % 2]:
                res = comparison[model_name]
                fig = plot_regression(x_data, y_data, res, f"{model_name} (R²={res['r_squared']:.4f})")
                st.plotly_chart(fig, use_container_width=True)

        result = None  # Da ne prikazuje korake ispod

    # Prikaz rezultata (za pojedinačne metode)
    if result and 'error_message' not in result:
        # Grafikon
        with col2:
            st.subheader("📈 Grafikon")
            title = f"{method} (R² = {result['r_squared']:.4f})"
            fig = plot_regression(x_data, y_data, result, title)
            st.plotly_chart(fig, use_container_width=True)

        # Metrike
        st.subheader("📊 Rezultati")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² (Koeficijent determinacije)", f"{result['r_squared']:.6f}")
            if 'r_squared_adjusted' in result:
                st.metric("Adjusted R²", f"{result['r_squared_adjusted']:.6f}")

        with col2:
            st.markdown("**Jednačina:**")
            st.latex(result['equation'].replace('y = ', 'y = ').replace('·', '\\cdot '))

        with col3:
            if 'a' in result and 'b' in result and 'A' not in result:
                st.markdown(f"**Nagib (a):** {result['a']:.6f}")
                st.markdown(f"**Odsječak (b):** {result['b']:.6f}")
            elif 'A' in result:
                st.markdown(f"**A:** {result['A']:.6f}")
                st.markdown(f"**B:** {result['B']:.6f}")

        # Interpretacija R²
        r2 = result['r_squared']
        if r2 >= 0.9:
            st.success(f"Odličan fit! Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        elif r2 >= 0.7:
            st.info(f"Dobar fit. Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        elif r2 >= 0.5:
            st.warning(f"Umjeren fit. Model objašnjava {r2*100:.1f}% varijabilnosti podataka.")
        else:
            st.error(f"Slab fit. Model objašnjava samo {r2*100:.1f}% varijabilnosti podataka.")

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result.get('steps', []):
            with st.expander(step['title'], expanded=(step['step'] <= 2)):
                if 'formulas' in step:
                    for name, val in step['formulas'].items():
                        st.markdown(f"${name} = {val}$")

                if 'formula' in step:
                    st.markdown(f"**Formula:** {step['formula']}")

                if 'calculation' in step:
                    st.markdown(f"**Računanje:** {step['calculation']}")

                if 'equation' in step:
                    st.success(f"**Jednačina:** {step['equation']}")

                if 'interpretation' in step:
                    st.info(step['interpretation'])

        # Tabela predviđenih vrijednosti i reziduala
        st.subheader("📋 Predviđene Vrijednosti i Reziduali")

        import pandas as pd
        pred_df = pd.DataFrame({
            'x': x_data,
            'y (stvarno)': y_data,
            'ŷ (predviđeno)': result['y_predicted'],
            'Rezidual (y - ŷ)': result['residuals']
        })
        pred_df['Rezidual (y - ŷ)'] = pred_df['Rezidual (y - ŷ)'].apply(lambda x: f"{x:.6f}")
        st.dataframe(pred_df, use_container_width=True)

    elif result and 'error_message' in result:
        st.error(result['error_message'])
