"""
Stranica: Numerička Derivacija iz Tablice
=========================================

Numerička derivacija kada je ulaz tablica podataka (x, y).
Koristi konačne diferencije ili aproksimaciju podataka.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.differentiation import (auto_differentiate, differentiate_from_table)
from methods.regression import (linear_regression, polynomial_regression,
                                 exponential_regression, power_regression,
                                 logarithmic_regression, compare_regression_models)

st.set_page_config(page_title="Derivacija iz Tablice", page_icon="∂", layout="wide")

st.title("∂ Numerička Derivacija iz Tablice")
st.markdown("*Izračunavanje derivacija kada je ulaz tablica podataka*")

st.info("""
**Princip:** Kada nemamo eksplicitnu funkciju f(x), već samo tablicu vrijednosti (x, y),
možemo izračunati derivaciju koristeći konačne diferencije ili prvo aproksimirati
podatke funkcijom pa diferencirati tu funkciju.
""")

# Sidebar
st.sidebar.header("⚙️ Postavke")

# Način derivacije
diff_approach = st.sidebar.radio(
    "Pristup derivaciji:",
    ["Konačne diferencije", "Derivacija preko aproksimacije"]
)

if diff_approach == "Konačne diferencije":
    st.sidebar.info("""
    **Automatski odabir metode:**
    - Forward difference: lijevi rub
    - Backward difference: desni rub
    - Central difference: unutrašnje tačke
    """)
    use_interpolation = st.sidebar.checkbox("Kreiraj interpoliranu funkciju (kubni spline)")
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
    "Put (m) → Brzina (m/s)": {
        'x': [0, 1, 2, 3, 4, 5, 6],
        'y': [0, 2, 8, 18, 32, 50, 72],
        'description': 'Put tokom vremena. Derivacija = brzina.',
        'interpretation': 'dy/dx = brzina kretanja'
    },
    "Kvadratna funkcija (x²)": {
        'x': [0, 1, 2, 3, 4, 5],
        'y': [0, 1, 4, 9, 16, 25],
        'description': 'y = x². Tačna derivacija: y\' = 2x',
        'interpretation': 'Očekivana derivacija: 0, 2, 4, 6, 8, 10'
    },
    "Sinusna funkcija": {
        'x': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'y': [0.0, 0.479, 0.841, 0.997, 0.909, 0.598, 0.141],
        'description': 'y = sin(x). Derivacija: y\' = cos(x)',
        'interpretation': 'Očekivana derivacija: cos(x)'
    },
    "Eksponencijalna funkcija": {
        'x': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'y': [1.0, 1.649, 2.718, 4.482, 7.389, 12.182, 20.086],
        'description': 'y = e^x. Derivacija: y\' = e^x',
        'interpretation': 'Derivacija jednaka originalnoj funkciji'
    },
    "Temperatura hlađenja": {
        'x': [0, 5, 10, 15, 20, 25, 30],
        'y': [100, 80, 65, 53, 44, 37, 32],
        'description': 'Temperatura tokom hlađenja',
        'interpretation': 'Derivacija = brzina hlađenja (negativna)'
    }
}

if data_source == "Predefinisani primjer":
    selected_example = st.sidebar.selectbox("Primjer:", list(predefined_examples.keys()))
    example = predefined_examples[selected_example]
    x_data = np.array(example['x'])
    y_data = np.array(example['y'])
    st.sidebar.info(example['description'])
    st.sidebar.success(example['interpretation'])

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
    ### Konačne Diferencije

    #### Forward Difference (Unaprijedna)
    $$f'(x) \\approx \\frac{f(x+h) - f(x)}{h}$$
    - Greška: $O(h)$
    - Koristi se na lijevom rubu domene

    #### Backward Difference (Unazadna)
    $$f'(x) \\approx \\frac{f(x) - f(x-h)}{h}$$
    - Greška: $O(h)$
    - Koristi se na desnom rubu domene

    #### Central Difference (Centralna)
    $$f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}$$
    - Greška: $O(h^2)$ - **najpreciznija**
    - Koristi se za unutrašnje tačke

    ### Derivacija preko Aproksimacije
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

    if diff_approach == "Konačne diferencije":
        st.header("Derivacija Konačnim Diferencijama")

        # Računanje
        result = differentiate_from_table(x_data, y_data, return_function=use_interpolation)

        # Prikaz metoda za svaku tačku
        st.subheader("📊 Automatski Odabir Metode za Svaku Tačku")

        method_icons = {
            'forward': '➡️ Forward',
            'backward': '⬅️ Backward',
            'central': '↔️ Central'
        }

        cols_methods = st.columns(len(result['derivatives']))
        for i, d in enumerate(result['derivatives']):
            method_type = 'central'
            if 'Forward' in d.get('method', ''):
                method_type = 'forward'
            elif 'Backward' in d.get('method', ''):
                method_type = 'backward'

            with cols_methods[i]:
                st.metric(
                    f"x = {d['x']:.2f}",
                    f"{d['derivative']:.4f}",
                    delta=method_icons.get(method_type, d.get('method', ''))
                )

        # Tabela rezultata
        st.subheader("📋 Rezultati")

        table_data = []
        for d in result['derivatives']:
            table_data.append({
                'x': f"{d['x']:.4f}",
                "f'(x)": f"{d['derivative']:.6f}",
                'Metoda': d.get('method', 'N/A'),
                'h': f"{d.get('h', 'N/A')}"
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Vizualizacija
        with col2:
            st.subheader("📈 Vizualizacija")

            fig = go.Figure()

            # Originalna funkcija
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                mode='markers+lines',
                name='f(x) - Originalni podaci',
                line=dict(color='blue'),
                marker=dict(size=10)
            ))

            # Derivacija
            deriv_x = [d['x'] for d in result['derivatives']]
            deriv_y = [d['derivative'] for d in result['derivatives']]

            fig.add_trace(go.Scatter(
                x=deriv_x, y=deriv_y,
                mode='markers+lines',
                name="f'(x) - Derivacija",
                line=dict(color='red', dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))

            # Interpolirana derivacija ako postoji
            if use_interpolation and 'interpolated_derivative' in result:
                x_fine = np.linspace(x_data[0], x_data[-1], 200)
                dy_fine = result['interpolated_derivative'](x_fine)

                fig.add_trace(go.Scatter(
                    x=x_fine, y=dy_fine,
                    mode='lines',
                    name="f'(x) - Spline interpolacija",
                    line=dict(color='green', width=2)
                ))

            fig.update_layout(
                title="Funkcija i Njena Derivacija",
                xaxis_title='x',
                yaxis_title='y',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

    else:  # Derivacija preko aproksimacije
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

        # Kreiraj derivaciju na osnovu tipa aproksimacije
        x_fine = np.linspace(x_data[0], x_data[-1], 200)
        y_approx_fine = []
        dy_approx_fine = []
        derivative_equation = ""

        if 'a' in approx_result and 'b' in approx_result and 'A' not in approx_result:
            if 'coefficients' not in approx_result:
                # Linearna: y = ax + b -> y' = a
                a = approx_result['a']
                b = approx_result['b']
                y_approx_fine = a * x_fine + b
                dy_approx_fine = np.full_like(x_fine, a)
                derivative_equation = f"y' = {a:.6f} (konstanta)"
            else:
                # Za logaritamsku i sl.
                if 'ln(x)' in approx_result.get('equation', ''):
                    # y = a + b*ln(x) -> y' = b/x
                    a = approx_result['a']
                    b = approx_result['b']
                    y_approx_fine = a + b * np.log(x_fine)
                    dy_approx_fine = b / x_fine
                    derivative_equation = f"y' = {b:.6f}/x"
                elif '√x' in approx_result.get('equation', ''):
                    # y = a + b*sqrt(x) -> y' = b/(2*sqrt(x))
                    a = approx_result['a']
                    b = approx_result['b']
                    y_approx_fine = a + b * np.sqrt(x_fine)
                    dy_approx_fine = b / (2 * np.sqrt(x_fine))
                    derivative_equation = f"y' = {b:.6f}/(2√x)"

        elif 'A' in approx_result and 'B' in approx_result:
            A = approx_result['A']
            B = approx_result['B']

            if 'e^' in approx_result.get('equation', ''):
                # Eksponencijalna: y = A*e^(Bx) -> y' = A*B*e^(Bx)
                y_approx_fine = A * np.exp(B * x_fine)
                dy_approx_fine = A * B * np.exp(B * x_fine)
                derivative_equation = f"y' = {A*B:.6f}·e^({B:.6f}x)"
            else:
                # Stepena: y = A*x^B -> y' = A*B*x^(B-1)
                y_approx_fine = A * (x_fine ** B)
                dy_approx_fine = A * B * (x_fine ** (B - 1))
                derivative_equation = f"y' = {A*B:.6f}·x^{B-1:.6f}"

        elif 'coefficients' in approx_result:
            # Polinomijalna
            coeffs = approx_result['coefficients']
            degree = len(coeffs) - 1

            # y = a0 + a1*x + a2*x^2 + ...
            y_approx_fine = np.polyval(coeffs[::-1], x_fine)

            # y' = a1 + 2*a2*x + 3*a3*x^2 + ...
            deriv_coeffs = [i * coeffs[i] for i in range(1, len(coeffs))]
            if deriv_coeffs:
                dy_approx_fine = np.polyval(deriv_coeffs[::-1], x_fine)
            else:
                dy_approx_fine = np.zeros_like(x_fine)

            # Formatiraj jednačinu derivacije
            terms = []
            for i, c in enumerate(deriv_coeffs):
                if abs(c) > 1e-10:
                    if i == 0:
                        terms.append(f"{c:.4f}")
                    elif i == 1:
                        terms.append(f"{c:.4f}x")
                    else:
                        terms.append(f"{c:.4f}x^{i}")
            derivative_equation = "y' = " + " + ".join(terms) if terms else "y' = 0"

        # Za hiperboličku i racionalnu - numerička derivacija
        if len(dy_approx_fine) == 0:
            y_pred = np.array(approx_result['y_predicted'])
            from scipy.interpolate import interp1d
            interp_func = interp1d(x_data, y_pred, kind='cubic', fill_value='extrapolate')
            y_approx_fine = interp_func(x_fine)

            # Numerička derivacija
            h = x_fine[1] - x_fine[0]
            dy_approx_fine = np.gradient(y_approx_fine, h)
            derivative_equation = "y' (numerički izračunata)"

        st.success(f"**Derivacija:** {derivative_equation}")

        # Vrijednosti derivacije u tačkama podataka
        deriv_at_points = []
        for xi in x_data:
            idx = np.argmin(np.abs(x_fine - xi))
            deriv_at_points.append(dy_approx_fine[idx])

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
