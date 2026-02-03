"""
Stranica: Numerička Derivacija
==============================

Implementira:
- Automatska detekcija metode (Forward/Backward/Central)
- Unos funkcije ili tablice vrijednosti
- Poređenje grešaka za različite h
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.differentiation import (forward_diff, backward_diff, central_diff,
                                      compare_errors, higher_order_derivatives,
                                      auto_differentiate, differentiate_from_table)
from utils.plotting import plot_differentiation_comparison
from utils.latex_helpers import format_differentiation_formulas

st.set_page_config(page_title="Numerička Derivacija", page_icon="∂", layout="wide")

st.title("∂ Numerička Derivacija")
st.markdown("*Aproksimacija derivacija pomoću konačnih diferencija*")

# Sidebar
st.sidebar.header("⚙️ Postavke")
method = st.sidebar.selectbox(
    "Odaberite metodu:",
    ["🔄 Automatska detekcija", "Unos tablice vrijednosti",
     "Forward Difference", "Backward Difference", "Central Difference",
     "Poređenje grešaka", "Više derivacije"]
)

# Teorija za sve metode
with st.expander("📖 Teorija Numeričke Derivacije (Bonus - nije rađeno na nastavi)", expanded=False):
    st.markdown(format_differentiation_formulas())

st.markdown("---")

# ============ AUTOMATSKA DETEKCIJA ============
if method == "🔄 Automatska detekcija":
    st.header("🔄 Automatska Detekcija Metode")

    st.info("""
    **Automatski izbor metode:**
    - **Forward Difference**: Na lijevom rubu domene
    - **Backward Difference**: Na desnom rubu domene
    - **Central Difference**: U unutrašnjosti (najpreciznija - O(h²))
    """)

    input_type = st.radio("Način unosa:", ["Funkcija", "Tablica vrijednosti"], horizontal=True)

    if input_type == "Funkcija":
        col1, col2 = st.columns(2)

        with col1:
            func_str = st.text_input("f(x) =", value="np.sin(x)")
            deriv_str = st.text_input("f'(x) = (opcionalno, za provjeru)", value="np.cos(x)")

        with col2:
            a = st.number_input("Lijeva granica domene:", value=0.0)
            b = st.number_input("Desna granica domene:", value=np.pi)
            h = st.number_input("Korak h:", value=0.1, format="%.4f")

        if st.button("🚀 Izračunaj derivacije", type="primary"):
            try:
                f = lambda x: eval(func_str)
                df_exact = lambda x: eval(deriv_str) if deriv_str else None

                result = auto_differentiate(f=f, domain=(a, b), h=h)

                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.success(f"Izračunato {result['summary']['total_points']} derivacija")

                    # Sažetak metoda
                    st.subheader("📊 Korištene Metode")
                    method_counts = result['summary']['method_counts']
                    cols = st.columns(3)
                    for i, (m, count) in enumerate(method_counts.items()):
                        cols[i % 3].metric(m, f"{count} tačaka")

                    # Tabela rezultata
                    st.subheader("📋 Rezultati")

                    table_data = []
                    for d in result['derivatives']:
                        row = {
                            'x': f"{d['x']:.4f}",
                            "f'(x)": f"{d['derivative']:.6f}",
                            'Metoda': d['method'],
                            'Razlog': d['reason']
                        }
                        if df_exact:
                            exact_val = df_exact(d['x'])
                            row['Tačna vrijednost'] = f"{exact_val:.6f}"
                            row['Greška'] = f"{abs(d['derivative'] - exact_val):.2e}"
                        table_data.append(row)

                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                    # Objašnjenje
                    st.markdown(result['summary']['explanation'])

            except Exception as e:
                st.error(f"Greška: {e}")

    else:  # Tablica vrijednosti
        st.subheader("📊 Unos Tablice Vrijednosti")

        col1, col2 = st.columns(2)

        with col1:
            x_input = st.text_area("X vrijednosti (svaka u novom redu ili zarezom):",
                                   value="0\n0.5\n1.0\n1.5\n2.0\n2.5\n3.0")

        with col2:
            y_input = st.text_area("Y vrijednosti (f(x)):",
                                   value="0\n0.479\n0.841\n0.997\n0.909\n0.598\n0.141")

        if st.button("🚀 Izračunaj derivacije iz tablice", type="primary"):
            try:
                # Parsiranje unosa
                x_data = [float(x.strip()) for x in x_input.replace(',', '\n').split('\n') if x.strip()]
                y_data = [float(y.strip()) for y in y_input.replace(',', '\n').split('\n') if y.strip()]

                if len(x_data) != len(y_data):
                    st.error("Broj X i Y vrijednosti mora biti jednak!")
                else:
                    result = auto_differentiate(x_points=np.array(x_data), y_points=np.array(y_data))

                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        st.success(f"Izračunato {len(result['derivatives'])} derivacija")

                        # Sažetak metoda
                        st.subheader("📊 Automatski Izbor Metoda")

                        for choice in result['method_choices']:
                            icon = "⬅️" if "Forward" in choice['method'] else ("➡️" if "Backward" in choice['method'] else "↔️")
                            st.markdown(f"{icon} **x = {choice['x']:.4f}**: {choice['method']} - *{choice['reason']}*")

                        # Tabela rezultata
                        st.subheader("📋 Rezultati")

                        table_data = []
                        for d in result['derivatives']:
                            table_data.append({
                                'x': f"{d['x']:.4f}",
                                "f'(x)": f"{d['derivative']:.6f}",
                                'Metoda': d['method'],
                                'h': f"{d['h']:.4f}",
                                'Formula': d['formula']
                            })

                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            except Exception as e:
                st.error(f"Greška pri parsiranju: {e}")

# ============ UNOS TABLICE VRIJEDNOSTI ============
elif method == "Unos tablice vrijednosti":
    st.header("📊 Derivacija iz Tablice Vrijednosti")

    st.markdown("""
    Unesite tablicu vrijednosti (x, y) i program će automatski:
    1. Odrediti optimalnu metodu za svaku tačku
    2. Izračunati derivacije
    3. Opcionalno kreirati interpoliranu funkciju
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unos podataka")
        data_method = st.radio("Način unosa:", ["Tekst", "Primjer podataka"], horizontal=True)

        if data_method == "Tekst":
            x_input = st.text_area("X vrijednosti:", value="0, 1, 2, 3, 4, 5")
            y_input = st.text_area("Y vrijednosti:", value="0, 1, 4, 9, 16, 25")
        else:
            example = st.selectbox("Odaberite primjer:", [
                "Kvadratna funkcija (y = x²)",
                "Sinusna funkcija",
                "Eksponencijalna funkcija",
                "Eksperimentalni podaci"
            ])

            if example == "Kvadratna funkcija (y = x²)":
                x_input = "0, 1, 2, 3, 4, 5"
                y_input = "0, 1, 4, 9, 16, 25"
            elif example == "Sinusna funkcija":
                x_input = "0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.14"
                y_input = "0, 0.479, 0.841, 0.997, 0.909, 0.598, 0.141, 0.0"
            elif example == "Eksponencijalna funkcija":
                x_input = "0, 0.5, 1.0, 1.5, 2.0"
                y_input = "1, 1.649, 2.718, 4.482, 7.389"
            else:
                x_input = "0, 2, 4, 6, 8, 10"
                y_input = "0, 5.2, 18.1, 37.5, 62.8, 95.0"

            st.text_area("X vrijednosti:", value=x_input, disabled=True)
            st.text_area("Y vrijednosti:", value=y_input, disabled=True)

    with_interpolation = st.checkbox("Kreiraj interpoliranu funkciju (kubni spline)")

    if st.button("🚀 Izračunaj", type="primary"):
        try:
            x_data = np.array([float(x.strip()) for x in x_input.split(',')])
            y_data = np.array([float(y.strip()) for y in y_input.split(',')])

            result = differentiate_from_table(x_data, y_data, return_function=with_interpolation)

            with col2:
                st.subheader("Rezultati")

                # Tabela
                table_data = []
                for d in result['derivatives']:
                    table_data.append({
                        'x': d['x'],
                        "f'(x)": d['derivative'],
                        'Metoda': d['method']
                    })

                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                # Graf ako imamo interpolaciju
                if with_interpolation and 'interpolated_function' in result:
                    import plotly.graph_objects as go

                    cs = result['interpolated_function']
                    cs_deriv = result['interpolated_derivative']

                    x_fine = np.linspace(x_data[0], x_data[-1], 200)
                    y_fine = cs(x_fine)
                    dy_fine = cs_deriv(x_fine)

                    fig = go.Figure()

                    # Originalni podaci
                    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                                            name='Podaci', marker=dict(size=10)))

                    # Interpolirana funkcija
                    fig.add_trace(go.Scatter(x=x_fine, y=y_fine, mode='lines',
                                            name='Interpolacija', line=dict(color='blue')))

                    # Derivacije (točke)
                    deriv_x = [d['x'] for d in result['derivatives']]
                    deriv_y = [d['derivative'] for d in result['derivatives']]
                    fig.add_trace(go.Scatter(x=deriv_x, y=deriv_y, mode='markers',
                                            name="f'(x) (numerički)", marker=dict(size=8, symbol='diamond')))

                    # Derivacija splina
                    fig.add_trace(go.Scatter(x=x_fine, y=dy_fine, mode='lines',
                                            name="f'(x) (spline)", line=dict(color='red', dash='dash')))

                    fig.update_layout(title='Funkcija i Derivacija',
                                     xaxis_title='x', yaxis_title='y')
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(result.get('interpolation_info', ''))

        except Exception as e:
            st.error(f"Greška: {e}")

# ============ POJEDINAČNE METODE ============
elif method in ["Forward Difference", "Backward Difference", "Central Difference"]:
    # Predefinisane funkcije
    predefined_functions = {
        "sin(x)": ("np.sin(x)", "np.cos(x)"),
        "e^x": ("np.exp(x)", "np.exp(x)"),
        "x²": ("x**2", "2*x"),
        "x³": ("x**3", "3*x**2"),
        "cos(x)": ("np.cos(x)", "-np.sin(x)"),
        "Vlastita funkcija": ("", "")
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Funkcija")
    selected_func = st.sidebar.selectbox("Odaberite funkciju:", list(predefined_functions.keys()))

    if selected_func == "Vlastita funkcija":
        func_str = st.sidebar.text_input("f(x) =", value="np.sin(x)")
        deriv_str = st.sidebar.text_input("f'(x) =", value="np.cos(x)")
    else:
        func_str, deriv_str = predefined_functions[selected_func]
        st.sidebar.code(f"f(x) = {func_str}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Parametri")
    x_point = st.sidebar.number_input("Tačka x:", value=1.0, step=0.1)
    h = st.sidebar.number_input("Korak h:", value=0.01, format="%.6f")

    run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

    try:
        f = lambda x: eval(func_str)
        df_exact = lambda x: eval(deriv_str) if deriv_str else None
    except:
        st.error("Greška u funkciji")
        st.stop()

    if method == "Forward Difference":
        st.header("Forward Difference (Unaprijedna Diferencija)")
        st.latex(r"f'(x) \approx \frac{f(x+h) - f(x)}{h}")
        diff_func = forward_diff

    elif method == "Backward Difference":
        st.header("Backward Difference (Unazadna Diferencija)")
        st.latex(r"f'(x) \approx \frac{f(x) - f(x-h)}{h}")
        diff_func = backward_diff

    else:  # Central
        st.header("Central Difference (Centralna Diferencija)")
        st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}")
        st.info("Centralna diferencija ima grešku **O(h²)** - bolje od forward/backward!")
        diff_func = central_diff

    if run_button:
        result = diff_func(f, x_point, h, df_exact)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Aproksimacija f'(x)", f"{result['derivative']:.10f}")
        with col2:
            if 'exact' in result:
                st.metric("Tačna vrijednost", f"{result['exact']:.10f}")

        if 'error' in result:
            st.metric("Apsolutna greška", f"{result['error']:.2e}")

        st.subheader("📝 Koraci")
        for step in result['steps']:
            st.markdown(f"**{step['title']}**: {step.get('description', '')}")

# ============ POREĐENJE GREŠAKA ============
elif method == "Poređenje grešaka":
    st.header("Poređenje Grešaka za Različite h")

    predefined_functions = {
        "sin(x)": ("np.sin(x)", "np.cos(x)"),
        "e^x": ("np.exp(x)", "np.exp(x)"),
        "x²": ("x**2", "2*x"),
    }

    selected_func = st.selectbox("Funkcija:", list(predefined_functions.keys()))
    func_str, deriv_str = predefined_functions[selected_func]

    x_point = st.number_input("Tačka x:", value=1.0)

    if st.button("🚀 Analiziraj greške", type="primary"):
        f = lambda x: eval(func_str)
        df_exact = lambda x: eval(deriv_str)

        result = compare_errors(f, x_point, df_exact)

        st.subheader("📊 Log-Log Graf Grešaka")
        fig = plot_differentiation_comparison(result)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎯 Optimalne Vrijednosti h")
        col1, col2, col3 = st.columns(3)

        with col1:
            opt = result['forward_optimal']
            st.metric("Forward", f"h = {opt['h']:.0e}", f"Greška: {opt['error']:.2e}")
        with col2:
            opt = result['backward_optimal']
            st.metric("Backward", f"h = {opt['h']:.0e}", f"Greška: {opt['error']:.2e}")
        with col3:
            opt = result['central_optimal']
            st.metric("Central", f"h = {opt['h']:.0e}", f"Greška: {opt['error']:.2e}")

# ============ VIŠE DERIVACIJE ============
elif method == "Više derivacije":
    st.header("Više Derivacije")

    func_str = st.text_input("f(x) =", value="np.sin(x)")
    x_point = st.number_input("Tačka x:", value=1.0)
    h = st.number_input("Korak h:", value=0.01, format="%.4f")
    max_order = st.slider("Maksimalni red derivacije:", 1, 4, 4)

    if st.button("🚀 Izračunaj", type="primary"):
        f = lambda x: eval(func_str)
        result = higher_order_derivatives(f, x_point, h, max_order)

        derivs = {
            'first': "f'(x)", 'second': "f''(x)",
            'third': "f'''(x)", 'fourth': "f''''(x)"
        }

        for key, name in derivs.items():
            if key in result:
                st.metric(name, f"{result[key]['value']:.10f}")
