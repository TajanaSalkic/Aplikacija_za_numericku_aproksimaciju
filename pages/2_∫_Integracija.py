"""
Stranica: Numerička Integracija
===============================

Implementira:
- Trapezna metoda - bonus
- Simpsonova metoda (1/3) - rađeno na nastavi
- Romberg integracija - bonus (nije rađeno na nastavi)
- Gaussova kvadratura - rađeno na nastavi
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.integration import (trapezoidal, simpson, romberg, gauss_quadrature,
                                  compare_integration_methods, integrate_from_table,
                                  integrate_with_interpolation)
from utils.plotting import plot_integration_trapezoid, plot_integration_simpson
from utils.latex_helpers import format_integration_trapezoidal, format_integration_simpson, format_romberg_table, format_gauss_quadrature
import pandas as pd

st.set_page_config(page_title="Numerička Integracija", page_icon="∫", layout="wide")

st.title("∫ Numerička Integracija")
st.markdown("*Aproksimacija određenog integrala*")

# Sidebar
st.sidebar.header("⚙️ Postavke")
method = st.sidebar.selectbox(
    "Odaberite metodu:",
    ["📊 Integracija iz tablice", "Trapezna metoda", "Simpsonova metoda",
     "Romberg integracija", "Gaussova kvadratura", "Poređenje metoda"]
)

# Predefinisane funkcije
predefined_functions = {
    "x²": ("x**2", "Kvadratna funkcija"),
    "sin(x)": ("np.sin(x)", "Sinusna funkcija"),
    "e^x": ("np.exp(x)", "Eksponencijalna"),
    "1/(1+x²)": ("1/(1+x**2)", "Racionalna funkcija"),
    "√x": ("np.sqrt(x)", "Korijen (za x > 0)"),
    "x³ - 2x": ("x**3 - 2*x", "Kubna funkcija"),
    "cos(x)": ("np.cos(x)", "Kosinusna funkcija"),
    "Vlastita funkcija": ("", "")
}

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Funkcija")
selected_func = st.sidebar.selectbox("Odaberite funkciju:", list(predefined_functions.keys()))

if selected_func == "Vlastita funkcija":
    func_str = st.sidebar.text_input("f(x) =", value="x**2")
else:
    func_str, _ = predefined_functions[selected_func]
    st.sidebar.code(f"f(x) = {func_str}")

# Kreiranje funkcije
try:
    f = lambda x: eval(func_str)
except Exception as e:
    st.error(f"Greška u funkciji: {e}")
    st.stop()

# Granice integracije
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Granice")
col1, col2 = st.sidebar.columns(2)
a = col1.number_input("a (donja):", value=0.0, step=0.1)
b = col2.number_input("b (gornja):", value=1.0, step=0.1)

# Parametri specifični za metodu
if method in ["Trapezna metoda", "Simpsonova metoda"]:
    n = st.sidebar.slider("Broj podintervala (n):", min_value=2, max_value=100, value=10, step=2)
elif method == "Romberg integracija":
    max_order = st.sidebar.slider("Maksimalni red:", min_value=2, max_value=8, value=5)
elif method == "Gaussova kvadratura":
    n_gauss = st.sidebar.slider("Broj Gaussovih tačaka:", min_value=1, max_value=10, value=5)

run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

# Tačna vrijednost (ako je poznata)
st.sidebar.markdown("---")
show_exact = st.sidebar.checkbox("Prikaži tačnu vrijednost (ako je poznata)")
if show_exact:
    exact_value = st.sidebar.number_input("Tačna vrijednost:", value=0.333333, format="%.6f")

# Glavni sadržaj

# ============ INTEGRACIJA IZ TABLICE ============
if method == "📊 Integracija iz tablice":
    st.header("📊 Integracija iz Tablice Vrijednosti")

    st.info("""
    Unesite tablicu (x, y) vrijednosti - integral se računa bez poznate funkcije.
    Program automatski bira najbolju metodu na osnovu podataka.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Unos podataka")

        data_source = st.radio("Način unosa:", ["Tekst", "Primjer podataka"], horizontal=True)

        if data_source == "Tekst":
            x_input = st.text_area("X vrijednosti (zarezom odvojene):",
                                   value="0, 1, 2, 3, 4, 5")
            y_input = st.text_area("Y vrijednosti (f(x)):",
                                   value="0, 1, 4, 9, 16, 25")
        else:
            example = st.selectbox("Primjer:", [
                "Kvadratna funkcija",
                "Sinusna funkcija",
                "Brzina vozila",
                "Eksperimentalni podaci"
            ])

            if example == "Kvadratna funkcija":
                x_input = "0, 1, 2, 3, 4, 5"
                y_input = "0, 1, 4, 9, 16, 25"
                st.info("∫x² dx od 0 do 5 = 125/3 ≈ 41.67")
            elif example == "Sinusna funkcija":
                x_input = "0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.14159"
                y_input = "0, 0.479, 0.841, 0.997, 0.909, 0.598, 0.141, 0"
                st.info("∫sin(x) dx od 0 do π = 2")
            elif example == "Brzina vozila":
                x_input = "0, 1, 2, 3, 4, 5, 6"
                y_input = "0, 10, 18, 24, 28, 30, 30"
                st.info("Brzina (m/s) → Integral = pređeni put (m)")
            else:
                x_input = "0, 2, 4, 6, 8, 10"
                y_input = "0, 5.2, 18.1, 37.5, 62.8, 95.0"

            st.text_area("X:", value=x_input, disabled=True, key="x_disp")
            st.text_area("Y:", value=y_input, disabled=True, key="y_disp")

        integration_method = st.selectbox("Metoda integracije:",
                                          ["auto", "trapezoid", "simpson"])

        use_interpolation = st.checkbox("Koristi interpolaciju za veću preciznost")

    if st.button("🚀 Izračunaj integral", type="primary"):
        try:
            x_data = np.array([float(x.strip()) for x in x_input.split(',')])
            y_data = np.array([float(y.strip()) for y in y_input.split(',')])

            if len(x_data) != len(y_data):
                st.error("Broj X i Y vrijednosti mora biti jednak!")
            else:
                with col2:
                    st.subheader("Rezultati")

                    if use_interpolation:
                        result = integrate_with_interpolation(x_data, y_data,
                                                             method=integration_method,
                                                             n_interp=100)
                        st.success(f"**Integral (sa interpolacijom): I ≈ {result['integral']:.6f}**")
                        if result.get('direct_integral'):
                            st.info(f"Direktna integracija: I ≈ {result['direct_integral']:.6f}")
                    else:
                        result = integrate_from_table(x_data, y_data, method=integration_method)
                        st.success(f"**Integral: I ≈ {result['integral']:.6f}**")

                    st.markdown(f"**Metoda:** {result['method']}")
                    st.markdown(f"**Broj tačaka:** {result.get('n_points', len(x_data))}")

                    # Prikaz podataka
                    st.subheader("📋 Podaci")
                    df = pd.DataFrame({'x': x_data, 'y': y_data})
                    st.dataframe(df, use_container_width=True)

                    # Vizualizacija
                    import plotly.graph_objects as go

                    fig = go.Figure()

                    # Podaci kao scatter
                    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers+lines',
                                            name='Podaci', line=dict(color='blue')))

                    # Ispuna ispod krivulje
                    fig.add_trace(go.Scatter(x=list(x_data) + [x_data[-1], x_data[0]],
                                            y=list(y_data) + [0, 0],
                                            fill='toself', fillcolor='rgba(0,100,255,0.3)',
                                            line=dict(color='rgba(0,0,0,0)'),
                                            name='Integral'))

                    fig.update_layout(title=f'Integracija iz tablice (I ≈ {result["integral"]:.4f})',
                                     xaxis_title='x', yaxis_title='y')
                    st.plotly_chart(fig, use_container_width=True)

                    # Koraci
                    if 'steps' in result:
                        st.subheader("📝 Koraci")
                        for step in result['steps']:
                            with st.expander(step['title']):
                                for key, val in step.items():
                                    if key != 'title':
                                        st.write(f"**{key}:** {val}")

        except Exception as e:
            st.error(f"Greška: {e}")

elif method == "Trapezna metoda":
    st.header("Trapezna Metoda")

    with st.expander("📖 Teorija (Bonus metoda)", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Trapezna metoda aproksimira integral **površinom trapeza** ispod krivulje.

        ### Formula

        $$I \\approx \\frac{h}{2} \\left[ f(x_0) + 2\\sum_{i=1}^{n-1} f(x_i) + f(x_n) \\right]$$

        gdje je:
        - $h = \\frac{b-a}{n}$ (širina podintervala)
        - $x_i = a + ih$ (tačke podjele)

        ### Greška

        Greška trapezne metode je $O(h^2)$:
        $$E = -\\frac{(b-a)^3}{12n^2} f''(\\xi)$$

        za neki $\\xi \\in [a,b]$.
        """)

    if run_button:
        result = trapezoidal(f, a, b, n)

        st.success(f"✅ Integral: **I ≈ {result['integral']:.10f}**")

        if show_exact:
            error = abs(result['integral'] - exact_value)
            st.info(f"Greška u odnosu na tačnu vrijednost: {error:.2e}")

        # Vizualizacija
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Vizualizacija")
            fig = plot_integration_trapezoid(f, a, b, n, result['integral'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📋 Parametri")
            st.markdown(f"""
            - **n** = {n} podintervala
            - **h** = {result['h']:.6f}
            - **a** = {a}, **b** = {b}
            """)

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")
        st.markdown(format_integration_trapezoidal(result['h'], result['y_points'], result['integral']))

        # Tabela trapeza
        st.subheader("📋 Detalji Trapeza")
        import pandas as pd
        trap_data = [{
            'Trapez': t['index'],
            'x_lijevo': f"{t['x_left']:.4f}",
            'x_desno': f"{t['x_right']:.4f}",
            'f(x_lijevo)': f"{t['y_left']:.6f}",
            'f(x_desno)': f"{t['y_right']:.6f}",
            'Površina': f"{t['area']:.6f}"
        } for t in result['trapezoids']]
        st.dataframe(pd.DataFrame(trap_data), use_container_width=True)

elif method == "Simpsonova metoda":
    st.header("Simpsonova Metoda (1/3)")

    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Simpsonova metoda koristi **paraboličnu** (kvadratnu) aproksimaciju funkcije
        umjesto linearne. Kroz svake tri uzastopne tačke provlači se parabola.

        ### Formula

        $$I \\approx \\frac{h}{3} \\left[ f(x_0) + 4\\sum_{\\text{neparni}} f(x_i) + 2\\sum_{\\text{parni}} f(x_i) + f(x_n) \\right]$$

        **NAPOMENA:** n mora biti **paran broj**!

        ### Greška

        Greška Simpsonove metode je $O(h^4)$:
        $$E = -\\frac{(b-a)^5}{180n^4} f^{(4)}(\\xi)$$

        Simpsonova metoda je **znatno preciznija** od trapezne za isti broj tačaka!
        """)

    if run_button:
        result = simpson(f, a, b, n)

        st.success(f"✅ Integral: **I ≈ {result['integral']:.10f}**")

        if show_exact:
            error = abs(result['integral'] - exact_value)
            st.info(f"Greška u odnosu na tačnu vrijednost: {error:.2e}")

        # Vizualizacija
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Vizualizacija")
            fig = plot_integration_simpson(f, a, b, n, result['integral'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📋 Parametri")
            st.markdown(f"""
            - **n** = {result['n']} podintervala (paran)
            - **h** = {result['h']:.6f}
            - **Broj parabola** = {result['n'] // 2}
            """)

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")
        st.markdown(format_integration_simpson(result['h'], result['y_points'], result['integral']))

elif method == "Romberg integracija":
    st.header("Romberg Integracija")

    with st.expander("📖 Teorija (Bonus metoda - nije rađena na nastavi)", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Romberg integracija kombinuje **trapeznu metodu** sa **Richardsonovom ekstrapolacijom**
        za postizanje visoke preciznosti.

        ### Richardsonova ekstrapolacija

        Ako je greška trapezne metode oblika $E = c_1h^2 + c_2h^4 + ...$, možemo eliminisati
        vodeći član kombinovanjem aproksimacija:

        $$S(h) = \\frac{4T(h/2) - T(h)}{3}$$

        Ovo daje **Simpsonovu metodu** sa greškom $O(h^4)$!

        ### Romberg shema

        Gradimo trougaonu tabelu $R[i,j]$:

        $$R[i,j] = \\frac{4^j R[i,j-1] - R[i-1,j-1]}{4^j - 1}$$

        ### Interpretacija kolona

        | Kolona | Metoda | Greška |
        |--------|--------|--------|
        | 0 | Trapezna | $O(h^2)$ |
        | 1 | Simpson | $O(h^4)$ |
        | 2 | Boole | $O(h^6)$ |
        | j | - | $O(h^{2j+2})$ |
        """)

    if run_button:
        result = romberg(f, a, b, max_order)

        st.success(f"✅ Integral: **I ≈ {result['integral']:.12f}**")
        st.info(f"Procjena greške: {result['error_estimate']:.2e}")

        if show_exact:
            actual_error = abs(result['integral'] - exact_value)
            st.info(f"Stvarna greška: {actual_error:.2e}")

        # Romberg tabela
        st.subheader("📊 Romberg Tabela")
        st.markdown(format_romberg_table(result['R_table']))

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            with st.expander(step['title'], expanded=(step['step'] <= 2)):
                if 'formula' in step:
                    st.markdown(f"**Formula:** {step['formula']}")
                if 'calculation' in step:
                    st.markdown(f"**Računanje:** {step['calculation']}")
                if 'extrapolations' in step:
                    for ext in step['extrapolations']:
                        st.markdown(f"- $R[{step['step']},{ext['j']}] = {ext['result']:.10f}$")

elif method == "Gaussova kvadratura":
    st.header("Gaussova Kvadratura")

    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Gaussova kvadratura je **optimalna** metoda numeričke integracije.
        Koristi posebno odabrane tačke i težine koji maksimiziraju preciznost.

        ### Formula

        $$\\int_a^b f(x)dx \\approx \\frac{b-a}{2} \\sum_{i=1}^{n} w_i \\cdot f(x_i)$$

        gdje su:
        - $t_i$ - Gaussove tačke (nule Legendreovih polinoma na [-1,1])
        - $w_i$ - odgovarajuće težine
        - $x_i = \\frac{b-a}{2}t_i + \\frac{a+b}{2}$ - transformirane tačke

        ### Preciznost

        **n-točkasta** Gaussova kvadratura je **TAČNA** za polinome stepena ≤ $2n-1$!

        | n tačaka | Tačno za stepen ≤ |
        |----------|-------------------|
        | 1 | 1 |
        | 2 | 3 |
        | 3 | 5 |
        | 5 | 9 |
        | 10 | 19 |

        ### Legendreovi polinomi

        Gaussove tačke su nule Legendreovih polinoma:
        - $P_0(x) = 1$
        - $P_1(x) = x$
        - $P_2(x) = \\frac{1}{2}(3x^2 - 1)$
        - $P_3(x) = \\frac{1}{2}(5x^3 - 3x)$
        """)

    if run_button:
        result = gauss_quadrature(f, a, b, n_gauss)

        st.success(f"✅ Integral: **I ≈ {result['integral']:.12f}**")
        st.info(f"Metoda je tačna za polinome stepena ≤ {result['exact_for_degree']}")

        if show_exact:
            error = abs(result['integral'] - exact_value)
            st.info(f"Greška u odnosu na tačnu vrijednost: {error:.2e}")

        # Gaussove tačke i težine
        st.subheader("📊 Gaussove Tačke i Težine")
        st.markdown(format_gauss_quadrature(n_gauss, result['gauss_points'], result['gauss_weights']))

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            with st.expander(step['title'], expanded=True):
                if 'formula' in step:
                    st.markdown(f"**Formula:** ${step['formula']}$")
                if 'transformation' in step:
                    st.markdown(f"**Transformacija:** ${step['transformation']}$")
                if 'details' in step:
                    import pandas as pd
                    df = pd.DataFrame(step['details'])
                    st.dataframe(df, use_container_width=True)
                if 'integral' in step:
                    st.markdown(f"**Rezultat:** I = {step['integral']:.10f}")

elif method == "Poređenje metoda":
    st.header("Poređenje Metoda Integracije")

    if run_button:
        # Trapezna i Simpson za različite n
        st.subheader("📊 Konvergencija sa Povećanjem n")

        import pandas as pd

        n_values = [4, 8, 16, 32, 64]
        data = []

        for n_val in n_values:
            trap = trapezoidal(f, a, b, n_val)
            simp = simpson(f, a, b, n_val)

            row = {
                'n': n_val,
                'Trapezna': trap['integral'],
                'Simpson': simp['integral']
            }

            if show_exact:
                row['Greška (Trap)'] = abs(trap['integral'] - exact_value)
                row['Greška (Simp)'] = abs(simp['integral'] - exact_value)

            data.append(row)

        st.dataframe(pd.DataFrame(data), use_container_width=True)

        # Romberg
        st.subheader("📊 Romberg Tabela")
        rom = romberg(f, a, b, 5)
        st.markdown(format_romberg_table(rom['R_table']))

        # Gauss za različite n
        st.subheader("📊 Gaussova Kvadratura")

        gauss_data = []
        for n_g in [2, 3, 5, 7, 10]:
            g = gauss_quadrature(f, a, b, n_g)
            row = {
                'n tačaka': n_g,
                'Integral': g['integral'],
                'Tačno za stepen ≤': g['exact_for_degree']
            }
            if show_exact:
                row['Greška'] = abs(g['integral'] - exact_value)
            gauss_data.append(row)

        st.dataframe(pd.DataFrame(gauss_data), use_container_width=True)

        # Zaključak
        st.markdown("""
        ### 🔍 Analiza

        | Metoda | Greška | Prednosti | Mane |
        |--------|--------|-----------|------|
        | **Trapezna** | $O(h^2)$ | Jednostavna | Niska preciznost |
        | **Simpson** | $O(h^4)$ | Dobra preciznost | n mora biti paran |
        | **Romberg** | Do $O(h^{2k})$ | Visoka preciznost | Više računanja |
        | **Gauss** | Optimalna | Najbolja preciznost | Neravnomjerne tačke |
        """)
