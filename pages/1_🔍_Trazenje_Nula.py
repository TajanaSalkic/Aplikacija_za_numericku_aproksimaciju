"""
Stranica: Traženje Nula Funkcije (Root Finding)
===============================================

Implementira:
- Metoda Bisekcije (Dihotomija) - rađeno na nastavi
- Newton-Raphson metoda - bonus
- Metoda Sekante - rađeno na nastavi
"""

import streamlit as st
import numpy as np
import sys
import os

# Dodaj parent folder u path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.root_finding import bisection, newton_raphson, secant, compare_methods
from utils.plotting import plot_root_finding_bisection, plot_root_finding_newton, plot_convergence

st.set_page_config(page_title="Traženje Nula", page_icon="🔍", layout="wide")

st.title("🔍 Traženje Nula Funkcije")
st.markdown("*Pronalaženje korijena jednačine f(x) = 0*")

# Sidebar za izbor metode
st.sidebar.header("⚙️ Postavke")
method = st.sidebar.selectbox(
    "Odaberite metodu:",
    ["Metoda Bisekcije", "Newton-Raphson", "Metoda Sekante", "Poređenje svih metoda"]
)

# Predefinisane funkcije
predefined_functions = {
    "x³ - x - 2": ("x**3 - x - 2", "3*x**2 - 1", "Kubna funkcija"),
    "x² - 4": ("x**2 - 4", "2*x", "Kvadratna funkcija"),
    "sin(x) - 0.5": ("np.sin(x) - 0.5", "np.cos(x)", "Trigonometrijska"),
    "e^x - 3x": ("np.exp(x) - 3*x", "np.exp(x) - 3", "Eksponencijalna"),
    "x³ - 2x - 5": ("x**3 - 2*x - 5", "3*x**2 - 2", "Klasični primjer"),
    "cos(x) - x": ("np.cos(x) - x", "-np.sin(x) - 1", "Fiksna tačka"),
    "Vlastita funkcija": ("", "", "")
}

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Funkcija")
selected_func = st.sidebar.selectbox("Odaberite funkciju:", list(predefined_functions.keys()))

if selected_func == "Vlastita funkcija":
    func_str = st.sidebar.text_input("f(x) =", value="x**3 - x - 2")
    deriv_str = st.sidebar.text_input("f'(x) = (opcionalno)", value="3*x**2 - 1")
else:
    func_str, deriv_str, _ = predefined_functions[selected_func]
    st.sidebar.code(f"f(x) = {func_str}")

# Kreiranje funkcija
try:
    f = lambda x: eval(func_str)
    if deriv_str:
        df = lambda x: eval(deriv_str)
    else:
        df = None
except Exception as e:
    st.error(f"Greška u funkciji: {e}")
    st.stop()

# Parametri
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Parametri")

if method in ["Metoda Bisekcije", "Poređenje svih metoda"]:
    col1, col2 = st.sidebar.columns(2)
    a = col1.number_input("a (lijeva granica):", value=1.0, step=0.1)
    b = col2.number_input("b (desna granica):", value=2.0, step=0.1)

if method in ["Newton-Raphson", "Metoda Sekante", "Poređenje svih metoda"]:
    x0 = st.sidebar.number_input("x₀ (početna tačka):", value=1.5, step=0.1)
    if method == "Metoda Sekante":
        x1 = st.sidebar.number_input("x₁ (druga tačka):", value=2.0, step=0.1)

tol = st.sidebar.number_input("Tolerancija:", value=1e-6, format="%.1e", step=1e-7)
max_iter = st.sidebar.slider("Max iteracija:", min_value=10, max_value=100, value=50)

# Dugme za pokretanje
run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

# Glavni sadržaj
if method == "Metoda Bisekcije":
    st.header("Metoda Bisekcije (Dihotomija)")

    # Teorija
    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Metoda bisekcije koristi **teoremu o međuvrijednosti**: ako je funkcija $f$
        kontinuirana na $[a,b]$ i $f(a) \\cdot f(b) < 0$, tada postoji korijen u $(a,b)$.

        ### Algoritam

        1. Izračunaj srednju tačku: $c = \\frac{a + b}{2}$
        2. Ako je $|f(c)| < \\varepsilon$, c je rješenje
        3. Ako je $f(a) \\cdot f(c) < 0$, korijen je u $[a,c]$
        4. Inače, korijen je u $[c,b]$
        5. Ponovi

        ### Formula

        $$c_{n+1} = \\frac{a_n + b_n}{2}$$

        ### Konvergencija

        - **Linearna** konvergencija
        - Greška se prepolavlja: $e_{n+1} \\leq \\frac{e_n}{2}$
        - Potreban broj iteracija: $n \\geq \\log_2\\left(\\frac{b-a}{\\varepsilon}\\right)$
        """)

    if run_button:
        result = bisection(f, a, b, tol, max_iter)

        if result['converged']:
            st.success(f"✅ Korijen pronađen: **x = {result['root']:.10f}** nakon {result['iterations']} iteracija")
        elif result['error_message']:
            st.error(result['error_message'])
        else:
            st.warning(f"⚠️ Nije konvergiralo. Aproksimacija: x ≈ {result['root']:.10f}")

        # Grafikon
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Vizualizacija")
            fig = plot_root_finding_bisection(f, a, b, result['steps'], result['root'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Konvergencija")
            if result['steps']:
                fig_conv = plot_convergence(result['steps'], "Bisekcija")
                st.plotly_chart(fig_conv, use_container_width=True)

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            with st.expander(f"Iteracija {step['iteration']}", expanded=(step['iteration'] <= 3)):
                if step['c'] is not None:
                    st.markdown(f"""
                    **Interval:** $[{step['a']:.6f}, {step['b']:.6f}]$

                    **Srednja tačka:**
                    $$c = \\frac{{{step['a']:.6f} + {step['b']:.6f}}}{{2}} = {step['c']:.6f}$$

                    **Vrijednost funkcije:** $f(c) = f({step['c']:.6f}) = {step['fc']:.6f}$

                    **Greška:** $|b - a| = {step['error']:.6f}$

                    {step.get('description', '')}
                    """)
                else:
                    st.markdown(step.get('description', ''))

        # Tabela iteracija
        st.subheader("📋 Tabela Iteracija")
        import pandas as pd
        table_data = []
        for step in result['steps']:
            if step['c'] is not None:
                table_data.append({
                    'Iteracija': step['iteration'],
                    'a': f"{step['a']:.6f}",
                    'b': f"{step['b']:.6f}",
                    'c': f"{step['c']:.6f}",
                    'f(c)': f"{step['fc']:.6f}",
                    'Greška': f"{step['error']:.2e}"
                })
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

elif method == "Newton-Raphson":
    st.header("Newton-Raphson Metoda")

    # Teorija
    with st.expander("📖 Teorija (Bonus metoda - nije rađena na nastavi)", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Newton-Raphson metoda koristi **tangentnu liniju** funkcije za aproksimaciju korijena.
        Nova aproksimacija je presječna tačka tangente sa x-osom.

        ### Izvod formule

        Jednačina tangente u tački $(x_n, f(x_n))$:
        $$y - f(x_n) = f'(x_n)(x - x_n)$$

        Presječna tačka sa x-osom $(y = 0)$:
        $$x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}$$

        ### Numerička derivacija

        Ako derivacija nije poznata:
        $$f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}$$

        ### Konvergencija

        - **Kvadratna** konvergencija (kada konvergira)
        - $e_{n+1} \\approx C \\cdot e_n^2$
        - Broj tačnih decimala se udvostručuje u svakoj iteraciji

        ### ⚠️ Upozorenja

        - Može **divergirati** ako je početna tačka daleko od korijena
        - Problemi ako je $f'(x_n) \\approx 0$
        """)

    if run_button:
        result = newton_raphson(f, x0, tol, max_iter, df)

        if result['converged']:
            st.success(f"✅ Korijen pronađen: **x = {result['root']:.10f}** nakon {result['iterations']} iteracija")
            st.info(f"Korištena derivacija: {result['derivative_type']}")
        elif result['error_message']:
            st.error(result['error_message'])
        else:
            st.warning(f"⚠️ Nije konvergiralo. Aproksimacija: x ≈ {result['root']:.10f}")

        # Grafikon
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Vizualizacija Tangenti")
            fig = plot_root_finding_newton(f, result['steps'], result['root'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📈 Konvergencija")
            if result['steps']:
                errors = [{'iteration': s['iteration'], 'error': abs(s['fx'])} for s in result['steps'] if 'fx' in s]
                fig_conv = plot_convergence(errors, "Newton-Raphson")
                st.plotly_chart(fig_conv, use_container_width=True)

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            with st.expander(f"Iteracija {step['iteration']}", expanded=(step['iteration'] <= 3)):
                st.markdown(f"""
                **Trenutna aproksimacija:** $x_{{{step['iteration']}}} = {step['x']:.10f}$

                **Vrijednost funkcije:** $f(x_{{{step['iteration']}}}) = {step['fx']:.10f}$

                **Derivacija:** $f'(x_{{{step['iteration']}}}) = {step['dfx']:.10f}$
                """)

                if 'x_new' in step:
                    st.markdown(f"""
                    **Newton-Raphson formula:**
                    $$x_{{n+1}} = x_n - \\frac{{f(x_n)}}{{f'(x_n)}}$$

                    $$x_{{{step['iteration']+1}}} = {step['x']:.6f} - \\frac{{{step['fx']:.6f}}}{{{step['dfx']:.6f}}} = {step['x_new']:.10f}$$
                    """)

elif method == "Metoda Sekante":
    st.header("Metoda Sekante")

    # Teorija
    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Osnovna ideja

        Metoda sekante je slična Newton-Raphson metodi, ali **ne zahtijeva derivaciju**.
        Umjesto tangente, koristi **sekantu** kroz dvije prethodne tačke.

        ### Formula

        Aproksimacija derivacije:
        $$f'(x_n) \\approx \\frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}$$

        Iterativna formula:
        $$x_{n+1} = x_n - f(x_n) \\cdot \\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

        ### Konvergencija

        - **Superlinearna** konvergencija
        - Red konvergencije: $\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.618$ (zlatni rez)
        - Sporija od Newton-Raphson, ali ne zahtijeva derivaciju

        ### Prednosti i mane

        ✅ Ne zahtijeva derivaciju
        ✅ Brža od bisekcije
        ❌ Zahtijeva dvije početne tačke
        ❌ Može divergirati
        """)

    if run_button:
        result = secant(f, x0, x1, tol, max_iter)

        if result['converged']:
            st.success(f"✅ Korijen pronađen: **x = {result['root']:.10f}** nakon {result['iterations']} iteracija")
        elif result['error_message']:
            st.error(result['error_message'])
        else:
            st.warning(f"⚠️ Nije konvergiralo.")

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            with st.expander(f"Iteracija {step['iteration']}", expanded=(step['iteration'] <= 3)):
                st.markdown(f"""
                **Prethodne tačke:** $x_{{n-1}} = {step['x_prev']:.6f}$, $x_n = {step['x_curr']:.6f}$

                **Vrijednosti:** $f(x_{{n-1}}) = {step['f_prev']:.6f}$, $f(x_n) = {step['f_curr']:.6f}$
                """)

                if step['x_new'] is not None:
                    st.markdown(f"""
                    **Formula sekante:**
                    $$x_{{n+1}} = x_n - f(x_n) \\cdot \\frac{{x_n - x_{{n-1}}}}{{f(x_n) - f(x_{{n-1}})}}$$

                    **Nova aproksimacija:** $x_{{n+1}} = {step['x_new']:.10f}$

                    **Greška:** $|x_{{n+1}} - x_n| = {step.get('error', 0):.2e}$
                    """)

elif method == "Poređenje svih metoda":
    st.header("Poređenje Metoda")

    if run_button:
        results = compare_methods(f, a, b, x0, tol, max_iter, df)

        # Sumarno poređenje
        st.subheader("📊 Rezultati")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Bisekcija",
                f"{results['bisection']['iterations']} iter.",
                f"x = {results['bisection']['root']:.6f}" if results['bisection']['root'] else "Nije konvergiralo"
            )

        with col2:
            st.metric(
                "Newton-Raphson",
                f"{results['newton']['iterations']} iter.",
                f"x = {results['newton']['root']:.6f}" if results['newton']['root'] else "Nije konvergiralo"
            )

        with col3:
            st.metric(
                "Sekanta",
                f"{results['secant']['iterations']} iter.",
                f"x = {results['secant']['root']:.6f}" if results['secant']['root'] else "Nije konvergiralo"
            )

        # Tabela poređenja
        st.subheader("📋 Tabela Poređenja")
        import pandas as pd
        comparison_df = pd.DataFrame(results['summary'])
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("""
        ### 🔍 Analiza

        | Metoda | Konvergencija | Zahtjevi | Stabilnost |
        |--------|---------------|----------|------------|
        | **Bisekcija** | Linearna O(n) | f(a)·f(b) < 0 | Vrlo stabilna |
        | **Newton-Raphson** | Kvadratna O(n²) | Derivacija, dobra početna tačka | Može divergirati |
        | **Sekanta** | Superlinearna O(n^1.618) | Dvije početne tačke | Srednje stabilna |
        """)
