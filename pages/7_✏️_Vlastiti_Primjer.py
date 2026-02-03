"""
Stranica: Vlastiti Primjer iz Stvarnog Života
==============================================

Omogućava korisnicima da unesu vlastite primjere iz stvarnog života
i primijene bilo koju implementiranu numeričku metodu.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.root_finding import bisection, newton_raphson, secant
from methods.integration import trapezoidal, simpson, gauss_quadrature, integrate_from_table
from methods.differentiation import auto_differentiate
from methods.regression import linear_regression, exponential_regression, polynomial_regression
from methods.linear_systems import jacobi, gauss_seidel

st.set_page_config(page_title="Vlastiti Primjer", page_icon="✏️", layout="wide")

st.title("✏️ Kreiraj Vlastiti Primjer")
st.markdown("*Definiši problem iz stvarnog života i primijeni numeričke metode*")

# Izbor tipa problema
problem_type = st.selectbox(
    "🎯 Tip problema:",
    ["Traženje nule (Root Finding)",
     "Numerička integracija",
     "Numerička derivacija",
     "Regresija/Aproksimacija podataka",
     "Sistem linearnih jednačina"]
)

st.markdown("---")

# ============ TRAŽENJE NULE ============
if problem_type == "Traženje nule (Root Finding)":
    st.header("🔍 Traženje Nule - Vlastiti Primjer")

    st.markdown("""
    ### Primjeri iz stvarnog života:
    - **Fizika**: Kada tijelo dostiže određenu brzinu?
    - **Ekonomija**: Kada investicija dostiže ciljnu vrijednost?
    - **Inženjerstvo**: Pronalaženje tačke ravnoteže
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Definiši Problem")

        problem_name = st.text_input("Naziv problema:",
                                     value="Kada tijelo dostiže brzinu 50 m/s?")

        st.markdown("**Unesite jednačinu f(x) = 0**")
        st.markdown("*Primjer: brzina(t) - 50 = 0, gdje je brzina(t) = 60*(1 - e^(-0.1*t))*")

        func_str = st.text_area("f(x) = 0, gdje je f(x):",
                                value="60*(1 - np.exp(-0.1*x)) - 50",
                                help="Koristite 'x' kao varijablu. Dostupno: np.sin, np.cos, np.exp, np.log, np.sqrt")

        st.markdown("**Parametri metode**")
        method = st.selectbox("Metoda:", ["Bisekcija", "Newton-Raphson", "Sekanta"])

        if method == "Bisekcija":
            a = st.number_input("Lijeva granica (a):", value=0.0)
            b = st.number_input("Desna granica (b):", value=100.0)
        elif method == "Newton-Raphson":
            x0 = st.number_input("Početna aproksimacija (x₀):", value=10.0)
            df_str = st.text_input("f'(x) = (opcionalno):", value="6*np.exp(-0.1*x)")
        else:  # Sekanta
            x0 = st.number_input("Prva tačka (x₀):", value=1.0)
            x1 = st.number_input("Druga tačka (x₁):", value=50.0)

        tol = st.number_input("Tolerancija:", value=1e-6, format="%.1e")

    with col2:
        st.subheader("📊 Rezultat")

        if st.button("🚀 Riješi problem", type="primary", key="root"):
            try:
                f = lambda x: eval(func_str)

                if method == "Bisekcija":
                    result = bisection(f, a, b, tol)
                elif method == "Newton-Raphson":
                    df = lambda x: eval(df_str) if df_str else None
                    result = newton_raphson(f, x0, tol, df=df)
                else:
                    result = secant(f, x0, x1, tol)

                if result['converged']:
                    st.success(f"✅ **Rješenje pronađeno!**")
                    st.metric("x =", f"{result['root']:.6f}")
                    st.metric("Iteracije", result['iterations'])

                    # Provjera
                    st.markdown(f"**Provjera:** f({result['root']:.6f}) = {f(result['root']):.2e}")

                    # Interpretacija
                    st.markdown("---")
                    st.markdown(f"### 📋 Odgovor na pitanje")
                    st.info(f"**{problem_name}**\n\nOdgovor: **x = {result['root']:.4f}**")

                else:
                    st.error(result.get('error_message', 'Metoda nije konvergirala'))

                # Koraci
                with st.expander("📝 Detaljni koraci"):
                    for step in result.get('steps', [])[:10]:
                        st.write(step)

            except Exception as e:
                st.error(f"Greška: {e}")

# ============ NUMERIČKA INTEGRACIJA ============
elif problem_type == "Numerička integracija":
    st.header("∫ Numerička Integracija - Vlastiti Primjer")

    st.markdown("""
    ### Primjeri iz stvarnog života:
    - **Fizika**: Pređeni put iz brzine, rad iz sile
    - **Ekonomija**: Ukupna potrošnja iz marginalne potrošnje
    - **Biologija**: Ukupna doza lijeka
    """)

    input_type = st.radio("Način unosa:", ["Funkcija", "Tablica podataka"], horizontal=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Definiši Problem")

        problem_name = st.text_input("Naziv problema:",
                                     value="Pređeni put vozila")

        if input_type == "Funkcija":
            func_str = st.text_area("Funkcija f(x):",
                                    value="20*x - 2*x**2",
                                    help="Npr. brzina vozila: v(t) = 20t - 2t²")
            a = st.number_input("Donja granica (a):", value=0.0)
            b = st.number_input("Gornja granica (b):", value=5.0)
            n = st.slider("Broj podintervala:", 2, 100, 20)
        else:
            st.markdown("**Unesite podatke (npr. vrijeme i brzina)**")
            x_input = st.text_area("X vrijednosti (vrijeme):",
                                   value="0, 1, 2, 3, 4, 5")
            y_input = st.text_area("Y vrijednosti (brzina):",
                                   value="0, 18, 32, 42, 48, 50")

        method = st.selectbox("Metoda:", ["Simpson", "Trapezna", "Gauss"])

    with col2:
        st.subheader("📊 Rezultat")

        if st.button("🚀 Izračunaj integral", type="primary", key="int"):
            try:
                if input_type == "Funkcija":
                    f = lambda x: eval(func_str)

                    if method == "Simpson":
                        result = simpson(f, a, b, n)
                    elif method == "Trapezna":
                        result = trapezoidal(f, a, b, n)
                    else:
                        result = gauss_quadrature(f, a, b, 5)

                    integral = result['integral']

                else:
                    x_data = np.array([float(x.strip()) for x in x_input.split(',')])
                    y_data = np.array([float(y.strip()) for y in y_input.split(',')])
                    result = integrate_from_table(x_data, y_data)
                    integral = result['integral']

                st.success(f"✅ **Integral izračunat!**")
                st.metric("I =", f"{integral:.6f}")

                # Interpretacija
                st.markdown("---")
                st.markdown(f"### 📋 Odgovor")
                st.info(f"**{problem_name}**\n\nRezultat: **{integral:.4f}** (jedinica zavisi od konteksta)")

            except Exception as e:
                st.error(f"Greška: {e}")

# ============ NUMERIČKA DERIVACIJA ============
elif problem_type == "Numerička derivacija":
    st.header("∂ Numerička Derivacija - Vlastiti Primjer")

    st.markdown("""
    ### Primjeri iz stvarnog života:
    - **Fizika**: Brzina iz položaja, ubrzanje iz brzine
    - **Ekonomija**: Marginalni trošak, marginalni prihod
    - **Biologija**: Brzina rasta populacije
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Definiši Problem")

        problem_name = st.text_input("Naziv problema:",
                                     value="Brzina vozila u trenutku t=2s")

        input_type = st.radio("Unos:", ["Funkcija", "Tablica"], horizontal=True)

        if input_type == "Funkcija":
            func_str = st.text_area("Položaj s(t):",
                                    value="5*x**2 + 3*x",
                                    help="Npr. s(t) = 5t² + 3t → v(t) = 10t + 3")
            x_point = st.number_input("Tačka t:", value=2.0)
            h = st.number_input("Korak h:", value=0.001, format="%.6f")
        else:
            x_input = st.text_area("Vrijeme (t):", value="0, 1, 2, 3, 4, 5")
            y_input = st.text_area("Položaj s(t):", value="0, 8, 26, 54, 92, 140")

    with col2:
        st.subheader("📊 Rezultat")

        if st.button("🚀 Izračunaj derivaciju", type="primary", key="deriv"):
            try:
                if input_type == "Funkcija":
                    f = lambda x: eval(func_str)
                    result = auto_differentiate(f=f, x=x_point, h=h, domain=(x_point-1, x_point+1))

                    if result['derivatives']:
                        deriv = result['derivatives'][0]['derivative']
                        method_used = result['derivatives'][0]['method']

                        st.success(f"✅ **Derivacija izračunata!**")
                        st.metric("f'(x) =", f"{deriv:.6f}")
                        st.markdown(f"*Korištena metoda: {method_used}*")

                        st.info(f"**{problem_name}**\n\nBrzina u t={x_point}: **v = {deriv:.4f}**")

                else:
                    x_data = np.array([float(x.strip()) for x in x_input.split(',')])
                    y_data = np.array([float(y.strip()) for y in y_input.split(',')])
                    result = auto_differentiate(x_points=x_data, y_points=y_data)

                    st.success(f"✅ **Derivacije izračunate!**")

                    df = pd.DataFrame([{
                        't': d['x'],
                        'v(t) = ds/dt': f"{d['derivative']:.4f}",
                        'Metoda': d['method']
                    } for d in result['derivatives']])
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Greška: {e}")

# ============ REGRESIJA ============
elif problem_type == "Regresija/Aproksimacija podataka":
    st.header("📈 Regresija - Vlastiti Primjer")

    st.markdown("""
    ### Primjeri iz stvarnog života:
    - **Fizika**: Ovisnost temperature o vremenu
    - **Ekonomija**: Trend prodaje, rast investicije
    - **Biologija**: Rast populacije, raspadanje lijeka
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Unesi Podatke")

        problem_name = st.text_input("Naziv problema:", value="Rast prodaje tokom godina")

        x_input = st.text_area("X vrijednosti (nezavisna varijabla):",
                               value="1, 2, 3, 4, 5, 6")
        y_input = st.text_area("Y vrijednosti (zavisna varijabla):",
                               value="100, 150, 210, 280, 360, 450")

        model = st.selectbox("Model:", ["Linearna", "Eksponencijalna", "Polinomijalna"])

        if model == "Polinomijalna":
            degree = st.slider("Stepen polinoma:", 1, 5, 2)

    with col2:
        st.subheader("📊 Rezultat")

        if st.button("🚀 Fituj model", type="primary", key="reg"):
            try:
                x_data = np.array([float(x.strip()) for x in x_input.split(',')])
                y_data = np.array([float(y.strip()) for y in y_input.split(',')])

                if model == "Linearna":
                    result = linear_regression(x_data, y_data)
                elif model == "Eksponencijalna":
                    if np.any(y_data <= 0):
                        st.error("Za eksponencijalnu regresiju, sve Y vrijednosti moraju biti pozitivne!")
                        st.stop()
                    result = exponential_regression(x_data, y_data)
                else:
                    result = polynomial_regression(x_data, y_data, degree)

                st.success(f"✅ **Model fitovan!**")
                st.markdown(f"**Jednačina:** {result['equation']}")
                st.metric("R²", f"{result['r_squared']:.4f}")

                # Graf
                import plotly.graph_objects as go
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                                        name='Podaci', marker=dict(size=10)))

                x_fit = np.linspace(min(x_data), max(x_data)*1.2, 100)
                if model == "Linearna":
                    y_fit = result['a'] * x_fit + result['b']
                elif model == "Eksponencijalna":
                    y_fit = result['A'] * np.exp(result['B'] * x_fit)
                else:
                    y_fit = sum(c * x_fit**i for i, c in enumerate(result['coefficients']))

                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines',
                                        name='Fit', line=dict(color='red')))

                fig.update_layout(title=problem_name, xaxis_title='X', yaxis_title='Y')
                st.plotly_chart(fig, use_container_width=True)

                # Predikcija
                st.subheader("🔮 Predikcija")
                x_pred = st.number_input("Predvidi za X =", value=float(max(x_data)+1))
                if model == "Linearna":
                    y_pred = result['a'] * x_pred + result['b']
                elif model == "Eksponencijalna":
                    y_pred = result['A'] * np.exp(result['B'] * x_pred)
                else:
                    y_pred = sum(c * x_pred**i for i, c in enumerate(result['coefficients']))
                st.metric(f"Y({x_pred})", f"{y_pred:.4f}")

            except Exception as e:
                st.error(f"Greška: {e}")

# ============ SISTEM JEDNAČINA ============
elif problem_type == "Sistem linearnih jednačina":
    st.header("🔢 Sistem Jednačina - Vlastiti Primjer")

    st.markdown("""
    ### Primjeri iz stvarnog života:
    - **Inženjerstvo**: Analiza električnih kola
    - **Ekonomija**: Ravnoteža tržišta
    - **Fizika**: Distribucija temperatura
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Definiši Sistem")

        problem_name = st.text_input("Naziv problema:", value="Analiza električnog kola")

        n = st.slider("Dimenzija sistema:", 2, 5, 3)

        st.markdown("**Matrica A:**")
        A = []
        for i in range(n):
            cols = st.columns(n)
            row = []
            for j in range(n):
                default = 4.0 if i == j else (-1.0 if abs(i-j) == 1 else 0.0)
                val = cols[j].number_input(f"A[{i+1},{j+1}]", value=default,
                                          key=f"a_{i}_{j}", label_visibility="collapsed")
                row.append(val)
            A.append(row)
        A = np.array(A)

        st.markdown("**Vektor b:**")
        b = []
        cols = st.columns(n)
        for i in range(n):
            default = 10.0 if i == 0 else 0.0
            val = cols[i].number_input(f"b[{i+1}]", value=default, key=f"b_{i}",
                                      label_visibility="collapsed")
            b.append(val)
        b = np.array(b)

        method = st.selectbox("Metoda:", ["Gauss-Seidel", "Jacobi"])

    with col2:
        st.subheader("📊 Rezultat")

        if st.button("🚀 Riješi sistem", type="primary", key="sys"):
            try:
                if method == "Gauss-Seidel":
                    result = gauss_seidel(A, b)
                else:
                    result = jacobi(A, b)

                if result['converged']:
                    st.success(f"✅ **Rješenje pronađeno!** ({result['iterations']} iteracija)")

                    st.markdown("**Rješenje x:**")
                    for i, xi in enumerate(result['solution']):
                        st.metric(f"x{i+1}", f"{xi:.6f}")

                    # Verifikacija
                    Ax = A @ result['solution']
                    st.markdown("**Verifikacija (Ax = b):**")
                    verify_df = pd.DataFrame({
                        'Ax': [f"{v:.4f}" for v in Ax],
                        'b': [f"{v:.4f}" for v in b],
                        'Razlika': [f"{abs(Ax[i]-b[i]):.2e}" for i in range(n)]
                    })
                    st.dataframe(verify_df, use_container_width=True)

                else:
                    st.error(result.get('error_message', 'Metoda nije konvergirala'))

            except Exception as e:
                st.error(f"Greška: {e}")

# Footer
st.markdown("---")
st.info("💡 **Savjet:** Probajte modificirati parametre i vidjeti kako se rezultati mijenjaju!")
