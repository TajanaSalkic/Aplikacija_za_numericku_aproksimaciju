"""
Stranica: Sistemi Linearnih Jednačina
=====================================

Implementira:
- Jacobijeva metoda - rađeno na nastavi
- Gauss-Seidelova metoda - rađeno na nastavi
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.linear_systems import jacobi, gauss_seidel, compare_methods, check_diagonal_dominance, create_example_systems
from utils.plotting import plot_linear_system_convergence
from utils.latex_helpers import create_matrix_latex, create_vector_latex

st.set_page_config(page_title="Sistemi Jednačina", page_icon="🔢", layout="wide")

st.title("🔢 Sistemi Linearnih Jednačina")
st.markdown("*Iterativne metode za rješavanje Ax = b*")

# Sidebar
st.sidebar.header("⚙️ Postavke")
method = st.sidebar.selectbox(
    "Odaberite metodu:",
    ["Jacobijeva metoda", "Gauss-Seidelova metoda", "Poređenje metoda"]
)

# Izbor sistema
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Sistem")

examples = create_example_systems()
example_names = ["Vlastiti sistem"] + [ex['name'] for ex in examples.values()]
selected_example = st.sidebar.selectbox("Odaberite primjer:", example_names)

if selected_example == "Vlastiti sistem":
    n = st.sidebar.slider("Dimenzija sistema (n):", 2, 5, 3)

    st.sidebar.markdown("**Matrica A:**")
    A = []
    for i in range(n):
        row = []
        cols = st.sidebar.columns(n)
        for j in range(n):
            val = cols[j].number_input(f"a[{i+1},{j+1}]", value=float(4 if i==j else -1 if abs(i-j)==1 else 0),
                                       key=f"a_{i}_{j}", label_visibility="collapsed")
            row.append(val)
        A.append(row)
    A = np.array(A)

    st.sidebar.markdown("**Vektor b:**")
    b = []
    cols = st.sidebar.columns(n)
    for i in range(n):
        val = cols[i].number_input(f"b[{i+1}]", value=float(i+1), key=f"b_{i}", label_visibility="collapsed")
        b.append(val)
    b = np.array(b)

else:
    # Pronađi odabrani primjer
    for key, ex in examples.items():
        if ex['name'] == selected_example:
            A = ex['A']
            b = ex['b']
            st.sidebar.info(ex['description'])
            break
    n = len(b)

# Parametri
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Parametri")
tol = st.sidebar.number_input("Tolerancija:", value=1e-6, format="%.1e")
max_iter = st.sidebar.slider("Max iteracija:", 10, 200, 50)

# Početna aproksimacija
use_custom_x0 = st.sidebar.checkbox("Vlastita početna aproksimacija")
if use_custom_x0:
    x0 = []
    cols = st.sidebar.columns(n)
    for i in range(n):
        val = cols[i].number_input(f"x0[{i+1}]", value=0.0, key=f"x0_{i}", label_visibility="collapsed")
        x0.append(val)
    x0 = np.array(x0)
else:
    x0 = np.zeros(n)

run_button = st.sidebar.button("🚀 Izračunaj", type="primary", use_container_width=True)

# Glavni sadržaj
st.markdown("---")

# Prikaz sistema
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sistem Jednačina")
    st.markdown(create_matrix_latex(A, "A"))
    st.markdown(create_vector_latex(b, "b"))

with col2:
    # Provjera dijagonalne dominantnosti
    diag_check = check_diagonal_dominance(A)
    st.subheader("✓ Dijagonalna Dominantnost")

    if diag_check['is_diagonally_dominant']:
        st.success("Matrica JE dijagonalno dominantna ✓")
        st.info("Konvergencija je garantovana!")
    else:
        st.warning("Matrica NIJE dijagonalno dominantna")
        st.info("Metoda može konvergirati, ali nije garantovano.")

    with st.expander("Detalji provjere"):
        for det in diag_check['details']:
            if det['is_dominant']:
                st.markdown(f"✓ Red {det['row']}: |{det['diagonal']:.4f}| > {det['off_diagonal_sum']:.4f}")
            else:
                st.markdown(f"✗ Red {det['row']}: |{det['diagonal']:.4f}| ≤ {det['off_diagonal_sum']:.4f}")

# Teorija
if method == "Jacobijeva metoda":
    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Jacobijeva Metoda

        **Razlaganje matrice:** $A = D + L + U$
        - $D$: dijagonalna matrica
        - $L$: strogo donja trougaona
        - $U$: strogo gornja trougaona

        **Iterativna formula:**
        $$x_i^{(k+1)} = \\frac{1}{a_{ii}} \\left( b_i - \\sum_{j \\neq i} a_{ij} x_j^{(k)} \\right)$$

        **Karakteristika:** Koristi SAMO vrijednosti iz prethodne iteracije $x^{(k)}$.

        **Konvergencija:**
        - Garantovana za strogo dijagonalno dominantne matrice
        - Garantovana za simetrične pozitivno definitne matrice
        """)

elif method == "Gauss-Seidelova metoda":
    with st.expander("📖 Teorija", expanded=False):
        st.markdown("""
        ### Gauss-Seidelova Metoda

        **Iterativna formula:**
        $$x_i^{(k+1)} = \\frac{1}{a_{ii}} \\left( b_i - \\sum_{j < i} a_{ij} x_j^{(k+1)} - \\sum_{j > i} a_{ij} x_j^{(k)} \\right)$$

        **Razlika od Jacobi:**
        - Jacobi: koristi SAMO stare vrijednosti $x^{(k)}$
        - Gauss-Seidel: koristi **nove** $x_j^{(k+1)}$ za $j < i$ čim su izračunate

        **Prednosti:**
        - Obično konvergira **brže** od Jacobijeve metode
        - Zahtijeva manje memorije (može raditi "in-place")
        """)

# Rezultati
if run_button:
    if method == "Jacobijeva metoda":
        st.header("Jacobijeva Metoda")
        result = jacobi(A, b, x0, tol, max_iter)

    elif method == "Gauss-Seidelova metoda":
        st.header("Gauss-Seidelova Metoda")
        result = gauss_seidel(A, b, x0, tol, max_iter)

    elif method == "Poređenje metoda":
        st.header("Poređenje Metoda")
        comparison = compare_methods(A, b, x0, tol, max_iter)

        col1, col2, col3 = st.columns(3)

        with col1:
            j_res = comparison['jacobi']
            st.metric("Jacobi - Iteracije", j_res['iterations'],
                     "Konvergiralo ✓" if j_res['converged'] else "Nije konvergiralo ✗")

        with col2:
            gs_res = comparison['gauss_seidel']
            st.metric("Gauss-Seidel - Iteracije", gs_res['iterations'],
                     "Konvergiralo ✓" if gs_res['converged'] else "Nije konvergiralo ✗")

        with col3:
            if comparison['exact_solution'] is not None:
                st.markdown("**Tačno rješenje:**")
                st.markdown(create_vector_latex(comparison['exact_solution'], "x"))

        # Graf konvergencije
        st.subheader("📈 Poređenje Konvergencije")
        fig = plot_linear_system_convergence(j_res['steps'], gs_res['steps'])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### 🔍 Analiza

        Gauss-Seidelova metoda tipično konvergira **brže** od Jacobijeve jer koristi
        već izračunate nove vrijednosti. U prosjeku zahtijeva ~2x manje iteracija.
        """)

        result = None  # Da ne prikazuje korake ispod

    if result and 'solution' in result:
        # Rezultat
        if result['converged']:
            st.success(f"✅ Konvergiralo nakon **{result['iterations']}** iteracija")
        else:
            st.warning(f"⚠️ Nije konvergiralo nakon {result['iterations']} iteracija")

        if result['solution'] is not None:
            st.subheader("📊 Rješenje")
            st.markdown(create_vector_latex(result['solution'], "x"))

            # Verifikacija Ax = b
            Ax = A @ result['solution']
            st.markdown("**Verifikacija (Ax):**")
            st.markdown(create_vector_latex(Ax, "Ax"))

            residual = np.linalg.norm(Ax - b)
            st.info(f"Rezidual ||Ax - b|| = {residual:.2e}")

        # Koraci
        st.subheader("📝 Step-by-Step Prikaz")

        for step in result['steps']:
            if 'calculations' in step:
                with st.expander(step['title'], expanded=(step['step'] <= 3)):
                    if 'x_old' in step:
                        st.markdown(f"**Prethodna aproksimacija:** $x^{{({step['step']-2})}} = [{', '.join([f'{xi:.6f}' for xi in step['x_old']])}]^T$")

                    for calc in step['calculations']:
                        st.markdown(f"""
                        **Komponenta {calc['component']}:**
                        ```
                        {calc['calculation']}
                        ```
                        """)

                    if 'x_new' in step:
                        st.markdown(f"**Nova aproksimacija:** $x^{{({step['step']-1})}} = [{', '.join([f'{xi:.6f}' for xi in step['x_new']])}]^T$")

                    if 'error' in step:
                        st.markdown(f"**Greška:** $||x^{{(k+1)}} - x^{{(k)}}||_\\infty = {step['error']:.2e}$")

        # Tabela konvergencije
        st.subheader("📋 Tabela Konvergencije")
        import pandas as pd

        table_data = []
        for step in result['steps']:
            if 'x_new' in step:
                row = {'Iteracija': step['step'] - 1}
                for i, xi in enumerate(step['x_new']):
                    row[f'x_{i+1}'] = f"{xi:.6f}"
                row['Greška'] = f"{step.get('error', 0):.2e}"
                table_data.append(row)

        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
