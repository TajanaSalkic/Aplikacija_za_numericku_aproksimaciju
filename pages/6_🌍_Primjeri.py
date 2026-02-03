"""
Stranica: Primjeri iz Stvarnog Života
=====================================

Implementira praktične primjere primjene numeričkih metoda u:
- Fizika/Inženjerstvo
- Biologija/Medicina
- Ekonomija
"""

import streamlit as st
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.root_finding import bisection, newton_raphson
from methods.integration import simpson, gauss_quadrature
from methods.regression import exponential_regression, linear_regression
from methods.linear_systems import gauss_seidel
from utils.plotting import plot_regression

st.set_page_config(page_title="Primjeri", page_icon="🌍", layout="wide")

st.title("🌍 Primjeri iz Stvarnog Života")
st.markdown("*Praktična primjena numeričkih metoda*")

# Sidebar
st.sidebar.header("📚 Kategorije")
category = st.sidebar.selectbox(
    "Odaberite kategoriju:",
    ["Fizika/Inženjerstvo", "Biologija/Medicina", "Ekonomija"]
)

st.markdown("---")

if category == "Fizika/Inženjerstvo":
    st.header("⚡ Fizika i Inženjerstvo")

    example = st.selectbox(
        "Odaberite primjer:",
        ["Pad tijela sa otporom zraka", "Kretanje projektila", "Električna kola (Kirchhoff)"]
    )

    if example == "Pad tijela sa otporom zraka":
        st.subheader("🪂 Pad Tijela sa Otporom Zraka")

        st.markdown("""
        ### Problem

        Tijelo mase $m$ pada pod uticajem gravitacije, uz otpor zraka proporcionalan kvadratu brzine.
        Diferencijalna jednačina kretanja:

        $$m\\frac{dv}{dt} = mg - kv^2$$

        **Terminalna brzina** (kada $dv/dt = 0$):
        $$v_{term} = \\sqrt{\\frac{mg}{k}}$$

        ### Pitanje
        Za datu masu i koeficijent otpora, koliko vremena treba da tijelo dostigne 90% terminalne brzine?
        """)

        col1, col2 = st.columns(2)

        with col1:
            m = st.number_input("Masa (kg):", value=70.0, min_value=1.0)
            k = st.number_input("Koeficijent otpora (kg/m):", value=0.25, min_value=0.01)
            g = 9.81

        with col2:
            v_term = np.sqrt(m * g / k)
            st.metric("Terminalna brzina", f"{v_term:.2f} m/s")
            target_v = 0.9 * v_term
            st.metric("Ciljna brzina (90%)", f"{target_v:.2f} m/s")

        if st.button("🚀 Izračunaj vrijeme", key="fall"):
            # Rješenje: v(t) = v_term * tanh(gt/v_term)
            # Tražimo t kada je v(t) = 0.9 * v_term
            # tanh(gt/v_term) = 0.9
            # gt/v_term = atanh(0.9)
            # t = v_term * atanh(0.9) / g

            def velocity_equation(t):
                return v_term * np.tanh(g * t / v_term) - target_v

            result = bisection(velocity_equation, 0.1, 50, tol=1e-6)

            if result['converged']:
                t_result = result['root']
                st.success(f"✅ Vrijeme do 90% terminalne brzine: **t = {t_result:.3f} s**")

                # Prikaz grafa brzine
                import plotly.graph_objects as go

                t_range = np.linspace(0, 2*t_result, 100)
                v_range = v_term * np.tanh(g * t_range / v_term)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t_range, y=v_range, name='v(t)', line=dict(color='blue', width=2)))
                fig.add_hline(y=v_term, line_dash="dash", line_color="red",
                            annotation_text=f"v_term = {v_term:.1f} m/s")
                fig.add_hline(y=target_v, line_dash="dot", line_color="green",
                            annotation_text=f"90% v_term = {target_v:.1f} m/s")
                fig.add_vline(x=t_result, line_dash="dot", line_color="orange")
                fig.add_trace(go.Scatter(x=[t_result], y=[target_v], mode='markers',
                                       marker=dict(size=12, color='orange'),
                                       name=f't = {t_result:.3f} s'))

                fig.update_layout(title='Brzina Pada vs Vrijeme',
                                xaxis_title='Vrijeme (s)', yaxis_title='Brzina (m/s)')
                st.plotly_chart(fig, use_container_width=True)

                # Koraci bisekcije
                with st.expander("📝 Koraci Metode Bisekcije"):
                    for step in result['steps'][:5]:
                        if step['c'] is not None:
                            st.markdown(f"Iteracija {step['iteration']}: interval [{step['a']:.4f}, {step['b']:.4f}], c = {step['c']:.4f}")

    elif example == "Kretanje projektila":
        st.subheader("🎯 Domet Projektila")

        st.markdown("""
        ### Problem

        Projektil je ispaljen brzinom $v_0$ pod uglom $\\theta$. Zanemarujući otpor zraka,
        jednačine kretanja su:
        - $x(t) = v_0 \\cos(\\theta) \\cdot t$
        - $y(t) = v_0 \\sin(\\theta) \\cdot t - \\frac{1}{2}gt^2$

        **Pitanje:** Izračunaj domet projektila korištenjem numeričke integracije brzine.
        """)

        col1, col2 = st.columns(2)

        with col1:
            v0 = st.number_input("Početna brzina (m/s):", value=50.0, min_value=1.0)
            theta_deg = st.slider("Ugao (°):", 5, 85, 45)
            theta = np.radians(theta_deg)

        with col2:
            # Analitičko rješenje za provjeru
            g = 9.81
            R_analytical = v0**2 * np.sin(2*theta) / g
            T_flight = 2 * v0 * np.sin(theta) / g
            st.metric("Vrijeme leta (analitički)", f"{T_flight:.3f} s")
            st.metric("Domet (analitički)", f"{R_analytical:.2f} m")

        if st.button("🚀 Numerička Integracija", key="projectile"):
            # Numerička integracija vx(t) da dobijemo x(T)
            # vx = v0 * cos(theta) je konstantna
            # Integral je jednostavan, ali pokazujemo metodu

            def vx(t):
                return v0 * np.cos(theta)

            result = simpson(vx, 0, T_flight, n=20)

            st.success(f"✅ Domet (Simpson): **R = {result['integral']:.2f} m**")

            # Gauss
            result_gauss = gauss_quadrature(vx, 0, T_flight, n=5)
            st.info(f"Domet (Gauss 5 tačaka): R = {result_gauss['integral']:.2f} m")

            error = abs(result['integral'] - R_analytical)
            st.info(f"Greška u odnosu na analitičko: {error:.4f} m ({error/R_analytical*100:.4f}%)")

            # Graf trajektorije
            import plotly.graph_objects as go

            t = np.linspace(0, T_flight, 100)
            x = v0 * np.cos(theta) * t
            y = v0 * np.sin(theta) * t - 0.5 * g * t**2

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Trajektorija',
                                    line=dict(color='blue', width=2)))
            fig.update_layout(title=f'Trajektorija Projektila (θ = {theta_deg}°)',
                            xaxis_title='x (m)', yaxis_title='y (m)')
            st.plotly_chart(fig, use_container_width=True)

    elif example == "Električna kola (Kirchhoff)":
        st.subheader("⚡ Analiza Električnog Kola")

        st.markdown("""
        ### Problem

        Električna mreža sa tri čvora daje sistem jednačina prema Kirchhoffovim zakonima:

        $$\\begin{cases}
        3I_1 - I_2 - I_3 = 10 \\\\
        -I_1 + 3I_2 - I_3 = 0 \\\\
        -I_1 - I_2 + 3I_3 = 0
        \\end{cases}$$

        Koristimo Gauss-Seidelovu metodu za pronalaženje struja.
        """)

        A = np.array([
            [3, -1, -1],
            [-1, 3, -1],
            [-1, -1, 3]
        ])
        b = np.array([10, 0, 0])

        if st.button("🔌 Riješi Sistem", key="circuit"):
            result = gauss_seidel(A, b, tol=1e-8)

            if result['converged']:
                I = result['solution']
                st.success(f"✅ Rješenje pronađeno nakon {result['iterations']} iteracija")

                col1, col2, col3 = st.columns(3)
                col1.metric("I₁", f"{I[0]:.4f} A")
                col2.metric("I₂", f"{I[1]:.4f} A")
                col3.metric("I₃", f"{I[2]:.4f} A")

                # Verifikacija
                st.markdown("**Verifikacija (A·I):**")
                AI = A @ I
                st.markdown(f"- $3I_1 - I_2 - I_3 = {AI[0]:.4f}$ (trebalo bi biti 10)")
                st.markdown(f"- $-I_1 + 3I_2 - I_3 = {AI[1]:.4f}$ (trebalo bi biti 0)")
                st.markdown(f"- $-I_1 - I_2 + 3I_3 = {AI[2]:.4f}$ (trebalo bi biti 0)")

elif category == "Biologija/Medicina":
    st.header("🧬 Biologija i Medicina")

    example = st.selectbox(
        "Odaberite primjer:",
        ["Rast populacije", "Širenje epidemije (SIR model)", "Farmakokinetika"]
    )

    if example == "Rast populacije":
        st.subheader("🐰 Eksponencijalni Rast Populacije")

        st.markdown("""
        ### Model

        Eksponencijalni model rasta:
        $$P(t) = P_0 \\cdot e^{rt}$$

        gdje je:
        - $P_0$ - početna populacija
        - $r$ - stopa rasta
        - $t$ - vrijeme

        ### Podaci
        Populacija bakterija mjerena svaki sat:
        """)

        # Podaci
        t_data = np.array([0, 1, 2, 3, 4, 5, 6])
        P_data = np.array([100, 122, 149, 182, 222, 271, 331])

        import pandas as pd
        st.dataframe(pd.DataFrame({'Vrijeme (h)': t_data, 'Populacija': P_data}), use_container_width=True)

        if st.button("📈 Fituj Eksponencijalni Model", key="population"):
            result = exponential_regression(t_data, P_data)

            st.success(f"**Model:** P(t) = {result['A']:.2f} · e^({result['B']:.4f}t)")
            st.info(f"R² = {result['r_squared']:.6f}")

            # Stopa rasta
            r = result['B']
            doubling_time = np.log(2) / r
            st.metric("Stopa rasta", f"{r*100:.2f}% po satu")
            st.metric("Vrijeme udvostručavanja", f"{doubling_time:.2f} sati")

            # Predikcija
            t_pred = 10
            P_pred = result['A'] * np.exp(result['B'] * t_pred)
            st.warning(f"**Predikcija:** P({t_pred}) = {P_pred:.0f}")

            # Graf
            fig = plot_regression(t_data, P_data, result, "Rast Populacije Bakterija")
            st.plotly_chart(fig, use_container_width=True)

    elif example == "Širenje epidemije (SIR model)":
        st.subheader("🦠 SIR Model Epidemije")

        st.markdown("""
        ### Model

        SIR model dijeli populaciju na tri grupe:
        - **S** - Susceptible (podložni zarazi)
        - **I** - Infected (zaraženi)
        - **R** - Recovered (oporavljeni/imuni)

        Diferencijalne jednačine:
        $$\\frac{dS}{dt} = -\\beta \\frac{SI}{N}$$
        $$\\frac{dI}{dt} = \\beta \\frac{SI}{N} - \\gamma I$$
        $$\\frac{dR}{dt} = \\gamma I$$

        gdje je:
        - $\\beta$ - stopa zaraze
        - $\\gamma$ - stopa oporavka
        - $N$ - ukupna populacija

        **Numerička integracija** se koristi za simulaciju širenja bolesti.
        """)

        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("Populacija:", value=1000, min_value=100)
            I0 = st.number_input("Početni zaraženi:", value=1, min_value=1)
            beta = st.slider("β (stopa zaraze):", 0.1, 1.0, 0.3)
            gamma = st.slider("γ (stopa oporavka):", 0.01, 0.5, 0.1)

        with col2:
            R0 = beta / gamma
            st.metric("R₀ (Basic Reproduction Number)", f"{R0:.2f}")
            if R0 > 1:
                st.warning("R₀ > 1: Epidemija će se širiti!")
            else:
                st.success("R₀ < 1: Epidemija će se ugasiti.")

        if st.button("🦠 Simuliraj Epidemiju", key="sir"):
            # Euler metoda za integraciju
            dt = 0.1
            T = 100
            steps = int(T / dt)

            S = np.zeros(steps)
            I = np.zeros(steps)
            R_arr = np.zeros(steps)
            t = np.linspace(0, T, steps)

            S[0] = N - I0
            I[0] = I0
            R_arr[0] = 0

            for i in range(steps - 1):
                dS = -beta * S[i] * I[i] / N
                dI = beta * S[i] * I[i] / N - gamma * I[i]
                dR = gamma * I[i]

                S[i+1] = S[i] + dS * dt
                I[i+1] = I[i] + dI * dt
                R_arr[i+1] = R_arr[i] + dR * dt

            # Peak zaraženih
            max_I = np.max(I)
            t_peak = t[np.argmax(I)]
            st.metric("Maksimalni broj zaraženih", f"{max_I:.0f}")
            st.metric("Vrijeme vrha epidemije", f"{t_peak:.1f} dana")

            # Graf
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=S, name='Susceptible', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t, y=I, name='Infected', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=t, y=R_arr, name='Recovered', line=dict(color='green')))

            fig.update_layout(title='SIR Model - Simulacija Epidemije',
                            xaxis_title='Vrijeme (dani)', yaxis_title='Populacija')
            st.plotly_chart(fig, use_container_width=True)

    elif example == "Farmakokinetika":
        st.subheader("💊 Koncentracija Lijeka u Krvi")

        st.markdown("""
        ### Model

        Nakon intravenozne injekcije, koncentracija lijeka opada eksponencijalno:
        $$C(t) = C_0 \\cdot e^{-kt}$$

        **Polu-život** lijeka ($t_{1/2}$):
        $$t_{1/2} = \\frac{\\ln 2}{k}$$

        Tražimo konstantu eliminacije $k$ iz podataka.
        """)

        # Podaci
        t_data = np.array([0, 1, 2, 4, 6, 8, 12])
        C_data = np.array([100, 82, 67, 45, 30, 20, 9])

        import pandas as pd
        st.dataframe(pd.DataFrame({'Vrijeme (h)': t_data, 'Koncentracija (mg/L)': C_data}), use_container_width=True)

        if st.button("💊 Analiziraj Podatke", key="pharma"):
            result = exponential_regression(t_data, C_data)

            C0 = result['A']
            k = -result['B']  # Negativan jer je raspad
            half_life = np.log(2) / k

            st.success(f"**Model:** C(t) = {C0:.2f} · e^(-{k:.4f}t)")

            col1, col2, col3 = st.columns(3)
            col1.metric("C₀ (početna koncentracija)", f"{C0:.2f} mg/L")
            col2.metric("k (konstanta eliminacije)", f"{k:.4f} h⁻¹")
            col3.metric("Polu-život", f"{half_life:.2f} h")

            # Pronađi vrijeme kada C < 5 mg/L (npr. ispod terapeutskog nivoa)
            def conc_eq(t):
                return C0 * np.exp(-k * t) - 5

            t_below = bisection(conc_eq, 0, 24)
            if t_below['converged']:
                st.info(f"Koncentracija pada ispod 5 mg/L nakon **{t_below['root']:.2f} sati**")

            # Graf
            fig = plot_regression(t_data, C_data, result, "Farmakokinetika - Koncentracija vs Vrijeme")
            st.plotly_chart(fig, use_container_width=True)

elif category == "Ekonomija":
    st.header("💰 Ekonomija i Finansije")

    example = st.selectbox(
        "Odaberite primjer:",
        ["Složena kamata", "Break-even analiza", "Amortizacija"]
    )

    if example == "Složena kamata":
        st.subheader("📈 Rast Investicije (Složena Kamata)")

        st.markdown("""
        ### Model

        Vrijednost investicije sa složenom kamatom:
        $$A = P \\cdot (1 + r/n)^{nt}$$

        ili kontinuirano:
        $$A = P \\cdot e^{rt}$$

        **Pitanje:** Koliko vremena treba da se investicija udvostruči?
        """)

        col1, col2 = st.columns(2)

        with col1:
            P = st.number_input("Početna investicija (€):", value=10000.0)
            r_percent = st.slider("Godišnja kamatna stopa (%):", 1.0, 15.0, 5.0)
            r = r_percent / 100

        with col2:
            # Analitičko rješenje: t = ln(2)/r (kontinuirano)
            t_double_cont = np.log(2) / r
            st.metric("Vrijeme udvostručavanja (kontinuirano)", f"{t_double_cont:.2f} godina")

            # Pravilo 72
            rule_72 = 72 / r_percent
            st.metric("Pravilo 72 aproksimacija", f"{rule_72:.1f} godina")

        if st.button("📊 Simuliraj Rast", key="compound"):
            # Numeričko rješenje za tačno vrijeme
            target = 2 * P

            def investment_eq(t):
                return P * np.exp(r * t) - target

            result = newton_raphson(investment_eq, 10)

            if result['converged']:
                t_double = result['root']
                st.success(f"✅ Investicija se udvostručava za **{t_double:.3f} godina**")

            # Graf rasta
            import plotly.graph_objects as go

            t = np.linspace(0, 2*t_double, 100)
            A = P * np.exp(r * t)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=A, name='Vrijednost', line=dict(color='green', width=2)))
            fig.add_hline(y=2*P, line_dash="dash", line_color="red",
                        annotation_text=f"Cilj: {2*P:.0f} €")
            fig.add_vline(x=t_double, line_dash="dot", line_color="orange")

            fig.update_layout(title=f'Rast Investicije ({r_percent}% godišnje)',
                            xaxis_title='Vrijeme (godine)', yaxis_title='Vrijednost (€)')
            st.plotly_chart(fig, use_container_width=True)

    elif example == "Break-even analiza":
        st.subheader("⚖️ Break-even Analiza")

        st.markdown("""
        ### Problem

        Kompanija ima:
        - Fiksne troškove: $F$
        - Varijabilne troškove po jedinici: $v$
        - Prodajna cijena po jedinici: $p$

        **Break-even tačka** je broj jedinica $x$ gdje je profit = 0:
        $$px = F + vx$$
        $$x = \\frac{F}{p - v}$$
        """)

        col1, col2 = st.columns(2)

        with col1:
            F = st.number_input("Fiksni troškovi (€):", value=50000.0)
            v = st.number_input("Varijabilni trošak po jedinici (€):", value=30.0)
            p = st.number_input("Prodajna cijena po jedinici (€):", value=50.0)

        if p <= v:
            st.error("Prodajna cijena mora biti veća od varijabilnog troška!")
        else:
            with col2:
                x_breakeven = F / (p - v)
                st.metric("Break-even količina", f"{x_breakeven:.0f} jedinica")

                revenue_breakeven = p * x_breakeven
                st.metric("Break-even prihod", f"{revenue_breakeven:.0f} €")

            if st.button("📊 Prikaži Analizu", key="breakeven"):
                # Numeričko rješenje (demo)
                def profit(x):
                    return p * x - F - v * x

                result = bisection(profit, 1, 100000)

                import plotly.graph_objects as go

                x = np.linspace(0, 2*x_breakeven, 100)
                revenue = p * x
                cost = F + v * x
                profit_arr = revenue - cost

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=revenue, name='Prihod', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=x, y=cost, name='Troškovi', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=x, y=profit_arr, name='Profit', line=dict(color='blue', dash='dash')))
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=x_breakeven, line_dash="dot", line_color="orange",
                            annotation_text=f"Break-even: {x_breakeven:.0f}")

                fig.update_layout(title='Break-even Analiza',
                                xaxis_title='Količina', yaxis_title='€')
                st.plotly_chart(fig, use_container_width=True)

    elif example == "Amortizacija":
        st.subheader("🏠 Otplata Kredita")

        st.markdown("""
        ### Formula za mjesečnu ratu

        $$M = P \\cdot \\frac{r(1+r)^n}{(1+r)^n - 1}$$

        gdje je:
        - $M$ - mjesečna rata
        - $P$ - iznos kredita
        - $r$ - mjesečna kamatna stopa
        - $n$ - broj mjeseci
        """)

        col1, col2 = st.columns(2)

        with col1:
            P = st.number_input("Iznos kredita (€):", value=200000.0)
            annual_rate = st.slider("Godišnja kamatna stopa (%):", 1.0, 10.0, 4.0)
            years = st.slider("Rok otplate (godine):", 5, 30, 20)

        r = annual_rate / 100 / 12  # Mjesečna stopa
        n = years * 12  # Broj mjeseci

        with col2:
            M = P * r * (1+r)**n / ((1+r)**n - 1)
            total_paid = M * n
            total_interest = total_paid - P

            st.metric("Mjesečna rata", f"{M:.2f} €")
            st.metric("Ukupno plaćeno", f"{total_paid:.2f} €")
            st.metric("Ukupna kamata", f"{total_interest:.2f} €")

        if st.button("📊 Plan Otplate", key="amortization"):
            # Kreiraj plan otplate
            balance = P
            months = []
            balances = []
            interests = []
            principals = []

            for month in range(1, n+1):
                interest_payment = balance * r
                principal_payment = M - interest_payment
                balance -= principal_payment

                if month <= 24 or month > n - 12:  # Prikaži samo početak i kraj
                    months.append(month)
                    balances.append(max(0, balance))
                    interests.append(interest_payment)
                    principals.append(principal_payment)

            import plotly.graph_objects as go

            # Graf stanja duga
            all_months = list(range(1, n+1))
            all_balances = []
            bal = P
            for _ in range(n):
                all_balances.append(bal)
                bal -= (M - bal * r)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=all_months, y=all_balances, name='Preostali dug',
                                    fill='tozeroy', line=dict(color='blue')))

            fig.update_layout(title='Plan Otplate Kredita',
                            xaxis_title='Mjesec', yaxis_title='Preostali dug (€)')
            st.plotly_chart(fig, use_container_width=True)
