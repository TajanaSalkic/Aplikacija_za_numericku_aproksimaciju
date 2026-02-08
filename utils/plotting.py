"""
Funkcije za Vizualizaciju
=========================

Ovaj modul sadrži funkcije za kreiranje interaktivnih grafova
korištenjem Plotly biblioteke.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict


def plot_regression(x_data: np.ndarray, y_data: np.ndarray,
                    result: Dict, title: str = "Regresija") -> go.Figure:
    """
    Graf regresije sa podacima i fitovanom linijom/krivuljom

    Podržava sve metode aproksimacije:
    - Linearna: y = ax + b
    - Stepena: y = A*x^B
    - Eksponencijalna: y = A*e^(Bx)
    - Logaritamska: y = a + b*ln(x)
    - Racionalna: y = x/(a + bx)
    - Polinomijalna: y = a0 + a1*x + a2*x^2 + ...

    Args:
        x_data: X podaci
        y_data: Y podaci
        result: Rezultat regresije (sadrži 'method' ključ)
        title: Naslov grafa

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Originalni podaci
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data,
        mode='markers',
        name='Podaci',
        marker=dict(color='blue', size=10)
    ))

    # Fitovana linija/krivulja
    # Za logaritamske i stepene funkcije, izbjegavaj x <= 0
    x_min = max(min(x_data), 0.001) if any(k in result.get('method', '').lower()
                                            for k in ['logaritam', 'stepena', 'power']) else min(x_data)
    x_fit = np.linspace(x_min, max(x_data), 200)

    method = result.get('method', '').lower()

    try:
        if 'coefficients' in result:
            # Polinomijalna
            coeffs = result['coefficients']
            y_fit = sum(c * x_fit**i for i, c in enumerate(coeffs))

        elif 'A' in result and 'B' in result:
            A = result['A']
            B = result['B']
            if 'stepena' in method or 'power' in method:
                # Stepena: y = A*x^B
                y_fit = A * (x_fit ** B)
            else:
                # Eksponencijalna: y = A*e^(Bx)
                y_fit = A * np.exp(B * x_fit)

        elif 'a' in result and 'b' in result:
            a = result['a']
            b = result['b']
            equation = result.get('equation', '')

            if 'ln(x)' in equation or 'logaritam' in method:
                # Logaritamska: y = a + b*ln(x)
                y_fit = a + b * np.log(x_fit)
            elif 'x/(' in equation or 'racional' in method:
                # Racionalna: y = x/(a + bx)
                y_fit = x_fit / (a + b * x_fit)
            elif 'e^(' in equation or 'eksponenci' in method:
                # Eksponencijalna: y = a*e^(bx)
                y_fit = a * np.exp(b * x_fit)
            elif 'x^' in equation or 'stepena' in method or 'power' in method:
                # Stepena: y = a*x^b
                y_fit = a * (x_fit ** b)
            else:
                # Linearna: y = a + bx (a je odsječak, b je nagib)
                # Kao u regression.py: a = intercept, b = slope
                y_fit = a + b * x_fit

        else:
            # Fallback: koristi predviđene vrijednosti ako postoje
            if 'y_predicted' in result and len(result['y_predicted']) > 0:
                y_fit = result['y_predicted']
                x_fit = x_data
            else:
                y_fit = y_data
                x_fit = x_data

    except Exception as e:
        # Ako dođe do greške, koristi predviđene vrijednosti
        y_fit = result.get('y_predicted', y_data)
        x_fit = x_data if len(y_fit) == len(x_data) else np.linspace(min(x_data), max(x_data), len(y_fit))

    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        name=f'Fit: {result.get("equation", "")}',
        line=dict(color='red', width=2)
    ))

    # R² anotacija
    r2 = result.get('r_squared', 0)
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'R² = {r2:.4f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='y',
        showlegend=True
    )

    return fig
