"""
Funkcije za Vizualizaciju
=========================

Ovaj modul sadrži funkcije za kreiranje interaktivnih grafova
korištenjem Plotly biblioteke.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Callable, Dict, List, Optional, Tuple


def plot_root_finding_bisection(f: Callable, a: float, b: float,
                                 steps: List[Dict], root: float) -> go.Figure:
    """
    Vizualizacija metode bisekcije

    Args:
        f: Funkcija
        a, b: Početni interval
        steps: Lista koraka
        root: Pronađeni korijen

    Returns:
        Plotly Figure
    """
    # Generisanje tačaka za funkciju
    x_range = np.linspace(a - 0.5, b + 0.5, 500)
    y_range = [f(x) for x in x_range]

    fig = go.Figure()

    # Funkcija
    fig.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=2)
    ))

    # X-osa
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Intervali kroz iteracije (samo prvih nekoliko)
    colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)',
              'rgba(255,255,0,0.3)', 'rgba(255,0,255,0.3)']

    for i, step in enumerate(steps[:5]):
        if 'a' in step and 'b' in step and step['a'] is not None:
            fig.add_vrect(
                x0=step['a'], x1=step['b'],
                fillcolor=colors[i % len(colors)],
                layer="below",
                line_width=0,
                annotation_text=f"Iter {step.get('iteration', i)}",
                annotation_position="top left"
            )

    # Korijen
    if root is not None:
        fig.add_trace(go.Scatter(
            x=[root], y=[0],
            mode='markers',
            name=f'Korijen: x = {root:.6f}',
            marker=dict(color='red', size=12, symbol='star')
        ))

    fig.update_layout(
        title='Metoda Bisekcije - Vizualizacija',
        xaxis_title='x',
        yaxis_title='f(x)',
        showlegend=True,
        hovermode='x unified'
    )

    return fig


def plot_root_finding_newton(f: Callable, steps: List[Dict],
                              root: float, x_range: Tuple[float, float] = None) -> go.Figure:
    """
    Vizualizacija Newton-Raphson metode sa tangentama

    Args:
        f: Funkcija
        steps: Lista koraka
        root: Pronađeni korijen
        x_range: Raspon x-ose

    Returns:
        Plotly Figure
    """
    # Određivanje raspona
    if x_range is None:
        x_vals = [s['x'] for s in steps if 'x' in s]
        if x_vals:
            margin = max(abs(max(x_vals) - min(x_vals)) * 0.5, 1)
            x_range = (min(x_vals) - margin, max(x_vals) + margin)
        else:
            x_range = (-5, 5)

    x_plot = np.linspace(x_range[0], x_range[1], 500)
    y_plot = [f(x) for x in x_plot]

    fig = go.Figure()

    # Funkcija
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=2)
    ))

    # X-osa
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Tangente za prvih nekoliko iteracija
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, step in enumerate(steps[:5]):
        if 'x' in step and 'fx' in step and 'dfx' in step:
            x_n = step['x']
            fx_n = step['fx']
            dfx_n = step['dfx']

            if dfx_n != 0 and 'x_new' in step:
                x_new = step['x_new']
                # Tangenta od x_n do x_new
                x_tang = np.array([x_n - 0.5, x_new + 0.5])
                y_tang = fx_n + dfx_n * (x_tang - x_n)

                fig.add_trace(go.Scatter(
                    x=x_tang, y=y_tang,
                    mode='lines',
                    name=f'Tangenta {i+1}',
                    line=dict(color=colors[i % len(colors)], dash='dot', width=1)
                ))

                # Tačka na funkciji
                fig.add_trace(go.Scatter(
                    x=[x_n], y=[fx_n],
                    mode='markers',
                    name=f'x_{i} = {x_n:.4f}',
                    marker=dict(color=colors[i % len(colors)], size=8),
                    showlegend=False
                ))

    # Korijen
    if root is not None:
        fig.add_trace(go.Scatter(
            x=[root], y=[0],
            mode='markers',
            name=f'Korijen: x = {root:.6f}',
            marker=dict(color='red', size=12, symbol='star')
        ))

    fig.update_layout(
        title='Newton-Raphson Metoda - Vizualizacija Tangenti',
        xaxis_title='x',
        yaxis_title='f(x)',
        showlegend=True
    )

    return fig


def plot_integration_trapezoid(f: Callable, a: float, b: float,
                                n: int, integral: float) -> go.Figure:
    """
    Vizualizacija trapezne metode

    Args:
        f: Funkcija
        a, b: Granice integracije
        n: Broj podintervala
        integral: Vrijednost integrala

    Returns:
        Plotly Figure
    """
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [f(x) for x in x_points]

    # Glatka funkcija
    x_smooth = np.linspace(a, b, 500)
    y_smooth = [f(x) for x in x_smooth]

    fig = go.Figure()

    # Funkcija
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=2)
    ))

    # Trapezi
    for i in range(n):
        x_trap = [x_points[i], x_points[i], x_points[i+1], x_points[i+1]]
        y_trap = [0, y_points[i], y_points[i+1], 0]

        fig.add_trace(go.Scatter(
            x=x_trap, y=y_trap,
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.3)',
            line=dict(color='rgba(0, 100, 255, 0.8)', width=1),
            name=f'Trapez {i+1}',
            showlegend=(i == 0)
        ))

    # Tačke podjele
    fig.add_trace(go.Scatter(
        x=x_points, y=y_points,
        mode='markers',
        name='Tačke podjele',
        marker=dict(color='red', size=8)
    ))

    fig.update_layout(
        title=f'Trapezna Metoda (n={n}), I ≈ {integral:.6f}',
        xaxis_title='x',
        yaxis_title='f(x)',
        showlegend=True
    )

    return fig


def plot_integration_simpson(f: Callable, a: float, b: float,
                              n: int, integral: float) -> go.Figure:
    """
    Vizualizacija Simpsonove metode sa parabolama

    Args:
        f: Funkcija
        a, b: Granice integracije
        n: Broj podintervala (paran)
        integral: Vrijednost integrala

    Returns:
        Plotly Figure
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [f(x) for x in x_points]

    # Glatka funkcija
    x_smooth = np.linspace(a, b, 500)
    y_smooth = [f(x) for x in x_smooth]

    fig = go.Figure()

    # Funkcija
    fig.add_trace(go.Scatter(
        x=x_smooth, y=y_smooth,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=2)
    ))

    # Parabole za svaki par intervala
    colors = ['rgba(255, 0, 0, 0.3)', 'rgba(0, 255, 0, 0.3)',
              'rgba(255, 165, 0, 0.3)', 'rgba(128, 0, 128, 0.3)']

    for i in range(0, n, 2):
        # Tri tačke za parabolu
        x0, x1, x2 = x_points[i], x_points[i+1], x_points[i+2]
        y0, y1, y2 = y_points[i], y_points[i+1], y_points[i+2]

        # Lagrangeova interpolacija
        x_para = np.linspace(x0, x2, 50)
        y_para = []
        for x in x_para:
            L0 = (x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2))
            L1 = (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2))
            L2 = (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))
            y_para.append(y0 * L0 + y1 * L1 + y2 * L2)

        # Ispuna ispod parabole
        x_fill = list(x_para) + [x2, x0]
        y_fill = y_para + [0, 0]

        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill='toself',
            fillcolor=colors[(i//2) % len(colors)],
            line=dict(color='rgba(0,0,0,0)', width=0),
            name=f'Parabola {i//2 + 1}',
            showlegend=(i == 0)
        ))

        # Parabola linija
        fig.add_trace(go.Scatter(
            x=x_para, y=y_para,
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            showlegend=False
        ))

    # Tačke
    fig.add_trace(go.Scatter(
        x=x_points, y=y_points,
        mode='markers',
        name='Tačke podjele',
        marker=dict(color='red', size=8)
    ))

    fig.update_layout(
        title=f'Simpsonova Metoda (n={n}), I ≈ {integral:.6f}',
        xaxis_title='x',
        yaxis_title='f(x)',
        showlegend=True
    )

    return fig


def plot_differentiation_comparison(results: Dict) -> go.Figure:
    """
    Graf poređenja grešaka numeričke derivacije za različite h

    Args:
        results: Rezultati iz compare_errors funkcije

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    h_values = results['h_values']

    # Forward difference
    forward_errors = [r['error'] for r in results['forward']]
    fig.add_trace(go.Scatter(
        x=h_values, y=forward_errors,
        mode='lines+markers',
        name='Forward Difference O(h)',
        line=dict(color='blue')
    ))

    # Backward difference
    backward_errors = [r['error'] for r in results['backward']]
    fig.add_trace(go.Scatter(
        x=h_values, y=backward_errors,
        mode='lines+markers',
        name='Backward Difference O(h)',
        line=dict(color='green')
    ))

    # Central difference
    central_errors = [r['error'] for r in results['central']]
    fig.add_trace(go.Scatter(
        x=h_values, y=central_errors,
        mode='lines+markers',
        name='Central Difference O(h²)',
        line=dict(color='red')
    ))

    # Optimalne tačke
    for method, color in [('forward', 'blue'), ('central', 'red')]:
        opt = results[f'{method}_optimal']
        fig.add_trace(go.Scatter(
            x=[opt['h']], y=[opt['error']],
            mode='markers',
            name=f'{method} optimal h={opt["h"]:.0e}',
            marker=dict(color=color, size=15, symbol='star')
        ))

    fig.update_layout(
        title='Poređenje Grešaka Numeričke Derivacije',
        xaxis_title='h (korak)',
        yaxis_title='Apsolutna greška',
        xaxis_type='log',
        yaxis_type='log',
        showlegend=True
    )

    return fig


def plot_regression(x_data: np.ndarray, y_data: np.ndarray,
                    result: Dict, title: str = "Regresija") -> go.Figure:
    """
    Graf regresije sa podacima i fitovanom linijom/krivuljom

    Args:
        x_data: X podaci
        y_data: Y podaci
        result: Rezultat regresije
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
    x_fit = np.linspace(min(x_data), max(x_data), 200)

    if 'a' in result and 'b' in result and 'A' not in result:
        # Linearna regresija
        y_fit = result['a'] * x_fit + result['b']
    elif 'A' in result and 'B' in result:
        # Eksponencijalna
        y_fit = result['A'] * np.exp(result['B'] * x_fit)
    elif 'coefficients' in result:
        # Polinomijalna
        coeffs = result['coefficients']
        y_fit = sum(c * x_fit**i for i, c in enumerate(coeffs))
    else:
        y_fit = result.get('y_predicted', [])
        x_fit = x_data

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


def plot_convergence(steps: List[Dict], method_name: str) -> go.Figure:
    """
    Graf konvergencije iterativne metode

    Args:
        steps: Lista koraka sa greškama
        method_name: Ime metode

    Returns:
        Plotly Figure
    """
    iterations = []
    errors = []

    for step in steps:
        if 'iteration' in step and 'error' in step:
            iterations.append(step['iteration'])
            errors.append(step['error'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iterations, y=errors,
        mode='lines+markers',
        name='Greška',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=f'Konvergencija - {method_name}',
        xaxis_title='Iteracija',
        yaxis_title='Greška',
        yaxis_type='log',
        showlegend=True
    )

    return fig


def plot_linear_system_convergence(jacobi_steps: List[Dict],
                                    gs_steps: List[Dict]) -> go.Figure:
    """
    Poređenje konvergencije Jacobi i Gauss-Seidel metode

    Args:
        jacobi_steps: Koraci Jacobi metode
        gs_steps: Koraci Gauss-Seidel metode

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Jacobi
    j_iters = [s['step'] for s in jacobi_steps if 'error' in s]
    j_errors = [s['error'] for s in jacobi_steps if 'error' in s]

    if j_iters and j_errors:
        fig.add_trace(go.Scatter(
            x=j_iters, y=j_errors,
            mode='lines+markers',
            name='Jacobi',
            line=dict(color='blue', width=2)
        ))

    # Gauss-Seidel
    gs_iters = [s['step'] for s in gs_steps if 'error' in s]
    gs_errors = [s['error'] for s in gs_steps if 'error' in s]

    if gs_iters and gs_errors:
        fig.add_trace(go.Scatter(
            x=gs_iters, y=gs_errors,
            mode='lines+markers',
            name='Gauss-Seidel',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        title='Poređenje Konvergencije: Jacobi vs Gauss-Seidel',
        xaxis_title='Iteracija',
        yaxis_title='Greška (||x^(k+1) - x^(k)||)',
        yaxis_type='log',
        showlegend=True
    )

    return fig
