"""
Primjeri iz Fizike i Inženjerstva
=================================

Ovaj modul sadrži funkcije i podatke za primjere primjene
numeričkih metoda u fizici i inženjerstvu.
"""

import numpy as np
from typing import Dict, List, Callable


def get_physics_examples() -> Dict:
    """
    Vraća rječnik sa primjerima iz fizike

    Returns:
        Dict sa primjerima
    """
    return {
        'falling_body': {
            'name': 'Pad tijela sa otporom zraka',
            'description': '''
            Tijelo mase m pada pod uticajem gravitacije, uz otpor zraka
            proporcionalan kvadratu brzine: F_drag = kv²

            Jednačina kretanja: m·dv/dt = mg - kv²
            Rješenje: v(t) = v_term · tanh(gt/v_term)
            gdje je v_term = √(mg/k)
            ''',
            'parameters': {
                'm': {'name': 'Masa', 'unit': 'kg', 'default': 70},
                'k': {'name': 'Koeficijent otpora', 'unit': 'kg/m', 'default': 0.25},
                'g': {'name': 'Gravitacija', 'unit': 'm/s²', 'default': 9.81}
            },
            'method': 'root_finding',
            'question': 'Vrijeme za dostizanje 90% terminalne brzine'
        },
        'projectile': {
            'name': 'Kretanje projektila',
            'description': '''
            Projektil ispaljen brzinom v₀ pod uglom θ.
            Zanemarujući otpor zraka:
            x(t) = v₀·cos(θ)·t
            y(t) = v₀·sin(θ)·t - ½gt²

            Domet: R = v₀²·sin(2θ)/g
            ''',
            'parameters': {
                'v0': {'name': 'Početna brzina', 'unit': 'm/s', 'default': 50},
                'theta': {'name': 'Ugao', 'unit': '°', 'default': 45}
            },
            'method': 'integration',
            'question': 'Domet projektila'
        },
        'circuit': {
            'name': 'Električna kola (Kirchhoff)',
            'description': '''
            Električna mreža sa tri čvora daje sistem jednačina
            prema Kirchhoffovim zakonima.

            Sistem: Ax = b
            gdje je A matrica vodljivosti, x vektor struja, b vektor izvora.
            ''',
            'A': np.array([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]]),
            'b': np.array([10, 0, 0]),
            'method': 'linear_system',
            'question': 'Struje u granama mreže'
        },
        'pendulum': {
            'name': 'Matematičko klatno',
            'description': '''
            Period matematičkog klatna za male oscilacije: T ≈ 2π√(L/g)

            Za veće amplitude, period zavisi od amplitude i računa se
            korištenjem eliptičkog integrala:
            T = 4√(L/g)·K(sin²(θ₀/2))

            Numerička integracija je potrebna za tačno računanje.
            ''',
            'parameters': {
                'L': {'name': 'Dužina klatna', 'unit': 'm', 'default': 1.0},
                'theta0': {'name': 'Početna amplituda', 'unit': '°', 'default': 30}
            },
            'method': 'integration',
            'question': 'Tačan period oscilacije'
        },
        'heat_conduction': {
            'name': 'Provođenje toplote',
            'description': '''
            1D jednačina provođenja toplote u šipci:
            ∂u/∂t = α·∂²u/∂x²

            Diskretizacija vodi na sistem linearnih jednačina
            koji se rješava iterativnim metodama.
            ''',
            'method': 'linear_system',
            'question': 'Temperaturna distribucija'
        }
    }


def terminal_velocity(m: float, k: float, g: float = 9.81) -> float:
    """
    Računanje terminalne brzine

    Args:
        m: Masa tijela (kg)
        k: Koeficijent otpora (kg/m)
        g: Gravitacija (m/s²)

    Returns:
        Terminalna brzina (m/s)
    """
    return np.sqrt(m * g / k)


def velocity_with_drag(t: float, m: float, k: float, g: float = 9.81) -> float:
    """
    Brzina tijela u slobodnom padu sa otporom zraka

    Args:
        t: Vrijeme (s)
        m: Masa (kg)
        k: Koeficijent otpora (kg/m)
        g: Gravitacija (m/s²)

    Returns:
        Brzina u trenutku t (m/s)
    """
    v_term = terminal_velocity(m, k, g)
    return v_term * np.tanh(g * t / v_term)


def projectile_range(v0: float, theta: float, g: float = 9.81) -> float:
    """
    Analitički domet projektila

    Args:
        v0: Početna brzina (m/s)
        theta: Ugao lansiranja (radijani)
        g: Gravitacija (m/s²)

    Returns:
        Domet (m)
    """
    return v0**2 * np.sin(2 * theta) / g


def projectile_trajectory(v0: float, theta: float, g: float = 9.81,
                          num_points: int = 100) -> Dict:
    """
    Generisanje trajektorije projektila

    Args:
        v0: Početna brzina (m/s)
        theta: Ugao (radijani)
        g: Gravitacija (m/s²)
        num_points: Broj tačaka

    Returns:
        Dict sa x, y koordinatama i vremenom
    """
    # Vrijeme leta
    T = 2 * v0 * np.sin(theta) / g

    t = np.linspace(0, T, num_points)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2

    return {
        't': t,
        'x': x,
        'y': y,
        'T_flight': T,
        'max_height': v0**2 * np.sin(theta)**2 / (2 * g),
        'range': projectile_range(v0, theta, g)
    }


def create_heat_conduction_system(n: int, alpha: float = 1.0,
                                   dx: float = 0.1, dt: float = 0.01) -> Dict:
    """
    Kreira sistem jednačina za provođenje toplote (implicitna metoda)

    Args:
        n: Broj unutrašnjih tačaka
        alpha: Koeficijent difuzije
        dx: Korak u prostoru
        dt: Korak u vremenu

    Returns:
        Dict sa matricom A i opisom
    """
    r = alpha * dt / dx**2

    # Tridiagonalna matrica
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 1 + 2*r
        if i > 0:
            A[i, i-1] = -r
        if i < n-1:
            A[i, i+1] = -r

    return {
        'A': A,
        'r': r,
        'description': f'Implicitna metoda za jednačinu toplote, r = {r:.4f}'
    }
