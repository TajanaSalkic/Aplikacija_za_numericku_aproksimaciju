"""
Primjeri iz Biologije i Medicine
================================

Ovaj modul sadrži funkcije i podatke za primjere primjene
numeričkih metoda u biologiji i medicini.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable


def get_biology_examples() -> Dict:
    """
    Vraća rječnik sa primjerima iz biologije

    Returns:
        Dict sa primjerima
    """
    return {
        'population_growth': {
            'name': 'Eksponencijalni rast populacije',
            'description': '''
            Eksponencijalni model rasta: P(t) = P₀·e^(rt)

            Koristi se za modeliranje rasta bakterija, populacija
            u uslovima neograničenih resursa.

            Polu-život udvostručavanja: t₂ = ln(2)/r
            ''',
            'sample_data': {
                't': [0, 1, 2, 3, 4, 5, 6],
                'P': [100, 122, 149, 182, 222, 271, 331]
            },
            'method': 'exponential_regression',
            'question': 'Stopa rasta i vrijeme udvostručavanja'
        },
        'logistic_growth': {
            'name': 'Logistički rast populacije',
            'description': '''
            Logistički model uzima u obzir ograničene resurse:
            dP/dt = rP(1 - P/K)

            Rješenje: P(t) = K / (1 + ((K-P₀)/P₀)·e^(-rt))

            K je nosivi kapacitet (carrying capacity)
            ''',
            'parameters': {
                'r': {'name': 'Stopa rasta', 'unit': '1/dan', 'default': 0.5},
                'K': {'name': 'Nosivi kapacitet', 'default': 1000},
                'P0': {'name': 'Početna populacija', 'default': 10}
            },
            'method': 'root_finding',
            'question': 'Vrijeme za dostizanje 90% kapaciteta'
        },
        'sir_model': {
            'name': 'SIR model epidemije',
            'description': '''
            SIR model dijeli populaciju na:
            - S (Susceptible) - podložni zarazi
            - I (Infected) - zaraženi
            - R (Recovered) - oporavljeni/imuni

            dS/dt = -βSI/N
            dI/dt = βSI/N - γI
            dR/dt = γI

            R₀ = β/γ (basic reproduction number)
            ''',
            'parameters': {
                'beta': {'name': 'Stopa zaraze', 'default': 0.3},
                'gamma': {'name': 'Stopa oporavka', 'default': 0.1},
                'N': {'name': 'Populacija', 'default': 1000},
                'I0': {'name': 'Početni zaraženi', 'default': 1}
            },
            'method': 'integration',
            'question': 'Dinamika epidemije'
        },
        'pharmacokinetics': {
            'name': 'Farmakokinetika',
            'description': '''
            Koncentracija lijeka u krvi nakon intravenozne injekcije:
            C(t) = C₀·e^(-kt)

            k - konstanta eliminacije
            Polu-život: t₁/₂ = ln(2)/k

            AUC (Area Under Curve) - mjera izloženosti lijeku
            ''',
            'sample_data': {
                't': [0, 1, 2, 4, 6, 8, 12],
                'C': [100, 82, 67, 45, 30, 20, 9]
            },
            'method': 'exponential_regression',
            'question': 'Konstanta eliminacije i polu-život'
        },
        'enzyme_kinetics': {
            'name': 'Enzimska kinetika (Michaelis-Menten)',
            'description': '''
            Michaelis-Menten jednačina:
            v = Vmax·[S] / (Km + [S])

            Lineweaver-Burk linearizacija:
            1/v = (Km/Vmax)·(1/[S]) + 1/Vmax
            ''',
            'sample_data': {
                'S': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                'v': [0.91, 1.67, 3.33, 5.00, 6.67, 8.33]
            },
            'method': 'linear_regression',
            'question': 'Vmax i Km parametri'
        }
    }


def exponential_growth(t: float, P0: float, r: float) -> float:
    """
    Eksponencijalni rast populacije

    Args:
        t: Vrijeme
        P0: Početna populacija
        r: Stopa rasta

    Returns:
        Populacija u trenutku t
    """
    return P0 * np.exp(r * t)


def logistic_growth(t: float, P0: float, r: float, K: float) -> float:
    """
    Logistički rast populacije

    Args:
        t: Vrijeme
        P0: Početna populacija
        r: Stopa rasta
        K: Nosivi kapacitet

    Returns:
        Populacija u trenutku t
    """
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))


def sir_model(t_max: float, dt: float, beta: float, gamma: float,
              N: int, I0: int) -> Dict:
    """
    Simulacija SIR modela korištenjem Euler metode

    Args:
        t_max: Maksimalno vrijeme simulacije
        dt: Vremenski korak
        beta: Stopa zaraze
        gamma: Stopa oporavka
        N: Ukupna populacija
        I0: Početni broj zaraženih

    Returns:
        Dict sa S, I, R nizovima i statistikama
    """
    steps = int(t_max / dt)

    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    t = np.linspace(0, t_max, steps)

    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for i in range(steps - 1):
        dS = -beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]

        S[i+1] = max(0, S[i] + dS * dt)
        I[i+1] = max(0, I[i] + dI * dt)
        R[i+1] = max(0, R[i] + dR * dt)

    # Statistike
    max_infected = np.max(I)
    peak_time = t[np.argmax(I)]
    total_infected = N - S[-1]

    return {
        't': t,
        'S': S,
        'I': I,
        'R': R,
        'R0': beta / gamma,
        'max_infected': max_infected,
        'peak_time': peak_time,
        'total_infected': total_infected,
        'final_susceptible': S[-1]
    }


def drug_concentration(t: float, C0: float, k: float) -> float:
    """
    Koncentracija lijeka u krvi

    Args:
        t: Vrijeme nakon doze
        C0: Početna koncentracija
        k: Konstanta eliminacije

    Returns:
        Koncentracija u trenutku t
    """
    return C0 * np.exp(-k * t)


def calculate_auc(t: np.ndarray, C: np.ndarray) -> float:
    """
    Računa AUC (Area Under Curve) korištenjem trapezne metode

    Args:
        t: Vremena mjerenja
        C: Koncentracije

    Returns:
        AUC vrijednost
    """
    return np.trapz(C, t)


def michaelis_menten(S: float, Vmax: float, Km: float) -> float:
    """
    Michaelis-Menten kinetika

    Args:
        S: Koncentracija supstrata
        Vmax: Maksimalna brzina reakcije
        Km: Michaelis konstanta

    Returns:
        Brzina reakcije
    """
    return Vmax * S / (Km + S)


def lineweaver_burk_transform(S: np.ndarray, v: np.ndarray) -> Dict:
    """
    Lineweaver-Burk transformacija za linearizaciju
    Michaelis-Menten kinetike

    Args:
        S: Koncentracije supstrata
        v: Brzine reakcije

    Returns:
        Dict sa transformisanim podacima
    """
    # 1/v = (Km/Vmax)(1/S) + 1/Vmax
    inv_S = 1 / np.array(S)
    inv_v = 1 / np.array(v)

    return {
        'inv_S': inv_S,
        'inv_v': inv_v,
        'description': '1/v vs 1/[S] linearizacija'
    }


def population_doubling_time(r: float) -> float:
    """
    Vrijeme udvostručavanja populacije

    Args:
        r: Stopa rasta

    Returns:
        Vrijeme udvostručavanja
    """
    return np.log(2) / r


def drug_half_life(k: float) -> float:
    """
    Polu-život lijeka

    Args:
        k: Konstanta eliminacije

    Returns:
        Polu-život
    """
    return np.log(2) / k


def time_to_steady_state(k: float, n_half_lives: int = 5) -> float:
    """
    Vrijeme za dostizanje stacionarnog stanja
    (obično 5 polu-života)

    Args:
        k: Konstanta eliminacije
        n_half_lives: Broj polu-života

    Returns:
        Vrijeme do stacionarnog stanja
    """
    return n_half_lives * drug_half_life(k)
