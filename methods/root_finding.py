"""
Metode za traženje nula funkcije (Root Finding)
================================================

Ovaj modul implementira tri numeričke metode za pronalaženje korijena jednačine f(x) = 0:

1. Metoda Bisekcije (Dihotomija) - Rađeno na nastavi
2. Newton-Raphson metoda - Bonus metoda
3. Metoda Sekante - Rađeno na nastavi

Svaka metoda vraća detaljne korake za step-by-step prikaz u aplikaciji.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


def bisection(f: Callable[[float], float],
              a: float,
              b: float,
              tol: float = 1e-6,
              max_iter: int = 100) -> Dict:
    """
    Metoda Bisekcije (Dihotomija)
    =============================

    TEORIJA:
    --------
    Metoda bisekcije je najjednostavnija i najstabilnija metoda za pronalaženje
    korijena funkcije. Bazira se na teoremi o međuvrijednosti (Intermediate Value Theorem):

    Ako je f kontinuirana na [a,b] i f(a)·f(b) < 0, tada postoji barem jedan
    korijen c ∈ (a,b) takav da je f(c) = 0.

    ALGORITAM:
    ----------
    1. Provjeri da li f(a)·f(b) < 0 (različiti predznaci)
    2. Izračunaj srednju tačku: c = (a + b) / 2
    3. Ako je |c_n - c_{n-1}| < tolerancija, c je rješenje
    4. Ako je f(a)·f(c) < 0, korijen je u [a,c], postavi b = c
    5. Inače, korijen je u [c,b], postavi a = c
    6. Ponovi od koraka 2 (ili zaustavi nakon max_iter iteracija)

    KONVERGENCIJA:
    --------------
    - Linearna konvergencija
    - Greška se prepolavlja u svakoj iteraciji: e_{n+1} ≤ e_n / 2
    - Potreban broj iteracija: n ≥ log₂((b-a)/tol)

    Args:
        f: Funkcija čiji korijen tražimo
        a: Lijeva granica intervala
        b: Desna granica intervala
        tol: Tolerancija (preciznost rješenja)
        max_iter: Maksimalan broj iteracija

    Returns:
        Dict sa ključevima:
        - 'root': Pronađeni korijen
        - 'iterations': Broj iteracija
        - 'converged': Da li je metoda konvergirala
        - 'steps': Lista koraka za vizualizaciju
        - 'error_message': Poruka greške (ako postoji)
    """
    steps = []

    # Provjera početnih uslova
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        return {
            'root': None,
            'iterations': 0,
            'converged': False,
            'steps': [],
            'error_message': f'Funkcija ima isti predznak na krajevima intervala: f({a})={fa:.6f}, f({b})={fb:.6f}. Metoda bisekcije zahtijeva f(a)·f(b) < 0.'
        }

    # Poseban slučaj - korijen je na kraju intervala
    if abs(fa) < tol:
        return {
            'root': a,
            'iterations': 0,
            'converged': True,
            'steps': [{'iteration': 0, 'a': a, 'b': b, 'c': a, 'fc': fa, 'error': 0}],
            'error_message': None
        }
    if abs(fb) < tol:
        return {
            'root': b,
            'iterations': 0,
            'converged': True,
            'steps': [{'iteration': 0, 'a': a, 'b': b, 'c': b, 'fc': fb, 'error': 0}],
            'error_message': None
        }

    # Početni korak
    steps.append({
        'iteration': 0,
        'a': a,
        'b': b,
        'fa': fa,
        'fb': fb,
        'c': None,
        'fc': None,
        'error': b - a,
        'description': f'Početni interval: [{a}, {b}], f({a}) = {fa:.6f}, f({b}) = {fb:.6f}'
    })

    # Pratimo prethodnu vrijednost c za provjeru konvergencije
    c_prev = None

    for i in range(1, max_iter + 1):
        # Srednja tačka
        c = (a + b) / 2
        fc = f(c)

        # Greška je apsolutna razlika između dvije uzastopne iteracije
        if c_prev is not None:
            error = abs(c - c_prev)
        else:
            error = abs(b - a) / 2  # Za prvu iteraciju

        step_info = {
            'iteration': i,
            'a': a,
            'b': b,
            'c': c,
            'c_prev': c_prev,
            'fa': f(a),
            'fb': f(b),
            'fc': fc,
            'error': error,
            'description': ''
        }

        # Provjera konvergencije - apsolutna razlika dvije uzastopne iteracije
        if c_prev is not None and error < tol:
            step_info['description'] = f'KONVERGENCIJA! |c_{i} - c_{i-1}| = {error:.2e} < {tol:.2e}'
            steps.append(step_info)
            return {
                'root': c,
                'iterations': i,
                'converged': True,
                'steps': steps,
                'error_message': None
            }

        # Određivanje novog intervala
        if fa * fc < 0:
            step_info['description'] = f'f(a)·f(c) = {fa*fc:.6f} < 0, korijen u [{a:.6f}, {c:.6f}]'
            b = c
            fb = fc
        else:
            step_info['description'] = f'f(c)·f(b) = {fc*fb:.6f} < 0, korijen u [{c:.6f}, {b:.6f}]'
            a = c
            fa = fc

        steps.append(step_info)

        # Ažuriraj prethodnu vrijednost c za sljedeću iteraciju
        c_prev = c

    # Nije konvergiralo
    c = (a + b) / 2
    return {
        'root': c,
        'iterations': max_iter,
        'converged': False,
        'steps': steps,
        'error_message': f'Metoda nije konvergirala nakon {max_iter} iteracija. Trenutna aproksimacija: {c:.10f}'
    }


def newton_raphson(f: Callable[[float], float],
                   x0: float,
                   tol: float = 1e-6,
                   max_iter: int = 100,
                   df: Optional[Callable[[float], float]] = None,
                   h: float = 1e-8) -> Dict:
    """
    Newton-Raphson Metoda
    =====================

    TEORIJA:
    --------
    Newton-Raphson metoda je jedna od najbržih metoda za pronalaženje korijena.
    Koristi tangentnu liniju funkcije za aproksimaciju korijena.

    Ideja: Ako je x_n trenutna aproksimacija, nova aproksimacija x_{n+1} se
    dobija kao presječna tačka tangente u (x_n, f(x_n)) sa x-osom.

    IZVOD FORMULE:
    --------------
    Jednačina tangente u tački (x_n, f(x_n)):
        y - f(x_n) = f'(x_n)(x - x_n)

    Presječna tačka sa x-osom (y = 0):
        -f(x_n) = f'(x_n)(x_{n+1} - x_n)
        x_{n+1} = x_n - f(x_n)/f'(x_n)

    FORMULA:
    --------
        x_{n+1} = x_n - f(x_n) / f'(x_n)

    NUMERIČKA DERIVACIJA:
    ---------------------
    Ako derivacija nije poznata, koristi se centralna diferencija:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    KONVERGENCIJA:
    --------------
    - Kvadratna konvergencija (kada konvergira)
    - e_{n+1} ≈ C · e_n²
    - Broj tačnih decimala se približno udvostručuje u svakoj iteraciji

    UPOZORENJA:
    -----------
    - Može divergirati ako je početna tačka daleko od korijena
    - Problemi ako je f'(x_n) ≈ 0 (horizontalna tangenta)
    - Može oscilirati ili "pobjeći" u beskonačnost

    Args:
        f: Funkcija čiji korijen tražimo
        x0: Početna aproksimacija
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija
        df: Derivacija funkcije (opcionalno)
        h: Korak za numeričku derivaciju

    Returns:
        Dict sa detaljima rješenja i koracima
    """
    steps = []
    x = x0

    # Funkcija za računanje derivacije
    if df is None:
        def numerical_derivative(x):
            return (f(x + h) - f(x - h)) / (2 * h)
        derivative = numerical_derivative
        derivative_type = "numerička (centralna diferencija)"
    else:
        derivative = df
        derivative_type = "analitička"

    for i in range(max_iter + 1):
        fx = f(x)
        dfx = derivative(x)

        step_info = {
            'iteration': i,
            'x': x,
            'fx': fx,
            'dfx': dfx,
            'derivative_type': derivative_type,
            'description': ''
        }

        # Provjera konvergencije
        if abs(fx) < tol:
            step_info['description'] = f'KONVERGENCIJA! |f(x)| = {abs(fx):.2e} < {tol:.2e}'
            steps.append(step_info)
            return {
                'root': x,
                'iterations': i,
                'converged': True,
                'steps': steps,
                'error_message': None,
                'derivative_type': derivative_type
            }

        # Provjera za dijeljenje sa nulom
        if abs(dfx) < 1e-15:
            step_info['description'] = f'GREŠKA: f\'(x) ≈ 0, tangenta je horizontalna!'
            steps.append(step_info)
            return {
                'root': None,
                'iterations': i,
                'converged': False,
                'steps': steps,
                'error_message': f'Derivacija je približno 0 u tački x = {x:.6f}. Metoda ne može nastaviti.',
                'derivative_type': derivative_type
            }

        # Newton-Raphson korak
        x_new = x - fx / dfx

        step_info['x_new'] = x_new
        step_info['description'] = f'x_{{n+1}} = {x:.6f} - {fx:.6f}/{dfx:.6f} = {x_new:.6f}'
        steps.append(step_info)

        # Provjera za divergenciju
        if abs(x_new) > 1e10:
            return {
                'root': None,
                'iterations': i,
                'converged': False,
                'steps': steps,
                'error_message': f'Metoda divergira! x = {x_new:.2e}',
                'derivative_type': derivative_type
            }

        x = x_new

    return {
        'root': x,
        'iterations': max_iter,
        'converged': False,
        'steps': steps,
        'error_message': f'Metoda nije konvergirala nakon {max_iter} iteracija.',
        'derivative_type': derivative_type
    }


def secant(f: Callable[[float], float],
           x0: float,
           x1: float,
           tol: float = 1e-6,
           max_iter: int = 100) -> Dict:
    """
    Metoda Sekante
    ==============

    TEORIJA:
    --------
    Metoda sekante je slična Newton-Raphson metodi, ali ne zahtijeva
    poznavanje derivacije. Umjesto tangente, koristi sekantu kroz
    dvije prethodne tačke.

    IZVOD FORMULE:
    --------------
    Aproksimacija derivacije pomoću konačne razlike:
        f'(x_n) ≈ [f(x_n) - f(x_{n-1})] / [x_n - x_{n-1}]

    Zamjenom u Newton-Raphson formuli:
        x_{n+1} = x_n - f(x_n) · [x_n - x_{n-1}] / [f(x_n) - f(x_{n-1})]

    FORMULA:
    --------
        x_{n+1} = x_n - f(x_n) · (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    ili ekvivalentno:

        x_{n+1} = [x_{n-1}·f(x_n) - x_n·f(x_{n-1})] / [f(x_n) - f(x_{n-1})]

    KONVERGENCIJA:
    --------------
    - Superlinearna konvergencija
    - Red konvergencije: φ = (1 + √5)/2 ≈ 1.618 (zlatni rez)
    - Sporija od Newton-Raphson, ali ne zahtijeva derivaciju

    PREDNOSTI:
    ----------
    - Ne zahtijeva računanje derivacije
    - Brža od bisekcije
    - Jednostavna implementacija

    MANE:
    -----
    - Može divergirati
    - Zahtijeva dvije početne tačke
    - Manje stabilna od bisekcije

    Args:
        f: Funkcija čiji korijen tražimo
        x0: Prva početna aproksimacija
        x1: Druga početna aproksimacija
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija

    Returns:
        Dict sa detaljima rješenja i koracima
    """
    steps = []

    x_prev = x0
    x_curr = x1
    f_prev = f(x_prev)
    f_curr = f(x_curr)

    # Početni korak
    steps.append({
        'iteration': 0,
        'x_prev': x_prev,
        'x_curr': x_curr,
        'f_prev': f_prev,
        'f_curr': f_curr,
        'x_new': None,
        'description': f'Početne tačke: x₀ = {x_prev}, x₁ = {x_curr}'
    })

    for i in range(1, max_iter + 1):
        # Provjera za dijeljenje sa nulom
        denominator = f_curr - f_prev
        if abs(denominator) < 1e-15:
            return {
                'root': None,
                'iterations': i,
                'converged': False,
                'steps': steps,
                'error_message': f'f(x_n) ≈ f(x_{{n-1}}), sekanta je horizontalna!'
            }

        # Formula sekante
        x_new = x_curr - f_curr * (x_curr - x_prev) / denominator
        f_new = f(x_new)

        step_info = {
            'iteration': i,
            'x_prev': x_prev,
            'x_curr': x_curr,
            'f_prev': f_prev,
            'f_curr': f_curr,
            'x_new': x_new,
            'f_new': f_new,
            'error': abs(x_new - x_curr),
            'description': f'x_{{n+1}} = {x_curr:.6f} - {f_curr:.6f}·({x_curr:.6f}-{x_prev:.6f})/({f_curr:.6f}-{f_prev:.6f}) = {x_new:.6f}'
        }

        # Provjera konvergencije
        # if abs(f_new) < tol or abs(x_new - x_curr) < tol:
        if abs(x_new - x_curr) < tol:
            step_info['description'] += f'\nKONVERGENCIJA!'
            steps.append(step_info)
            return {
                'root': x_new,
                'iterations': i,
                'converged': True,
                'steps': steps,
                'error_message': None
            }

        # Provjera divergencije
        if abs(x_new) > 1e10:
            steps.append(step_info)
            return {
                'root': None,
                'iterations': i,
                'converged': False,
                'steps': steps,
                'error_message': f'Metoda divergira!'
            }

        steps.append(step_info)

        # Ažuriranje za sljedeću iteraciju
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new

    return {
        'root': x_curr,
        'iterations': max_iter,
        'converged': False,
        'steps': steps,
        'error_message': f'Metoda nije konvergirala nakon {max_iter} iteracija.'
    }


def compare_methods(f: Callable[[float], float],
                    a: float, b: float,
                    x0: float = None,
                    tol: float = 1e-6,
                    max_iter: int = 100,
                    df: Callable[[float], float] = None) -> Dict:
    """
    Poređenje svih metoda za istu funkciju

    Args:
        f: Funkcija
        a, b: Interval za bisekciju
        x0: Početna tačka za Newton (ako None, koristi sredinu intervala)
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija
        df: Derivacija (opcionalno)

    Returns:
        Dict sa rezultatima svih metoda
    """
    if x0 is None:
        x0 = (a + b) / 2

    results = {
        'bisection': bisection(f, a, b, tol, max_iter),
        'newton': newton_raphson(f, x0, tol, max_iter, df),
        'secant': secant(f, a, b, tol, max_iter)
    }

    # Dodaj sumarnu tabelu
    summary = []
    for name, result in results.items():
        summary.append({
            'method': name,
            'root': result['root'],
            'iterations': result['iterations'],
            'converged': result['converged']
        })

    results['summary'] = summary
    return results
