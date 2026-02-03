"""
Metode Numeričke Derivacije
===========================

Ovaj modul implementira metode za numeričko računanje derivacija:

1. Forward Difference (Unaprijedna diferencija)
2. Backward Difference (Unazadna diferencija)
3. Central Difference (Centralna diferencija)
4. Poređenje grešaka za različite h

Sve metode su BONUS (nisu rađene na nastavi) sa detaljnim objašnjenjima.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Union


def forward_diff(f: Callable[[float], float],
                 x: float,
                 h: float = 0.001,
                 df_exact: Optional[Callable[[float], float]] = None) -> Dict:
    """
    Forward Difference (Unaprijedna Diferencija)
    ============================================

    TEORIJA:
    --------
    Forward difference aproksimira derivaciju koristeći vrijednost funkcije
    u tački x i tački x+h (unaprijed).

    IZVOD IZ TAYLOROVOG REDA:
    -------------------------
    Taylorov razvoj f(x+h) oko tačke x:
        f(x+h) = f(x) + h·f'(x) + h²/2·f''(x) + h³/6·f'''(x) + ...

    Rješavamo za f'(x):
        f'(x) = [f(x+h) - f(x)]/h - h/2·f''(x) - h²/6·f'''(x) - ...
        f'(x) ≈ [f(x+h) - f(x)]/h + O(h)

    FORMULA:
    --------
        f'(x) ≈ [f(x+h) - f(x)] / h

    GREŠKA:
    -------
    Greška rezanja (truncation error): O(h)
        E_t ≈ -h/2 · f''(x)

    Greška zaokruživanja (round-off error): O(ε/h)
        E_r ≈ ε·f(x) / h

    gdje je ε mašinska preciznost (~10⁻¹⁶ za double precision)

    UKUPNA GREŠKA:
    --------------
    E = E_t + E_r ≈ h/2·|f''(x)| + ε·|f(x)|/h

    Optimalni h minimizira ukupnu grešku:
        h_opt ≈ √(2ε·|f(x)|/|f''(x)|) ≈ √ε ≈ 10⁻⁸

    PREDNOSTI:
    ----------
    - Jednostavna za implementaciju
    - Zahtijeva samo jednu dodatnu evaluaciju funkcije

    MANE:
    -----
    - Greška prvog reda O(h)
    - Asimetrična (koristi samo tačke sa jedne strane)

    Args:
        f: Funkcija
        x: Tačka u kojoj računamo derivaciju
        h: Korak
        df_exact: Tačna derivacija (opcionalno, za računanje greške)

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    fx = f(x)
    fxh = f(x + h)
    derivative = (fxh - fx) / h

    steps = [
        {
            'step': 1,
            'title': 'Evaluacija funkcije',
            'f_x': fx,
            'f_x_plus_h': fxh,
            'description': f'f({x}) = {fx:.10f}, f({x}+{h}) = f({x + h}) = {fxh:.10f}'
        },
        {
            'step': 2,
            'title': 'Primjena formule',
            'formula': "f'(x) ≈ [f(x+h) - f(x)] / h",
            'calculation': f"f'({x}) ≈ [{fxh:.10f} - {fx:.10f}] / {h}",
            'result': derivative
        }
    ]

    result = {
        'derivative': derivative,
        'x': x,
        'h': h,
        'steps': steps,
        'method': 'Forward Difference',
        'order': 'O(h) - prvi red',
        'formula': "f'(x) ≈ [f(x+h) - f(x)] / h"
    }

    if df_exact is not None:
        exact = df_exact(x)
        error = abs(derivative - exact)
        result['exact'] = exact
        result['error'] = error
        result['relative_error'] = error / abs(exact) if exact != 0 else float('inf')
        steps.append({
            'step': 3,
            'title': 'Računanje greške',
            'exact': exact,
            'error': error,
            'description': f'Tačna vrijednost: {exact:.10f}, Greška: {error:.2e}'
        })

    return result


def backward_diff(f: Callable[[float], float],
                  x: float,
                  h: float = 0.001,
                  df_exact: Optional[Callable[[float], float]] = None) -> Dict:
    """
    Backward Difference (Unazadna Diferencija)
    ==========================================

    TEORIJA:
    --------
    Backward difference koristi vrijednosti funkcije u tački x i x-h (unazad).

    IZVOD IZ TAYLOROVOG REDA:
    -------------------------
    Taylorov razvoj f(x-h) oko tačke x:
        f(x-h) = f(x) - h·f'(x) + h²/2·f''(x) - h³/6·f'''(x) + ...

    Rješavamo za f'(x):
        f'(x) ≈ [f(x) - f(x-h)] / h + O(h)

    FORMULA:
    --------
        f'(x) ≈ [f(x) - f(x-h)] / h

    GREŠKA:
    -------
    Greška rezanja: O(h)
        E_t ≈ h/2 · f''(x)

    Napomena: Greška ima suprotan predznak od forward difference.

    PRIMJENA:
    ---------
    Backward difference je korisna kada nemamo pristup tačkama
    ispred x (npr. na desnom rubu domene).

    Args:
        f: Funkcija
        x: Tačka u kojoj računamo derivaciju
        h: Korak
        df_exact: Tačna derivacija (opcionalno)

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    fx = f(x)
    fxmh = f(x - h)
    derivative = (fx - fxmh) / h

    steps = [
        {
            'step': 1,
            'title': 'Evaluacija funkcije',
            'f_x': fx,
            'f_x_minus_h': fxmh,
            'description': f'f({x}) = {fx:.10f}, f({x}-{h}) = f({x - h}) = {fxmh:.10f}'
        },
        {
            'step': 2,
            'title': 'Primjena formule',
            'formula': "f'(x) ≈ [f(x) - f(x-h)] / h",
            'calculation': f"f'({x}) ≈ [{fx:.10f} - {fxmh:.10f}] / {h}",
            'result': derivative
        }
    ]

    result = {
        'derivative': derivative,
        'x': x,
        'h': h,
        'steps': steps,
        'method': 'Backward Difference',
        'order': 'O(h) - prvi red',
        'formula': "f'(x) ≈ [f(x) - f(x-h)] / h"
    }

    if df_exact is not None:
        exact = df_exact(x)
        error = abs(derivative - exact)
        result['exact'] = exact
        result['error'] = error
        result['relative_error'] = error / abs(exact) if exact != 0 else float('inf')
        steps.append({
            'step': 3,
            'title': 'Računanje greške',
            'exact': exact,
            'error': error,
            'description': f'Tačna vrijednost: {exact:.10f}, Greška: {error:.2e}'
        })

    return result


def central_diff(f: Callable[[float], float],
                 x: float,
                 h: float = 0.001,
                 df_exact: Optional[Callable[[float], float]] = None) -> Dict:
    """
    Central Difference (Centralna Diferencija)
    ==========================================

    TEORIJA:
    --------
    Central difference koristi vrijednosti funkcije sa obje strane tačke x.
    Ovo daje značajno bolju preciznost od forward/backward metoda.

    IZVOD IZ TAYLOROVOG REDA:
    -------------------------
    f(x+h) = f(x) + h·f'(x) + h²/2·f''(x) + h³/6·f'''(x) + h⁴/24·f⁽⁴⁾(x) + ...
    f(x-h) = f(x) - h·f'(x) + h²/2·f''(x) - h³/6·f'''(x) + h⁴/24·f⁽⁴⁾(x) - ...

    Oduzimanje:
    f(x+h) - f(x-h) = 2h·f'(x) + 2h³/6·f'''(x) + ...

    Rješavamo za f'(x):
        f'(x) = [f(x+h) - f(x-h)]/(2h) - h²/6·f'''(x) - ...
        f'(x) ≈ [f(x+h) - f(x-h)]/(2h) + O(h²)

    FORMULA:
    --------
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

    GREŠKA:
    -------
    Greška rezanja: O(h²)
        E_t ≈ -h²/6 · f'''(x)

    ZAŠTO JE BOLJA?
    ---------------
    - Članovi sa h i h² se poništavaju pri oduzimanju
    - Greška je drugog reda (h²), ne prvog (h)
    - Za h = 0.01: forward ima grešku ~0.01, central ~0.0001

    OPTIMALNI h:
    ------------
    h_opt ≈ ∛(3ε·|f(x)|/|f'''(x)|) ≈ ε^(1/3) ≈ 10⁻⁵ do 10⁻⁶

    DRUGA DERIVACIJA (bonus):
    -------------------------
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²  + O(h²)

    Args:
        f: Funkcija
        x: Tačka u kojoj računamo derivaciju
        h: Korak
        df_exact: Tačna derivacija (opcionalno)

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    fxph = f(x + h)
    fxmh = f(x - h)
    derivative = (fxph - fxmh) / (2 * h)

    steps = [
        {
            'step': 1,
            'title': 'Evaluacija funkcije',
            'f_x_plus_h': fxph,
            'f_x_minus_h': fxmh,
            'description': f'f({x}+{h}) = {fxph:.10f}, f({x}-{h}) = {fxmh:.10f}'
        },
        {
            'step': 2,
            'title': 'Primjena formule',
            'formula': "f'(x) ≈ [f(x+h) - f(x-h)] / (2h)",
            'calculation': f"f'({x}) ≈ [{fxph:.10f} - {fxmh:.10f}] / (2·{h})",
            'result': derivative
        }
    ]

    result = {
        'derivative': derivative,
        'x': x,
        'h': h,
        'steps': steps,
        'method': 'Central Difference',
        'order': 'O(h²) - drugi red',
        'formula': "f'(x) ≈ [f(x+h) - f(x-h)] / (2h)"
    }

    if df_exact is not None:
        exact = df_exact(x)
        error = abs(derivative - exact)
        result['exact'] = exact
        result['error'] = error
        result['relative_error'] = error / abs(exact) if exact != 0 else float('inf')
        steps.append({
            'step': 3,
            'title': 'Računanje greške',
            'exact': exact,
            'error': error,
            'description': f'Tačna vrijednost: {exact:.10f}, Greška: {error:.2e}'
        })

    return result


def second_derivative_central(f: Callable[[float], float],
                              x: float,
                              h: float = 0.001,
                              d2f_exact: Optional[Callable[[float], float]] = None) -> Dict:
    """
    Druga Derivacija - Centralna Diferencija
    ========================================

    FORMULA:
        f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²

    GREŠKA: O(h²)

    Args:
        f: Funkcija
        x: Tačka
        h: Korak
        d2f_exact: Tačna druga derivacija (opcionalno)

    Returns:
        Dict sa rezultatom
    """
    fxph = f(x + h)
    fx = f(x)
    fxmh = f(x - h)
    derivative = (fxph - 2 * fx + fxmh) / (h ** 2)

    result = {
        'derivative': derivative,
        'x': x,
        'h': h,
        'method': 'Central Difference (2nd derivative)',
        'formula': "f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²"
    }

    if d2f_exact is not None:
        exact = d2f_exact(x)
        result['exact'] = exact
        result['error'] = abs(derivative - exact)

    return result


def compare_errors(f: Callable[[float], float],
                   x: float,
                   df_exact: Callable[[float], float],
                   h_values: List[float] = None) -> Dict:
    """
    Poređenje Grešaka za Različite h
    =================================

    TEORIJA:
    --------
    Ova funkcija demonstrira ključni koncept numeričke derivacije:
    postoji OPTIMALNI h koji minimizira ukupnu grešku.

    DVA IZVORA GREŠKE:
    ------------------
    1. Greška rezanja (truncation error) - raste sa h
       - Forward/Backward: O(h)
       - Central: O(h²)

    2. Greška zaokruživanja (round-off error) - raste kada h → 0
       - Proporcionalna sa ε/h

    UKUPNA GREŠKA:
    --------------
    E_total = E_truncation + E_roundoff

    Za forward/backward:
        E ≈ h·|f''|/2 + ε·|f|/h
        h_opt ≈ √(2ε|f|/|f''|) ≈ 10⁻⁸

    Za central:
        E ≈ h²·|f'''|/6 + ε·|f|/h
        h_opt ≈ (3ε|f|/|f'''|)^(1/3) ≈ 10⁻⁵

    LOG-LOG GRAF:
    -------------
    Na log-log grafu:
    - Za velike h: nagib -1 (forward) ili -2 (central) (truncation dominira)
    - Za male h: nagib +1 (round-off dominira)
    - Optimalni h je u minimumu

    Args:
        f: Funkcija
        x: Tačka
        df_exact: Tačna derivacija
        h_values: Lista vrijednosti h za testiranje

    Returns:
        Dict sa rezultatima poređenja
    """
    if h_values is None:
        # Logaritamski raspoređene vrijednosti od 10^(-15) do 10^0
        h_values = [10 ** (-i) for i in range(16)]

    exact = df_exact(x)

    results = {
        'forward': [],
        'backward': [],
        'central': [],
        'h_values': h_values,
        'exact': exact,
        'x': x
    }

    for h in h_values:
        # Forward difference
        fwd = forward_diff(f, x, h)
        results['forward'].append({
            'h': h,
            'derivative': fwd['derivative'],
            'error': abs(fwd['derivative'] - exact)
        })

        # Backward difference
        bwd = backward_diff(f, x, h)
        results['backward'].append({
            'h': h,
            'derivative': bwd['derivative'],
            'error': abs(bwd['derivative'] - exact)
        })

        # Central difference
        cnt = central_diff(f, x, h)
        results['central'].append({
            'h': h,
            'derivative': cnt['derivative'],
            'error': abs(cnt['derivative'] - exact)
        })

    # Pronađi optimalni h za svaku metodu
    for method in ['forward', 'backward', 'central']:
        errors = [r['error'] for r in results[method]]
        min_idx = np.argmin(errors)
        results[f'{method}_optimal'] = {
            'h': h_values[min_idx],
            'error': errors[min_idx]
        }

    results['explanation'] = {
        'truncation': 'Greška rezanja (raste sa h): Forward O(h), Central O(h²)',
        'roundoff': 'Greška zaokruživanja (raste kada h→0): O(ε/h)',
        'optimal': 'Optimalni h balansira oba izvora greške',
        'forward_optimal': '~10⁻⁸ za forward/backward difference',
        'central_optimal': '~10⁻⁵ do 10⁻⁶ za central difference'
    }

    return results


def auto_differentiate(f: Callable[[float], float] = None,
                       x_points: np.ndarray = None,
                       y_points: np.ndarray = None,
                       x: float = None,
                       h: float = None,
                       domain: Tuple[float, float] = None) -> Dict:
    """
    Automatska Detekcija Metode Derivacije
    ======================================

    Automatski bira najbolju metodu derivacije na osnovu položaja tačke:
    - Na lijevom rubu domene: Forward Difference
    - Na desnom rubu domene: Backward Difference
    - U unutrašnjosti: Central Difference (najpreciznija)

    Može raditi sa:
    1. Funkcijom f(x) - ako je f zadana
    2. Tablicom vrijednosti (x_points, y_points) - ako funkcija nije zadana

    Args:
        f: Funkcija (opcionalno ako su dati x_points i y_points)
        x_points: Niz x vrijednosti (za tablični unos)
        y_points: Niz y vrijednosti (za tablični unos)
        x: Tačka u kojoj računamo derivaciju (ako nije dato, računa za sve tačke)
        h: Korak (ako nije dato, automatski se određuje)
        domain: (a, b) granice domene za funkciju

    Returns:
        Dict sa derivacijama i objašnjenjima izbora metode
    """
    results = {
        'derivatives': [],
        'method_choices': [],
        'steps': []
    }

    # Ako imamo tablicu vrijednosti
    if x_points is not None and y_points is not None:
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        n = len(x_points)

        if n < 2:
            return {'error': 'Potrebne su najmanje 2 tačke za derivaciju'}

        # Automatski h iz podataka
        h_values = np.diff(x_points)
        h_avg = np.mean(h_values)

        for i in range(n):
            x_i = x_points[i]
            y_i = y_points[i]

            if i == 0:
                # Lijevi rub - Forward Difference
                method = 'Forward Difference'
                reason = 'Lijevi rub domene - nema tačaka lijevo'
                h_used = x_points[1] - x_points[0]
                deriv = (y_points[1] - y_points[0]) / h_used
                formula = f"f'({x_i:.4f}) ≈ [f({x_points[1]:.4f}) - f({x_i:.4f})] / {h_used:.4f}"

            elif i == n - 1:
                # Desni rub - Backward Difference
                method = 'Backward Difference'
                reason = 'Desni rub domene - nema tačaka desno'
                h_used = x_points[-1] - x_points[-2]
                deriv = (y_points[-1] - y_points[-2]) / h_used
                formula = f"f'({x_i:.4f}) ≈ [f({x_i:.4f}) - f({x_points[-2]:.4f})] / {h_used:.4f}"

            else:
                # Unutrašnjost - Central Difference (najpreciznija)
                method = 'Central Difference'
                reason = 'Unutrašnja tačka - koristi se najpreciznija metoda'
                h_left = x_i - x_points[i-1]
                h_right = x_points[i+1] - x_i

                if abs(h_left - h_right) < 1e-10:
                    # Uniformni razmak
                    h_used = h_left
                    deriv = (y_points[i+1] - y_points[i-1]) / (2 * h_used)
                    formula = f"f'({x_i:.4f}) ≈ [f({x_points[i+1]:.4f}) - f({x_points[i-1]:.4f})] / (2·{h_used:.4f})"
                else:
                    # Neuniformni razmak - koristi težinsku formulu
                    deriv = (y_points[i+1] - y_points[i-1]) / (h_left + h_right)
                    h_used = (h_left + h_right) / 2
                    formula = f"f'({x_i:.4f}) ≈ [f({x_points[i+1]:.4f}) - f({x_points[i-1]:.4f})] / ({h_left:.4f} + {h_right:.4f})"

            results['derivatives'].append({
                'x': x_i,
                'derivative': deriv,
                'method': method,
                'reason': reason,
                'formula': formula,
                'h': h_used
            })

            results['method_choices'].append({
                'x': x_i,
                'index': i,
                'method': method,
                'reason': reason
            })

        results['source'] = 'table'
        results['h_average'] = h_avg

    # Ako imamo funkciju
    elif f is not None:
        if domain is None:
            domain = (-10, 10)  # Default domena

        a, b = domain
        if h is None:
            h = (b - a) / 100  # Automatski h

        # Ako je zadana specifična tačka
        if x is not None:
            x_eval = [x]
        else:
            # Generiši tačke
            x_eval = np.linspace(a, b, 11)

        for x_i in x_eval:
            # Odredi položaj u domeni
            tol = h * 0.5

            if x_i <= a + tol:
                # Lijevi rub
                method = 'Forward Difference'
                reason = f'Blizu lijevog ruba domene (x ≈ {a})'
                deriv = (f(x_i + h) - f(x_i)) / h
                formula = f"f'({x_i:.4f}) ≈ [f({x_i + h:.4f}) - f({x_i:.4f})] / {h:.6f}"

            elif x_i >= b - tol:
                # Desni rub
                method = 'Backward Difference'
                reason = f'Blizu desnog ruba domene (x ≈ {b})'
                deriv = (f(x_i) - f(x_i - h)) / h
                formula = f"f'({x_i:.4f}) ≈ [f({x_i:.4f}) - f({x_i - h:.4f})] / {h:.6f}"

            else:
                # Unutrašnjost - Central (najpreciznija)
                method = 'Central Difference'
                reason = 'Unutrašnja tačka - koristi se najpreciznija metoda O(h²)'
                deriv = (f(x_i + h) - f(x_i - h)) / (2 * h)
                formula = f"f'({x_i:.4f}) ≈ [f({x_i + h:.4f}) - f({x_i - h:.4f})] / (2·{h:.6f})"

            results['derivatives'].append({
                'x': x_i,
                'derivative': deriv,
                'method': method,
                'reason': reason,
                'formula': formula,
                'h': h
            })

            results['method_choices'].append({
                'x': x_i,
                'method': method,
                'reason': reason
            })

        results['source'] = 'function'
        results['domain'] = domain
        results['h'] = h

    else:
        return {'error': 'Morate zadati ili funkciju f ili tablicu (x_points, y_points)'}

    # Sažetak
    method_counts = {}
    for choice in results['method_choices']:
        m = choice['method']
        method_counts[m] = method_counts.get(m, 0) + 1

    results['summary'] = {
        'total_points': len(results['derivatives']),
        'method_counts': method_counts,
        'explanation': '''
Automatska detekcija metode:
• Forward Difference: Koristi se na lijevom rubu gdje nema tačaka lijevo
• Backward Difference: Koristi se na desnom rubu gdje nema tačaka desno
• Central Difference: Koristi se u unutrašnjosti - najpreciznija metoda O(h²)
'''
    }

    return results


def differentiate_from_table(x_points: np.ndarray,
                             y_points: np.ndarray,
                             return_function: bool = False) -> Dict:
    """
    Derivacija iz Tablice Vrijednosti
    ==================================

    Računa derivaciju iz tablice vrijednosti sa automatskim izborom metode.
    Može vratiti i interpoliranu funkciju za daljnje računanje.

    Args:
        x_points: X vrijednosti
        y_points: Y vrijednosti
        return_function: Ako True, vraća i interpoliranu funkciju

    Returns:
        Dict sa derivacijama i opcionalno funkcijom
    """
    from scipy.interpolate import CubicSpline

    x_points = np.array(x_points)
    y_points = np.array(y_points)

    # Automatska derivacija
    result = auto_differentiate(x_points=x_points, y_points=y_points)

    if return_function:
        # Kreiraj kubni spline za interpolaciju
        cs = CubicSpline(x_points, y_points)
        result['interpolated_function'] = cs
        result['interpolated_derivative'] = cs.derivative()

        # Dodaj objašnjenje
        result['interpolation_info'] = '''
Kubni spline interpolacija:
• Kreira glatku funkciju koja prolazi kroz sve tačke
• Derivacija splina daje glatku aproksimaciju derivacije
• Pogodno za daljnju analizu i vizualizaciju
'''

    return result


def higher_order_derivatives(f: Callable[[float], float],
                             x: float,
                             h: float = 0.001,
                             order: int = 4) -> Dict:
    """
    Više derivacije korištenjem konačnih diferencija
    ================================================

    Računa f', f'', f''', f'''' korištenjem centralnih diferencija.

    FORMULE:
    --------
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    f'''(x) ≈ [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h³)
    f''''(x) ≈ [f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)] / h⁴

    Args:
        f: Funkcija
        x: Tačka
        h: Korak
        order: Maksimalni red derivacije (1-4)

    Returns:
        Dict sa svim derivacijama
    """
    results = {}

    # Potrebne vrijednosti funkcije
    f_vals = {
        -2: f(x - 2 * h),
        -1: f(x - h),
        0: f(x),
        1: f(x + h),
        2: f(x + 2 * h)
    }

    if order >= 1:
        # Prva derivacija
        d1 = (f_vals[1] - f_vals[-1]) / (2 * h)
        results['first'] = {
            'value': d1,
            'formula': "[f(x+h) - f(x-h)] / (2h)"
        }

    if order >= 2:
        # Druga derivacija
        d2 = (f_vals[1] - 2 * f_vals[0] + f_vals[-1]) / (h ** 2)
        results['second'] = {
            'value': d2,
            'formula': "[f(x+h) - 2f(x) + f(x-h)] / h²"
        }

    if order >= 3:
        # Treća derivacija
        d3 = (f_vals[2] - 2 * f_vals[1] + 2 * f_vals[-1] - f_vals[-2]) / (2 * h ** 3)
        results['third'] = {
            'value': d3,
            'formula': "[f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h³)"
        }

    if order >= 4:
        # Četvrta derivacija
        d4 = (f_vals[2] - 4 * f_vals[1] + 6 * f_vals[0] - 4 * f_vals[-1] + f_vals[-2]) / (h ** 4)
        results['fourth'] = {
            'value': d4,
            'formula': "[f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)] / h⁴"
        }

    results['x'] = x
    results['h'] = h
    results['f_values'] = f_vals

    return results
