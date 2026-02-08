"""
Metode Numeričke Derivacije
===========================

Ovaj modul implementira metode za numeričko računanje derivacija:

1. Forward Difference (Unaprijedna diferencija)
2. Backward Difference (Unazadna diferencija)
3. Central Difference (Centralna diferencija)
4. Automatska detekcija metode (auto_differentiate)

"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional

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


