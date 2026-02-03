"""
Metode Regresije i Aproksimacije
================================

Ovaj modul implementira metode za aproksimaciju podataka:

1. Linearna Regresija (Metoda Najmanjih Kvadrata) - Rađeno na nastavi
2. Eksponencijalna Aproksimacija - Rađeno na nastavi
3. Polinomijalna Regresija - Bonus metoda

Sve metode vraćaju detaljne korake za step-by-step prikaz.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def linear_regression(x_data: np.ndarray,
                      y_data: np.ndarray) -> Dict:
    """
    Linearna Regresija (Metoda Najmanjih Kvadrata)
    ==============================================

    TEORIJA:
    --------
    Linearna regresija traži pravu y = ax + b koja najbolje
    "fituje" date tačke u smislu minimizacije sume kvadrata odstupanja.

    MODEL:
    ------
    y = ax + b

    gdje je:
    - a: nagib (slope)
    - b: odsječak na y-osi (intercept)

    METODA NAJMANJIH KVADRATA:
    --------------------------
    Cilj: Minimizirati sumu kvadrata reziduala (odstupanja):
        S = Σ(y_i - (ax_i + b))²

    IZVOD FORMULA:
    --------------
    Parcijalni izvodi:
        ∂S/∂a = -2Σx_i(y_i - ax_i - b) = 0
        ∂S/∂b = -2Σ(y_i - ax_i - b) = 0

    Normalne jednačine:
        a·Σx_i² + b·Σx_i = Σx_i·y_i
        a·Σx_i + b·n = Σy_i

    RJEŠENJE:
    ---------
        a = [n·Σx_i·y_i - Σx_i·Σy_i] / [n·Σx_i² - (Σx_i)²]
        b = [Σy_i - a·Σx_i] / n

    ili korištenjem sredina:
        a = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
        b = ȳ - a·x̄

    R² (KOEFICIJENT DETERMINACIJE):
    -------------------------------
    R² mjeri koliko dobro model opisuje podatke.
        R² = 1 - SS_res/SS_tot
        SS_res = Σ(y_i - ŷ_i)²  (rezidualna suma kvadrata)
        SS_tot = Σ(y_i - ȳ)²    (ukupna suma kvadrata)

    R² ∈ [0, 1]:
    - R² = 1: savršen fit
    - R² = 0: model nije bolji od srednje vrijednosti

    Args:
        x_data: Nezavisna varijabla (n tačaka)
        y_data: Zavisna varijabla (n tačaka)

    Returns:
        Dict sa parametrima, koracima i statistikama
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    n = len(x)

    steps = []

    # Korak 1: Računanje suma
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    mean_x = sum_x / n
    mean_y = sum_y / n

    steps.append({
        'step': 1,
        'title': 'Računanje potrebnih suma',
        'n': n,
        'sum_x': sum_x,
        'sum_y': sum_y,
        'sum_xy': sum_xy,
        'sum_x2': sum_x2,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'formulas': {
            'Σx': f'{sum_x:.4f}',
            'Σy': f'{sum_y:.4f}',
            'Σxy': f'{sum_xy:.4f}',
            'Σx²': f'{sum_x2:.4f}',
            'x̄': f'{mean_x:.4f}',
            'ȳ': f'{mean_y:.4f}'
        }
    })

    # Korak 2: Računanje koeficijenata
    denominator = n * sum_x2 - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - a * sum_x) / n

    steps.append({
        'step': 2,
        'title': 'Računanje koeficijenata',
        'formula_a': 'a = [n·Σxy - Σx·Σy] / [n·Σx² - (Σx)²]',
        'calculation_a': f'a = [{n}·{sum_xy:.4f} - {sum_x:.4f}·{sum_y:.4f}] / [{n}·{sum_x2:.4f} - ({sum_x:.4f})²]',
        'a': a,
        'formula_b': 'b = (Σy - a·Σx) / n',
        'calculation_b': f'b = ({sum_y:.4f} - {a:.4f}·{sum_x:.4f}) / {n}',
        'b': b,
        'equation': f'y = {a:.6f}·x + {b:.6f}'
    })

    # Korak 3: Predviđene vrijednosti i reziduali
    y_pred = a * x + b
    residuals = y - y_pred

    steps.append({
        'step': 3,
        'title': 'Predviđene vrijednosti i reziduali',
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'description': 'Rezidual = y_i - ŷ_i (razlika između stvarne i predviđene vrijednosti)'
    })

    # Korak 4: R² koeficijent
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - mean_y) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Standardna greška
    if n > 2:
        std_error = np.sqrt(ss_res / (n - 2))
    else:
        std_error = 0

    steps.append({
        'step': 4,
        'title': 'Ocjena kvalitete modela',
        'ss_res': ss_res,
        'ss_tot': ss_tot,
        'r_squared': r_squared,
        'r_squared_formula': 'R² = 1 - SS_res/SS_tot',
        'r_squared_calculation': f'R² = 1 - {ss_res:.4f}/{ss_tot:.4f} = {r_squared:.6f}',
        'std_error': std_error,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,  # nagib
        'b': b,  # odsječak
        'equation': f'y = {a:.6f}x + {b:.6f}',
        'r_squared': r_squared,
        'std_error': std_error,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'steps': steps,
        'method': 'Linearna regresija (metoda najmanjih kvadrata)'
    }


def interpret_r_squared(r2: float) -> str:
    """Interpretacija R² vrijednosti"""
    if r2 >= 0.9:
        return f'Odličan fit (R² = {r2:.4f}): Model objašnjava {r2*100:.1f}% varijabilnosti.'
    elif r2 >= 0.7:
        return f'Dobar fit (R² = {r2:.4f}): Model objašnjava {r2*100:.1f}% varijabilnosti.'
    elif r2 >= 0.5:
        return f'Umjeren fit (R² = {r2:.4f}): Model objašnjava {r2*100:.1f}% varijabilnosti.'
    else:
        return f'Slab fit (R² = {r2:.4f}): Model objašnjava samo {r2*100:.1f}% varijabilnosti.'


def exponential_regression(x_data: np.ndarray,
                           y_data: np.ndarray) -> Dict:
    """
    Eksponencijalna Aproksimacija
    =============================

    TEORIJA:
    --------
    Eksponencijalna aproksimacija fituje podatke na model:
        y = A · e^(Bx)  ili  y = A · B^x

    LINEARIZACIJA:
    --------------
    Logaritmovanjem transformišemo eksponencijalni model u linearni:
        ln(y) = ln(A) + Bx

    Supstitucija:
        Y = ln(y), a = B, b = ln(A)

    Dobijamo linearni model:
        Y = ax + b

    PROCEDURA:
    ----------
    1. Transformiši y → Y = ln(y)
    2. Primijeni linearnu regresiju na (x, Y)
    3. Izračunaj: B = a, A = e^b

    NAPOMENA:
    ---------
    y vrijednosti moraju biti POZITIVNE (ln(y) nije definisan za y ≤ 0)

    PRIMJENE:
    ---------
    - Rast populacije
    - Radioaktivni raspad
    - Kamate i investicije
    - Širenje bolesti

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla (mora biti > 0)

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    steps = []

    # Provjera pozitivnosti
    if np.any(y <= 0):
        return {
            'A': None,
            'B': None,
            'error_message': 'Sve y vrijednosti moraju biti pozitivne za eksponencijalnu aproksimaciju!',
            'steps': []
        }

    # Korak 1: Linearizacija
    Y = np.log(y)

    steps.append({
        'step': 1,
        'title': 'Linearizacija - logaritmovanje',
        'original_y': y.tolist(),
        'transformed_Y': Y.tolist(),
        'description': 'Y = ln(y) transformiše eksponencijalni model u linearni',
        'model_original': 'y = A·e^(Bx)',
        'model_linearized': 'Y = ln(y) = ln(A) + Bx = b + ax'
    })

    # Korak 2: Linearna regresija na transformisanim podacima
    linear_result = linear_regression(x, Y)

    a = linear_result['a']
    b_lin = linear_result['b']

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'linear_equation': f'Y = {a:.6f}x + {b_lin:.6f}',
        'a_linear': a,
        'b_linear': b_lin,
        'r_squared_linear': linear_result['r_squared']
    })

    # Korak 3: Povratak na originalne parametre
    B = a
    A = np.exp(b_lin)

    steps.append({
        'step': 3,
        'title': 'Transformacija nazad u eksponencijalne parametre',
        'formulas': {
            'B': 'B = a (nagib linearne regresije)',
            'A': 'A = e^b (eksponent odsječka)'
        },
        'calculations': {
            'B': f'B = {a:.6f}',
            'A': f'A = e^{b_lin:.6f} = {A:.6f}'
        },
        'final_equation': f'y = {A:.6f}·e^({B:.6f}x)'
    })

    # Predviđene vrijednosti
    y_pred = A * np.exp(B * x)
    residuals = y - y_pred

    # R² za originalne podatke
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Ocjena kvalitete modela (originalni podaci)',
        'r_squared': r_squared,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'A': A,
        'B': B,
        'equation': f'y = {A:.6f}·e^({B:.6f}x)',
        'equation_alternative': f'y = {A:.6f}·{np.exp(B):.6f}^x',
        'r_squared': r_squared,
        'r_squared_linear': linear_result['r_squared'],
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'steps': steps,
        'method': 'Eksponencijalna aproksimacija (linearizacija)'
    }


def polynomial_regression(x_data: np.ndarray,
                          y_data: np.ndarray,
                          degree: int = 2) -> Dict:
    """
    Polinomijalna Regresija
    =======================

    TEORIJA (BONUS - nije rađeno na nastavi):
    -----------------------------------------
    Polinomijalna regresija fituje podatke na polinom stepena n:
        y = a_n·x^n + a_{n-1}·x^{n-1} + ... + a_1·x + a_0

    METODA NAJMANJIH KVADRATA:
    --------------------------
    Cilj: Minimizirati S = Σ(y_i - p(x_i))²

    To vodi na sistem normalnih jednačina koje se mogu napisati
    u matričnom obliku kao:
        X^T·X·a = X^T·y

    gdje je X Vandermondova matrica:
        X = [1  x_1  x_1²  ...  x_1^n]
            [1  x_2  x_2²  ...  x_2^n]
            [...                     ]
            [1  x_m  x_m²  ...  x_m^n]

    RJEŠENJE:
    ---------
        a = (X^T·X)^{-1}·X^T·y

    UPOZORENJE - OVERFITTING:
    -------------------------
    - Visoki stepen polinoma može dovesti do "overfittinga"
    - Model savršeno prolazi kroz tačke, ali loše predviđa
    - Pravilo: stepen << broj tačaka
    - Koristi R² i vizualizaciju za provjeru

    IZBOR STEPENA:
    --------------
    - Stepen 1: Linearna regresija
    - Stepen 2: Kvadratna (parabola) - često dovoljna
    - Stepen 3+: Koristiti oprezno

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla
        degree: Stepen polinoma

    Returns:
        Dict sa koeficijentima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    n = len(x)

    steps = []

    # Upozorenje za overfitting
    if degree >= n - 1:
        steps.append({
            'step': 0,
            'title': '⚠️ UPOZORENJE',
            'message': f'Stepen polinoma ({degree}) je previsok za {n} tačaka. Rizik od overfittinga!'
        })

    # Korak 1: Kreiranje Vandermondove matrice
    X = np.vander(x, degree + 1, increasing=True)

    steps.append({
        'step': 1,
        'title': 'Kreiranje Vandermondove matrice',
        'description': f'Matrica X dimenzija {n}×{degree+1}',
        'X_shape': X.shape,
        'note': 'X[i,j] = x_i^j za j = 0, 1, ..., degree'
    })

    # Korak 2: Rješavanje normalnih jednačina
    # a = (X^T X)^(-1) X^T y
    XtX = X.T @ X
    Xty = X.T @ y

    steps.append({
        'step': 2,
        'title': 'Formiranje normalnih jednačina',
        'formula': 'X^T·X·a = X^T·y',
        'XtX_shape': XtX.shape,
        'description': 'Rješavamo sistem za koeficijente a'
    })

    # Rješavanje sistema
    try:
        coefficients = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return {
            'coefficients': None,
            'error_message': 'Sistem je singularan - nije moguće riješiti.',
            'steps': steps
        }

    steps.append({
        'step': 3,
        'title': 'Rješenje - koeficijenti polinoma',
        'coefficients': coefficients.tolist(),
        'equation': format_polynomial(coefficients)
    })

    # Korak 4: Predviđene vrijednosti i R²
    y_pred = X @ coefficients
    residuals = y - y_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Adjusted R²
    if n > degree + 1:
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - degree - 1)
    else:
        r_squared_adj = r_squared

    steps.append({
        'step': 4,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_adjusted': r_squared_adj,
        'note': 'Adjusted R² penalizira dodavanje više parametara',
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'coefficients': coefficients.tolist(),
        'degree': degree,
        'equation': format_polynomial(coefficients),
        'r_squared': r_squared,
        'r_squared_adjusted': r_squared_adj,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'steps': steps,
        'method': f'Polinomijalna regresija (stepen {degree})'
    }


def format_polynomial(coeffs: np.ndarray) -> str:
    """Formatira polinom za prikaz"""
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-10:
            continue
        if i == 0:
            terms.append(f'{c:.6f}')
        elif i == 1:
            sign = '+' if c >= 0 else ''
            terms.append(f'{sign}{c:.6f}x')
        else:
            sign = '+' if c >= 0 else ''
            terms.append(f'{sign}{c:.6f}x^{i}')

    return 'y = ' + ''.join(terms) if terms else 'y = 0'


def power_regression(x_data: np.ndarray,
                     y_data: np.ndarray) -> Dict:
    """
    Stepena Aproksimacija (Power Regression)
    ========================================

    MODEL:
    ------
    y = A · x^B

    LINEARIZACIJA:
    --------------
    ln(y) = ln(A) + B·ln(x)

    Supstitucija: Y = ln(y), X = ln(x), a = B, b = ln(A)
    Linearni model: Y = aX + b

    Args:
        x_data: Nezavisna varijabla (mora biti > 0)
        y_data: Zavisna varijabla (mora biti > 0)

    Returns:
        Dict sa parametrima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    if np.any(x <= 0) or np.any(y <= 0):
        return {
            'error_message': 'Sve x i y vrijednosti moraju biti pozitivne!'
        }

    # Linearizacija
    X = np.log(x)
    Y = np.log(y)

    # Linearna regresija
    linear_result = linear_regression(X, Y)

    B = linear_result['a']
    A = np.exp(linear_result['b'])

    y_pred = A * (x ** B)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        'A': A,
        'B': B,
        'equation': f'y = {A:.6f}·x^{B:.6f}',
        'r_squared': r_squared,
        'y_predicted': y_pred.tolist(),
        'method': 'Stepena aproksimacija'
    }


def compare_regression_models(x_data: np.ndarray,
                              y_data: np.ndarray) -> Dict:
    """
    Poređenje različitih modela regresije

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla

    Returns:
        Dict sa svim modelima i poređenjem
    """
    results = {}

    # Linearna
    results['linear'] = linear_regression(x_data, y_data)

    # Kvadratna
    results['quadratic'] = polynomial_regression(x_data, y_data, degree=2)

    # Kubna
    results['cubic'] = polynomial_regression(x_data, y_data, degree=3)

    # Eksponencijalna (ako su y > 0)
    if np.all(np.array(y_data) > 0):
        results['exponential'] = exponential_regression(x_data, y_data)

    # Stepena (ako su x > 0 i y > 0)
    if np.all(np.array(x_data) > 0) and np.all(np.array(y_data) > 0):
        results['power'] = power_regression(x_data, y_data)

    # Sumarno poređenje
    summary = []
    for name, result in results.items():
        if 'r_squared' in result:
            summary.append({
                'model': name,
                'r_squared': result['r_squared'],
                'equation': result.get('equation', 'N/A')
            })

    summary.sort(key=lambda x: x['r_squared'], reverse=True)
    results['summary'] = summary

    return results
