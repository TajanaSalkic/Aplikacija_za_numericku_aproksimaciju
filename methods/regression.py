"""
Metode Aproksimacije Funkcija
=============================

Ovaj modul implementira metode za aproksimaciju podataka korištenjem
metode najmanjih kvadrata i linearizacije.

NOTACIJA (prema slajdovima sa predavanja):
------------------------------------------
- Model: y = a + bx (a = odsječak, b = nagib)
- Normalne jednačine prikazane eksplicitno

METODE APROKSIMACIJE:
---------------------

1. LINEARNA APROKSIMACIJA (Metoda Najmanjih Kvadrata)
   Model: y = a + bx
   - Direktna primjena metode najmanjih kvadrata
   - Rađeno na nastavi (slajdovi 105-110)

2. NELINEARNE APROKSIMACIJE (putem linearizacije):

   a) STEPENA (Power) APROKSIMACIJA
      Model: y = a · x^b
      Linearizacija: ln(y) = ln(a) + b·ln(x)
      - Rađeno na nastavi (slajd 113)

   b) EKSPONENCIJALNA APROKSIMACIJA
      Model: y = a · e^(bx)
      Linearizacija: ln(y) = ln(a) + bx
      - Rađeno na nastavi (slajdovi 114-117)

   c) LOGARITAMSKA APROKSIMACIJA
      Model: y = a + b·ln(x)
      Već linearan oblik: Y = a + b·X, gdje je X = ln(x)
      - Bonus metoda

   d) HIPERBOLIČKA APROKSIMACIJA
      Model: y = 1/(a + bx)
      Linearizacija: 1/y = a + bx
      - Bonus metoda

   e) RACIONALNA APROKSIMACIJA
      Model: y = x/(a + bx)
      Linearizacija: x/y = a + bx
      - Bonus metoda

   f) KVADRATNA KORIJEN APROKSIMACIJA
      Model: y = a + b·√x
      Već linearan oblik: Y = a + b·X, gdje je X = √x
      - Bonus metoda

3. POLINOMIJALNA APROKSIMACIJA
   Model: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
   - Koristi sistem normalnih jednačina
   - Rađeno na nastavi (slajdovi 105-112)

Sve metode vraćaju detaljne korake za step-by-step prikaz.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def linear_regression(x_data: np.ndarray,
                      y_data: np.ndarray) -> Dict:
    """
    Linearna Regresija (Metoda Najmanjih Kvadrata)
    ==============================================

    Prema slajdovima sa predavanja (str. 105-110)

    TEORIJA:
    --------
    Linearna regresija traži pravu y = a + bx koja najbolje
    "fituje" date tačke u smislu minimizacije sume kvadrata odstupanja.

    MODEL:
    ------
    y = a + bx

    gdje je:
    - a: odsječak na y-osi (intercept)
    - b: nagib (slope)

    METODA NAJMANJIH KVADRATA:
    --------------------------
    Cilj: Minimizirati sumu kvadrata reziduala (odstupanja):
        S = Σ(Yᵢ - yᵢ)² = Σ(Yᵢ - a - bxᵢ)²

    gdje je:
    - Yᵢ: izmjerena (eksperimentalna) vrijednost
    - yᵢ = a + bxᵢ: aproksimirana vrijednost

    IZVOD FORMULA:
    --------------
    Parcijalni izvodi:
        ∂S/∂a = -2Σ(Yᵢ - a - bxᵢ) = 0
        ∂S/∂b = -2Σxᵢ(Yᵢ - a - bxᵢ) = 0

    NORMALNE JEDNAČINE (kao na slajdu 108):
    ---------------------------------------
        a·N + b·Σxᵢ = ΣYᵢ
        a·Σxᵢ + b·Σxᵢ² = ΣxᵢYᵢ

    RJEŠENJE:
    ---------
    Iz sistema se dobija:
        b = [N·ΣxᵢYᵢ - Σxᵢ·ΣYᵢ] / [N·Σxᵢ² - (Σxᵢ)²]
        a = [ΣYᵢ - b·Σxᵢ] / N

    Args:
        x_data: Nezavisna varijabla (n tačaka)
        y_data: Zavisna varijabla (n tačaka)

    Returns:
        Dict sa parametrima, koracima i statistikama
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    # Korak 1: Računanje suma
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    steps.append({
        'step': 1,
        'title': 'Računanje potrebnih suma',
        'description': 'Računamo sume potrebne za normalne jednačine',
        'N': N,
        'sum_x': sum_x,
        'sum_y': sum_y,
        'sum_xy': sum_xy,
        'sum_x2': sum_x2,
        'formulas': {
            'N': f'{N}',
            'Σxᵢ': f'{sum_x:.6f}',
            'ΣYᵢ': f'{sum_y:.6f}',
            'ΣxᵢYᵢ': f'{sum_xy:.6f}',
            'Σxᵢ²': f'{sum_x2:.6f}'
        }
    })

    # Korak 2: Formiranje sistema normalnih jednačina
    # a·N + b·Σxᵢ = ΣYᵢ
    # a·Σxᵢ + b·Σxᵢ² = ΣxᵢYᵢ
    steps.append({
        'step': 2,
        'title': 'Sistem normalnih jednačina',
        'description': 'Formiramo sistem linearnih jednačina (kao na slajdu 108)',
        'system': [
            f'a·{N} + b·{sum_x:.4f} = {sum_y:.4f}',
            f'a·{sum_x:.4f} + b·{sum_x2:.4f} = {sum_xy:.4f}'
        ],
        'matrix_form': {
            'equation': 'A·[a, b]ᵀ = B',
            'A': [[N, sum_x], [sum_x, sum_x2]],
            'B': [sum_y, sum_xy]
        }
    })

    # Korak 3: Rješavanje sistema - računanje koeficijenata
    denominator = N * sum_x2 - sum_x ** 2
    b = (N * sum_xy - sum_x * sum_y) / denominator
    a = (sum_y - b * sum_x) / N

    steps.append({
        'step': 3,
        'title': 'Rješavanje sistema - računanje koeficijenata',
        'formula_b': 'b = [N·ΣxᵢYᵢ - Σxᵢ·ΣYᵢ] / [N·Σxᵢ² - (Σxᵢ)²]',
        'calculation_b': f'b = [{N}·{sum_xy:.4f} - {sum_x:.4f}·{sum_y:.4f}] / [{N}·{sum_x2:.4f} - ({sum_x:.4f})²]',
        'b': b,
        'formula_a': 'a = (ΣYᵢ - b·Σxᵢ) / N',
        'calculation_a': f'a = ({sum_y:.4f} - {b:.6f}·{sum_x:.4f}) / {N}',
        'a': a,
        'equation': f'y = {a:.6f} + {b:.6f}·x'
    })

    # Korak 4: Predviđene vrijednosti, reziduali i greške
    y_pred = a + b * x
    residuals = y - y_pred

    # Računanje greške u procentima kao na slajdovima 109-110
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja kao na slajdu 109-110
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške (kao na slajdu 109-110)',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent
    })

    # Korak 5: Ocjena kvalitete - R² koeficijent
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Standardna greška
    if N > 2:
        std_error = np.sqrt(ss_res / (N - 2))
    else:
        std_error = 0

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'ss_res': ss_res,
        'ss_tot': ss_tot,
        'r_squared': r_squared,
        'r_squared_formula': 'R² = 1 - SS_res/SS_tot',
        'r_squared_calculation': f'R² = 1 - {ss_res:.6f}/{ss_tot:.6f} = {r_squared:.6f}',
        'std_error': std_error,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,  # odsječak
        'b': b,  # nagib
        'equation': f'y = {a:.6f} + {b:.6f}·x',
        'r_squared': r_squared,
        'std_error': std_error,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
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

    Prema slajdovima sa predavanja (str. 114-117)

    MODEL:
    ------
    y = a · e^(bx)

    LINEARIZACIJA (slajd 114):
    --------------------------
    Logaritmovanjem obje strane:
        ln(y) = ln(a) + bx

    Supstitucija:
        Y = ln(y)
        A = ln(a)
        X = x
        B = b

    Dobijamo linearni model:
        Y = A + BX

    PROCEDURA:
    ----------
    1. Transformiši y → Y = ln(y)
    2. Primijeni linearnu regresiju na (X, Y) = (x, ln(y))
    3. Izračunaj: a = e^A, b = B

    PRIMJER SA SLAJDA 115-117:
    --------------------------
    Za podatke x = [0,1,2,3,4], y = [3,6,12,24,48]:
    Y = ln(y) = [1.0986, 1.7918, 2.4849, 3.1781, 3.8712]
    Linearna regresija daje: Y = 1.09861 + 0.69315·x
    Transformacija: a = e^1.09861 = 3, b = 0.69315
    Rezultat: y = 3·e^(0.69315x) = 3·2^x

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla (mora biti > 0)

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    # Provjera pozitivnosti
    if np.any(y <= 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'Sve y vrijednosti moraju biti pozitivne za eksponencijalnu aproksimaciju!',
            'steps': []
        }

    # Korak 1: Linearizacija - transformacija Y = ln(y)
    Y = np.log(y)

    # Prikaz tabele transformacije kao na slajdu 116
    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'y': y[i],
            'Y=ln(y)': Y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Linearizacija - logaritmovanje (slajd 114-116)',
        'model_original': 'y = a·e^(bx)',
        'linearization': 'ln(y) = ln(a) + bx',
        'substitution': {
            'Y': 'ln(y)',
            'A': 'ln(a)',
            'X': 'x',
            'B': 'b'
        },
        'model_linearized': 'Y = A + BX',
        'transform_table': transform_table,
        'description': 'Transformišemo eksponencijalni model u linearni primjenom prirodnog logaritma'
    })

    # Korak 2: Linearna regresija na transformisanim podacima (X, Y) = (x, ln(y))
    # Računamo sume
    sum_X = np.sum(x)
    sum_Y = np.sum(Y)
    sum_XY = np.sum(x * Y)
    sum_X2 = np.sum(x ** 2)

    # Sistem jednačina: A·N + B·ΣX = ΣY i A·ΣX + B·ΣX² = ΣXY
    denominator = N * sum_X2 - sum_X ** 2
    B = (N * sum_XY - sum_X * sum_Y) / denominator
    A = (sum_Y - B * sum_X) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'description': 'Primjenjujemo metodu najmanjih kvadrata na (X, Y) = (x, ln(y))',
        'sums': {
            'N': N,
            'ΣX': f'{sum_X:.6f}',
            'ΣY': f'{sum_Y:.6f}',
            'ΣXY': f'{sum_XY:.6f}',
            'ΣX²': f'{sum_X2:.6f}'
        },
        'system': [
            f'A·{N} + B·{sum_X:.4f} = {sum_Y:.4f}',
            f'A·{sum_X:.4f} + B·{sum_X2:.4f} = {sum_XY:.4f}'
        ],
        'solution': {
            'A': A,
            'B': B
        },
        'linear_equation': f'Y = {A:.6f} + {B:.6f}·X'
    })

    # Korak 3: Povratak na originalne parametre (slajd 117)
    a = np.exp(A)
    b = B

    steps.append({
        'step': 3,
        'title': 'Transformacija nazad u originalne parametre (slajd 117)',
        'description': 'Iz linearnih koeficijenata računamo originalne parametre',
        'formulas': {
            'a': 'a = e^A',
            'b': 'b = B'
        },
        'calculations': {
            'a': f'a = e^{A:.6f} = {a:.6f}',
            'b': f'b = {B:.6f}'
        },
        'final_equation': f'y = {a:.6f}·e^({b:.6f}·x)'
    })

    # Predviđene vrijednosti
    y_pred = a * np.exp(b * x)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    # R² za originalne podatke
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # R² za linearizirane podatke
    Y_pred = A + B * x
    ss_res_lin = np.sum((Y - Y_pred) ** 2)
    ss_tot_lin = np.sum((Y - np.mean(Y)) ** 2)
    r_squared_linear = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'interpretation': interpret_r_squared(r_squared)
    })

    # Alternativni oblik y = a·c^x gdje c = e^b
    c = np.exp(b)

    return {
        'a': a,
        'b': b,
        'equation': f'y = {a:.6f}·e^({b:.6f}·x)',
        'equation_alternative': f'y = {a:.6f}·{c:.6f}^x',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Eksponencijalna aproksimacija (linearizacija)'
    }


def power_regression(x_data: np.ndarray,
                     y_data: np.ndarray) -> Dict:
    """
    Stepena Aproksimacija (Power Regression)
    ========================================

    Prema slajdovima sa predavanja (str. 113)

    MODEL:
    ------
    y = a · x^b

    LINEARIZACIJA (slajd 113):
    --------------------------
    Logaritmovanjem obje strane:
        ln(y) = ln(a) + b·ln(x)

    Supstitucija:
        Y = ln(y)
        A = ln(a)
        X = ln(x)
        B = b

    Linearni model:
        Y = A + BX

    PROCEDURA:
    ----------
    1. Transformiši: X = ln(x), Y = ln(y)
    2. Primijeni linearnu regresiju na (X, Y)
    3. Izračunaj: a = e^A, b = B

    Args:
        x_data: Nezavisna varijabla (mora biti > 0)
        y_data: Zavisna varijabla (mora biti > 0)

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    if np.any(x <= 0) or np.any(y <= 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'Sve x i y vrijednosti moraju biti pozitivne za stepenu aproksimaciju!',
            'steps': []
        }

    # Korak 1: Linearizacija - transformacija X = ln(x), Y = ln(y)
    X = np.log(x)
    Y = np.log(y)

    # Prikaz tabele transformacije
    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'y': y[i],
            'X=ln(x)': X[i],
            'Y=ln(y)': Y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Linearizacija - logaritmovanje obje varijable (slajd 113)',
        'model_original': 'y = a·x^b',
        'linearization': 'ln(y) = ln(a) + b·ln(x)',
        'substitution': {
            'Y': 'ln(y)',
            'A': 'ln(a)',
            'X': 'ln(x)',
            'B': 'b'
        },
        'model_linearized': 'Y = A + BX',
        'transform_table': transform_table,
        'description': 'Transformišemo stepeni model u linearni primjenom prirodnog logaritma na obje varijable'
    })

    # Korak 2: Linearna regresija na transformisanim podacima
    sum_X = np.sum(X)
    sum_Y = np.sum(Y)
    sum_XY = np.sum(X * Y)
    sum_X2 = np.sum(X ** 2)

    denominator = N * sum_X2 - sum_X ** 2
    B = (N * sum_XY - sum_X * sum_Y) / denominator
    A = (sum_Y - B * sum_X) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'description': 'Primjenjujemo metodu najmanjih kvadrata na (X, Y) = (ln(x), ln(y))',
        'sums': {
            'N': N,
            'ΣX': f'{sum_X:.6f}',
            'ΣY': f'{sum_Y:.6f}',
            'ΣXY': f'{sum_XY:.6f}',
            'ΣX²': f'{sum_X2:.6f}'
        },
        'system': [
            f'A·{N} + B·{sum_X:.4f} = {sum_Y:.4f}',
            f'A·{sum_X:.4f} + B·{sum_X2:.4f} = {sum_XY:.4f}'
        ],
        'solution': {
            'A': A,
            'B': B
        },
        'linear_equation': f'Y = {A:.6f} + {B:.6f}·X'
    })

    # Korak 3: Povratak na originalne parametre
    a = np.exp(A)
    b = B

    steps.append({
        'step': 3,
        'title': 'Transformacija nazad u originalne parametre',
        'formulas': {
            'a': 'a = e^A',
            'b': 'b = B'
        },
        'calculations': {
            'a': f'a = e^{A:.6f} = {a:.6f}',
            'b': f'b = {B:.6f}'
        },
        'final_equation': f'y = {a:.6f}·x^{b:.6f}'
    })

    # Predviđene vrijednosti
    y_pred = a * (x ** b)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # R² za linearizirane podatke
    Y_pred = A + B * X
    ss_res_lin = np.sum((Y - Y_pred) ** 2)
    ss_tot_lin = np.sum((Y - np.mean(Y)) ** 2)
    r_squared_linear = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,
        'b': b,
        'equation': f'y = {a:.6f}·x^{b:.6f}',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Stepena aproksimacija (linearizacija)'
    }


def polynomial_regression(x_data: np.ndarray,
                          y_data: np.ndarray,
                          degree: int = 2) -> Dict:
    """
    Polinomijalna Regresija
    =======================

    Prema slajdovima sa predavanja (str. 105-112)

    MODEL:
    ------
    y = a₀ + a₁x + a₂x² + ... + aₙxⁿ

    Za stepen 1 (linearna): y = a + bx
    Za stepen 2 (kvadratna): y = a + bx + cx²

    METODA NAJMANJIH KVADRATA:
    --------------------------
    Cilj: Minimizirati S = Σ(Yᵢ - yᵢ)²

    Parcijalni izvodi ∂S/∂aₖ = 0 daju sistem normalnih jednačina.

    SISTEM NORMALNIH JEDNAČINA (slajd 107):
    ---------------------------------------
    Za polinom stepena n sa koeficijentima a₀, a₁, ..., aₙ:

    a₀·N + a₁·Σxᵢ + ... + aₙ·Σxᵢⁿ = ΣYᵢ
    a₀·Σxᵢ + a₁·Σxᵢ² + ... + aₙ·Σxᵢⁿ⁺¹ = ΣxᵢYᵢ
    ...
    a₀·Σxᵢⁿ + a₁·Σxᵢⁿ⁺¹ + ... + aₙ·Σxᵢ²ⁿ = ΣxᵢⁿYᵢ

    PRIMJER ZA STEPEN 2 (slajd 111-112):
    ------------------------------------
    Model: Cp = a + bT + cT²

    Sistem jednačina:
    5a + b·Σxᵢ + c·Σxᵢ² = ΣYᵢ
    a·Σxᵢ + b·Σxᵢ² + c·Σxᵢ³ = ΣxᵢYᵢ
    a·Σxᵢ² + b·Σxᵢ³ + c·Σxᵢ⁴ = Σxᵢ²Yᵢ

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla
        degree: Stepen polinoma (1=linearna, 2=kvadratna, ...)

    Returns:
        Dict sa koeficijentima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    # Upozorenje za overfitting
    if degree >= N - 1:
        steps.append({
            'step': 0,
            'title': 'UPOZORENJE',
            'message': f'Stepen polinoma ({degree}) je previsok za {N} tačaka. Rizik od overfittinga!'
        })

    # Korak 1: Računanje suma potencija x i proizvoda
    # Potrebne sume: Σxⁱ za i = 0, 1, ..., 2n i Σxⁱy za i = 0, 1, ..., n
    sums_x = {}
    sums_xy = {}

    for i in range(2 * degree + 1):
        sums_x[f'Σx^{i}'] = np.sum(x ** i)

    for i in range(degree + 1):
        sums_xy[f'Σx^{i}·Y'] = np.sum((x ** i) * y)

    steps.append({
        'step': 1,
        'title': 'Računanje potrebnih suma',
        'description': f'Za polinom stepena {degree} računamo sume potencija x i proizvode sa Y',
        'N': N,
        'sums_x': sums_x,
        'sums_xy': sums_xy
    })

    # Korak 2: Formiranje sistema normalnih jednačina
    # Matrica koeficijenata A i vektor desne strane B
    A_matrix = np.zeros((degree + 1, degree + 1))
    B_vector = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A_matrix[i, j] = np.sum(x ** (i + j))
        B_vector[i] = np.sum((x ** i) * y)

    # Formatiranje sistema za prikaz
    system_equations = []
    coef_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][:degree + 1]

    for i in range(degree + 1):
        eq_parts = []
        for j in range(degree + 1):
            coef_val = A_matrix[i, j]
            if j == 0:
                eq_parts.append(f'{coef_names[j]}·{coef_val:.4f}')
            else:
                eq_parts.append(f'{coef_names[j]}·{coef_val:.4f}')
        system_equations.append(' + '.join(eq_parts) + f' = {B_vector[i]:.4f}')

    steps.append({
        'step': 2,
        'title': 'Sistem normalnih jednačina (slajd 107, 112)',
        'description': f'Formiramo sistem od {degree + 1} linearnih jednačina',
        'system': system_equations,
        'matrix_A': A_matrix.tolist(),
        'vector_B': B_vector.tolist()
    })

    # Korak 3: Rješavanje sistema
    try:
        coefficients = np.linalg.solve(A_matrix, B_vector)
    except np.linalg.LinAlgError:
        return {
            'coefficients': None,
            'error_message': 'Sistem je singularan - nije moguće riješiti.',
            'steps': steps
        }

    # Formatiranje rezultata
    coef_dict = {}
    for i, name in enumerate(coef_names):
        coef_dict[name] = coefficients[i]

    steps.append({
        'step': 3,
        'title': 'Rješenje sistema - koeficijenti polinoma',
        'coefficients': coef_dict,
        'equation': format_polynomial_professor(coefficients)
    })

    # Korak 4: Predviđene vrijednosti i R²
    y_pred = np.zeros(N)
    for i, c in enumerate(coefficients):
        y_pred += c * (x ** i)

    residuals = y - y_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # Adjusted R²
    if N > degree + 1:
        r_squared_adj = 1 - (1 - r_squared) * (N - 1) / (N - degree - 1)
    else:
        r_squared_adj = r_squared

    # Računanje greške za svaku tačku (kao na slajdu 109-112)
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja kao na slajdovima 109-112
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške (slajd 109-112)',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_adjusted': r_squared_adj,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'coefficients': coefficients.tolist(),
        'coefficients_named': coef_dict,
        'degree': degree,
        'equation': format_polynomial_professor(coefficients),
        'r_squared': r_squared,
        'r_squared_adjusted': r_squared_adj,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': f'Polinomijalna regresija (stepen {degree})'
    }


def format_polynomial_professor(coeffs: np.ndarray) -> str:
    """
    Formatira polinom u stilu profesora: y = a + bx + cx² + ...
    """
    coef_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    terms = []

    for i, c in enumerate(coeffs):
        if abs(c) < 1e-10:
            continue

        if i < len(coef_names):
            name = coef_names[i]
        else:
            name = f'a_{i}'

        if i == 0:
            terms.append(f'{c:.6f}')
        elif i == 1:
            if c >= 0 and terms:
                terms.append(f' + {c:.6f}·x')
            elif c < 0:
                terms.append(f' - {abs(c):.6f}·x')
            else:
                terms.append(f'{c:.6f}·x')
        else:
            if c >= 0 and terms:
                terms.append(f' + {c:.6f}·x^{i}')
            elif c < 0:
                terms.append(f' - {abs(c):.6f}·x^{i}')
            else:
                terms.append(f'{c:.6f}·x^{i}')

    return 'y = ' + ''.join(terms) if terms else 'y = 0'


def format_polynomial(coeffs: np.ndarray) -> str:
    """Formatira polinom za prikaz (stari format za kompatibilnost)"""
    return format_polynomial_professor(coeffs)


def logarithmic_regression(x_data: np.ndarray,
                           y_data: np.ndarray) -> Dict:
    """
    Logaritamska Aproksimacija
    ==========================

    MODEL:
    ------
    y = a + b·ln(x)

    LINEARIZACIJA:
    --------------
    Model je već u linearnom obliku!
    Supstitucija: X = ln(x)
    Linearni model: y = a + b·X

    PROCEDURA:
    ----------
    1. Transformiši: X = ln(x)
    2. Primijeni linearnu regresiju na (X, y)
    3. Koeficijenti su direktno a i b

    Args:
        x_data: Nezavisna varijabla (mora biti > 0)
        y_data: Zavisna varijabla

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    if np.any(x <= 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'Sve x vrijednosti moraju biti pozitivne za logaritamsku aproksimaciju!',
            'steps': []
        }

    # Korak 1: Transformacija X = ln(x)
    X = np.log(x)

    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'X=ln(x)': X[i],
            'y': y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Transformacija x varijable',
        'model_original': 'y = a + b·ln(x)',
        'substitution': 'X = ln(x)',
        'model_linearized': 'y = a + b·X (već linearan oblik)',
        'transform_table': transform_table
    })

    # Korak 2: Linearna regresija na (X, y)
    sum_X = np.sum(X)
    sum_Y = np.sum(y)
    sum_XY = np.sum(X * y)
    sum_X2 = np.sum(X ** 2)

    denominator = N * sum_X2 - sum_X ** 2
    b = (N * sum_XY - sum_X * sum_Y) / denominator
    a = (sum_Y - b * sum_X) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'sums': {
            'N': N,
            'ΣX': f'{sum_X:.6f}',
            'Σy': f'{sum_Y:.6f}',
            'ΣXy': f'{sum_XY:.6f}',
            'ΣX²': f'{sum_X2:.6f}'
        },
        'system': [
            f'a·{N} + b·{sum_X:.4f} = {sum_Y:.4f}',
            f'a·{sum_X:.4f} + b·{sum_X2:.4f} = {sum_XY:.4f}'
        ],
        'solution': {'a': a, 'b': b},
        'equation': f'y = {a:.6f} + {b:.6f}·X'
    })

    steps.append({
        'step': 3,
        'title': 'Konačni rezultat',
        'final_equation': f'y = {a:.6f} + {b:.6f}·ln(x)',
        'description': 'Koeficijenti su direktno iz linearne regresije (bez dodatne transformacije)'
    })

    # Predviđene vrijednosti
    y_pred = a + b * np.log(x)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,
        'b': b,
        'equation': f'y = {a:.6f} + {b:.6f}·ln(x)',
        'r_squared': r_squared,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Logaritamska aproksimacija'
    }


def hyperbolic_regression(x_data: np.ndarray,
                          y_data: np.ndarray) -> Dict:
    """
    Hiperbolička Aproksimacija
    ==========================

    MODEL:
    ------
    y = 1/(a + bx)

    LINEARIZACIJA:
    --------------
    1/y = a + bx

    Supstitucija: Y = 1/y
    Linearni model: Y = a + bx

    PROCEDURA:
    ----------
    1. Transformiši: Y = 1/y
    2. Primijeni linearnu regresiju na (x, Y)
    3. Koeficijenti a i b su direktno iz regresije

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla (mora biti != 0)

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    if np.any(y == 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'y vrijednosti ne smiju biti nula za hiperboličku aproksimaciju!',
            'steps': []
        }

    # Korak 1: Transformacija Y = 1/y
    Y = 1 / y

    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'y': y[i],
            'Y=1/y': Y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Linearizacija - recipročna transformacija',
        'model_original': 'y = 1/(a + bx)',
        'linearization': '1/y = a + bx',
        'substitution': 'Y = 1/y',
        'model_linearized': 'Y = a + bx',
        'transform_table': transform_table
    })

    # Korak 2: Linearna regresija na (x, Y)
    sum_x = np.sum(x)
    sum_Y = np.sum(Y)
    sum_xY = np.sum(x * Y)
    sum_x2 = np.sum(x ** 2)

    denominator = N * sum_x2 - sum_x ** 2
    b = (N * sum_xY - sum_x * sum_Y) / denominator
    a = (sum_Y - b * sum_x) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'sums': {
            'N': N,
            'Σx': f'{sum_x:.6f}',
            'ΣY': f'{sum_Y:.6f}',
            'ΣxY': f'{sum_xY:.6f}',
            'Σx²': f'{sum_x2:.6f}'
        },
        'system': [
            f'a·{N} + b·{sum_x:.4f} = {sum_Y:.4f}',
            f'a·{sum_x:.4f} + b·{sum_x2:.4f} = {sum_xY:.4f}'
        ],
        'solution': {'a': a, 'b': b},
        'equation': f'Y = {a:.6f} + {b:.6f}·x'
    })

    steps.append({
        'step': 3,
        'title': 'Konačni rezultat',
        'final_equation': f'y = 1/({a:.6f} + {b:.6f}·x)',
        'description': 'Parametri a i b su direktno iz linearne regresije'
    })

    # Predviđene vrijednosti
    y_pred = 1 / (a + b * x)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # R² za linearizirane podatke
    Y_pred = a + b * x
    ss_res_lin = np.sum((Y - Y_pred) ** 2)
    ss_tot_lin = np.sum((Y - np.mean(Y)) ** 2)
    r_squared_linear = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,
        'b': b,
        'equation': f'y = 1/({a:.6f} + {b:.6f}·x)',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Hiperbolička aproksimacija'
    }


def rational_regression(x_data: np.ndarray,
                        y_data: np.ndarray) -> Dict:
    """
    Racionalna Aproksimacija
    ========================

    MODEL:
    ------
    y = x/(a + bx)

    LINEARIZACIJA:
    --------------
    x/y = a + bx

    Supstitucija: Y = x/y
    Linearni model: Y = a + bx

    PROCEDURA:
    ----------
    1. Transformiši: Y = x/y
    2. Primijeni linearnu regresiju na (x, Y)
    3. Koeficijenti a i b su direktno iz regresije

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla (mora biti != 0)

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    if np.any(y == 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'y vrijednosti ne smiju biti nula za racionalnu aproksimaciju!',
            'steps': []
        }

    # Korak 1: Transformacija Y = x/y
    Y = x / y

    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'y': y[i],
            'Y=x/y': Y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Linearizacija - transformacija x/y',
        'model_original': 'y = x/(a + bx)',
        'linearization': 'x/y = a + bx',
        'substitution': 'Y = x/y',
        'model_linearized': 'Y = a + bx',
        'transform_table': transform_table
    })

    # Korak 2: Linearna regresija na (x, Y)
    sum_x = np.sum(x)
    sum_Y = np.sum(Y)
    sum_xY = np.sum(x * Y)
    sum_x2 = np.sum(x ** 2)

    denominator = N * sum_x2 - sum_x ** 2
    b = (N * sum_xY - sum_x * sum_Y) / denominator
    a = (sum_Y - b * sum_x) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'sums': {
            'N': N,
            'Σx': f'{sum_x:.6f}',
            'ΣY': f'{sum_Y:.6f}',
            'ΣxY': f'{sum_xY:.6f}',
            'Σx²': f'{sum_x2:.6f}'
        },
        'system': [
            f'a·{N} + b·{sum_x:.4f} = {sum_Y:.4f}',
            f'a·{sum_x:.4f} + b·{sum_x2:.4f} = {sum_xY:.4f}'
        ],
        'solution': {'a': a, 'b': b},
        'equation': f'Y = {a:.6f} + {b:.6f}·x'
    })

    steps.append({
        'step': 3,
        'title': 'Konačni rezultat',
        'final_equation': f'y = x/({a:.6f} + {b:.6f}·x)',
        'description': 'Parametri a i b su direktno iz linearne regresije'
    })

    # Predviđene vrijednosti
    y_pred = x / (a + b * x)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    # R² za linearizirane podatke
    Y_pred = a + b * x
    ss_res_lin = np.sum((Y - Y_pred) ** 2)
    ss_tot_lin = np.sum((Y - np.mean(Y)) ** 2)
    r_squared_linear = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,
        'b': b,
        'equation': f'y = x/({a:.6f} + {b:.6f}·x)',
        'r_squared': r_squared,
        'r_squared_linear': r_squared_linear,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Racionalna aproksimacija'
    }


def sqrt_regression(x_data: np.ndarray,
                    y_data: np.ndarray) -> Dict:
    """
    Aproksimacija Kvadratnim Korijenom
    ===================================

    MODEL:
    ------
    y = a + b·√x

    LINEARIZACIJA:
    --------------
    Model je već u linearnom obliku!
    Supstitucija: X = √x
    Linearni model: y = a + b·X

    PROCEDURA:
    ----------
    1. Transformiši: X = √x
    2. Primijeni linearnu regresiju na (X, y)
    3. Koeficijenti su direktno a i b

    Args:
        x_data: Nezavisna varijabla (mora biti >= 0)
        y_data: Zavisna varijabla

    Returns:
        Dict sa parametrima i koracima
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    N = len(x)

    steps = []

    if np.any(x < 0):
        return {
            'a': None,
            'b': None,
            'error_message': 'x vrijednosti moraju biti >= 0 za aproksimaciju kvadratnim korijenom!',
            'steps': []
        }

    # Korak 1: Transformacija X = √x
    X = np.sqrt(x)

    transform_table = []
    for i in range(N):
        transform_table.append({
            'x': x[i],
            'X=√x': X[i],
            'y': y[i]
        })

    steps.append({
        'step': 1,
        'title': 'Transformacija x varijable',
        'model_original': 'y = a + b·√x',
        'substitution': 'X = √x',
        'model_linearized': 'y = a + b·X (već linearan oblik)',
        'transform_table': transform_table
    })

    # Korak 2: Linearna regresija na (X, y)
    sum_X = np.sum(X)
    sum_Y = np.sum(y)
    sum_XY = np.sum(X * y)
    sum_X2 = np.sum(X ** 2)

    denominator = N * sum_X2 - sum_X ** 2
    b = (N * sum_XY - sum_X * sum_Y) / denominator
    a = (sum_Y - b * sum_X) / N

    steps.append({
        'step': 2,
        'title': 'Linearna regresija na transformisanim podacima',
        'sums': {
            'N': N,
            'ΣX': f'{sum_X:.6f}',
            'Σy': f'{sum_Y:.6f}',
            'ΣXy': f'{sum_XY:.6f}',
            'ΣX²': f'{sum_X2:.6f}'
        },
        'system': [
            f'a·{N} + b·{sum_X:.4f} = {sum_Y:.4f}',
            f'a·{sum_X:.4f} + b·{sum_X2:.4f} = {sum_XY:.4f}'
        ],
        'solution': {'a': a, 'b': b},
        'equation': f'y = {a:.6f} + {b:.6f}·X'
    })

    steps.append({
        'step': 3,
        'title': 'Konačni rezultat',
        'final_equation': f'y = {a:.6f} + {b:.6f}·√x',
        'description': 'Koeficijenti su direktno iz linearne regresije'
    })

    # Predviđene vrijednosti
    y_pred = a + b * np.sqrt(x)
    residuals = y - y_pred

    # Računanje greške u procentima
    errors_percent = []
    for i in range(N):
        if y[i] != 0:
            err = ((y_pred[i] - y[i]) / y[i]) * 100
        else:
            err = 0
        errors_percent.append(err)

    # Tabela poređenja
    comparison_table = []
    for i in range(N):
        comparison_table.append({
            'x': x[i],
            'Y (izmjereno)': y[i],
            'y (aproksimacija)': y_pred[i],
            'Greška (%)': errors_percent[i]
        })

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    steps.append({
        'step': 4,
        'title': 'Predviđene vrijednosti i greške',
        'description': 'Greška (%) = (y_pred - Y) / Y × 100',
        'comparison_table': comparison_table,
        'errors_percent': errors_percent
    })

    steps.append({
        'step': 5,
        'title': 'Ocjena kvalitete modela',
        'r_squared': r_squared,
        'interpretation': interpret_r_squared(r_squared)
    })

    return {
        'a': a,
        'b': b,
        'equation': f'y = {a:.6f} + {b:.6f}·√x',
        'r_squared': r_squared,
        'y_predicted': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'errors_percent': errors_percent,
        'steps': steps,
        'method': 'Aproksimacija kvadratnim korijenom'
    }


def compare_regression_models(x_data: np.ndarray,
                              y_data: np.ndarray,
                              include_polynomial: bool = True) -> Dict:
    """
    Poređenje različitih modela aproksimacije

    Uspoređuje sve dostupne metode aproksimacije i rangira ih
    prema R² vrijednosti (koeficijentu determinacije).

    Args:
        x_data: Nezavisna varijabla
        y_data: Zavisna varijabla
        include_polynomial: Da li uključiti polinomijalne modele

    Returns:
        Dict sa svim modelima i poređenjem
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    results = {}

    # 1. Linearna aproksimacija (uvijek dostupna)
    results['Linearna (y=a+bx)'] = linear_regression(x, y)

    # 2. Polinomijalna aproksimacija
    if include_polynomial:
        results['Kvadratna (y=a+bx+cx²)'] = polynomial_regression(x, y, degree=2)
        if len(x) > 4:
            results['Kubna (y=a+bx+cx²+dx³)'] = polynomial_regression(x, y, degree=3)

    # 3. Eksponencijalna (zahtijeva y > 0)
    if np.all(y > 0):
        results['Eksponencijalna (y=a·e^(bx))'] = exponential_regression(x, y)

    # 4. Stepena (zahtijeva x > 0 i y > 0)
    if np.all(x > 0) and np.all(y > 0):
        results['Stepena (y=a·x^b)'] = power_regression(x, y)

    # 5. Logaritamska (zahtijeva x > 0)
    if np.all(x > 0):
        results['Logaritamska (y=a+b·ln(x))'] = logarithmic_regression(x, y)

    # 6. Hiperbolička (zahtijeva y != 0)
    if not np.any(y == 0):
        results['Hiperbolička (y=1/(a+bx))'] = hyperbolic_regression(x, y)

    # 7. Racionalna (zahtijeva y != 0)
    if not np.any(y == 0):
        results['Racionalna (y=x/(a+bx))'] = rational_regression(x, y)

    # 8. Kvadratni korijen (zahtijeva x >= 0)
    if np.all(x >= 0):
        results['Korijen (y=a+b·√x)'] = sqrt_regression(x, y)

    # Sumarno poređenje - filtriranje uspješnih modela
    summary = []
    for name, result in results.items():
        if 'r_squared' in result and 'error_message' not in result:
            summary.append({
                'model': name,
                'r_squared': result['r_squared'],
                'equation': result.get('equation', 'N/A')
            })

    # Sortiranje po R² (od najboljeg)
    summary.sort(key=lambda x: x['r_squared'], reverse=True)
    results['summary'] = summary

    # Preporuka najboljeg modela
    if summary:
        best = summary[0]
        results['recommendation'] = {
            'best_model': best['model'],
            'r_squared': best['r_squared'],
            'equation': best['equation'],
            'interpretation': interpret_r_squared(best['r_squared'])
        }

    return results


def get_all_approximation_methods() -> List[Dict]:
    """
    Vraća listu svih dostupnih metoda aproksimacije sa opisima.

    Returns:
        Lista rječnika sa informacijama o metodama
    """
    return [
        {
            'name': 'Linearna aproksimacija',
            'model': 'y = a + bx',
            'function': 'linear_regression',
            'linearization': 'Direktna primjena (već linearan)',
            'requirements': 'Nema posebnih zahtjeva',
            'description': 'Metoda najmanjih kvadrata za linearni model',
            'slides': '105-110'
        },
        {
            'name': 'Stepena aproksimacija',
            'model': 'y = a·x^b',
            'function': 'power_regression',
            'linearization': 'ln(y) = ln(a) + b·ln(x)',
            'requirements': 'x > 0, y > 0',
            'description': 'Koristi se za zakone proporcionalnosti',
            'slides': '113'
        },
        {
            'name': 'Eksponencijalna aproksimacija',
            'model': 'y = a·e^(bx)',
            'function': 'exponential_regression',
            'linearization': 'ln(y) = ln(a) + bx',
            'requirements': 'y > 0',
            'description': 'Eksponencijalni rast/opadanje',
            'slides': '114-117'
        },
        {
            'name': 'Logaritamska aproksimacija',
            'model': 'y = a + b·ln(x)',
            'function': 'logarithmic_regression',
            'linearization': 'Y = a + bX, X = ln(x)',
            'requirements': 'x > 0',
            'description': 'Logaritamske veze i saturacija'
        },
        {
            'name': 'Hiperbolička aproksimacija',
            'model': 'y = 1/(a + bx)',
            'function': 'hyperbolic_regression',
            'linearization': '1/y = a + bx',
            'requirements': 'y ≠ 0',
            'description': 'Michaelis-Menten i slične kinetike'
        },
        {
            'name': 'Racionalna aproksimacija',
            'model': 'y = x/(a + bx)',
            'function': 'rational_regression',
            'linearization': 'x/y = a + bx',
            'requirements': 'y ≠ 0',
            'description': 'Langmuir adsorpcija'
        },
        {
            'name': 'Aproksimacija kvadratnim korijenom',
            'model': 'y = a + b·√x',
            'function': 'sqrt_regression',
            'linearization': 'Y = a + bX, X = √x',
            'requirements': 'x ≥ 0',
            'description': 'Procesi sa korijenom'
        },
        {
            'name': 'Polinomijalna aproksimacija',
            'model': 'y = a + bx + cx² + ... ',
            'function': 'polynomial_regression',
            'linearization': 'Sistem normalnih jednačina',
            'requirements': 'stepen < broj tačaka - 1',
            'description': 'Opšta polinomijalna aproksimacija',
            'slides': '105-112'
        }
    ]
