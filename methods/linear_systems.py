"""
Iterativne Metode za Sisteme Linearnih Jednačina
================================================

Ovaj modul implementira iterativne metode za rješavanje sistema Ax = b:

1. Jacobijeva Metoda - Rađeno na nastavi
2. Gauss-Seidelova Metoda - Rađeno na nastavi

Obje metode vraćaju detaljne korake za step-by-step prikaz.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def check_diagonal_dominance(A: np.ndarray) -> Dict:
    """
    Provjera Dijagonalne Dominantnosti
    ==================================

    DEFINICIJA:
    -----------
    Matrica A je strogo dijagonalno dominantna ako za svaki red i važi:
        |a_{ii}| > Σ_{j≠i} |a_{ij}|

    tj. apsolutna vrijednost dijagonalnog elementa je veća od sume
    apsolutnih vrijednosti svih ostalih elemenata u tom redu.

    ZNAČAJ:
    -------
    Ako je matrica strogo dijagonalno dominantna, Jacobijeva i
    Gauss-Seidelova metoda GARANTOVANO konvergiraju.

    Args:
        A: Matrica koeficijenata

    Returns:
        Dict sa rezultatom provjere
    """
    n = A.shape[0]
    is_dominant = True
    details = []

    for i in range(n):
        diagonal = abs(A[i, i])
        off_diagonal_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        row_dominant = diagonal > off_diagonal_sum

        details.append({
            'row': i + 1,
            'diagonal': diagonal,
            'off_diagonal_sum': off_diagonal_sum,
            'is_dominant': row_dominant,
            'description': f'|a_{{{i+1}{i+1}}}| = {diagonal:.4f} {">" if row_dominant else "≤"} {off_diagonal_sum:.4f} = Σ|a_{{{i+1}j}}|'
        })

        if not row_dominant:
            is_dominant = False

    return {
        'is_diagonally_dominant': is_dominant,
        'details': details,
        'conclusion': 'Matrica JE dijagonalno dominantna - konvergencija garantovana!' if is_dominant
        else 'Matrica NIJE dijagonalno dominantna - konvergencija nije garantovana (ali metoda može i dalje konvergirati)'
    }


def jacobi(A: np.ndarray,
           b: np.ndarray,
           x0: np.ndarray = None,
           tol: float = 1e-6,
           max_iter: int = 100) -> Dict:
    """
    Jacobijeva Metoda
    =================

    TEORIJA:
    --------
    Jacobijeva metoda je iterativna metoda za rješavanje sistema Ax = b.
    Razlaže matricu A na: A = D + L + U
    - D: dijagonalna matrica
    - L: strogo donja trougaona matrica
    - U: strogo gornja trougaona matrica

    IZVOD:
    ------
    Ax = b
    (D + L + U)x = b
    Dx = b - (L + U)x
    x = D⁻¹(b - (L + U)x)

    ITERATIVNA FORMULA:
    -------------------
    x^{(k+1)} = D⁻¹(b - (L + U)x^{(k)})

    ili za komponentu i:
        x_i^{(k+1)} = (b_i - Σ_{j≠i} a_{ij}·x_j^{(k)}) / a_{ii}

    KARAKTERISTIKA:
    ---------------
    U Jacobijevoj metodi, za računanje x^{(k+1)} koristimo SAMO
    vrijednosti iz prethodne iteracije x^{(k)}.

    KONVERGENCIJA:
    --------------
    - Garantovana za strogo dijagonalno dominantne matrice
    - Garantovana za simetrične pozitivno definitne matrice
    - Sporija od Gauss-Seidel metode

    Args:
        A: Matrica koeficijenata (n×n)
        b: Vektor desne strane (n×1)
        x0: Početna aproksimacija (default: nula vektor)
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija

    Returns:
        Dict sa rješenjem i detaljnim koracima
    """
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if x0 is None:
        x0 = np.zeros(n)
    x = np.array(x0, dtype=float)

    steps = []

    # Provjera dijagonalne dominantnosti
    diag_check = check_diagonal_dominance(A)
    steps.append({
        'step': 0,
        'title': 'Provjera dijagonalne dominantnosti',
        'diagonal_check': diag_check
    })

    # Provjera da dijagonalni elementi nisu nula
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            return {
                'solution': None,
                'iterations': 0,
                'converged': False,
                'steps': steps,
                'error_message': f'Dijagonalni element a[{i+1},{i+1}] = 0. Potrebno preurediti sistem.'
            }

    # Početna iteracija
    steps.append({
        'step': 1,
        'title': 'Početna aproksimacija',
        'x': x.copy(),
        'description': f'x⁽⁰⁾ = {x}'
    })

    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)

        step_details = {
            'step': k + 1,
            'title': f'Iteracija {k}',
            'calculations': [],
            'x_old': x.copy()
        }

        # Jacobijeva formula za svaku komponentu
        for i in range(n):
            sum_term = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_term) / A[i, i]

            step_details['calculations'].append({
                'component': i + 1,
                'formula': f'x_{{{i+1}}}^{{({k})}} = (b_{{{i+1}}} - Σ_{{j≠{i+1}}} a_{{{i+1}j}}·x_j^{{({k-1})}}) / a_{{{i+1}{i+1}}}',
                'calculation': f'x_{{{i+1}}}^{{({k})}} = ({b[i]:.4f} - {sum_term:.4f}) / {A[i,i]:.4f} = {x_new[i]:.6f}'
            })

        step_details['x_new'] = x_new.copy()

        # Računanje greške (norma razlike)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        step_details['error'] = error

        steps.append(step_details)

        # Provjera konvergencije
        if error < tol:
            return {
                'solution': x_new,
                'iterations': k,
                'converged': True,
                'steps': steps,
                'error_message': None,
                'final_error': error
            }

        x = x_new.copy()

    return {
        'solution': x,
        'iterations': max_iter,
        'converged': False,
        'steps': steps,
        'error_message': f'Metoda nije konvergirala nakon {max_iter} iteracija.',
        'final_error': error
    }


def gauss_seidel(A: np.ndarray,
                 b: np.ndarray,
                 x0: np.ndarray = None,
                 tol: float = 1e-6,
                 max_iter: int = 100) -> Dict:
    """
    Gauss-Seidelova Metoda
    ======================

    TEORIJA:
    --------
    Gauss-Seidelova metoda je poboljšanje Jacobijeve metode.
    Ključna razlika: koristi već izračunate vrijednosti iz TRENUTNE
    iteracije čim postanu dostupne.

    IZVOD:
    ------
    Razlaganje: A = L* + U
    - L*: donja trougaona matrica (uključuje dijagonalu)
    - U: strogo gornja trougaona matrica

    L*x^{(k+1)} = b - Ux^{(k)}
    x^{(k+1)} = (L*)⁻¹(b - Ux^{(k)})

    ITERATIVNA FORMULA:
    -------------------
    Za komponentu i:
        x_i^{(k+1)} = (b_i - Σ_{j<i} a_{ij}·x_j^{(k+1)} - Σ_{j>i} a_{ij}·x_j^{(k)}) / a_{ii}
                            ↑ nove vrijednosti      ↑ stare vrijednosti

    RAZLIKA OD JACOBI:
    ------------------
    - Jacobi: koristi SAMO stare vrijednosti x^{(k)}
    - Gauss-Seidel: koristi nove x_j^{(k+1)} za j < i čim su izračunate

    PREDNOSTI:
    ----------
    - Obično konvergira brže od Jacobijeve metode
    - Zahtijeva manje memorije (može raditi "in-place")
    - Konvergira za širu klasu matrica

    KONVERGENCIJA:
    --------------
    - Garantovana za strogo dijagonalno dominantne matrice
    - Garantovana za simetrične pozitivno definitne matrice
    - Obično 2x brža od Jacobijeve metode

    Args:
        A: Matrica koeficijenata
        b: Vektor desne strane
        x0: Početna aproksimacija
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija

    Returns:
        Dict sa rješenjem i detaljnim koracima
    """
    n = len(b)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if x0 is None:
        x0 = np.zeros(n)
    x = np.array(x0, dtype=float)

    steps = []

    # Provjera dijagonalne dominantnosti
    diag_check = check_diagonal_dominance(A)
    steps.append({
        'step': 0,
        'title': 'Provjera dijagonalne dominantnosti',
        'diagonal_check': diag_check
    })

    # Provjera dijagonalnih elemenata
    for i in range(n):
        if abs(A[i, i]) < 1e-15:
            return {
                'solution': None,
                'iterations': 0,
                'converged': False,
                'steps': steps,
                'error_message': f'Dijagonalni element a[{i+1},{i+1}] = 0.'
            }

    # Početna iteracija
    steps.append({
        'step': 1,
        'title': 'Početna aproksimacija',
        'x': x.copy(),
        'description': f'x⁽⁰⁾ = {x}'
    })

    for k in range(1, max_iter + 1):
        x_old = x.copy()

        step_details = {
            'step': k + 1,
            'title': f'Iteracija {k}',
            'calculations': [],
            'x_old': x_old.copy()
        }

        # Gauss-Seidel formula za svaku komponentu
        for i in range(n):
            # Suma sa novim vrijednostima (j < i)
            sum_new = sum(A[i, j] * x[j] for j in range(i))
            # Suma sa starim vrijednostima (j > i)
            sum_old = sum(A[i, j] * x_old[j] for j in range(i + 1, n))

            x[i] = (b[i] - sum_new - sum_old) / A[i, i]

            step_details['calculations'].append({
                'component': i + 1,
                'formula': f'x_{{{i+1}}}^{{({k})}} = (b_{{{i+1}}} - Σ_{{j<{i+1}}} a_{{{i+1}j}}·x_j^{{({k})}} - Σ_{{j>{i+1}}} a_{{{i+1}j}}·x_j^{{({k-1})}}) / a_{{{i+1}{i+1}}}',
                'sum_new_terms': sum_new,
                'sum_old_terms': sum_old,
                'calculation': f'x_{{{i+1}}}^{{({k})}} = ({b[i]:.4f} - {sum_new:.4f} - {sum_old:.4f}) / {A[i,i]:.4f} = {x[i]:.6f}',
                'note': 'Koristi već izračunate vrijednosti iz ove iteracije!'
            })

        step_details['x_new'] = x.copy()

        # Računanje greške
        error = np.linalg.norm(x - x_old, ord=np.inf)
        step_details['error'] = error

        steps.append(step_details)

        # Provjera konvergencije
        if error < tol:
            return {
                'solution': x,
                'iterations': k,
                'converged': True,
                'steps': steps,
                'error_message': None,
                'final_error': error
            }

    return {
        'solution': x,
        'iterations': max_iter,
        'converged': False,
        'steps': steps,
        'error_message': f'Metoda nije konvergirala nakon {max_iter} iteracija.',
        'final_error': error
    }


def compare_methods(A: np.ndarray,
                    b: np.ndarray,
                    x0: np.ndarray = None,
                    tol: float = 1e-6,
                    max_iter: int = 100) -> Dict:
    """
    Poređenje Jacobijeve i Gauss-Seidelove metode

    Args:
        A: Matrica koeficijenata
        b: Vektor desne strane
        x0: Početna aproksimacija
        tol: Tolerancija
        max_iter: Maksimalan broj iteracija

    Returns:
        Dict sa rezultatima obje metode i poređenjem
    """
    jacobi_result = jacobi(A, b, x0, tol, max_iter)
    gs_result = gauss_seidel(A, b, x0, tol, max_iter)

    # Tačno rješenje (ako je moguće izračunati)
    try:
        exact = np.linalg.solve(A, b)
    except:
        exact = None

    comparison = {
        'jacobi': jacobi_result,
        'gauss_seidel': gs_result,
        'exact_solution': exact,
        'summary': {
            'jacobi_iterations': jacobi_result['iterations'],
            'gauss_seidel_iterations': gs_result['iterations'],
            'jacobi_converged': jacobi_result['converged'],
            'gauss_seidel_converged': gs_result['converged']
        }
    }

    if exact is not None and jacobi_result['converged'] and gs_result['converged']:
        comparison['summary']['jacobi_error'] = np.linalg.norm(jacobi_result['solution'] - exact)
        comparison['summary']['gauss_seidel_error'] = np.linalg.norm(gs_result['solution'] - exact)

    return comparison


def create_example_systems() -> Dict:
    """
    Kreiranje primjera sistema za demonstraciju

    Returns:
        Dict sa primjerima sistema
    """
    examples = {
        'example_1': {
            'name': 'Dijagonalno dominantan sistem (3×3)',
            'A': np.array([
                [10, -1, 2],
                [-1, 11, -1],
                [2, -1, 10]
            ]),
            'b': np.array([6, 25, -11]),
            'description': 'Klasičan primjer dijagonalno dominantnog sistema.'
        },
        'example_2': {
            'name': 'Sistem iz fizike (toplota)',
            'A': np.array([
                [4, -1, 0, -1],
                [-1, 4, -1, 0],
                [0, -1, 4, -1],
                [-1, 0, -1, 4]
            ]),
            'b': np.array([100, 0, 0, 100]),
            'description': 'Sistem koji nastaje pri diskretizaciji jednačine provođenja toplote.'
        },
        'example_3': {
            'name': 'Električna mreža (Kirchhoff)',
            'A': np.array([
                [3, -1, -1],
                [-1, 3, -1],
                [-1, -1, 3]
            ]),
            'b': np.array([1, 0, 0]),
            'description': 'Sistem koji nastaje primjenom Kirchhoffovih zakona na električnu mrežu.'
        }
    }

    return examples
