"""
Validatori Unosa
================

Funkcije za validaciju korisničkog unosa u aplikaciji.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


def validate_function(func_str: str) -> Tuple[bool, str, Optional[Callable]]:
    """
    Validira string reprezentaciju funkcije

    Args:
        func_str: String funkcije (npr. "x**2 - 4")

    Returns:
        Tuple (valid, message, function)
    """
    try:
        # Pokušaj kreirati lambda funkciju
        f = lambda x: eval(func_str)

        # Testiraj na nekoliko vrijednosti
        test_values = [0, 1, -1, 0.5]
        for val in test_values:
            try:
                result = f(val)
                if not np.isfinite(result):
                    continue  # Neki unosi mogu biti inf/nan za određene x
            except:
                pass

        return True, "Funkcija je validna", f

    except SyntaxError as e:
        return False, f"Sintaksna greška: {e}", None
    except NameError as e:
        return False, f"Nepoznato ime: {e}", None
    except Exception as e:
        return False, f"Greška: {e}", None


def validate_interval(a: float, b: float) -> Tuple[bool, str]:
    """
    Validira interval [a, b]

    Args:
        a: Lijeva granica
        b: Desna granica

    Returns:
        Tuple (valid, message)
    """
    if not np.isfinite(a) or not np.isfinite(b):
        return False, "Granice moraju biti konačni brojevi"

    if a >= b:
        return False, "Donja granica mora biti manja od gornje (a < b)"

    if b - a > 1e10:
        return False, "Interval je prevelik"

    return True, "Interval je validan"


def validate_tolerance(tol: float) -> Tuple[bool, str]:
    """
    Validira toleranciju

    Args:
        tol: Tolerancija

    Returns:
        Tuple (valid, message)
    """
    if not np.isfinite(tol):
        return False, "Tolerancija mora biti konačan broj"

    if tol <= 0:
        return False, "Tolerancija mora biti pozitivna"

    if tol > 1:
        return False, "Tolerancija je prevelika (preporučeno < 1)"

    if tol < 1e-15:
        return False, "Tolerancija je premala (preporučeno > 1e-15)"

    return True, "Tolerancija je validna"


def validate_matrix(A: np.ndarray) -> Tuple[bool, str]:
    """
    Validira matricu za sisteme jednačina

    Args:
        A: Matrica koeficijenata

    Returns:
        Tuple (valid, message)
    """
    if A.ndim != 2:
        return False, "Matrica mora biti 2D"

    n, m = A.shape
    if n != m:
        return False, "Matrica mora biti kvadratna"

    if n < 2:
        return False, "Matrica mora imati barem 2 reda"

    # Provjera singularnosti
    try:
        det = np.linalg.det(A)
        if abs(det) < 1e-15:
            return False, "Matrica je singularna (determinanta ≈ 0)"
    except:
        return False, "Greška pri računanju determinante"

    return True, "Matrica je validna"


def validate_data_points(x: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
    """
    Validira podatke za regresiju

    Args:
        x: X koordinate
        y: Y koordinate

    Returns:
        Tuple (valid, message)
    """
    if len(x) != len(y):
        return False, "X i Y moraju imati isti broj elemenata"

    if len(x) < 2:
        return False, "Potrebno je najmanje 2 tačke"

    if not np.all(np.isfinite(x)):
        return False, "Sve X vrijednosti moraju biti konačni brojevi"

    if not np.all(np.isfinite(y)):
        return False, "Sve Y vrijednosti moraju biti konačni brojevi"

    # Provjera duplikata u X
    if len(np.unique(x)) != len(x):
        return False, "X vrijednosti moraju biti jedinstvene"

    return True, "Podaci su validni"


def validate_positive_data(y: np.ndarray) -> Tuple[bool, str]:
    """
    Validira da su svi Y pozitivni (za eksponencijalnu regresiju)

    Args:
        y: Y vrijednosti

    Returns:
        Tuple (valid, message)
    """
    if np.any(y <= 0):
        return False, "Sve Y vrijednosti moraju biti pozitivne za eksponencijalnu aproksimaciju"

    return True, "Podaci su validni za eksponencijalnu aproksimaciju"


def validate_polynomial_degree(n: int, num_points: int) -> Tuple[bool, str]:
    """
    Validira stepen polinoma za regresiju

    Args:
        n: Stepen polinoma
        num_points: Broj tačaka podataka

    Returns:
        Tuple (valid, message)
    """
    if n < 1:
        return False, "Stepen polinoma mora biti najmanje 1"

    if n >= num_points:
        return False, f"Stepen polinoma ({n}) mora biti manji od broja tačaka ({num_points})"

    if n >= num_points - 1:
        return False, f"Upozorenje: Stepen polinoma ({n}) je blizu broja tačaka ({num_points}), rizik od overfittinga!"

    return True, "Stepen polinoma je validan"


def validate_integration_n(n: int, method: str = "simpson") -> Tuple[bool, str]:
    """
    Validira broj podintervala za integraciju

    Args:
        n: Broj podintervala
        method: Metoda integracije

    Returns:
        Tuple (valid, message)
    """
    if n < 1:
        return False, "Broj podintervala mora biti pozitivan"

    if method.lower() == "simpson" and n % 2 != 0:
        return False, "Za Simpsonovu metodu, n mora biti paran broj"

    if n > 10000:
        return False, "Broj podintervala je prevelik (max 10000)"

    return True, "Broj podintervala je validan"


def safe_eval_function(func_str: str, x_values: List[float]) -> Tuple[List[float], List[str]]:
    """
    Sigurno evaluira funkciju na listi vrijednosti

    Args:
        func_str: String funkcije
        x_values: Lista x vrijednosti

    Returns:
        Tuple (results, errors)
    """
    results = []
    errors = []

    try:
        f = lambda x: eval(func_str)

        for x in x_values:
            try:
                y = f(x)
                if np.isfinite(y):
                    results.append(y)
                    errors.append(None)
                else:
                    results.append(None)
                    errors.append(f"f({x}) nije konačan broj")
            except Exception as e:
                results.append(None)
                errors.append(str(e))

    except Exception as e:
        errors = [str(e)] * len(x_values)
        results = [None] * len(x_values)

    return results, errors
