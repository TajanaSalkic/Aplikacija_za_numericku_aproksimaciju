"""
Metode Numeričke Integracije
============================

Ovaj modul implementira metode za numeričko računanje određenog integrala:

1. Trapezna metoda
2. Simpsonova metoda (1/3)

Sve metode vraćaju detaljne korake za step-by-step prikaz.
"""

from typing import Callable, Dict


def trapezoidal(f: Callable[[float], float],
                a: float,
                b: float,
                n: int = 10) -> Dict:
    """
    Trapezna Metoda (Trapezoidal Rule)
    ===================================

    TEORIJA:
    --------
    Trapezna metoda aproksimira integral površinom trapeza.
    Interval [a,b] se dijeli na n jednakih podintervala širine h = (b-a)/n.
    Na svakom podintervalu, funkcija se aproksimira linearnom funkcijom,
    a integral je površina trapeza.

    IZVOD:
    ------
    Površina jednog trapeza sa osnovicama f(x_i) i f(x_{i+1}) i visinom h:
        A_i = h · [f(x_i) + f(x_{i+1})] / 2

    Ukupna površina (suma svih trapeza):
        I ≈ h/2 · [f(x_0) + 2f(x_1) + 2f(x_2) + ... + 2f(x_{n-1}) + f(x_n)]

    FORMULA:
    --------
        I ≈ h/2 · [f(a) + 2·Σf(x_i) + f(b)]

    gdje je h = (b-a)/n i x_i = a + i·h

    GREŠKA:
    -------
    Greška trapezne metode je O(h²):
        E = -(b-a)³/(12n²) · f''(ξ) za neki ξ ∈ [a,b]

    Za glatke funkcije, greška opada kvadratno sa povećanjem n.

    Args:
        f: Funkcija koju integriramo
        a: Donja granica integracije
        b: Gornja granica integracije
        n: Broj podintervala

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [f(x) for x in x_points]

    steps = []

    # Korak 1: Računanje koraka h
    steps.append({
        'step': 1,
        'title': 'Računanje koraka h',
        'formula': f'h = (b - a) / n = ({b} - {a}) / {n} = {h:.6f}',
        'description': f'Dijelimo interval [{a}, {b}] na {n} jednakih dijelova.'
    })

    # Korak 2: Određivanje tačaka
    steps.append({
        'step': 2,
        'title': 'Određivanje tačaka podjele',
        'x_points': x_points.copy(),
        'y_points': y_points.copy(),
        'description': f'Tačke podjele: x_i = a + i·h, za i = 0, 1, ..., {n}'
    })

    # Korak 3: Primjena formule
    # I = h/2 * [f(a) + 2*sum(f(x_i)) + f(b)]
    middle_sum = sum(y_points[1:-1])
    integral = h / 2 * (y_points[0] + 2 * middle_sum + y_points[-1])

    steps.append({
        'step': 3,
        'title': 'Primjena trapezne formule',
        'formula': f'I = h/2 · [f(a) + 2·Σf(x_i) + f(b)]',
        'calculation': f'I = {h:.6f}/2 · [{y_points[0]:.6f} + 2·{middle_sum:.6f} + {y_points[-1]:.6f}]',
        'result': integral,
        'description': 'Sumiramo površine svih trapeza.'
    })

    # Detalji o trapezima
    trapezoids = []
    for i in range(n):
        area = h * (y_points[i] + y_points[i + 1]) / 2
        trapezoids.append({
            'index': i + 1,
            'x_left': x_points[i],
            'x_right': x_points[i + 1],
            'y_left': y_points[i],
            'y_right': y_points[i + 1],
            'area': area
        })

    return {
        'integral': integral,
        'n': n,
        'h': h,
        'x_points': x_points,
        'y_points': y_points,
        'trapezoids': trapezoids,
        'steps': steps,
        'method': 'Trapezna metoda'
    }


def simpson(f: Callable[[float], float],
            a: float,
            b: float,
            n: int = 10) -> Dict:
    """
    Simpsonova Metoda (Simpson's 1/3 Rule)
    ======================================

    TEORIJA:
    --------
    Simpsonova metoda koristi paraboličnu (kvadratnu) aproksimaciju funkcije
    umjesto linearne. Kroz svake tri uzastopne tačke provlači se parabola,
    čija se površina može egzaktno izračunati.

    IZVOD:
    ------
    Za tri tačke x_0, x_1, x_2 sa vrijednostima f_0, f_1, f_2,
    Lagrangeova interpolaciona parabola daje integral:

        ∫[x_0 do x_2] p(x)dx = h/3 · [f_0 + 4f_1 + f_2]

    FORMULA (kompozitna):
    ---------------------
        I ≈ h/3 · [f_0 + 4f_1 + 2f_2 + 4f_3 + 2f_4 + ... + 4f_{n-1} + f_n]

    ili: I ≈ h/3 · [f_0 + 4·(suma neparnih) + 2·(suma parnih) + f_n]

    NAPOMENA: n mora biti PARAN broj!

    GREŠKA:
    -------
    Greška Simpsonove metode je O(h⁴):
        E = -(b-a)⁵/(180n⁴) · f⁽⁴⁾(ξ)

    Simpsonova metoda je znatno preciznija od trapezne za isti broj tačaka.

    Args:
        f: Funkcija koju integriramo
        a: Donja granica
        b: Gornja granica
        n: Broj podintervala (mora biti paran!)

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    # n mora biti paran
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    y_points = [f(x) for x in x_points]

    steps = []

    # Korak 1: Provjera i računanje h
    steps.append({
        'step': 1,
        'title': 'Priprema i računanje koraka',
        'formula': f'h = (b - a) / n = ({b} - {a}) / {n} = {h:.6f}',
        'note': 'NAPOMENA: n mora biti paran broj za Simpsonovu metodu!',
        'description': f'Dijelimo interval na {n} podintervala (n = {n} je paran).'
    })

    # Korak 2: Tačke
    steps.append({
        'step': 2,
        'title': 'Vrijednosti funkcije u tačkama',
        'x_points': x_points.copy(),
        'y_points': y_points.copy(),
        'description': 'Računamo f(x_i) za sve tačke podjele.'
    })

    # Korak 3: Simpsonova formula
    odd_sum = sum(y_points[1:-1:2])    # Neparni indeksi: 1, 3, 5, ...
    even_sum = sum(y_points[2:-1:2])   # Parni indeksi: 2, 4, 6, ...

    integral = h / 3 * (y_points[0] + 4 * odd_sum + 2 * even_sum + y_points[-1])

    steps.append({
        'step': 3,
        'title': 'Primjena Simpsonove formule',
        'formula': 'I = h/3 · [f₀ + 4·(Σ neparni) + 2·(Σ parni) + f_n]',
        'odd_indices': list(range(1, n, 2)),
        'even_indices': list(range(2, n, 2)),
        'odd_sum': odd_sum,
        'even_sum': even_sum,
        'calculation': f'I = {h:.6f}/3 · [{y_points[0]:.6f} + 4·{odd_sum:.6f} + 2·{even_sum:.6f} + {y_points[-1]:.6f}]',
        'result': integral
    })

    # Detalji o parabolama
    parabolas = []
    for i in range(0, n, 2):
        area = h / 3 * (y_points[i] + 4 * y_points[i + 1] + y_points[i + 2])
        parabolas.append({
            'index': i // 2 + 1,
            'x_points': [x_points[i], x_points[i + 1], x_points[i + 2]],
            'y_points': [y_points[i], y_points[i + 1], y_points[i + 2]],
            'area': area
        })

    return {
        'integral': integral,
        'n': n,
        'h': h,
        'x_points': x_points,
        'y_points': y_points,
        'parabolas': parabolas,
        'steps': steps,
        'method': 'Simpsonova metoda 1/3'
    }


