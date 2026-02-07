"""
Metode Numeričke Integracije
============================

Ovaj modul implementira metode za numeričko računanje određenog integrala:

1. Trapezna metoda
2. Simpsonova metoda (1/3)
3. Romberg integracija 
4. Gaussova kvadratura

Sve metode vraćaju detaljne korake za step-by-step prikaz.
"""

import numpy as np
from typing import Callable, Dict, List, Tuple


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


def romberg(f: Callable[[float], float],
            a: float,
            b: float,
            max_order: int = 5) -> Dict:
    """
    Romberg Integracija
    ===================

    TEORIJA (DETALJNO - nije rađeno na nastavi):
    ---------------------------------------------
    Romberg integracija kombinuje trapeznu metodu sa Richardsonovom
    ekstrapolacijom za postizanje visoke preciznosti.

    OSNOVNA IDEJA:
    --------------
    1. Računamo trapezne aproksimacije za različit broj podintervala:
       T(h), T(h/2), T(h/4), ...

    2. Greška trapezne metode ima oblik:
       I - T(h) = c₁h² + c₂h⁴ + c₃h⁶ + ...

    3. Richardsonovom ekstrapolacijom eliminišemo vodeći član greške:
       S(h) = [4T(h/2) - T(h)] / 3  →  greška O(h⁴)

    4. Nastavljamo proces za eliminaciju viših redova greške.

    ROMBERG SHEMA:
    --------------
    Gradimo trougaonu tabelu R[i,j]:

    R[0,0] = T(h)
    R[1,0] = T(h/2)         R[1,1] = S(h) = (4R[1,0] - R[0,0])/3
    R[2,0] = T(h/4)         R[2,1]          R[2,2]
    ...

    Opća formula:
        R[i,j] = [4^j · R[i,j-1] - R[i-1,j-1]] / (4^j - 1)

    INTERPRETACIJA KOLONA:
    ----------------------
    - Kolona 0: Trapezne aproksimacije (greška O(h²))
    - Kolona 1: Simpsonove aproksimacije (greška O(h⁴))
    - Kolona 2: Booleove aproksimacije (greška O(h⁶))
    - Kolona j: Greška O(h^{2j+2})

    PREDNOSTI:
    ----------
    - Vrlo visoka preciznost sa malo evaluacija funkcije
    - Automatska procjena greške (razlika susjednih aproksimacija)
    - Kombinuje jednostavnost trapezne metode sa visokom preciznošću

    Args:
        f: Funkcija
        a: Donja granica
        b: Gornja granica
        max_order: Maksimalni red (broj redova u tabeli)

    Returns:
        Dict sa Romberg tabelom i detaljnim koracima
    """
    R = np.zeros((max_order, max_order))
    steps = []

    # Korak 1: Inicijalna trapezna aproksimacija
    h = b - a
    R[0, 0] = h * (f(a) + f(b)) / 2

    steps.append({
        'step': 1,
        'title': 'Inicijalna trapezna aproksimacija (1 interval)',
        'formula': 'R[0,0] = h·[f(a) + f(b)]/2',
        'h': h,
        'calculation': f'R[0,0] = {h}·[{f(a):.6f} + {f(b):.6f}]/2 = {R[0,0]:.10f}',
        'R_table': R[:1, :1].copy()
    })

    # Gradimo tabelu
    for i in range(1, max_order):
        # Računamo novu trapeznu aproksimaciju sa više tačaka
        h = h / 2
        n = 2 ** i

        # Dodajemo nove tačke (samo one koje nisu već izračunate)
        sum_new = sum(f(a + (2 * k - 1) * h) for k in range(1, n // 2 + 1))
        R[i, 0] = R[i - 1, 0] / 2 + h * sum_new

        step_info = {
            'step': i + 1,
            'title': f'Red {i}: Trapezna aproksimacija ({n} intervala)',
            'h': h,
            'n_intervals': n,
            'R_i0': R[i, 0],
            'extrapolations': []
        }

        # Richardsonova ekstrapolacija
        for j in range(1, i + 1):
            factor = 4 ** j
            R[i, j] = (factor * R[i, j - 1] - R[i - 1, j - 1]) / (factor - 1)

            step_info['extrapolations'].append({
                'j': j,
                'formula': f'R[{i},{j}] = (4^{j}·R[{i},{j - 1}] - R[{i - 1},{j - 1}])/(4^{j}-1)',
                'calculation': f'R[{i},{j}] = ({factor}·{R[i, j - 1]:.10f} - {R[i - 1, j - 1]:.10f})/{factor - 1}',
                'result': R[i, j]
            })

        step_info['R_table'] = R[:i + 1, :i + 1].copy()
        steps.append(step_info)

    # Procjena greške
    error_estimate = abs(R[max_order - 1, max_order - 1] - R[max_order - 2, max_order - 2])

    return {
        'integral': R[max_order - 1, max_order - 1],
        'R_table': R,
        'max_order': max_order,
        'error_estimate': error_estimate,
        'steps': steps,
        'method': 'Romberg integracija',
        'interpretation': {
            'column_0': 'Trapezna pravila (greška O(h²))',
            'column_1': 'Simpsonova pravila (greška O(h⁴))',
            'column_2': 'Booleova pravila (greška O(h⁶))',
            'diagonal': 'Najpreciznija aproksimacija u svakom redu'
        }
    }


def gauss_quadrature(f: Callable[[float], float],
                     a: float,
                     b: float,
                     n: int = 5) -> Dict:
    """
    Gaussova Kvadratura
    ===================

    TEORIJA:
    --------
    Gaussova kvadratura je optimalna metoda numeričke integracije.
    Koristi posebno odabrane tačke (Gaussove tačke) i težine koje
    maksimiziraju preciznost za dati broj evaluacija funkcije.

    OSNOVNA IDEJA:
    --------------
    Integral se aproksimira sumom:
        ∫[a,b] f(x)dx ≈ Σ w_i · f(x_i)

    gdje su x_i Gaussove tačke, a w_i odgovarajuće težine.

    GAUSSOVE TAČKE:
    ---------------
    Gaussove tačke su nule Legendreovih polinoma na intervalu [-1, 1].
    Za integraciju na proizvoljnom intervalu [a, b], vrši se transformacija:
        x = (b-a)/2 · t + (a+b)/2,  t ∈ [-1, 1]

    PRECIZNOST:
    -----------
    n-točkasta Gaussova kvadratura je TAČNA za polinome stepena ≤ 2n-1!
    To znači da sa samo 3 tačke možemo tačno integrirati polinome do 5. stepena.

    LEGENDREOVI POLINOMI:
    ---------------------
    P_0(x) = 1
    P_1(x) = x
    P_2(x) = (3x² - 1)/2
    P_3(x) = (5x³ - 3x)/2
    ...

    Rekurentna formula:
    (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)

    Args:
        f: Funkcija
        a: Donja granica
        b: Gornja granica
        n: Broj Gaussovih tačaka (1-10)

    Returns:
        Dict sa rezultatom i detaljnim koracima
    """
    # Gaussove tačke i težine za interval [-1, 1]
    # Predefinirane vrijednosti za n = 1 do 10
    gauss_data = {
        1: {
            'points': [0.0],
            'weights': [2.0]
        },
        2: {
            'points': [-0.5773502691896257, 0.5773502691896257],
            'weights': [1.0, 1.0]
        },
        3: {
            'points': [-0.7745966692414834, 0.0, 0.7745966692414834],
            'weights': [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
        },
        4: {
            'points': [-0.8611363115940526, -0.3399810435848563,
                       0.3399810435848563, 0.8611363115940526],
            'weights': [0.3478548451374538, 0.6521451548625461,
                        0.6521451548625461, 0.3478548451374538]
        },
        5: {
            'points': [-0.9061798459386640, -0.5384693101056831, 0.0,
                       0.5384693101056831, 0.9061798459386640],
            'weights': [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891]
        },
        6: {
            'points': [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                       0.2386191860831969, 0.6612093864662645, 0.9324695142031521],
            'weights': [0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                        0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        },
        7: {
            'points': [-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
                       0.4058451513773972, 0.7415311855993945, 0.9491079123427585],
            'weights': [0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694,
                        0.3818300505051189, 0.2797053914892766, 0.1294849661688697]
        },
        8: {
            'points': [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                       0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363],
            'weights': [0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                        0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763]
        },
        9: {
            'points': [-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0,
                       0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261],
            'weights': [0.0812743883615744, 0.1806481606948574, 0.2606106964029354, 0.3123470770400029, 0.3302393550012598,
                        0.3123470770400029, 0.2606106964029354, 0.1806481606948574, 0.0812743883615744]
        },
        10: {
            'points': [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,
                       0.1488743389816312, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717],
            'weights': [0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        }
    }

    n = min(max(n, 1), 10)  # Ograniči na 1-10
    points = gauss_data[n]['points']
    weights = gauss_data[n]['weights']

    steps = []

    # Korak 1: Transformacija intervala
    # x = (b-a)/2 * t + (a+b)/2
    c1 = (b - a) / 2
    c2 = (a + b) / 2

    steps.append({
        'step': 1,
        'title': 'Transformacija intervala',
        'description': f'Transformišemo interval [{a}, {b}] u standardni interval [-1, 1]',
        'formula': 'x = (b-a)/2 · t + (a+b)/2',
        'transformation': f'x = {c1:.6f}·t + {c2:.6f}',
        'jacobian': f'dx = {c1:.6f}·dt'
    })

    # Korak 2: Gaussove tačke i težine
    steps.append({
        'step': 2,
        'title': f'Gaussove tačke i težine za n = {n}',
        'description': f'{n}-točkasta Gaussova kvadratura je tačna za polinome do stepena {2*n-1}',
        'points': points.copy(),
        'weights': weights.copy(),
        'note': 'Tačke su nule Legendreovog polinoma P_n(x)'
    })

    # Korak 3: Transformacija tačaka i računanje
    x_transformed = [c1 * t + c2 for t in points]
    f_values = [f(x) for x in x_transformed]
    contributions = [w * fval for w, fval in zip(weights, f_values)]

    steps.append({
        'step': 3,
        'title': 'Transformacija tačaka na [a, b]',
        'original_points': points.copy(),
        'transformed_points': x_transformed.copy(),
        'f_values': f_values.copy()
    })

    # Korak 4: Primjena formule
    integral = c1 * sum(contributions)

    step4_details = []
    for i, (t, x, w, fx, contrib) in enumerate(zip(points, x_transformed, weights, f_values, contributions)):
        step4_details.append({
            'i': i + 1,
            't_i': t,
            'x_i': x,
            'w_i': w,
            'f(x_i)': fx,
            'w_i * f(x_i)': contrib
        })

    steps.append({
        'step': 4,
        'title': 'Primjena Gaussove formule',
        'formula': 'I = (b-a)/2 · Σ w_i · f(x_i)',
        'details': step4_details,
        'sum_contributions': sum(contributions),
        'integral': integral
    })

    return {
        'integral': integral,
        'n_points': n,
        'exact_for_degree': 2 * n - 1,
        'gauss_points': points,
        'gauss_weights': weights,
        'transformed_points': x_transformed,
        'f_values': f_values,
        'steps': steps,
        'method': f'Gaussova kvadratura ({n} tačaka)'
    }


def compare_integration_methods(f: Callable[[float], float],
                                a: float,
                                b: float,
                                exact_value: float = None,
                                n_values: List[int] = [4, 8, 16, 32]) -> Dict:
    """
    Poređenje metoda integracije za različit broj podintervala

    Args:
        f: Funkcija
        a, b: Granice integracije
        exact_value: Tačna vrijednost integrala (ako je poznata)
        n_values: Lista vrijednosti n za poređenje

    Returns:
        Dict sa rezultatima poređenja
    """
    results = {'trapezoid': [], 'simpson': [], 'romberg': [], 'gauss': []}

    for n in n_values:
        trap = trapezoidal(f, a, b, n)
        simp = simpson(f, a, b, n)

        results['trapezoid'].append({'n': n, 'value': trap['integral']})
        results['simpson'].append({'n': n, 'value': simp['integral']})

    # Romberg za različite redove
    for order in [2, 3, 4, 5]:
        rom = romberg(f, a, b, order)
        results['romberg'].append({'order': order, 'value': rom['integral']})

    # Gauss za različit broj tačaka
    for n in [2, 3, 5, 7]:
        gauss = gauss_quadrature(f, a, b, n)
        results['gauss'].append({'n': n, 'value': gauss['integral']})

    # Dodaj greške ako je poznata tačna vrijednost
    if exact_value is not None:
        results['exact'] = exact_value
        for method in ['trapezoid', 'simpson']:
            for item in results[method]:
                item['error'] = abs(item['value'] - exact_value)
        for item in results['romberg']:
            item['error'] = abs(item['value'] - exact_value)
        for item in results['gauss']:
            item['error'] = abs(item['value'] - exact_value)

    return results


def integrate_from_table(x_points: np.ndarray,
                         y_points: np.ndarray,
                         method: str = 'auto') -> Dict:
    """
    Integracija iz Tablice Vrijednosti
    ===================================

    Računa integral iz tablice vrijednosti (x, y) bez poznate funkcije.
    Automatski bira najbolju metodu na osnovu broja tačaka.

    METODE:
    -------
    - 'trapezoid': Trapezna metoda (za bilo koji broj tačaka)
    - 'simpson': Simpsonova metoda (zahtijeva neparan broj tačaka)
    - 'auto': Automatski izbor najbolje metode

    Args:
        x_points: X vrijednosti (moraju biti sortirane)
        y_points: Y vrijednosti
        method: Metoda integracije ('trapezoid', 'simpson', 'auto')

    Returns:
        Dict sa integralom i detaljima
    """
    x = np.array(x_points)
    y = np.array(y_points)
    n = len(x)

    if n < 2:
        return {'error': 'Potrebne su najmanje 2 tačke za integraciju'}

    # Provjeri da li su x sortirane
    if not np.all(np.diff(x) > 0):
        # Sortiraj po x
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    steps = []

    # Provjeri uniformnost razmaka
    h_values = np.diff(x)
    is_uniform = np.allclose(h_values, h_values[0], rtol=1e-5)

    steps.append({
        'step': 1,
        'title': 'Analiza podataka',
        'n_points': n,
        'x_range': f'[{x[0]:.4f}, {x[-1]:.4f}]',
        'is_uniform': is_uniform,
        'h_values': h_values.tolist() if not is_uniform else [h_values[0]],
        'note': 'Uniformni razmak' if is_uniform else 'Neuniformni razmak - koristi se trapezna metoda'
    })

    # Automatski izbor metode
    if method == 'auto':
        if is_uniform and n >= 3 and (n - 1) % 2 == 0:
            method = 'simpson'
        else:
            method = 'trapezoid'

        steps.append({
            'step': 2,
            'title': 'Automatski izbor metode',
            'chosen_method': method,
            'reason': f"Simpson: uniformni={is_uniform}, n={n}, (n-1) paran={(n-1) % 2 == 0}" if method == 'simpson'
                     else "Trapezna: univerzalna metoda za bilo kakve podatke"
        })

    # Računanje integrala
    if method == 'trapezoid':
        # Trapezna metoda za neuniformni razmak
        integral = 0
        trapezoids = []

        for i in range(n - 1):
            h = x[i + 1] - x[i]
            area = h * (y[i] + y[i + 1]) / 2
            integral += area
            trapezoids.append({
                'i': i + 1,
                'x_left': x[i],
                'x_right': x[i + 1],
                'h': h,
                'area': area
            })

        steps.append({
            'step': 3,
            'title': 'Trapezna formula (neuniformni razmak)',
            'formula': 'I = Σ h_i · (y_i + y_{i+1}) / 2',
            'trapezoids': trapezoids,
            'integral': integral
        })

        method_name = 'Trapezna metoda (iz tablice)'

    elif method == 'simpson':
        if not is_uniform:
            return {'error': 'Simpsonova metoda zahtijeva uniformni razmak između tačaka'}

        if (n - 1) % 2 != 0:
            return {'error': 'Simpsonova metoda zahtijeva neparan broj tačaka (paran broj intervala)'}

        h = h_values[0]

        # Simpson 1/3
        odd_sum = sum(y[1:-1:2])
        even_sum = sum(y[2:-1:2])
        integral = h / 3 * (y[0] + 4 * odd_sum + 2 * even_sum + y[-1])

        steps.append({
            'step': 3,
            'title': 'Simpsonova formula',
            'formula': 'I = h/3 · [y₀ + 4·Σ(neparni) + 2·Σ(parni) + y_n]',
            'h': h,
            'y_0': y[0],
            'y_n': y[-1],
            'odd_sum': odd_sum,
            'even_sum': even_sum,
            'integral': integral
        })

        method_name = 'Simpsonova metoda (iz tablice)'

    else:
        return {'error': f'Nepoznata metoda: {method}'}

    return {
        'integral': integral,
        'method': method_name,
        'n_points': n,
        'x_range': (x[0], x[-1]),
        'is_uniform': is_uniform,
        'steps': steps,
        'x_points': x.tolist(),
        'y_points': y.tolist()
    }


def integrate_with_interpolation(x_points: np.ndarray,
                                  y_points: np.ndarray,
                                  method: str = 'simpson',
                                  n_interp: int = 100) -> Dict:
    """
    Integracija sa Interpolacijom
    =============================

    Prvo interpolira podatke (kubni spline), zatim integrira
    interpoliranu funkciju sa višom preciznošću.

    Args:
        x_points: X vrijednosti
        y_points: Y vrijednosti
        method: Metoda integracije
        n_interp: Broj tačaka za interpolaciju

    Returns:
        Dict sa integralom i poređenjem
    """
    from scipy.interpolate import CubicSpline

    x = np.array(x_points)
    y = np.array(y_points)

    # Kreiraj kubni spline
    cs = CubicSpline(x, y)

    # Generiši više tačaka za preciznu integraciju
    x_fine = np.linspace(x[0], x[-1], n_interp)
    y_fine = cs(x_fine)

    steps = []

    steps.append({
        'step': 1,
        'title': 'Interpolacija kubnim splineom',
        'original_points': len(x),
        'interpolated_points': n_interp,
        'description': 'Kubni spline kreira glatku funkciju kroz sve tačke'
    })

    # Integracija na finim podacima
    if method == 'simpson':
        # Osiguraj neparan broj tačaka
        if n_interp % 2 == 0:
            n_interp += 1
            x_fine = np.linspace(x[0], x[-1], n_interp)
            y_fine = cs(x_fine)

        h = (x[-1] - x[0]) / (n_interp - 1)
        odd_sum = sum(y_fine[1:-1:2])
        even_sum = sum(y_fine[2:-1:2])
        integral = h / 3 * (y_fine[0] + 4 * odd_sum + 2 * even_sum + y_fine[-1])
        method_name = 'Simpson (sa interpolacijom)'

    else:
        # Trapezna kao fallback
        integral = np.trapz(y_fine, x_fine)
        method_name = 'Trapezna (sa interpolacijom)'

    steps.append({
        'step': 2,
        'title': f'{method_name}',
        'n_points': n_interp,
        'integral': integral
    })

    # Poređenje sa direktnom integracijom
    direct_result = integrate_from_table(x, y)

    steps.append({
        'step': 3,
        'title': 'Poređenje',
        'direct_integral': direct_result.get('integral'),
        'interpolated_integral': integral,
        'difference': abs(integral - direct_result.get('integral', 0))
    })

    return {
        'integral': integral,
        'direct_integral': direct_result.get('integral'),
        'method': method_name,
        'n_interpolation_points': n_interp,
        'interpolated_function': cs,
        'steps': steps
    }
