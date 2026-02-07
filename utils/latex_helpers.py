"""
LaTeX Pomoćne Funkcije
======================

Funkcije za generisanje LaTeX formula za prikaz u Streamlit aplikaciji.
Streamlit koristi st.latex() za rendering matematičkih izraza.
"""

from typing import Dict, List, Optional
import numpy as np


def format_bisection_step(step: Dict) -> str:
    """
    Formatira korak metode bisekcije u LaTeX

    Args:
        step: Dict sa podacima koraka

    Returns:
        LaTeX string
    """
    i = step.get('iteration', 0)
    a = step.get('a', 0)
    b = step.get('b', 0)
    c = step.get('c')
    fc = step.get('fc')
    error = step.get('error', 0)

    latex = f"""
**Iteracija {i}**

Interval: $[a, b] = [{a:.6f}, {b:.6f}]$

"""
    if c is not None:
        latex += f"""
Srednja tačka: $c = \\frac{{a + b}}{{2}} = \\frac{{{a:.6f} + {b:.6f}}}{{2}} = {c:.6f}$

$f(c) = f({c:.6f}) = {fc:.6f}$

Greška: $|b - a| = {error:.6f}$
"""
    return latex


def format_newton_step(step: Dict) -> str:
    """
    Formatira korak Newton-Raphson metode u LaTeX

    Args:
        step: Dict sa podacima koraka

    Returns:
        LaTeX string
    """
    i = step.get('iteration', 0)
    x = step.get('x', 0)
    fx = step.get('fx', 0)
    dfx = step.get('dfx', 0)
    x_new = step.get('x_new')

    latex = f"""
**Iteracija {i}**

Trenutna aproksimacija: $x_{{{i}}} = {x:.6f}$

$f(x_{{{i}}}) = f({x:.6f}) = {fx:.6f}$

$f'(x_{{{i}}}) = f'({x:.6f}) = {dfx:.6f}$

"""
    if x_new is not None:
        latex += f"""
Newton-Raphson formula:
$$x_{{n+1}} = x_n - \\frac{{f(x_n)}}{{f'(x_n)}}$$

$$x_{{{i+1}}} = {x:.6f} - \\frac{{{fx:.6f}}}{{{dfx:.6f}}} = {x_new:.6f}$$
"""
    return latex


def format_integration_trapezoidal(h: float, y_points: List[float], integral: float) -> str:
    """
    Formatira trapeznu formulu u LaTeX

    Args:
        h: Korak
        y_points: Vrijednosti funkcije
        integral: Rezultat

    Returns:
        LaTeX string
    """
    n = len(y_points) - 1

    latex = f"""
## Trapezna Formula

$$I \\approx \\frac{{h}}{{2}} \\left[ f(x_0) + 2\\sum_{{i=1}}^{{n-1}} f(x_i) + f(x_n) \\right]$$

Gdje je:
- $h = {h:.6f}$ (korak)
- $n = {n}$ (broj podintervala)

### Računanje:

$f(x_0) = {y_points[0]:.6f}$

$f(x_n) = {y_points[-1]:.6f}$

$\\sum_{{i=1}}^{{n-1}} f(x_i) = {sum(y_points[1:-1]):.6f}$

### Rezultat:

$$I \\approx \\frac{{{h:.6f}}}{{2}} \\left[ {y_points[0]:.6f} + 2 \\cdot {sum(y_points[1:-1]):.6f} + {y_points[-1]:.6f} \\right] = {integral:.6f}$$
"""
    return latex


def format_integration_simpson(h: float, y_points: List[float], integral: float) -> str:
    """
    Formatira Simpsonovu formulu u LaTeX

    Args:
        h: Korak
        y_points: Vrijednosti funkcije
        integral: Rezultat

    Returns:
        LaTeX string
    """
    n = len(y_points) - 1

    odd_sum = sum(y_points[1:-1:2])
    even_sum = sum(y_points[2:-1:2])

    latex = f"""
## Simpsonova Formula (1/3)

$$I \\approx \\frac{{h}}{{3}} \\left[ f(x_0) + 4\\sum_{{\\text{{neparni}}}} f(x_i) + 2\\sum_{{\\text{{parni}}}} f(x_i) + f(x_n) \\right]$$

Gdje je:
- $h = {h:.6f}$ (korak)
- $n = {n}$ (broj podintervala, mora biti paran)

### Računanje:

$f(x_0) = {y_points[0]:.6f}$

$f(x_n) = {y_points[-1]:.6f}$

$\\sum_{{\\text{{neparni}}}} f(x_i) = {odd_sum:.6f}$ (indeksi 1, 3, 5, ...)

$\\sum_{{\\text{{parni}}}} f(x_i) = {even_sum:.6f}$ (indeksi 2, 4, 6, ...)

### Rezultat:

$$I \\approx \\frac{{{h:.6f}}}{{3}} \\left[ {y_points[0]:.6f} + 4 \\cdot {odd_sum:.6f} + 2 \\cdot {even_sum:.6f} + {y_points[-1]:.6f} \\right] = {integral:.6f}$$
"""
    return latex


def format_romberg_table(R: np.ndarray) -> str:
    """
    Formatira Romberg tabelu u LaTeX

    Args:
        R: Romberg matrica

    Returns:
        LaTeX string
    """
    n = R.shape[0]

    latex = """
## Romberg Tabela

| Red | R[i,0] | R[i,1] | R[i,2] | R[i,3] | R[i,4] |
|-----|--------|--------|--------|--------|--------|
"""
    for i in range(n):
        row = f"| {i} |"
        for j in range(n):
            if j <= i:
                row += f" {R[i,j]:.8f} |"
            else:
                row += " - |"
        latex += row + "\n"

    latex += f"""

### Interpretacija kolona:
- **Kolona 0**: Trapezna pravila $O(h^2)$
- **Kolona 1**: Simpsonova pravila $O(h^4)$
- **Kolona 2**: Booleova pravila $O(h^6)$
- **Dijagonala**: Najpreciznija aproksimacija

**Najbolja aproksimacija**: $I \\approx {R[n-1, n-1]:.10f}$
"""
    return latex


def format_gauss_quadrature(n: int, points: List[float], weights: List[float]) -> str:
    """
    Formatira Gaussovu kvadraturu u LaTeX

    Args:
        n: Broj tačaka
        points: Gaussove tačke
        weights: Težine

    Returns:
        LaTeX string
    """
    latex = f"""
## Gaussova Kvadratura ({n} tačaka)

$$\\int_a^b f(x) dx \\approx \\frac{{b-a}}{{2}} \\sum_{{i=1}}^{{{n}}} w_i \\cdot f\\left( \\frac{{b-a}}{{2}} t_i + \\frac{{a+b}}{{2}} \\right)$$

### Gaussove tačke i težine za interval [-1, 1]:

| i | Tačka $t_i$ | Težina $w_i$ |
|---|-------------|--------------|
"""
    for i, (t, w) in enumerate(zip(points, weights)):
        latex += f"| {i+1} | {t:.10f} | {w:.10f} |\n"

    latex += f"""

### Preciznost:
- {n}-točkasta Gaussova kvadratura je **tačna** za polinome stepena ≤ {2*n-1}
"""
    return latex


def format_differentiation_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za numeričku derivaciju

    Returns:
        LaTeX string
    """
    return """
## Formule Numeričke Derivacije

### 1. Forward Difference (Unaprijedna)
$$f'(x) \\approx \\frac{f(x+h) - f(x)}{h}$$
- Greška: $O(h)$ - prvi red
- Koristi tačke: $x$, $x+h$

### 2. Backward Difference (Unazadna)
$$f'(x) \\approx \\frac{f(x) - f(x-h)}{h}$$
- Greška: $O(h)$ - prvi red
- Koristi tačke: $x-h$, $x$

### 3. Central Difference (Centralna)
$$f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}$$
- Greška: $O(h^2)$ - drugi red
- Koristi tačke: $x-h$, $x+h$

### Druga derivacija (Central):
$$f''(x) \\approx \\frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$

---

## Izvori Greške

1. **Greška rezanja** (truncation): Raste sa $h$
2. **Greška zaokruživanja** (round-off): Raste kada $h \\to 0$

**Optimalni $h$**:
- Forward/Backward: $h_{opt} \\approx \\sqrt{\\epsilon} \\approx 10^{-8}$
- Central: $h_{opt} \\approx \\epsilon^{1/3} \\approx 10^{-5}$

gdje je $\\epsilon \\approx 10^{-16}$ mašinska preciznost.
"""


def format_jacobi_iteration(step: Dict, A: np.ndarray, b: np.ndarray) -> str:
    """
    Formatira iteraciju Jacobijeve metode u LaTeX

    Args:
        step: Podaci koraka
        A: Matrica sistema
        b: Vektor desne strane

    Returns:
        LaTeX string
    """
    n = len(b)
    k = step.get('step', 0) - 1

    latex = f"""
## Iteracija {k}

### Jacobijeva Formula:
$$x_i^{{(k+1)}} = \\frac{{1}}{{a_{{ii}}}} \\left( b_i - \\sum_{{j \\neq i}} a_{{ij}} x_j^{{(k)}} \\right)$$

### Računanje:
"""

    if 'calculations' in step:
        for calc in step['calculations']:
            i = calc['component']
            latex += f"\n$x_{{{i}}}^{{({k+1})}} = {calc['calculation'].split('=')[-1].strip()}$\n"

    if 'x_new' in step:
        x_new = step['x_new']
        latex += f"\n\n**Novo rješenje**: $\\mathbf{{x}}^{{({k+1})}} = [{', '.join([f'{xi:.6f}' for xi in x_new])}]^T$"

    if 'error' in step:
        latex += f"\n\n**Greška**: $||\\mathbf{{x}}^{{({k+1})}} - \\mathbf{{x}}^{{({k})}}||_\\infty = {step['error']:.2e}$"

    return latex


def format_linear_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za linearnu regresiju

    Returns:
        LaTeX string
    """
    return """
## Linearna Regresija - Metoda Najmanjih Kvadrata

### Model:
$$y = ax + b$$

### Cilj:
Minimizirati sumu kvadrata reziduala:
$$S = \\sum_{i=1}^{n} (y_i - ax_i - b)^2$$

### Normalne jednačine:
$$\\frac{\\partial S}{\\partial a} = 0 \\quad \\Rightarrow \\quad a \\cdot \\sum x_i^2 + b \\cdot \\sum x_i = \\sum x_i y_i$$

$$\\frac{\\partial S}{\\partial b} = 0 \\quad \\Rightarrow \\quad a \\cdot \\sum x_i + b \\cdot n = \\sum y_i$$

### Rješenje:
$$a = \\frac{n \\sum x_i y_i - \\sum x_i \\sum y_i}{n \\sum x_i^2 - (\\sum x_i)^2}$$

$$b = \\frac{\\sum y_i - a \\sum x_i}{n} = \\bar{y} - a\\bar{x}$$

### Koeficijent determinacije $R^2$:
$$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$

- $R^2 = 1$: Savršen fit
- $R^2 = 0$: Model nije bolji od srednje vrijednosti
"""


def format_exponential_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za eksponencijalnu regresiju

    Returns:
        LaTeX string
    """
    return """
## Eksponencijalna Aproksimacija

### Model:
$$y = A \\cdot e^{Bx}$$

### Linearizacija:
Logaritmovanjem:
$$\\ln(y) = \\ln(A) + Bx$$

Supstitucija: $Y = \\ln(y)$, $a = B$, $b = \\ln(A)$

Linearni model:
$$Y = ax + b$$

### Procedura:
1. Transformiši: $Y_i = \\ln(y_i)$
2. Primijeni linearnu regresiju na $(x_i, Y_i)$
3. Izračunaj: $B = a$, $A = e^b$

### Napomena:
- Sve $y$ vrijednosti moraju biti **pozitivne**!
"""


def format_power_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za stepenu aproksimaciju

    Returns:
        LaTeX string
    """
    return """
## Stepena Aproksimacija

### Model:
$$y = A \\cdot x^B$$

### Linearizacija:
Logaritmovanjem obje strane:
$$\\ln(y) = \\ln(A) + B \\cdot \\ln(x)$$

Supstitucija: $Y = \\ln(y)$, $X = \\ln(x)$, $a = B$, $b = \\ln(A)$

Linearni model:
$$Y = aX + b$$

### Procedura:
1. Transformiši: $X_i = \\ln(x_i)$, $Y_i = \\ln(y_i)$
2. Primijeni linearnu regresiju na $(X_i, Y_i)$
3. Izračunaj: $B = a$, $A = e^b$

### Napomena:
- Sve $x$ i $y$ vrijednosti moraju biti **pozitivne**!

### Primjene:
- Keplerovi zakoni planetarnog kretanja
- Veza veličine i metabolizma kod životinja
- Fizički zakoni proporcionalnosti
"""


def format_logarithmic_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za logaritamsku aproksimaciju

    Returns:
        LaTeX string
    """
    return """
## Logaritamska Aproksimacija

### Model:
$$y = a + b \\cdot \\ln(x)$$

### Linearizacija:
Model je već u linearnom obliku!

Supstitucija: $X = \\ln(x)$

Linearni model:
$$y = a + bX$$

### Procedura:
1. Transformiši: $X_i = \\ln(x_i)$
2. Primijeni linearnu regresiju na $(X_i, y_i)$
3. Koeficijenti $a$ i $b$ su direktno iz regresije

### Napomena:
- Sve $x$ vrijednosti moraju biti **pozitivne**!

### Primjene:
- Psihofizički zakoni (Weber-Fechner)
- Logaritamske skale (pH, decibeli)
- Saturacijske krive
"""


def format_hyperbolic_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za hiperboličku aproksimaciju

    Returns:
        LaTeX string
    """
    return """
## Hiperbolička Aproksimacija

### Model:
$$y = \\frac{1}{a + bx}$$

### Linearizacija:
Recipročna vrijednost:
$$\\frac{1}{y} = a + bx$$

Supstitucija: $Y = \\frac{1}{y}$

Linearni model:
$$Y = a + bx$$

### Procedura:
1. Transformiši: $Y_i = \\frac{1}{y_i}$
2. Primijeni linearnu regresiju na $(x_i, Y_i)$
3. Koeficijenti $a$ i $b$ su direktno iz regresije

### Napomena:
- $y$ vrijednosti ne smiju biti **nula**!

### Primjene:
- Michaelis-Menten kinetika enzima
- Odziv sistema sa zasićenjem
"""


def format_rational_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za racionalnu aproksimaciju

    Returns:
        LaTeX string
    """
    return """
## Racionalna Aproksimacija

### Model:
$$y = \\frac{x}{a + bx}$$

### Linearizacija:
Dijeljenjem sa $y$ i množenjem sa $x$:
$$\\frac{x}{y} = a + bx$$

Supstitucija: $Y = \\frac{x}{y}$

Linearni model:
$$Y = a + bx$$

### Procedura:
1. Transformiši: $Y_i = \\frac{x_i}{y_i}$
2. Primijeni linearnu regresiju na $(x_i, Y_i)$
3. Koeficijenti $a$ i $b$ su direktno iz regresije

### Napomena:
- $y$ vrijednosti ne smiju biti **nula**!

### Primjene:
- Langmuir adsorpcija
- Zasićenje kapaciteta
- Procesna kinetika
"""


def format_sqrt_regression_formulas() -> str:
    """
    Vraća LaTeX objašnjenje formula za aproksimaciju kvadratnim korijenom

    Returns:
        LaTeX string
    """
    return """
## Aproksimacija Kvadratnim Korijenom

### Model:
$$y = a + b \\cdot \\sqrt{x}$$

### Linearizacija:
Model je već u linearnom obliku!

Supstitucija: $X = \\sqrt{x}$

Linearni model:
$$y = a + bX$$

### Procedura:
1. Transformiši: $X_i = \\sqrt{x_i}$
2. Primijeni linearnu regresiju na $(X_i, y_i)$
3. Koeficijenti $a$ i $b$ su direktno iz regresije

### Napomena:
- Sve $x$ vrijednosti moraju biti **nenegativne** ($x \\geq 0$)!

### Primjene:
- Fizički procesi sa korijenom
- Vrijeme reakcije
- Difuzijski procesi
"""


def format_polynomial_regression_formulas(degree: int = 2) -> str:
    """
    Vraća LaTeX objašnjenje formula za polinomijalnu aproksimaciju

    Args:
        degree: Stepen polinoma

    Returns:
        LaTeX string
    """
    return f"""
## Polinomijalna Aproksimacija (stepen {degree})

### Model:
$$y = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$$

### Metoda Najmanjih Kvadrata:
Cilj: Minimizirati $S = \\sum(y_i - p(x_i))^2$

U matričnom obliku:
$$\\mathbf{{X}}^T \\mathbf{{X}} \\cdot \\mathbf{{a}} = \\mathbf{{X}}^T \\mathbf{{y}}$$

gdje je $\\mathbf{{X}}$ **Vandermondova matrica**:
$$\\mathbf{{X}} = \\begin{{bmatrix}} 1 & x_1 & x_1^2 & \\cdots & x_1^n \\\\ 1 & x_2 & x_2^2 & \\cdots & x_2^n \\\\ \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\ 1 & x_m & x_m^2 & \\cdots & x_m^n \\end{{bmatrix}}$$

### Rješenje:
$$\\mathbf{{a}} = (\\mathbf{{X}}^T \\mathbf{{X}})^{{-1}} \\mathbf{{X}}^T \\mathbf{{y}}$$

### Upozorenje - Overfitting:
- Visoki stepen polinoma može dovesti do "overfittinga"
- Model savršeno prolazi kroz tačke, ali loše predviđa
- **Pravilo**: stepen << broj tačaka

### Adjusted $R^2$:
$$R^2_{{adj}} = 1 - (1-R^2)\\frac{{n-1}}{{n-p-1}}$$

gdje je $p$ broj parametara (stepen + 1).
"""


def format_all_approximation_methods() -> str:
    """
    Vraća LaTeX pregled svih metoda aproksimacije

    Returns:
        LaTeX string
    """
    return """
## Pregled Metoda Aproksimacije

### Linearne vs Nelinearne Aproksimacije

**Linearna aproksimacija** (direktna primjena metode najmanjih kvadrata):
- $y = ax + b$

**Nelinearne aproksimacije** (zahtijevaju linearizaciju):

| Model | Linearizacija | Transformirane varijable |
|-------|---------------|--------------------------|
| $y = Ax^B$ | $\\ln(y) = \\ln(A) + B\\ln(x)$ | $Y = \\ln(y), X = \\ln(x)$ |
| $y = Ae^{Bx}$ | $\\ln(y) = \\ln(A) + Bx$ | $Y = \\ln(y)$ |
| $y = a + b\\ln(x)$ | Već linearno | $X = \\ln(x)$ |
| $y = \\frac{1}{a+bx}$ | $\\frac{1}{y} = a + bx$ | $Y = \\frac{1}{y}$ |
| $y = \\frac{x}{a+bx}$ | $\\frac{x}{y} = a + bx$ | $Y = \\frac{x}{y}$ |
| $y = a + b\\sqrt{x}$ | Već linearno | $X = \\sqrt{x}$ |

### Postupak Linearizacije:
1. Transformiši podatke prema formuli
2. Primijeni linearnu regresiju na transformirane podatke
3. Izračunaj originalne parametre iz dobijenih koeficijenata

### Odabir Metode:
- Koristite **poređenje modela** za automatski odabir najboljeg modela
- Najveći $R^2$ ukazuje na najbolji fit
- Razmotrite i fizički/logički smisao modela
"""


def create_matrix_latex(A: np.ndarray, name: str = "A") -> str:
    """
    Kreira LaTeX prikaz matrice

    Args:
        A: NumPy matrica
        name: Ime matrice

    Returns:
        LaTeX string
    """
    n, m = A.shape
    matrix_content = " \\\\ ".join([" & ".join([f"{A[i,j]:.4f}" for j in range(m)]) for i in range(n)])
    return f"${name} = \\begin{{bmatrix}} {matrix_content} \\end{{bmatrix}}$"


def create_vector_latex(v: np.ndarray, name: str = "x") -> str:
    """
    Kreira LaTeX prikaz vektora

    Args:
        v: NumPy vektor
        name: Ime vektora

    Returns:
        LaTeX string
    """
    vector_content = " \\\\ ".join([f"{vi:.6f}" for vi in v])
    return f"${name} = \\begin{{bmatrix}} {vector_content} \\end{{bmatrix}}$"
