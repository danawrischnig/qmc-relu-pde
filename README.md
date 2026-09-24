# QMC Experiments for Parametric PDEs

This repository contains randomized quasi-Monte Carlo (QMC) experiments for two parametric elliptic PDEs on $D=(0,1)^2$.

For each parameter vector $y\in[0,1]^s$, the quantity of interest is

```math
g(y)=\int_D u(x,y)\,dx.
```

The experiments compare QMC integration of the nonsmooth ReLU functional with smooth Softplus approximations.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate qmc-pde
```

Then run everything from the repository root.

The main dependencies are NumPy, SciPy, Matplotlib, Joblib, and tqdm.

`qmc.py` currently loads the lattice generating vector from the internet, so an internet connection is required when starting the experiment.

## Model problems

The considered problems are of the form

```math
-\nabla\cdot(a(y)\nabla u(x,y)) = f(x),
\qquad
u|_{\partial D}=b,
```

with

```math
a(y)
=
a_0+\sum_{j=1}^s\frac{y_j}{j^\alpha}\psi_j(x),
```

and $b \in \mathbb R$ constant.

In the notation below, $a_0$ corresponds to the command-line argument ``--baseline``, $\alpha$ to ``--decay``, and $s$ to ``--s``.


### Constant diffusion

For the first problem, consider $\psi_j \equiv 1$, $j\ge 1$, and

```math
f(x)
=
72\left[x_1(1-x_1)+x_2(1-x_2)\right].
```

This problem has the analytical solution

```math
u(x,y)
=
b+
\frac{36x_1(1-x_1)x_2(1-x_2)}{a(y)},
```

and therefore

```math
g(y)=b+\frac{1}{a(y)}.
```

No FEM solve is required for this model.

### Sine diffusion

The second problem is given by

```math
\psi_j(x_1, x_2) = \sin(j\pi x_1)\sin(j\pi x_2), \qquad j \ge 1,
```

and $f \equiv 100$.

There is no analytical solution in the implementation. The PDE is solved using linear P1 finite elements on a uniform triangular mesh.

The mesh resolution is controlled by ``--level L`` with $2^L$ subdivisions per coordinate direction.

### Boundary condition
In both problems, the boundary data $b \in \mathbb R$ is chosen as a function of the PDE parameters such that less than 1 percent of the QMC values are positive. 

## Running the experiments

Show all available options with

```bash
python experiment.py --help
```

### Constant diffusion

A small test run:

```bash
python experiment.py \
    --pde-name constant_diffusion \
    --decay 1.5 \
    --s 20 \
    --log-theta 12 \
    --log-n 8 \
    --log-n-ref 10 \
    --n-jobs 4
```

For a larger run: `--log-n 14` and `--log-n-ref 18`.

### Sine diffusion

A small test run:

```bash
python experiment.py \
    --pde-name sine_diffusion \
    --decay 1.5 \
    --s 20 \
    --level 5 \
    --log-theta 12 \
    --log-n 6 \
    --log-n-ref 8 \
    --n-jobs 4
```

## Experiment output

The experiment estimates

```math
I(g) = \int_{[0,1)^s} \text{ReLU}(g(y)) \mathrm dy
```

and the Softplus approximations

```math
I_\theta(g)
=
\int_{[0,1)^s}
\Psi_{\theta}(g(y))\,\mathrm{d}y
=
\int_{[0,1)^s}
\frac{\log(1+\exp(\theta g(y)))}{\theta}
\,\mathrm{d}y,
\qquad
\theta = 1,2,4,\ldots
```

The script saves

- experiment parameters as a `.json` file;
- QMC estimates, reference values, sample sizes, and estimated standard errors as a `.npz` file.
- RMSE plots as a `.png` file.

The output filenames are generated automatically unless explicitly specified with
``--filename-data``,
``--filename-params``,
``--filename-plot``.
