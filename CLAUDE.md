# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a code package for **parametric modeling of airfoils and propeller blades** (螺旋桨参数化建模方法), implementing classic geometry parameterization, blade-element aerodynamics, and design-optimization methods in parallel **Python** and **MATLAB**. It is reference/educational code: each file is self-contained, runs standalone, and produces visualizations. Comments are written in Chinese.

## Layout

Top-level files form the numbered method series (`p01`–`p12` Python, `m01`–`m12` MATLAB). Files with matching numbers are equivalent implementations of the same method in the two languages.

| # | Method | Python | MATLAB |
|---|--------|--------|--------|
| 01 | CST parameterization | `p01_cst.py` | `m01_cst.m` |
| 02 | PARSEC | `p02_parsec.py` | `m02_parsec.m` |
| 03 | Hicks-Henne bump functions | `p03_hicks_henne.py` | `m03_hicks_henne.m` |
| 04 | B-Spline / NURBS | `p04_bspline.py` | `m04_bspline.m` |
| 05 | Free-Form Deformation (FFD) | `p05_ffd.py` | `m05_ffd.m` |
| 06 | Spanwise dist. + Betz-Prandtl + Adkins-Liebeck | `p06_spanwise_betz_adkins.py` | `m06_*.m` |
| 07 | CCBlade-style BEM solver (Ning 2014) | `p07_ccblade_bem.py` | `m07_*.m` |
| 08 | Kriging surrogate optimization | `p08_kriging_optimization.py` | `m08_*.m` |
| 09 | 3D blade geometry generator | `p09_blade_3d_generator.py` | `m09_*.m` |
| 10 | NSGA-II multi-objective optimization | `p10_nsga2_multiobjective.py` | `m10_*.m` |
| 11 | NACA airfoil tools | `p11_naca_tools.py` | `m11_*.m` |
| 12 | Integrated end-to-end pipeline | `p12_integrated_pipeline.py` | `m12_*.m` |

The `python/` and `matlab/` subdirectories hold a separate, more focused pair of CST modules (`cst_airfoil`, `cst_propeller`) split into reusable library functions plus a fitting routine.

## Architecture

- **`p12_integrated_pipeline.py` is the capstone.** It chains the other methods into one design flow: NACA airfoil → CST fit → spanwise distribution → Adkins-Liebeck initial design → BEM analysis → Hicks-Henne + FFD shape tuning → 3D blade geometry → coordinate-file export (for ANSYS import). To keep it standalone it **inlines** copies of the core functions from each module rather than importing them — so a change to a core algorithm (e.g. `bernstein_b`, `cst_profile`, `prandtl_F`) must be mirrored both in its source module and in `p12`.

- **CST is the foundational parameterization** used throughout. The convention is the class function `C(ψ) = ψ^0.5 · (1-ψ)^1.0` (round LE, sharp TE) times a Bernstein-polynomial shape function. Fitting (`fit_cst`) is least-squares on the Bernstein basis, masking the endpoints to avoid the singular class-function values.

- **BEM (`p07`) follows Ning 2014**: the blade-element momentum equations are recast as a single 1-D residual `f(φ)=0` solved with Brent's method (`scipy.optimize.brentq`) for guaranteed convergence, instead of classic fixed-point iteration.

- **Each top-level file shares the same structure**: header comment (method origin + references), standalone core functions, then a demo block that generates a multi-panel figure saved as `<name>_demo.png`.

## Running

Python (each script is an executable demo that writes a PNG):
```bash
pip install numpy matplotlib scipy
python p01_cst.py        # → cst_demo.png
python p12_integrated_pipeline.py
```

MATLAB — open and run any `.m` file directly. All helpers are local functions inside the script, so there are no path/dependency requirements.

There is no build system, test suite, or linter; correctness is verified by inspecting the generated demo figures.
