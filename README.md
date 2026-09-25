# On the Convergence of the Extended Kalman Filter and of its Riccati Solution — numerical example

Code and figures of Section VII of

> A. Cellier-Devaux, D. Astolfi, V. Andrieu, *On the Convergence of the Extended Kalman Filter and of its Riccati Solution*, submitted to IEEE Transactions on Automatic Control, 2026. Long version: HAL (to appear).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vincent69410/EKFPaper/blob/main/example_oscillator.ipynb)

## The example

Harmonic oscillator with a quadratic output,

    x1' = x2,   x2' = -x1,   y = x1^2,

with `Q = I`, `R = 1`, on the annulus `0.4 < |x| < 1.6`. The solutions are circles,
so that the annulus is invariant in both time directions, and the linearisation
loses observability twice per period along each orbit (where `x1 = 0`).

The script `example_oscillator.py`

1. integrates the information Riccati equation along the orbits of radius
   `r` in `[0.4, 1.6]` and extracts its `2π`-periodic solution, which is the
   observability metric `Π(x)` (Proposition 7 of the paper);
2. computes the bounds of `Π` on the annulus, the certified upper bound
   `π̄`, and the gradient bound `sup ‖∇ₓΠ‖` on `0.45 ≤ |x| ≤ 1.55`;
3. computes the contraction rate `α*(r)` of the linearised error dynamics
   along each orbit from its monodromy matrix, and the guaranteed rate
   `q λ_min(Π)/2`;
4. computes the observability Gramian over one period, the constant `α_o`,
   and the certified lower bound `π̲ = exp(-q̄ π̄ T_o) α_o`;
5. simulates the EKF driven by the solution through `x(0) = (1, 0)`, from
   `x̂(0) = (1.25, 0.25)` and `P_0 ∈ {10⁻² I, I, 10² I}`, and measures the
   decay rates of `‖x̃‖` and of `‖P⁻¹ − Π(x̂)‖`.

## Running it

    pip install -r requirements.txt
    python example_oscillator.py

About three minutes on a laptop. The figures are written to `figures/`:

| file | content | paper |
|---|---|---|
| `fig_metric_oscillator.pdf` | eigenvalues of `Π` along three orbits; `α*(r)` and the guaranteed rate | long version, Fig. 1 |
| `fig_ekf_oscillator.pdf` | estimation error and `‖P⁻¹ − Π(x̂)‖` for three `P_0` | long version, Fig. 2 |
| `fig_oscillator_all.pdf` | the four panels in one row | TAC version, Fig. 1 |

## Correspondence with the numbers of Section VII

The script prints, in this order:

| printed value | value in the paper |
|---|---|
| `true bounds of Pi on the annulus: 0.2454 <= Pi <= 3.0672` | `0.245 I ⪯ Π ⪯ 3.07 I` |
| `certified upper bound pibar = 4.3526` | `π̄ = 4.35` |
| `numerical L_Pi = sup ‖grad_x Pi‖ ~ 2.391` | `‖∇ₓΠ‖ ≤ 2.4` |
| `alpha*(r=0.4) = 0.277`, `alpha*(r=1.0) = 0.646`, `alpha*(r=1.6) = 0.99` | `α*` from `0.28` to `0.99`; `α*(1) = 0.646` |
| `guaranteed rate with the TRUE floor, q*pi_lo/2 = 0.1227` | `q λ_min(Π)/2 = 0.12` |
| `alpha_o (window 2pi) ~ 0.5027 (exact pi*r1^2)` | `α_o = π r_1² ≈ 0.50` |
| `certified lower bound pi_lo = 6.67e-13` | `π̲ ≈ 7·10⁻¹³` |
| `Gramian eigenvalues at r=1: [π, 3π]` | eigenvalues `π r²` and `3π r²` |
| `P0=...: rate |xt| ~ -0.646, rate Pi ~ -0.64…` | observed rate `0.65` for the three `P_0` |

`run_output.txt` is the output of the run used for the paper.

## Files

- `example_oscillator.py` — the script (numpy, scipy, matplotlib).
- `example_oscillator.ipynb` — the same script as a notebook, for Colab.
- `figures/` — the three figures.
- `run_output.txt` — printed output of the reference run.
- `requirements.txt`, `LICENSE`.

## Citation

If you use this code, please cite the paper above.
