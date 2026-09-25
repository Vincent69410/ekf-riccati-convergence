"""Numerical example of Section VII of

    A. Cellier-Devaux, D. Astolfi, V. Andrieu,
    "On the Convergence of the Extended Kalman Filter and of its Riccati Solution".

Harmonic oscillator x1' = x2, x2' = -x1 with output y = x1^2, on the annulus
0.4 < |x| < 1.6, with Q = I and R = 1.  The script computes the observability
metric Pi along the orbits, its bounds and gradient bound, the contraction
rate alpha*(r) of the linearised error dynamics, the observability Gramian,
and simulates the EKF for three initial covariances.  It prints every number
quoted in Section VII and writes the figures to ./figures/.

Run:  python example_oscillator.py     (about 3 minutes, numpy/scipy/matplotlib)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})

import os
OUT = "figures/"
os.makedirs(OUT, exist_ok=True)

q, r = 1.0, 1.0
A = np.array([[0.0, 1.0], [-1.0, 0.0]])
Q = q * np.eye(2)
Rinv = 1.0 / r
rho_min, rho_max = 0.4, 1.6          # annulus (the tube)

def f(x): return A @ x
def h(x): return x[0] ** 2
def C(x): return np.array([[2 * x[0], 0.0]])
def S(x): return C(x).T @ C(x) * Rinv

# ---------------------------------------------------------------- metric Pi
# along the orbit of radius rho: x(theta) = rho (cos theta, sin theta)
def riccati_info(t, p, rho):
    Pi = p.reshape(2, 2)
    x = rho * np.array([np.cos(t), -np.sin(t)])      # solution of xdot = A x
    dPi = -(Pi @ A + A.T @ Pi) - Pi @ Q @ Pi + S(x)
    return dPi.reshape(-1)

n_rho, n_th = 97, 721
rhos = np.linspace(rho_min, rho_max, n_rho)
thetas = np.linspace(0, 2 * np.pi, n_th)
abar, cbar = 1.0, 4 * rho_max ** 2 * Rinv
pibar_cert = (abar + np.sqrt(abar ** 2 + q * cbar)) / q
PiGrid = np.zeros((n_rho, n_th, 2, 2))
period_err = 0.0
for i, rho in enumerate(rhos):
    n_per = 8
    sol = solve_ivp(riccati_info, [0, 2 * np.pi * n_per], (pibar_cert * np.eye(2)).reshape(-1),
                    args=(rho,), rtol=1e-10, atol=1e-12, dense_output=True)
    last = sol.sol(2 * np.pi * (n_per - 1) + thetas).T.reshape(n_th, 2, 2)
    prev = sol.sol(2 * np.pi * (n_per - 2) + thetas).T.reshape(n_th, 2, 2)
    period_err = max(period_err, np.max(np.abs(last - prev)))
    PiGrid[i] = 0.5 * (last + last.transpose(0, 2, 1))
print(f"periodicity error of Pi along orbits: {period_err:.2e}")

eigs = np.linalg.eigvalsh(PiGrid)
pi_lo, pi_hi = eigs.min(), eigs.max()
print(f"true bounds of Pi on the annulus: {pi_lo:.4f} <= Pi <= {pi_hi:.4f}")
print(f"certified upper bound pibar = {pibar_cert:.4f}")

interp = RegularGridInterpolator((rhos, thetas), PiGrid, bounds_error=False, fill_value=None)
# spectral representation in theta (periodic) + cubic spline in rho
from scipy.interpolate import CubicSpline
samples = PiGrid[:, :-1, :, :]                      # drop duplicated endpoint 2*pi
coef = np.fft.rfft(samples, axis=1) / samples.shape[1]   # (n_rho, n_freq, 2, 2)
spl_re = CubicSpline(rhos, coef.real, axis=0)
spl_im = CubicSpline(rhos, coef.imag, axis=0)
freqs = np.arange(coef.shape[1])
def Pi_of(x):
    rho = np.hypot(x[0], x[1]); th = (-np.arctan2(x[1], x[0])) % (2 * np.pi)   # time along the orbit
    c = spl_re(rho) + 1j * spl_im(rho)                # (n_freq, 2, 2)
    e = np.exp(1j * freqs * th)
    val = c[0].real + 2 * np.real(np.tensordot(e[1:], c[1:], axes=(0, 0)))
    if samples.shape[1] % 2 == 0:                     # Nyquist term counted once
        val -= np.real(c[-1] * e[-1])
    return val

# gradient of Pi (Frobenius on the matrix side), by finite differences in cartesian coords
eps = 1e-4
LPi = 0.0
for rho in np.linspace(rho_min + 0.05, rho_max - 0.05, 25):
    for th in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        x = rho * np.array([np.cos(th), np.sin(th)])
        d1 = (Pi_of(x + [eps, 0]) - Pi_of(x - [eps, 0])) / (2 * eps)
        d2 = (Pi_of(x + [0, eps]) - Pi_of(x - [0, eps])) / (2 * eps)
        # sup over unit v of ||d1 v1 + d2 v2||_F
        M = np.array([[np.sum(d1 * d1), np.sum(d1 * d2)], [np.sum(d1 * d2), np.sum(d2 * d2)]])
        LPi = max(LPi, np.sqrt(np.linalg.eigvalsh(M).max()))
print(f"numerical L_Pi = sup ||grad_x Pi|| ~ {LPi:.3f}")

# ---------------------------------------------------------------- rates
def monodromy(rho, i):
    def rhs(t, z):
        Z = z.reshape(2, 2)
        Pi = interp(np.array([[rho, t % (2 * np.pi)]]))[0]
        M = A + Q @ Pi
        return (-(M.T) @ Z).reshape(-1)
    sol = solve_ivp(rhs, [0, 2 * np.pi], np.eye(2).reshape(-1), rtol=1e-10, atol=1e-12)
    return sol.y[:, -1].reshape(2, 2)

alpha_star = np.inf
for i, rho in enumerate(rhos[::4]):
    Mo = monodromy(rho, i)
    ev = np.linalg.eigvals(Mo)
    rate = -np.log(np.abs(ev)).max() / (2 * np.pi)
    alpha_star = min(alpha_star, rate)
print(f"true contraction rate alpha* (min over annulus) ~ {alpha_star:.4f}")
for rr in (rho_min, 1.0, rho_max):
    ev = np.linalg.eigvals(monodromy(rr, 0))
    print(f"  alpha*(r={rr:.1f}) = {-np.log(np.abs(ev)).max() / (2 * np.pi):.4f}")
print(f"guaranteed rate with the TRUE floor, q*pi_lo/2 = {q*pi_lo/2:.4f}")

# Gramian over one period along solutions, and certified floor
def gramian(rho, To=2 * np.pi):
    # O(t) = int_{t-To}^t Phi(s,t)^T S(x(s)) Phi(s,t) ds, t = 0; Phi(s,0)=rotation
    ss = np.linspace(-To, 0, 2001)
    integrand = []
    for s in ss:
        Phi = np.array([[np.cos(s), np.sin(s)], [-np.sin(s), np.cos(s)]])   # exp(A s)
        x = rho * np.array([np.cos(s), -np.sin(s)])
        integrand.append(Phi.T @ S(x) @ Phi)
    w = np.full(len(ss), ss[1] - ss[0]); w[0] *= 0.5; w[-1] *= 0.5        # trapezoid rule
    return np.tensordot(w, np.array(integrand), axes=(0, 0))
alpha_o = min(np.linalg.eigvalsh(gramian(rho)).min() for rho in rhos[::6])
To = 2 * np.pi
pi_lo_cert = np.exp(-q * pibar_cert * To) * alpha_o
print(f"alpha_o (window 2pi) ~ {alpha_o:.4f}  (exact value pi*r1^2 = {np.pi*rho_min**2:.4f});  "
      f"certified lower bound pi_lo = exp(-q*pibar*To)*alpha_o = {pi_lo_cert:.2e}")
Og = gramian(1.0)
print(f"Gramian eigenvalues at r=1 over one period: {np.linalg.eigvalsh(Og)}  (exact: pi, 3pi)")

# ---------------------------------------------------------------- EKF
def simulate(P0, xhat0, x_true0, T=40.0):
    def rhs(t, z):
        x = z[0:2]; xh = z[2:4]; P = z[4:8].reshape(2, 2)
        y = h(x)
        K = P @ C(xh).T * Rinv
        dx = f(x)
        dxh = f(xh) + (K * (y - h(xh))).reshape(-1)
        dP = A @ P + P @ A.T + Q - P @ S(xh) @ P
        return np.concatenate([dx, dxh, dP.reshape(-1)])
    z0 = np.concatenate([x_true0, xhat0, P0.reshape(-1)])
    sol = solve_ivp(rhs, [0, T], z0, rtol=1e-10, atol=1e-12, dense_output=True)
    ts = np.linspace(0, T, 2001)
    Z = sol.sol(ts)
    err_x = np.linalg.norm(Z[0:2] - Z[2:4], axis=0)
    err_P = np.zeros_like(ts)
    for k, t in enumerate(ts):
        P = Z[4:8, k].reshape(2, 2); xh = Z[2:4, k]
        err_P[k] = np.linalg.norm(np.linalg.inv(P) - Pi_of(xh), 2)
    return ts, err_x, err_P

x_true0 = np.array([1.0, 0.0])
xhat0 = np.array([1.25, 0.25])
cases = [(1e-2, r"$P_0=10^{-2}I$"), (1.0, r"$P_0=I$"), (1e2, r"$P_0=10^{2}I$")]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.15))
styles = ["-", "--", ":"]
SIM = []
for (p0, lab), st in zip(cases, styles):
    ts, ex, eP = simulate(p0 * np.eye(2), xhat0, x_true0)
    SIM.append((lab, st, ts, ex, eP))
    ax[0].semilogy(ts, ex, st, color="k", lw=1.2, label=lab)
    ax[1].semilogy(ts, eP, st, color="k", lw=1.2, label=lab)
    print(f"P0={p0:g}: final |xt|={ex[-1]:.2e}, final |P^-1-Pi|={eP[-1]:.2e}, "
          f"rate |xt| ~ {np.polyfit(ts[500:1500], np.log(ex[500:1500]+1e-300), 1)[0]:.3f}, "
          f"rate Pi ~ {np.polyfit(ts[500:1500], np.log(eP[500:1500]+1e-300), 1)[0]:.3f}")
ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"$\|\tilde x(t)\|$")
ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$\|P(t)^{-1}-\Pi(\hat x(t))\|$")
ax[1].legend(frameon=False, loc="upper right")
for a in ax:
    a.grid(alpha=0.3, which="both", lw=0.4)
fig.tight_layout()
fig.savefig(OUT + "fig_ekf_oscillator.pdf")
print("figure saved")

# metric eigenvalues along the annulus (second figure)
fig2, ax2 = plt.subplots(1, 2, figsize=(7.0, 2.15))
for i, rho in enumerate([0.5, 1.0, 1.5]):
    j = np.argmin(np.abs(rhos - rho))
    ev = np.linalg.eigvalsh(PiGrid[j])
    ax2[0].plot(thetas, ev[:, 0], styles[i], color="k", lw=1.2, label=rf"$r={rho}$")
    ax2[0].plot(thetas, ev[:, 1], styles[i], color="k", lw=1.2)
ax2[0].set_xlabel(r"$t$ along the orbit"); ax2[0].set_ylabel(r"eigenvalues of $\Pi$")
ax2[0].set_xticks([0, np.pi, 2 * np.pi]); ax2[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax2[0].legend(frameon=False, loc="upper right")
rates = []
for rho in rhos[::2]:
    ev = np.linalg.eigvals(monodromy(rho, 0))
    rates.append(-np.log(np.abs(ev)).max() / (2 * np.pi))
ax2[1].plot(rhos[::2], rates, "k-", lw=1.2, label=r"$\alpha^\star(r)$")
ax2[1].axhline(q * pi_lo / 2, color="k", ls="--", lw=1.0, label=r"$q\,\lambda_{\min}(\Pi)/2$")
ax2[1].set_xlabel(r"$r=\|x\|$"); ax2[1].set_ylabel("rate"); ax2[1].legend(frameon=False, loc="center right")
for a in ax2:
    a.grid(alpha=0.3, lw=0.4)
fig2.tight_layout()
fig2.savefig(OUT + "fig_metric_oscillator.pdf")
print("figure 2 saved")

# ---------------------------------------------------------------- combined 1x4 figure (short version)
fig3, ax3 = plt.subplots(1, 4, figsize=(7.1, 1.75))
for i, rho in enumerate([0.5, 1.0, 1.5]):
    j = np.argmin(np.abs(rhos - rho))
    ev = np.linalg.eigvalsh(PiGrid[j])
    ax3[0].plot(thetas, ev[:, 0], styles[i], color="k", lw=1.0, label=rf"$r={rho}$")
    ax3[0].plot(thetas, ev[:, 1], styles[i], color="k", lw=1.0)
ax3[0].set_xlabel(r"$t$ along the orbit"); ax3[0].set_ylabel(r"eigenvalues of $\Pi$")
ax3[0].set_xticks([0, np.pi, 2 * np.pi]); ax3[0].set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax3[0].legend(frameon=False, loc="upper right")
ax3[1].plot(rhos[::2], rates, "k-", lw=1.0, label=r"$\alpha^\star(r)$")
ax3[1].axhline(q * pi_lo / 2, color="k", ls="--", lw=0.9, label=r"$q\,\lambda_{\min}(\Pi)/2$")
ax3[1].set_xlabel(r"$r=\|x\|$"); ax3[1].set_ylabel("rate"); ax3[1].legend(frameon=False, loc="center right")
for lab, st, ts, ex, eP in SIM:
    ax3[2].semilogy(ts, ex, st, color="k", lw=1.0, label=lab)
    ax3[3].semilogy(ts, eP, st, color="k", lw=1.0, label=lab)
ax3[2].set_xlabel(r"$t$"); ax3[2].set_ylabel(r"$\|\tilde x(t)\|$")
ax3[3].set_xlabel(r"$t$"); ax3[3].set_ylabel(r"$\|P(t)^{-1}-\Pi(\hat x(t))\|$")
ax3[3].legend(frameon=False, loc="upper right")
for a in ax3:
    a.grid(alpha=0.3, which="both", lw=0.4)
fig3.tight_layout(w_pad=0.6)
fig3.savefig(OUT + "fig_oscillator_all.pdf")
print("figure 3 saved")
