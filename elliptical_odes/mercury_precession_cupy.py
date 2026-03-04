# %%
# =============================================================================
# mercury_precession_cupy.py
# CuPy-accelerated version of mercury_precession.py
#
# Key GPU acceleration strategy:
#   The Dormand-Prince ODE stepper is inherently sequential (each step depends
#   on the previous one), so we cannot parallelise a single orbit over time.
#   Instead we run N_batch orbits simultaneously as a batch, i.e. the state
#   vector has shape (N_batch, n_eqs).  All arithmetic is done on device via
#   CuPy, so each RK stage evaluation processes N_batch trajectories in one
#   CUDA kernel call.
#
#   For the single-orbit use-cases the code falls back to batch-size 1,
#   which still benefits from the cuBLAS / cuRAND kernels used by CuPy.
# =============================================================================

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# %%
# -------------------------  Physical constants  ------------------------------
year_conversion = 3.156e7       # s in one year
rad_to_arcsec   = 180/np.pi * 3600
c    = 299792458.0              # m/s
G    = 6.67430e-11              # m^3/(kg s^2)
Msun = 1.98847e30               # kg
M    = 1.0                      # Solar masses
Mlen = G * Msun / (c * c)       # GM/c^2  [m]

# -------------------------  Mercury orbital parameters  ---------------------
a         = 5.7909227e10        # semi-major axis [m]
e         = 0.20563             # eccentricity
period    = 0.240846            # orbital period [yr]

rp = a * (1.0 - e)             # perihelion distance [m]
ra = a * (1.0 + e)             # aphelion distance   [m]

N_orbits_century = 100.0 / period

# %%
# -------------------------  Helper functions  --------------------------------
def r_s(M_=1.0):
    """Schwarzschild radius  [m]"""
    return 2.0 * Mlen * M_

def half_r_s(M_=1.0):
    """Half-Schwarzschild radius = GM/c^2  [m]"""
    return Mlen * M_

def f_schwarz(r, M_=1.0):
    """(-g_tt) for Schwarzschild metric"""
    return 1.0 - r_s(M_) / r

# -------------------------  Conserved quantities  ---------------------------
gtt_rp   = f_schwarz(rp, M)
gtt_ra   = f_schwarz(ra, M)
L_sq     = (gtt_ra - gtt_rp) / (gtt_rp / rp**2 - gtt_ra / ra**2)
E_sq_rp  = gtt_rp * (1.0 + L_sq / rp**2)
E_sq_ra  = gtt_ra * (1.0 + L_sq / ra**2)
L        = np.sqrt(L_sq)
E        = np.sqrt(E_sq_rp)

print("------------- Check L, E -------------")
print("L squared =", L_sq)
print("E squared from rp =", E_sq_rp)
print("E squared from ra =", E_sq_ra)
print("Energy:", E)
print("Angular Momentum:", L)

dphi_analytical = 6 * np.pi * half_r_s(M) / (a * (1.0 - e * e))

# Move scalar constants to GPU once
L_gpu   = cp.float64(L)
M_gpu   = cp.float64(M)
Mlen_gpu = cp.float64(Mlen)

# %%
# =============================================================================
#  RHS functions  (operate on batched state arrays of shape (N_batch, n_eq))
# =============================================================================

def rhs_tau_batch(tau, Y):
    """
    Batched RHS for the proper-time formulation.

    Parameters
    ----------
    tau : float   (scalar, same for every trajectory in the batch)
    Y   : cp.ndarray  shape (N_batch, 3)  columns = [r, phi, pr]

    Returns
    -------
    dY  : cp.ndarray  shape (N_batch, 3)
    """
    r   = Y[:, 0]
    pr  = Y[:, 2]

    dr_dtau  = pr
    dphi_dtau = L_gpu / r**2
    dpr_dtau  = (L_gpu**2 / r**3
                 - Mlen_gpu * M_gpu / r**2
                 - 3.0 * Mlen_gpu * M_gpu * L_gpu**2 / r**4)

    return cp.stack([dr_dtau, dphi_dtau, dpr_dtau], axis=1)


def rhs_phi_batch(phi, Y):
    """
    Batched RHS for the Binet (phi) formulation.

    Parameters
    ----------
    phi : float   (scalar)
    Y   : cp.ndarray  shape (N_batch, 2)  columns = [u, du/dphi]

    Returns
    -------
    dY  : cp.ndarray  shape (N_batch, 2)
    """
    u  = Y[:, 0]
    up = Y[:, 1]
    upp = Mlen_gpu * M_gpu / L_gpu**2 + 3.0 * Mlen_gpu * M_gpu * u**2 - u
    return cp.stack([up, upp], axis=1)


# %%
# =============================================================================
#  Dormand-Prince coefficients  (on CPU, applied as Python scalars)
# =============================================================================
c2, a21                         = 1/5,  1/5
c3, a31, a32                    = 3/10, 3/40, 9/40
c4, a41, a42, a43               = 4/5,  44/45, -56/15, 32/9
c5, c51, c52, c53, c54          = 8/9,  19372/6561, -25360/2187, 64448/6561, -212/729
c6, c61, c62, c63, c64, c65     = 1.0,  9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
c7, c71, c73, c74, c75, c76     = 1.0,  35/384, 500/1113, 125/192, -2187/6784, 11/84

e1 = 35/384   - 5179/57600
e3 = 500/1113 - 7571/16695
e4 = 125/192  - 393/640
e5 = -2187/6784 + 92097/339200
e6 = 11/84    - 187/2100
e7 = 0.0      - 1/40


# %%
def dormand_prince_batch(var, Y, h, variable):
    """
    One Dormand-Prince step for a batch of trajectories.

    Parameters
    ----------
    var      : float
    Y        : cp.ndarray  shape (N_batch, n_eq)
    h        : float
    variable : 'tau' | 'phi'

    Returns
    -------
    Y5   : cp.ndarray  (N_batch, n_eq)  — 5th-order solution
    err  : cp.ndarray  (N_batch, n_eq)  — local error estimate
    k    : list of 7 cp.ndarrays
    """
    rhs = rhs_tau_batch if variable == 'tau' else rhs_phi_batch

    k = [None] * 7
    k[0] = rhs(var,          Y)
    k[1] = rhs(var + c2*h,   Y + h*a21*k[0])
    k[2] = rhs(var + c3*h,   Y + h*(a31*k[0] + a32*k[1]))
    k[3] = rhs(var + c4*h,   Y + h*(a41*k[0] + a42*k[1] + a43*k[2]))
    k[4] = rhs(var + c5*h,   Y + h*(c51*k[0] + c52*k[1] + c53*k[2] + c54*k[3]))
    k[5] = rhs(var + c6*h,   Y + h*(c61*k[0] + c62*k[1] + c63*k[2] + c64*k[3] + c65*k[4]))
    k[6] = rhs(var + c7*h,   Y + h*(c71*k[0]             + c73*k[2] + c74*k[3] + c75*k[4] + c76*k[5]))

    Y5  = Y + h*(c71*k[0] + c73*k[2] + c74*k[3] + c75*k[4] + c76*k[5])
    err = h*(e1*k[0] + e3*k[2] + e4*k[3] + e5*k[4] + e6*k[5] + e7*k[6])

    return Y5, err, k


def adaptive_dormand_prince_batch(var_0, var_f, Y0, h_init,
                                  rtol=1e-10, atol=1e-15,
                                  variable='phi'):
    """
    Adaptive Dormand-Prince integrator for a batch of trajectories.

    Parameters
    ----------
    var_0, var_f : float   — integration interval
    Y0           : cp.ndarray  shape (N_batch, n_eq)  — initial states
    h_init       : float
    rtol, atol   : float
    variable     : 'tau' | 'phi'

    Returns
    -------
    vars_out : list of float
    Ys_out   : list of cp.ndarray  shape (N_batch, n_eq)
    """
    Y    = Y0.copy()
    var  = var_0
    h    = h_init
    safety  = 0.9
    exp_err = 1.0 / 5.0

    vars_out = [var]
    Ys_out   = [Y.copy()]

    while var < var_f:
        if var + h > var_f:
            h = var_f - var

        Y5, err, _ = dormand_prince_batch(var, Y, h, variable)

        # Error norm: max over batch and equations
        scale    = atol + cp.maximum(cp.abs(Y), cp.abs(Y5)) * rtol   # (N_batch, n_eq)
        err_norm_batch = cp.sqrt(cp.mean((err / scale)**2, axis=1))   # (N_batch,)
        err_norm = float(cp.max(err_norm_batch))                      # scalar

        if err_norm <= 1.0:
            var += h
            Y    = Y5
            vars_out.append(var)
            Ys_out.append(Y.copy())
            if err_norm > 0:
                h = h * safety * err_norm**(-exp_err)
        else:
            h = h * max(safety * err_norm**(-exp_err), 0.1)

    return vars_out, Ys_out


# %%
# =============================================================================
#  Perihelion utilities  (work on CPU numpy arrays)
# =============================================================================

def find_perihelia_idx(r):
    """Find indices of perihelia in a 1-D radius array."""
    return np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1


def compute_mean_perihelia(phi, idx):
    """Mean angular advance between successive perihelia."""
    phi_mod = phi % (2 * np.pi)
    dphi    = np.diff(phi_mod[idx])
    return np.mean(dphi)


# %%
# =============================================================================
#  Example 1 — Single orbit (batch size 1), proper-time formulation
# =============================================================================
print("\n========== Single orbit (tau formulation) ==========")

tf          = 2 * np.pi * np.sqrt(a**3 / half_r_s(M))
h_init      = tf / 1000

Y0_np = np.array([[rp, 0.0, 0.0]], dtype=np.float64)   # shape (1, 3)
Y0    = cp.asarray(Y0_np)

vars_out, Ys_out = adaptive_dormand_prince_batch(
    0.0, tf, Y0, h_init, rtol=1e-20, atol=1e-25, variable='tau')

# Bring results back to CPU
Ys_cpu = np.array([cp.asnumpy(Y) for Y in Ys_out])   # (n_steps, 1, 3)
r_1orb   = Ys_cpu[:, 0, 0]
phi_1orb = Ys_cpu[:, 0, 1]

x1 = r_1orb * np.cos(phi_1orb)
y1 = r_1orb * np.sin(phi_1orb)

plt.figure()
plt.scatter(0., 0., color='orange', s=70, label='Sun')
plt.scatter(x1[-1], y1[-1], color='red', s=20, label='End')
plt.plot(x1, y1, color='grey', linestyle='--', alpha=0.6, zorder=-1)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Single Mercury orbit (CuPy, tau)')
plt.legend()
plt.tight_layout()
plt.savefig('mercury_single_orbit_cupy.png', dpi=150)
plt.show()


# %%
# =============================================================================
#  Example 2 — Convergence test: 4 tolerance pairs, 10 orbits
# =============================================================================
print("\n========== Convergence test (10 orbits) ==========")

tf_10 = 10 * 2 * np.pi * np.sqrt(a**3 / half_r_s(M))
h_ini  = tf_10 / 1000

tol_pairs = [
    (1e-10, 1e-15),
    (1e-15, 1e-20),
    (1e-20, 1e-25),
    (1e-25, 1e-30),
]

Y0 = cp.asarray(np.array([[rp, 0.0, 0.0]], dtype=np.float64))

for rtol, atol in tol_pairs:
    vars_out, Ys_out = adaptive_dormand_prince_batch(
        0.0, tf_10, Y0, h_ini, rtol=rtol, atol=atol, variable='tau')

    Ys_cpu   = np.array([cp.asnumpy(Y) for Y in Ys_out])
    r_arr    = Ys_cpu[:, 0, 0]
    phi_arr  = Ys_cpu[:, 0, 1]

    idx_peri = find_perihelia_idx(r_arr)
    dphi     = compute_mean_perihelia(phi_arr, idx_peri)
    print(f"  rtol={rtol:.0e}  atol={atol:.0e} | "
          f"Precession/orbit={dphi:.6e} rad  "
          f"→ {dphi * N_orbits_century * rad_to_arcsec:.4f} arcsec/century")


# %%
# =============================================================================
#  Example 3 — Full century integration (~451 orbits), adaptive step
# =============================================================================
print("\n========== Full century integration (adaptive) ==========")

tf_century = 451 * 2 * np.pi * np.sqrt(a**3 / half_r_s(M))
h_ini      = tf_century / 100

Y0 = cp.asarray(np.array([[rp, 0.0, 0.0]], dtype=np.float64))

vars_out, Ys_out = adaptive_dormand_prince_batch(
    0.0, tf_century, Y0, h_ini, rtol=1e-25, atol=1e-30, variable='tau')

print(f"  Total steps taken: {len(vars_out)}")

Ys_cpu  = np.array([cp.asnumpy(Y) for Y in Ys_out])
r_cen   = Ys_cpu[:, 0, 0]
phi_cen = Ys_cpu[:, 0, 1]

idx_peri = find_perihelia_idx(r_cen)
dphi     = compute_mean_perihelia(phi_cen, idx_peri)

print("------------- No Interpolation ----------------")
print("  Precession per orbit [rad]:", dphi)
print("  Precession per century [arcsec]:", dphi * N_orbits_century * rad_to_arcsec)
print("\n------------- Analytical Solution ----------------")
print("  Precession per orbit [rad]:", dphi_analytical)
print("  Precession per century [arcsec]:",
      dphi_analytical * N_orbits_century * rad_to_arcsec)


# %%
# =============================================================================
#  Example 4 — Batch integration: many perturbed initial conditions at once
#              This is where the GPU shines most.
# =============================================================================
print("\n========== Batch integration over perturbed ICs ==========")

N_batch   = 512                  # simultaneous trajectories on GPU
tf_batch  = 10 * 2 * np.pi * np.sqrt(a**3 / half_r_s(M))
h_ini     = tf_batch / 1000

# Small random perturbations around the fiducial perihelion IC
rng      = np.random.default_rng(42)
delta_r  = rng.uniform(-1e6, 1e6, size=N_batch)   # ±1000 km perturbation
delta_pr = rng.uniform(-1e2, 1e2, size=N_batch)   # ±100 m/s

r_batch  = rp  + delta_r
pr_batch = 0.0 + delta_pr
phi_batch = np.zeros(N_batch)

Y0_batch = cp.asarray(
    np.stack([r_batch, phi_batch, pr_batch], axis=1).astype(np.float64)
)   # shape (N_batch, 3)

vars_out, Ys_out = adaptive_dormand_prince_batch(
    0.0, tf_batch, Y0_batch, h_ini, rtol=1e-10, atol=1e-15, variable='tau')

print(f"  Integrated {N_batch} trajectories simultaneously.")
print(f"  Total adaptive steps: {len(vars_out)}")

# Summarise per-trajectory perihelion advances
Ys_cpu = np.array([cp.asnumpy(Y) for Y in Ys_out])   # (steps, N_batch, 3)
dphi_per_traj = []
for b in range(N_batch):
    r_b   = Ys_cpu[:, b, 0]
    phi_b = Ys_cpu[:, b, 1]
    idx   = find_perihelia_idx(r_b)
    if len(idx) >= 2:
        dphi_per_traj.append(compute_mean_perihelia(phi_b, idx))

dphi_per_traj = np.array(dphi_per_traj)
print(f"  Mean precession/orbit across batch: "
      f"{dphi_per_traj.mean():.6e} ± {dphi_per_traj.std():.2e} rad")


# %%
# =============================================================================
#  Example 5 — Binet (phi) formulation, single trajectory
# =============================================================================
print("\n========== Binet (phi) formulation ==========")

phif      = 10 * 2 * np.pi
h_phi_ini = phif / 100

u_0      = 1.0 / rp
du_dphi_0 = 0.0

Y0_phi = cp.asarray(np.array([[u_0, du_dphi_0]], dtype=np.float64))

vars_phi, Ys_phi = adaptive_dormand_prince_batch(
    0.0, phif, Y0_phi, h_phi_ini, rtol=1e-20, atol=1e-25, variable='phi')

phi_arr = np.array(vars_phi)
Ys_phi_cpu = np.array([cp.asnumpy(Y) for Y in Ys_phi])   # (steps, 1, 2)
r_phi   = 1.0 / Ys_phi_cpu[:, 0, 0]

# Spline-based perihelion detection (same as original)
tck   = splrep(phi_arr, r_phi, s=0)
phi_new = np.linspace(phi_arr[0], phi_arr[-1], phi_arr.size * 500)
r_new   = splev(phi_new, tck, der=0)
dr_new  = splev(phi_new, tck, der=1)

idx_min = np.where(np.diff(np.sign(dr_new)))[0][::2]
dphi_binet = compute_mean_perihelia(phi_new, idx_min)

print("  Precession per orbit [rad]:", dphi_binet)
print("  Precession per century [arcsec]:",
      dphi_binet * N_orbits_century * rad_to_arcsec)
print("\n  Analytical value [arcsec/century]:",
      dphi_analytical * N_orbits_century * rad_to_arcsec)

x_phi = r_phi * np.cos(phi_arr)
y_phi = r_phi * np.sin(phi_arr)

plt.figure()
plt.plot(x_phi, y_phi, lw=0.5)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Mercury orbit — Binet equation (CuPy)')
plt.tight_layout()
plt.savefig('mercury_binet_cupy.png', dpi=150)
plt.show()

print("\nDone.")
