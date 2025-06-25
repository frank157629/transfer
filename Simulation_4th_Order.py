# ================================================================
# 4-state large-signal GFL model – strict Eq. (8)
# states X = [ delta , Dw , i_gd , i_gq ]ᵀ
# ================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# ---------- base parameters -------------------------------------
base = dict(Vg = 1.0,  wN = 1.0,  wg = 1.0,
            Lg = 0.15, Rg = 0.015,
            Kp = 20.0, Ki = 400.0,
            iLd = 0.0, iLq = -0.8, #i_{Ld}^*,i_{Lq}^*
            )
# Example functions for v_od^g(t) and v_oq^g(t).
# Here i choose:
#   v_od^g(t) = 1.0 + 0.1 * sin(2π·50Hz·t)
#   v_oq^g(t) =       0.1 * cos(2π·50Hz·t)
# (just as placeholders; Hussein may suggest a different shape or frequency)
def vodg_function(t):
    # 50Hz = 50*2πrad/s → in p.u. (if base frequency is 50Hz, wN=1 p.u.),
    # 2π·50Hz corresponds to 2π·(50/50) = 2πrad/s in p.u.
    # return 1.0 + 0.1 * np.sin(2 * np.pi * 1.0 * t)
    return 1.0
def dvodg_function(t):
    # derivative of 0.1*sin(2π·1·t) is 0.1 * (2π·1) * cos(2π·1·t)
    # return 0.1 * 2 * np.pi * np.cos(2 * np.pi * 1.0 * t)
    return 0.0

def voqg_function(t):
    # return       0.1 * np.cos(2 * np.pi * 1.0 * t)
    return 1.0

def dvoqg_function(t):
    # derivative of 0.1*cos(2π·1·t) is -0.1 * (2π·1) * sin(2π·1·t)
    # return -0.1 * 2 * np.pi * np.sin(2 * np.pi * 1.0 * t)
    return 0.0

# ---------- RHS --------------------------------------------------
def rhs(t, X, p):
    δ, Dw, igd, igq = X                 # unpack states

    # --------- grid-frame voltages (balanced) --------------------
    vg_d, vg_q = p['Vg'], 0.0           #  v_g^g components

    # --------- reference currents  (Eq. 6) -----------------------
    iLd_g =  np.cos(δ)*p['iLd'] - np.sin(δ)*p['iLq']
    iLq_g =  np.sin(δ)*p['iLd'] + np.cos(δ)*p['iLq']
    # Now let v_od^g(t), v_oq^g(t) be time functions instead of constants:
    vod = vodg_function(t)
    voq = voqg_function(t)
    #Derivatives of v_od^g and v_oq^g from the chosen functions
    dvod = dvodg_function(t)
    dvoq = dvoqg_function(t)
    # --------- GFL_2nd_order dynamics  (Eq. 3) -----------------------------
    dδ = Dw + p['wN'] - p['wg']

    pll_P = p['Kp'] * (-np.sin(δ) * dvod
                       - np.cos(δ) * dδ * vod
                       + np.cos(δ) * dvoq
                       - np.sin(δ) * dδ * voq)

    pll_I = p['Ki'] * (-np.sin(δ) * vod + np.cos(δ) * voq)

    dDw = pll_P + pll_I
    # --------- RL dynamics   (Eq. 4) -----------------------------

    # true inductor derivatives ( *include* the L_g·i̇ term! )
    di_gd = (vod - vg_d + p['wN']*p['Lg']*igq - p['Rg']*igd) / p['Lg']
    di_gq = (voq - vg_q - p['wN']*p['Lg']*igd - p['Rg']*igq) / p['Lg']




    return [dδ, dDw, di_gd, di_gq]

# ---------- single run helper -----------------------------------
def run_case(p, X0, t_end=1.0, h=0.001):
    t_eval = np.linspace(0, t_end, int(t_end/h)+1)
    sol = solve_ivp(rhs, (0, t_end), X0,
                    args=(p,), method='RK45', t_eval=t_eval,
                    max_step=h)
    return sol.t, sol.y

# ---------- batch + PDF output (like your 2-state code) ---------
def main(N=10, seed=42,
         pdf_name=str(Path.home()/ "Desktop"/"results_4th_order.pdf")):
    rng = np.random.default_rng(seed)
    with PdfPages(pdf_name) as pdf:
        for idx in range(N):
            p          = base.copy()
            p['wg']    = 1.0 + rng.uniform(-0.02, 0.02)
            p['iLd']   = rng.uniform(-1.0, 1.0)
            p['iLq']   = rng.uniform(-1.0, 1.0)
            X0 = np.zeros(4)

            t, y = run_case(p, X0)

            fig, ax = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
            ax[0].plot(t, y[0] * 180 / np.pi)
            ax[0].set_ylabel(r'$\delta$ (deg)')
            ax[1].plot(t, y[1])
            ax[1].set_ylabel(r'$\Delta\omega$ (p.u.)')
            ax[2].plot(t, y[2])
            ax[2].set_ylabel(r'$i_{gd}$ (p.u.)')
            ax[3].plot(t, y[3])
            ax[3].set_ylabel(r'$i_{gq}$ (p.u.)')
            ax[3].set_xlabel('time (s)')

            subtitle = (
                f"Vg={p['Vg']:.2f}, wN={p['wN']:.2f}, wg={p['wg']:.2f}, "
                f"Lg={p['Lg']:.3f}, Rg={p['Rg']:.3f}, "
                f"Kp={p['Kp']:.1f}, Ki={p['Ki']:.1f}, "
                f"iLd*={p['iLd']:+.2f}, iLq*={p['iLd']:+.2f},\n"
                f"x(0)"
                f"= [δ0={X0[0] * 180 / np.pi:+.1f}°, Δω0={X0[1]:+.3f}, "
                f"i_gd0={X0[2]:+.2f}, i_gq0={X0[3]:+.2f}]"
            )
            fig.suptitle(subtitle, fontsize=9)
            fig.tight_layout(rect=[0, 0.04, 1, 0.94])
            pdf.savefig(fig)
            plt.show()
            plt.close(fig)

    print(f"\n  {N} cases saved to «{pdf_name}»")

# ---------------- entry-point -----------------------------------
if __name__ == "__main__":
    main()