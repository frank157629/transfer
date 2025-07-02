import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

desktop = Path.home() / "Desktop"
pdf_name = desktop / "results_2nd_order.pdf"

##Second-order non-linear Model of a GridFollowingConverterModels converter
#param
params = dict(
    Vg = 1.0,   ## Grid voltage magnitude (p.u.)
    wN = 1.0,   ## Nominal angular frequency (p.u.)
    wg = 1.0,   ## Actual grid frequency (p.u.)
    Lg = 0.15,  ## Filter inductance (p.u.)
    Rg = 0.015, ## Filter resistance  (p.u.)
    Kp = 20.0,  ## GFL_2nd_order proportional gain
    Ki = 400.0, ## GFL_2nd_order integral gain
    iLd = 0.0,  ## d-axis current reference (p.u.)
    iLq = -0.8  ## q-axis current reference (p.u.)
)

#Right hand side of ODE
def rhs(t, X , p):
    x1, x2 = X
    delta = x1
    Dw = x2
    ddelta = x2 + p['wN'] - p['wg']
    term1 = p['Kp'] * (-1) * (ddelta * np.cos(x1) * p['Vg'])
    term2 = p['Ki']*(p['wN']*p['Lg']*p['iLd'] + p['Rg']*p['iLq'] - np.sin(x1)*p['Vg'])
    dDw = term1 + term2
    return np.array([ddelta, dDw])

#Run RK45 ODE-solver, return time and states
def run_case(p, x0, t_end=1.0, h=0.01):
    n_pts = int(t_end / h) + 1
    t_eval = np.linspace(0.0, t_end, n_pts)
    sol = solve_ivp(fun=rhs,
                    t_span=(0, t_end),
                    y0=x0,
                    t_eval=t_eval,
                    args=(p,),
                    method='RK45'
                    )
    return sol.t, sol.y
#Run 10 simulations, each with different params. and same initial value
def main(N=10, seed=42, pdf_name=str(Path.home()/ "Desktop" / "results_2nd_order.pdf")):
    rng = np.random.default_rng(seed)
    with PdfPages(pdf_name) as pdf:
        for _ in range(N):
            #Copy base parameters
            p = params.copy()
            #Randomise parameters
            p["wg"] = 1.0 + rng.uniform(-0.02,0.02)   # 0.98 〜 1.02 p.u.
            X0 = np.array([0.0, 0.0])
            p["iLd"] = rng.uniform(-1.0, 1.0)
            p["iLq"] = rng.uniform(-1.0, 1.0)
            t, y = run_case(p, X0)
            fig, axs = plt.subplots(2, 1, figsize=(6,6))
            #delta
            axs[0].plot(t, y[0]*180/np.pi)
            axs[0].set_ylabel("x1_δ(deg)")
            #Δω
            axs[1].plot(t, y[1])
            axs[1].set_ylabel("x2_Δω(p.u.)")
            axs[1].set_xlabel("Time(s)")
            #subtitle
            subtitle = (
                f"Vg={p['Vg']:.2f}, wN={p['wN']:.2f}, wg={p['wg']:.2f}, "
                f"Lg={p['Lg']:.3f}, Rg={p['Rg']:.3f}, "
                f"Kp={p['Kp']:.1f}, Ki={p['Ki']:.1f}, "
                f"iLd*={p['iLd']:+.2f}, iLq*={p['iLd']:+.2f},\n"
                f"x(0)= [δ0={X0[0] * 180 / np.pi:+.1f}°, Δω0={X0[1]:+.3f}] "
            )
            fig.suptitle(subtitle, fontsize=9)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)  #adds page
            plt.show(block=True)
            plt.close(fig)  # free memory
    print(f"All {N} figures saved to '{pdf_name}'")

#-------------------Main Function --------------------
if __name__ == "__main__":
    main(pdf_name=str(Path.home()/ "Desktop" / "results_2nd_order.pdf"))





