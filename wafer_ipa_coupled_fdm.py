import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, eye, csc_matrix
from scipy.sparse.linalg import factorized


# ============================================================
# 1. OPERATING CONDITIONS
# ============================================================

# ------------------------------------------------------------
# Wafer
# ------------------------------------------------------------

R = 0.150                  # wafer radius [m]
D = 2.0 * R                # wafer diameter [m]

rpm = 500.0

omega = (
    rpm
    * 2.0
    * np.pi
    / 60.0
)                          # [rad/s]


# ------------------------------------------------------------
# IPA
# ------------------------------------------------------------

T_IPA_in = 110.0           # IPA supply temperature [degC]

Q_IPA_ccmin = 500.0        # [cc/min]

Q_IPA = (
    Q_IPA_ccmin
    * 1.0e-6
    / 60.0
)                          # [m3/s]


# ------------------------------------------------------------
# N2
# ------------------------------------------------------------

T_N2 = 25.0                # [degC]

P_bar = 4.0                # absolute pressure assumption

Q_N2_LPM = 500.0           # actual LPM at chamber condition

Q_N2 = (
    Q_N2_LPM
    * 1.0e-3
    / 60.0
)                          # [m3/s]


# ============================================================
# 2. SILICON WAFER PROPERTIES
# ============================================================

rho_w = 2330.0             # [kg/m3]

cp_w = 700.0               # [J/kg/K]

k_w = 130.0                # [W/m/K]

thickness_w = 0.000775     # [m]


alpha_w = (
    k_w
    /
    (
        rho_w
        * cp_w
    )
)


# wafer heat capacity per unit area

C_w_area = (
    rho_w
    * cp_w
    * thickness_w
)                          # [J/m2/K]


# ============================================================
# 3. INITIAL WAFER TEMPERATURE
# ============================================================

T_initial = 25.0            # [degC]


# ============================================================
# 4. RADIAL FDM GRID
# ============================================================

dr = 0.001                  # 1 mm

N = int(
    R / dr
) + 1


r = np.linspace(
    0.0,
    R,
    N
)


r_mm = (
    r
    * 1000.0
)


# ============================================================
# 5. TIME STEP
#
# Wafer conduction is treated implicitly.
# Therefore this can be considerably larger than explicit FDM.
# ============================================================

dt = 0.01                   # [s]


# ============================================================
# 6. N2 PROPERTIES AT 4 BAR
# ============================================================

rho_N2_1bar = 1.14          # [kg/m3]

nu_N2_1bar = 1.40e-5        # [m2/s]


# ideal-gas approximation

rho_N2 = (
    rho_N2_1bar
    * P_bar
)


nu_N2 = (
    nu_N2_1bar
    / P_bar
)


cp_N2 = 1040.0              # [J/kg/K]

k_N2 = 0.026                # [W/m/K]

Pr_N2 = 0.72

Le = 1.0

C_flow = 0.586


# ============================================================
# 7. N2 VELOCITY
# ============================================================

A_wafer = (
    np.pi
    * R**2
)


Vz = (
    Q_N2
    / A_wafer
)


# ============================================================
# 8. N2 HEAT-TRANSFER COEFFICIENT
# ============================================================

Re_D = (
    Vz
    * D
    / nu_N2
)


h_flow = (
    (k_N2 / D)
    * C_flow
    * np.sqrt(Re_D)
    * Pr_N2**0.33
)


h_rot = (
    0.396
    * k_N2
    * np.sqrt(
        omega
        / nu_N2
    )
)


h_gas = np.sqrt(
    h_flow**2
    + h_rot**2
)


# ============================================================
# 9. GAS-SIDE MASS TRANSFER
# ============================================================

h_m = (
    h_gas
    /
    (
        rho_N2
        * cp_N2
    )
    * Le**(-2.0 / 3.0)
)


# ============================================================
# 10. IPA CONSTANTS
# ============================================================

M_IPA = 0.06010             # [kg/mol]

R_gas = 8.314               # [J/mol/K]

Tc_IPA = 508.3              # [K]

Tb_IPA = (
    82.5
    + 273.15
)                          # [K]


hfg_ref_molar = (
    39.85e3
)                          # [J/mol]


hfg_ref = (
    hfg_ref_molar
    / M_IPA
)                          # [J/kg]


# ============================================================
# 11. IPA DENSITY rho(T)
# ============================================================

def rho_IPA(T):

    T = np.asarray(T)

    rho = (
        782.0
        - 0.85
        * (
            T - 25.0
        )
    )

    return np.maximum(
        rho,
        600.0
    )


# ============================================================
# 12. IPA Cp(T)
# ============================================================

def cp_IPA(T):

    T = np.asarray(T)

    return (
        2550.0
        + 5.5
        * (
            T - 25.0
        )
    )


# ============================================================
# 13. IPA VISCOSITY mu(T)
#
# Approximate engineering interpolation.
# Replace with validated property data if available.
# ============================================================

T_mu_data = np.array([
    25.0,
    60.0,
    80.0,
    100.0,
    110.0
])


mu_data = np.array([
    2.04e-3,
    0.80e-3,
    0.55e-3,
    0.40e-3,
    0.35e-3
])                         # [Pa s]


def mu_IPA(T):

    T_use = np.clip(
        T,
        T_mu_data[0],
        T_mu_data[-1]
    )

    # log interpolation is better for viscosity

    return np.exp(
        np.interp(
            T_use,
            T_mu_data,
            np.log(mu_data)
        )
    )


# ============================================================
# 14. IPA THERMAL CONDUCTIVITY k(T)
# ============================================================

def k_IPA(T):

    k = (
        0.135
        - 2.2e-4
        * (
            np.asarray(T)
            - 25.0
        )
    )

    return np.maximum(
        k,
        0.09
    )


# ============================================================
# 15. IPA LATENT HEAT hfg(T)
#
# Watson correlation
# ============================================================

def h_fg_IPA(T):

    T_K = (
        np.asarray(T)
        + 273.15
    )


    ratio = (
        (
            1.0
            - T_K / Tc_IPA
        )
        /
        (
            1.0
            - Tb_IPA / Tc_IPA
        )
    )


    ratio = np.maximum(
        ratio,
        1.0e-12
    )


    return (
        hfg_ref
        * ratio**0.38
    )


# ============================================================
# 16. IPA SATURATION PRESSURE
# ============================================================

A_IPA = 8.87829
B_IPA = 2010.33
C_IPA = 252.636


def Psat_IPA(T):

    T = np.asarray(T)


    P_mmHg = 10.0**(
        A_IPA
        - B_IPA
        /
        (
            C_IPA
            + T
        )
    )


    return (
        P_mmHg
        * 133.322
    )


# ============================================================
# 17. SATURATED IPA VAPOR DENSITY
# ============================================================

def rho_v_sat_IPA(T):

    T = np.asarray(T)


    return (
        Psat_IPA(T)
        * M_IPA
        /
        (
            R_gas
            * (
                T + 273.15
            )
        )
    )


# ============================================================
# 18. IPA INLET MASS FLOW
#
# 500 cc/min at 110 C
# ============================================================

rho_IPA_in = (
    rho_IPA(
        T_IPA_in
    )
)


mdot_IPA_in = (
    rho_IPA_in
    * Q_IPA
)


# ============================================================
# 19. EFFECTIVE NOZZLE / CENTER RADIUS
#
# Needed because centrifugal-film equation has r=0 singularity.
#
# This is a physical geometry input, NOT a fitting efficiency.
#
# Change this to actual nozzle/wetted footprint if known.
# ============================================================

r_nozzle_eff = 0.005        # 5 mm


# ============================================================
# 20. ROTATING IPA FILM THICKNESS
#
#
# Average radial velocity from centrifugal thin-film flow:
#
# u_bar =
#
# rho * omega^2 * r * delta^2
# --------------------------------
#             3 mu
#
#
# and
#
# Q = 2*pi*r*delta*u_bar
#
#
# Therefore:
#
# delta =
#
# [ 3 mu Q /
#   (2*pi*rho*omega^2*r^2) ]^(1/3)
# ============================================================

def film_thickness(
    T,
    mdot,
    radius
):

    rho_l = float(
        rho_IPA(T)
    )

    mu_l = float(
        mu_IPA(T)
    )


    # local liquid volume flow

    Q_local = (
        mdot
        / rho_l
    )


    r_eff = max(
        radius,
        r_nozzle_eff
    )


    delta = (
        (
            3.0
            * mu_l
            * Q_local
        )
        /
        (
            2.0
            * np.pi
            * rho_l
            * omega**2
            * r_eff**2
        )
    )**(1.0 / 3.0)


    return delta


# ============================================================
# 21. LIQUID-WAFER HEAT-TRANSFER COEFFICIENT
#
# Conduction through thin liquid film:
#
# h_lw = k_IPA / delta
# ============================================================

def h_liquid_wafer(
    T,
    mdot,
    radius
):

    delta = (
        film_thickness(
            T,
            mdot,
            radius
        )
    )


    return (
        float(
            k_IPA(T)
        )
        / delta
    )


# ============================================================
# 22. IPA EVAPORATION MASS FLUX
#
# Dry N2 approximation:
#
# m'' = hm * rho_v,sat
# ============================================================

def evaporation_mass_flux(T):

    return max(
        0.0,
        h_m
        * float(
            rho_v_sat_IPA(T)
        )
    )


# ============================================================
# 23. LIQUID FILM RADIAL SOLUTION
#
#
# Hot IPA enters at r_source.
#
# Because centrifugal force is outward,
# fresh IPA affects primarily:
#
#       r >= r_source
#
#
# For each annulus:
#
# mdot Cp dTl/dr
#
# =
#
# -2*pi*r [
#
#      h_lw(Tl-Tw)
#
#    + h_gas(Tl-TN2)
#
#    + m''evap hfg
#
# ]
#
#
# Also:
#
# d(mdot)/dr =
#
# -2*pi*r*m''evap
# ============================================================

def liquid_film_profile(
    T_wafer,
    r_source
):

    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    T_liquid = np.full(
        N,
        np.nan
    )


    mdot_array = np.zeros(
        N
    )


    delta_array = np.zeros(
        N
    )


    h_lw_array = np.zeros(
        N
    )


    q_lw_array = np.zeros(
        N
    )


    q_gas_array = np.zeros(
        N
    )


    q_latent_array = np.zeros(
        N
    )


    m_evap_array = np.zeros(
        N
    )


    wet = np.zeros(
        N,
        dtype=bool
    )


    # --------------------------------------------------------
    # Nearest radial grid point to nozzle
    # --------------------------------------------------------

    i0 = int(
        np.argmin(
            np.abs(
                r
                - r_source
            )
        )
    )


    # --------------------------------------------------------
    # IPA enters here
    # --------------------------------------------------------

    T_liquid[i0] = (
        T_IPA_in
    )


    mdot_array[i0] = (
        mdot_IPA_in
    )


    wet[i0] = True


    # ========================================================
    # RADIAL MARCHING
    # ========================================================

    for i in range(
        i0,
        N - 1
    ):

        Tl = (
            T_liquid[i]
        )


        mdot = (
            mdot_array[i]
        )


        if (
            not np.isfinite(Tl)
            or mdot <= 1.0e-9
        ):

            break


        wet[i] = True


        # ----------------------------------------------------
        # Film thickness
        # ----------------------------------------------------

        delta = (
            film_thickness(
                Tl,
                mdot,
                r[i]
            )
        )


        delta_array[i] = (
            delta
        )


        # ----------------------------------------------------
        # Liquid-wafer heat transfer
        # ----------------------------------------------------

        h_lw = (
            h_liquid_wafer(
                Tl,
                mdot,
                r[i]
            )
        )


        h_lw_array[i] = (
            h_lw
        )


        q_lw = (
            h_lw
            * (
                Tl
                - T_wafer[i]
            )
        )


        # positive:
        # heat from IPA -> wafer

        q_lw_array[i] = (
            q_lw
        )


        # ----------------------------------------------------
        # Liquid -> N2 sensible heat
        # ----------------------------------------------------

        q_gas = (
            h_gas
            * (
                Tl
                - T_N2
            )
        )


        q_gas_array[i] = (
            q_gas
        )


        # ----------------------------------------------------
        # Evaporation
        # ----------------------------------------------------

        m_evap = (
            evaporation_mass_flux(
                Tl
            )
        )


        m_evap_array[i] = (
            m_evap
        )


        q_latent = (
            m_evap
            * float(
                h_fg_IPA(Tl)
            )
        )


        q_latent_array[i] = (
            q_latent
        )


        # ----------------------------------------------------
        # Annulus area
        #
        # At center use half-cell radius
        # ----------------------------------------------------

        r_area = max(
            r[i],
            dr / 2.0
        )


        dA = (
            2.0
            * np.pi
            * r_area
            * dr
        )


        # ----------------------------------------------------
        # Evaporated liquid mass
        # ----------------------------------------------------

        dm_evap = (
            m_evap
            * dA
        )


        mdot_next = max(
            mdot
            - dm_evap,
            0.0
        )


        # ----------------------------------------------------
        # Liquid energy loss over this annulus
        # ----------------------------------------------------

        q_total_liquid = (
            q_lw
            + q_gas
            + q_latent
        )


        dQ = (
            q_total_liquid
            * dA
        )


        cp_local = float(
            cp_IPA(Tl)
        )


        # ----------------------------------------------------
        # Radial liquid temperature change
        # ----------------------------------------------------

        dT_liquid = (
            dQ
            /
            (
                max(
                    mdot,
                    1.0e-12
                )
                * cp_local
            )
        )


        T_next = (
            Tl
            - dT_liquid
        )


        # ----------------------------------------------------
        # Numerical safety only
        # ----------------------------------------------------

        T_next = np.clip(
            T_next,
            -20.0,
            T_IPA_in
        )


        if mdot_next <= 1.0e-9:

            break


        T_liquid[i + 1] = (
            T_next
        )


        mdot_array[i + 1] = (
            mdot_next
        )


        wet[i + 1] = True


    # ========================================================
    # Calculate quantities at last wet point
    # ========================================================

    wet_indices = np.where(
        wet
    )[0]


    if len(
        wet_indices
    ) > 0:

        i = (
            wet_indices[-1]
        )


        Tl = (
            T_liquid[i]
        )


        mdot = (
            mdot_array[i]
        )


        if (
            np.isfinite(Tl)
            and mdot > 1.0e-9
        ):

            delta = (
                film_thickness(
                    Tl,
                    mdot,
                    r[i]
                )
            )


            h_lw = (
                h_liquid_wafer(
                    Tl,
                    mdot,
                    r[i]
                )
            )


            delta_array[i] = delta

            h_lw_array[i] = h_lw


            q_lw_array[i] = (
                h_lw
                * (
                    Tl
                    - T_wafer[i]
                )
            )


            q_gas_array[i] = (
                h_gas
                * (
                    Tl
                    - T_N2
                )
            )


            m_evap = (
                evaporation_mass_flux(
                    Tl
                )
            )


            m_evap_array[i] = (
                m_evap
            )


            q_latent_array[i] = (
                m_evap
                * float(
                    h_fg_IPA(Tl)
                )
            )


    return (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw_array,
        q_gas_array,
        q_latent_array,
        m_evap_array,
        wet
    )


# ============================================================
# 24. IMPLICIT RADIAL FDM MATRIX FOR WAFER
#
#
# dTw/dt =
#
# alpha [
#
# d2Tw/dr2
#
# +
#
# (1/r)dTw/dr
#
# ]
#
# +
#
# surface heat flux / C_area
# ============================================================

L = lil_matrix(
    (
        N,
        N
    )
)


# ------------------------------------------------------------
# Center
#
# symmetry:
#
# dT/dr = 0
# ------------------------------------------------------------

L[0, 0] = (
    -4.0
    / dr**2
)

L[0, 1] = (
    4.0
    / dr**2
)


# ------------------------------------------------------------
# Interior
# ------------------------------------------------------------

for i in range(
    1,
    N - 1
):

    ri = (
        r[i]
    )


    L[i, i - 1] = (
        1.0
        / dr**2

        -

        1.0
        /
        (
            2.0
            * ri
            * dr
        )
    )


    L[i, i] = (
        -2.0
        / dr**2
    )


    L[i, i + 1] = (
        1.0
        / dr**2

        +

        1.0
        /
        (
            2.0
            * ri
            * dr
        )
    )


# ------------------------------------------------------------
# Edge
#
# radial adiabatic:
#
# dT/dr = 0
# ------------------------------------------------------------

L[-1, -2] = (
    2.0
    / dr**2
)

L[-1, -1] = (
    -2.0
    / dr**2
)


L = csc_matrix(
    L
)


A_fdm = (
    eye(
        N,
        format="csc"
    )
    -
    dt
    * alpha_w
    * L
)


solve_fdm = factorized(
    A_fdm
)


# ============================================================
# 25. ONE COUPLED FDM TIME STEP
# ============================================================

def coupled_step(
    T_wafer,
    r_source
):

    (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw,
        q_liquid_gas,
        q_latent,
        m_evap,
        wet
    ) = liquid_film_profile(
        T_wafer,
        r_source
    )


    # --------------------------------------------------------
    # Where fresh IPA is NOT present, expose wafer directly
    # to N2.
    # --------------------------------------------------------

    q_dry_N2 = np.zeros(
        N
    )


    q_dry_N2[
        ~wet
    ] = (
        h_gas
        * (
            T_wafer[
                ~wet
            ]
            - T_N2
        )
    )


    # --------------------------------------------------------
    # Net external heat flux TO wafer
    # --------------------------------------------------------

    q_surface = (
        q_lw
        - q_dry_N2
    )


    # --------------------------------------------------------
    # Implicit conduction + explicit surface coupling
    # --------------------------------------------------------

    rhs = (
        T_wafer
        +
        dt
        * q_surface
        / C_w_area
    )


    T_new = (
        solve_fdm(
            rhs
        )
    )


    return (
        T_new,
        T_liquid,
        mdot_array,
        delta_array,
        q_lw,
        q_latent,
        wet
    )


# ============================================================
# 26. STAGE 1
#
# 60 s PREHEAT
#
# Nozzle fixed at wafer center
# ============================================================

t_preheat = 60.0

n_preheat = int(
    t_preheat
    / dt
)


T_preheat = np.ones(
    N
) * T_initial


# ------------------------------------------------------------
# History
# ------------------------------------------------------------

time_preheat = []

center_history = []

edge_history = []


edge_66_time = None


record_every = max(
    1,
    int(
        0.10
        / dt
    )
)


print(
    "\n=============================================="
)

print(
    "STAGE 1 : 110 C IPA CENTER PREHEAT"
)

print(
    "=============================================="
)


for n in range(
    n_preheat
):

    time_now = (
        n
        * dt
    )


    (
        T_preheat,
        T_liquid_pre,
        mdot_pre,
        delta_pre,
        q_lw_pre,
        q_lat_pre,
        wet_pre
    ) = coupled_step(
        T_preheat,
        0.0
    )


    # --------------------------------------------------------
    # Detect when edge reaches 66 C
    # --------------------------------------------------------

    if (
        edge_66_time is None
        and T_preheat[-1] >= 66.0
    ):

        edge_66_time = (
            time_now
            + dt
        )


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    if n % record_every == 0:

        time_preheat.append(
            time_now
            + dt
        )


        center_history.append(
            T_preheat[0]
        )


        edge_history.append(
            T_preheat[-1]
        )


# ------------------------------------------------------------
# Recalculate liquid distribution for final preheat state
# ------------------------------------------------------------

(
    T_liquid_pre,
    mdot_pre,
    delta_pre,
    h_lw_pre,
    q_lw_pre,
    q_gas_pre,
    q_lat_pre,
    mevap_pre,
    wet_pre
) = liquid_film_profile(
    T_preheat,
    0.0
)


print(
    f"Wafer center @60 s = "
    f"{T_preheat[0]:.3f} C"
)

print(
    f"Wafer edge   @60 s = "
    f"{T_preheat[-1]:.3f} C"
)


if edge_66_time is None:

    print(
        "Wafer edge did NOT reach 66 C during 60 s."
    )

else:

    print(
        f"Wafer edge first reached 66 C at "
        f"{edge_66_time:.2f} s"
    )


# ============================================================
# 27. STAGE 2
#
# FAIR COMPARISON:
#
# Start both cases from exactly the same 60 s preheat profile.
#
# Case A:
# nozzle stays at center
#
# Case B:
# nozzle swings:
#
# 0 -> 150 -> 0 mm
#
# at 20 mm/s
#
# One complete cycle = 15 s
# ============================================================

v_swing = (
    20.0e-3
)                          # [m/s]


t_swing_cycle = (
    2.0
    * R
    / v_swing
)                          # 15 s


n_stage2 = int(
    t_swing_cycle
    / dt
)


# ============================================================
# 28. TRIANGULAR SWING MOTION
# ============================================================

def swing_position(t):

    distance = (
        v_swing
        * t
    )


    if distance <= R:

        return distance


    return (
        2.0
        * R
        - distance
    )


# ============================================================
# 29. NO-SWING CASE
# ============================================================

T_no_swing = (
    T_preheat.copy()
)


# history

time_stage2 = []

T0_no_history = []

T150_no_history = []


for n in range(
    n_stage2
):

    t = (
        n
        * dt
    )


    (
        T_no_swing,
        _,
        _,
        _,
        _,
        _,
        _
    ) = coupled_step(
        T_no_swing,
        0.0
    )


    if n % record_every == 0:

        time_stage2.append(
            t
        )


        T0_no_history.append(
            T_no_swing[0]
        )


        T150_no_history.append(
            T_no_swing[-1]
        )


# ============================================================
# 30. SWING CASE
# ============================================================

T_swing = (
    T_preheat.copy()
)


T_swing_sum = np.zeros(
    N
)


source_history = []

T0_swing_history = []

T150_swing_history = []


for n in range(
    n_stage2
):

    t = (
        n
        * dt
    )


    r_source = (
        swing_position(
            t
        )
    )


    (
        T_swing,
        _,
        _,
        _,
        _,
        _,
        _
    ) = coupled_step(
        T_swing,
        r_source
    )


    T_swing_sum += (
        T_swing
    )


    if n % record_every == 0:

        source_history.append(
            r_source
            * 1000.0
        )


        T0_swing_history.append(
            T_swing[0]
        )


        T150_swing_history.append(
            T_swing[-1]
        )


# ------------------------------------------------------------
# Cycle-average temperature
# ------------------------------------------------------------

T_swing_average = (
    T_swing_sum
    / n_stage2
)


# ============================================================
# 31. PRINT CONDITIONS
# ============================================================

print(
    "\n=============================================="
)

print(
    "OPERATING CONDITIONS"
)

print(
    "=============================================="
)


print(
    f"IPA inlet T      = "
    f"{T_IPA_in:.1f} C"
)


print(
    f"IPA flow         = "
    f"{Q_IPA_ccmin:.1f} cc/min"
)


print(
    f"IPA inlet rho    = "
    f"{rho_IPA_in:.2f} kg/m3"
)


print(
    f"IPA mass flow    = "
    f"{mdot_IPA_in:.6f} kg/s"
)


print(
    f"N2 pressure      = "
    f"{P_bar:.1f} bar"
)


print(
    f"N2 flow          = "
    f"{Q_N2_LPM:.1f} LPM"
)


print(
    f"Rotation         = "
    f"{rpm:.1f} rpm"
)


print(
    f"h_flow           = "
    f"{h_flow:.3f} W/m2/K"
)


print(
    f"h_rot            = "
    f"{h_rot:.3f} W/m2/K"
)


print(
    f"h_gas            = "
    f"{h_gas:.3f} W/m2/K"
)


print(
    f"Swing speed      = "
    f"{v_swing*1000:.1f} mm/s"
)


print(
    f"Swing cycle      = "
    f"{t_swing_cycle:.2f} s"
)


# ============================================================
# 32. IMPORTANT RADIAL RESULTS
# ============================================================

positions_mm = [
    0,
    30,
    60,
    90,
    120,
    150
]


print(
    "\n============================================================"
)

print(
    "RADIAL TEMPERATURE RESULTS"
)

print(
    "============================================================"
)


for pos in positions_mm:

    idx = np.argmin(
        np.abs(
            r_mm
            - pos
        )
    )


    print(
        f"\nr = {r_mm[idx]:.0f} mm"
    )


    print(
        f"  preheat 60 s     = "
        f"{T_preheat[idx]:8.3f} C"
    )


    print(
        f"  no swing +15 s   = "
        f"{T_no_swing[idx]:8.3f} C"
    )


    print(
        f"  swing final      = "
        f"{T_swing[idx]:8.3f} C"
    )


    print(
        f"  swing cycle avg  = "
        f"{T_swing_average[idx]:8.3f} C"
    )


# ============================================================
# 33. GRAPH 1
#
# MAIN WAFER TEMPERATURE COMPARISON
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    r_mm,
    T_preheat,
    linestyle="--",
    linewidth=2.5,
    label="60 s preheat - center nozzle"
)


plt.plot(
    r_mm,
    T_no_swing,
    linewidth=2,
    label="No swing - additional 15 s"
)


plt.plot(
    r_mm,
    T_swing,
    linewidth=2,
    label="Swing final - 20 mm/s"
)


plt.plot(
    r_mm,
    T_swing_average,
    linewidth=2,
    label="Swing cycle average"
)


plt.xlabel(
    "Radial position r [mm]"
)


plt.ylabel(
    "Wafer temperature [°C]"
)


plt.title(
    "Coupled IPA Film + Wafer FDM\n"
    "IPA 110°C, 500 cc/min, 4 bar N2, 500 rpm"
)


plt.ylim(
    20,
    115
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 34. GRAPH 2
#
# IPA TEMPERATURE PROFILE AFTER 60 s PREHEAT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    r_mm[
        wet_pre
    ],
    T_liquid_pre[
        wet_pre
    ],
    linewidth=2,
    label="IPA film"
)


plt.plot(
    r_mm,
    T_preheat,
    linewidth=2,
    linestyle="--",
    label="Wafer"
)


plt.xlabel(
    "Radial position r [mm]"
)


plt.ylabel(
    "Temperature [°C]"
)


plt.title(
    "IPA Film and Wafer Temperature After 60 s"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 35. GRAPH 3
#
# WAFER CENTER / EDGE DURING PREHEAT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    time_preheat,
    center_history,
    linewidth=2,
    label="r = 0 mm"
)


plt.plot(
    time_preheat,
    edge_history,
    linewidth=2,
    label="r = 150 mm"
)


plt.axhline(
    66.0,
    linestyle="--",
    linewidth=1.5,
    label="66 °C"
)


plt.xlabel(
    "Preheat time [s]"
)


plt.ylabel(
    "Wafer temperature [°C]"
)


plt.title(
    "Center and Edge Temperature During Preheat"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 36. GRAPH 4
#
# SWING POSITION
# ============================================================

plt.figure(
    figsize=(8, 4)
)


plt.plot(
    time_stage2,
    source_history,
    linewidth=2
)


plt.xlabel(
    "Time after preheat [s]"
)


plt.ylabel(
    "Nozzle radial position [mm]"
)


plt.title(
    "IPA Nozzle Swing - 20 mm/s"
)


plt.ylim(
    0,
    155
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()


# ============================================================
# 37. GRAPH 5
#
# CENTER TEMPERATURE AFTER PREHEAT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    time_stage2,
    T0_no_history,
    linewidth=2,
    label="No swing"
)


plt.plot(
    time_stage2,
    T0_swing_history,
    linewidth=2,
    label="Swing"
)


plt.xlabel(
    "Time after 60 s preheat [s]"
)


plt.ylabel(
    "Wafer T at r = 0 mm [°C]"
)


plt.title(
    "Center Temperature: No Swing vs Swing"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 38. GRAPH 6
#
# EDGE TEMPERATURE AFTER PREHEAT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    time_stage2,
    T150_no_history,
    linewidth=2,
    label="No swing"
)


plt.plot(
    time_stage2,
    T150_swing_history,
    linewidth=2,
    label="Swing"
)


plt.xlabel(
    "Time after 60 s preheat [s]"
)


plt.ylabel(
    "Wafer T at r = 150 mm [°C]"
)


plt.title(
    "Edge Temperature: No Swing vs Swing"
)


plt.grid(
    True
)


plt.legend()


plt.tight_layout()


plt.show()


# ============================================================
# 39. GRAPH 7
#
# FILM THICKNESS AFTER PREHEAT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


valid_delta = (
    delta_pre > 0.0
)


plt.plot(
    r_mm[
        valid_delta
    ],
    delta_pre[
        valid_delta
    ]
    * 1.0e6,
    linewidth=2
)


plt.xlabel(
    "Radial position r [mm]"
)


plt.ylabel(
    "IPA film thickness [um]"
)


plt.title(
    "Estimated Rotating IPA Film Thickness"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()

# ============================================================
# TEMPERATURE COMPARISON
#
# 1. 60 s preheat
# 2. No swing + 15 s
# 3. Swing 20 mm/s - final instant
# 4. Swing 20 mm/s - cycle average
# ============================================================

plt.figure(
    figsize=(9, 6)
)


# ------------------------------------------------------------
# 1. 60 s PREHEAT
# nozzle fixed at center
# ------------------------------------------------------------

plt.plot(
    r_mm,
    T_preheat,
    linestyle="--",
    linewidth=2.2,
    label="60 s preheat - center nozzle"
)


# ------------------------------------------------------------
# 2. NO SWING
# nozzle remains at center for additional 15 s
# ------------------------------------------------------------

plt.plot(
    r_mm,
    T_no_swing,
    linewidth=2.2,
    label="No swing - plus 15 s"
)


# ------------------------------------------------------------
# 3. SWING
#
# nozzle:
#
# 0 -> 150 -> 0 mm
#
# speed = 20 mm/s
#
# this is temperature at the END of one cycle
# ------------------------------------------------------------

plt.plot(
    r_mm,
    T_swing,
    linewidth=2.2,
    label="Swing 20 mm/s - final"
)


# ------------------------------------------------------------
# 4. SWING CYCLE-AVERAGED TEMPERATURE
#
# Average temperature during entire 15 s swing cycle
# ------------------------------------------------------------

plt.plot(
    r_mm,
    T_swing_average,
    linestyle="-.",
    linewidth=2.2,
    label="Swing 20 mm/s - cycle average"
)


# ------------------------------------------------------------
# Graph settings
# ------------------------------------------------------------

plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "Wafer temperature [°C]"
)

plt.title(
    "Coupled IPA Film + Wafer FDM\n"
    "IPA 110°C, 500 cc/min, 4 bar N2, 500 rpm"
)

plt.xlim(
    0,
    150
)

plt.ylim(
    0,
    120
)

plt.xticks(
    np.arange(
        0,
        151,
        20
    )
)

plt.yticks(
    np.arange(
        0,
        121,
        20
    )
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# EVAPORATION CONTRIBUTION STUDY
#
# Case 1 : evaporation ON  -> original model
# Case 2 : evaporation OFF -> h_m = 0
#
# Everything else is identical.
# ============================================================


# ------------------------------------------------------------
# Save original mass-transfer coefficient
# ------------------------------------------------------------

h_m_original = h_m


# ============================================================
# 1. FUNCTION : RUN 60 s PREHEAT
# ============================================================

def run_preheat_evap_case(
    h_m_value
):

    global h_m

    h_m = h_m_value


    # Start both cases from exactly same wafer temperature

    T_case = (
        np.ones(N)
        * T_initial
    )


    # 60 s preheat
    # nozzle fixed at center

    for n in range(
        n_preheat
    ):

        (
            T_case,
            _,
            _,
            _,
            _,
            _,
            _
        ) = coupled_step(
            T_case,
            0.0
        )


    # --------------------------------------------------------
    # Recalculate final liquid profile using final wafer T
    # --------------------------------------------------------

    (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw,
        q_gas,
        q_latent,
        m_evap,
        wet
    ) = liquid_film_profile(
        T_case,
        0.0
    )


    return (
        T_case,
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw,
        q_gas,
        q_latent,
        m_evap,
        wet
    )


# ============================================================
# 2. EVAPORATION ON
# ============================================================

(
    T_evap_on,
    Tl_evap_on,
    mdot_evap_on,
    delta_evap_on,
    hlw_evap_on,
    qlw_evap_on,
    qgas_evap_on,
    qlat_evap_on,
    mevap_on,
    wet_evap_on
) = run_preheat_evap_case(
    h_m_original
)


# ============================================================
# 3. EVAPORATION OFF
#
# h_m = 0
#
# Therefore:
# m''evap = 0
# q_latent = 0
# mdot does not decrease due to evaporation
# ============================================================

(
    T_evap_off,
    Tl_evap_off,
    mdot_evap_off,
    delta_evap_off,
    hlw_evap_off,
    qlw_evap_off,
    qgas_evap_off,
    qlat_evap_off,
    mevap_off,
    wet_evap_off
) = run_preheat_evap_case(
    0.0
)


# Restore original model
h_m = h_m_original


# ============================================================
# 4. PRINT TEMPERATURE COMPARISON
# ============================================================

positions_mm = [
    0,
    30,
    60,
    90,
    120,
    150
]


print(
    "\n=============================================================="
)

print(
    "EVAPORATION EFFECT - 60 s PREHEAT"
)

print(
    "=============================================================="
)

print(
    " r [mm]   Evap ON [C]   Evap OFF [C]   OFF - ON [C]"
)

print(
    "--------------------------------------------------------------"
)


for pos in positions_mm:

    idx = np.argmin(
        np.abs(
            r_mm
            - pos
        )
    )


    dT_evap = (
        T_evap_off[idx]
        - T_evap_on[idx]
    )


    print(
        f"{r_mm[idx]:7.0f}"
        f"{T_evap_on[idx]:14.3f}"
        f"{T_evap_off[idx]:15.3f}"
        f"{dT_evap:15.3f}"
    )


# ============================================================
# 5. GRAPH 1
#
# WAFER TEMPERATURE:
# evaporation ON vs OFF
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    r_mm,
    T_evap_on,
    linewidth=2.5,
    label="Evaporation ON"
)


plt.plot(
    r_mm,
    T_evap_off,
    linestyle="--",
    linewidth=2.5,
    label="Evaporation OFF"
)


plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "Wafer temperature after 60 s [°C]"
)

plt.title(
    "Effect of IPA Evaporation on Wafer Temperature\n"
    "60 s Preheat"
)

plt.xlim(
    0,
    150
)

plt.ylim(
    20,
    115
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 6. GRAPH 2
#
# TEMPERATURE REDUCTION CAUSED BY EVAPORATION
#
# positive value:
# evaporation lowers wafer temperature
# ============================================================

delta_T_evap = (
    T_evap_off
    - T_evap_on
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    r_mm,
    delta_T_evap,
    linewidth=2.5
)


plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0
)


plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "Temperature reduction by evaporation [°C]"
)

plt.title(
    "Contribution of Evaporation to Wafer Cooling\n"
    "T(no evaporation) - T(evaporation)"
)

plt.xlim(
    0,
    150
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 7. GRAPH 3
#
# q_lw WITH vs WITHOUT EVAPORATION
#
# Shows how evaporation indirectly changes
# IPA -> wafer heat transfer
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    r_mm[
        wet_evap_on
    ],
    qlw_evap_on[
        wet_evap_on
    ],
    linewidth=2.5,
    label="q_lw : evaporation ON"
)


plt.plot(
    r_mm[
        wet_evap_off
    ],
    qlw_evap_off[
        wet_evap_off
    ],
    linestyle="--",
    linewidth=2.5,
    label="q_lw : evaporation OFF"
)


plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "q_lw [W/m²]"
)

plt.title(
    "Effect of Evaporation on IPA-to-Wafer Heat Flux\n"
    "After 60 s Preheat"
)

plt.xlim(
    0,
    150
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 8. GRAPH 4
#
# LIQUID ENERGY-LOSS COMPONENTS
# evaporation ON case
#
# q_lw
# q_gas
# q_latent
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    r_mm[
        wet_evap_on
    ],
    qlw_evap_on[
        wet_evap_on
    ],
    linewidth=2.2,
    label="q_lw : liquid -> wafer"
)


plt.plot(
    r_mm[
        wet_evap_on
    ],
    qgas_evap_on[
        wet_evap_on
    ],
    linewidth=2.2,
    label="q_gas : liquid -> N2"
)


plt.plot(
    r_mm[
        wet_evap_on
    ],
    qlat_evap_on[
        wet_evap_on
    ],
    linewidth=2.2,
    label="q_latent : evaporation"
)


plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "Heat flux [W/m²]"
)

plt.title(
    "IPA Energy-Loss Contributions\n"
    "Evaporation ON, after 60 s"
)

plt.xlim(
    0,
    150
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 9. GRAPH 5
#
# LOCAL LATENT-HEAT CONTRIBUTION [%]
#
# q_latent /
# (q_lw + q_gas + q_latent)
# ============================================================

q_total_on = (
    qlw_evap_on
    + qgas_evap_on
    + qlat_evap_on
)


latent_fraction = np.zeros(
    N
)


valid = (
    wet_evap_on
    &
    (
        q_total_on
        > 1.0e-12
    )
)


latent_fraction[
    valid
] = (
    100.0
    * qlat_evap_on[
        valid
    ]
    / q_total_on[
        valid
    ]
)


plt.figure(
    figsize=(9, 5)
)


plt.plot(
    r_mm[
        valid
    ],
    latent_fraction[
        valid
    ],
    linewidth=2.5
)


plt.xlabel(
    "Radial position r [mm]"
)

plt.ylabel(
    "Latent heat contribution [%]"
)

plt.title(
    "Local Contribution of IPA Evaporation\n"
    "q_latent / (q_lw + q_gas + q_latent)"
)

plt.xlim(
    0,
    150
)

plt.ylim(
    bottom=0
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 10. INTEGRATED HEAT-RATE CONTRIBUTION
#
# P = integral q'' * 2*pi*r dr
# ============================================================

P_lw = np.trapezoid(
    qlw_evap_on
    * 2.0
    * np.pi
    * r,
    r
)


P_gas = np.trapezoid(
    qgas_evap_on
    * 2.0
    * np.pi
    * r,
    r
)


P_latent = np.trapezoid(
    qlat_evap_on
    * 2.0
    * np.pi
    * r,
    r
)


P_total = (
    P_lw
    + P_gas
    + P_latent
)


print(
    "\n=============================================="
)

print(
    "FINAL-INSTANT IPA ENERGY CONTRIBUTION @ 60 s"
)

print(
    "=============================================="
)

print(
    f"IPA -> wafer   : {P_lw:.3f} W"
)

print(
    f"IPA -> N2      : {P_gas:.3f} W"
)

print(
    f"Evaporation    : {P_latent:.3f} W"
)

print(
    f"Total          : {P_total:.3f} W"
)


if P_total > 0.0:

    print(
        f"Latent fraction: "
        f"{100.0 * P_latent / P_total:.2f} %"
    )


# ============================================================
# 11. BAR GRAPH:
# integrated contribution at 60 s
# ============================================================

labels = [
    "IPA -> wafer",
    "IPA -> N2",
    "Evaporation"
]

powers = [
    P_lw,
    P_gas,
    P_latent
]


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    labels,
    powers
)


plt.ylabel(
    "Integrated heat rate [W]"
)

plt.title(
    "IPA Energy-Loss Contribution at 60 s"
)

plt.grid(
    True,
    axis="y"
)

plt.tight_layout()

plt.show()

q_lw_loss = np.maximum(
    qlw_evap_on,
    0.0
)

q_lw_gain = np.maximum(
    -qlw_evap_on,
    0.0
)

q_loss_total = (
    q_lw_loss
    + qgas_evap_on
    + qlat_evap_on
)

latent_fraction = np.zeros(N)

valid = (
    wet_evap_on
    &
    (q_loss_total > 1.0e-12)
)

latent_fraction[valid] = (
    100.0
    * qlat_evap_on[valid]
    / q_loss_total[valid]
)


plt.figure(figsize=(9, 5))

plt.plot(
    r_mm[valid],
    latent_fraction[valid],
    linewidth=2.5
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("Evaporation contribution [%]")

plt.title(
    "Evaporation Share of IPA Heat Loss\n"
    "q_latent / (q_lw,loss + q_gas + q_latent)"
)

plt.xlim(0, 150)
plt.ylim(0, 100)

plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. liquid_film_profile wrapper
#    apply_evap_to_mdot 옵션 추가 버전
# ============================================================

def liquid_film_profile_modified(
    T_wafer,
    r_source,
    apply_evap_to_mdot=True
):
    """
    Same film model as liquid_film_profile(), with one diagnostic option:

    apply_evap_to_mdot=True
        Evaporation reduces the remaining liquid mass flow.

    apply_evap_to_mdot=False
        Latent heat is retained, but the remaining liquid mass flow is
        intentionally not reduced. This is a sensitivity test, not a
        fully mass-conservative physical model.
    """

    T_liquid = np.full(N, np.nan)
    mdot_array = np.zeros(N)
    delta_array = np.zeros(N)
    h_lw_array = np.zeros(N)
    q_lw_array = np.zeros(N)
    q_gas_array = np.zeros(N)
    q_latent_array = np.zeros(N)
    m_evap_array = np.zeros(N)
    wet = np.zeros(N, dtype=bool)

    i0 = int(np.argmin(np.abs(r - r_source)))

    T_liquid[i0] = T_IPA_in
    mdot_array[i0] = mdot_IPA_in
    wet[i0] = True

    for i in range(i0, N - 1):

        Tl = T_liquid[i]
        mdot = mdot_array[i]

        if (not np.isfinite(Tl)) or mdot <= 1.0e-9:
            break

        wet[i] = True

        delta = film_thickness(Tl, mdot, r[i])
        h_lw = h_liquid_wafer(Tl, mdot, r[i])

        q_lw = h_lw * (Tl - T_wafer[i])
        q_gas = h_gas * (Tl - T_N2)

        m_evap = evaporation_mass_flux(Tl)
        q_latent = m_evap * float(h_fg_IPA(Tl))

        delta_array[i] = delta
        h_lw_array[i] = h_lw
        q_lw_array[i] = q_lw
        q_gas_array[i] = q_gas
        q_latent_array[i] = q_latent
        m_evap_array[i] = m_evap

        r_area = max(r[i], dr / 2.0)
        dA = 2.0 * np.pi * r_area * dr

        dm_evap = m_evap * dA

        if apply_evap_to_mdot:
            mdot_next = max(mdot - dm_evap, 0.0)
        else:
            mdot_next = mdot

        q_total_liquid = q_lw + q_gas + q_latent
        dQ = q_total_liquid * dA

        cp_local = float(cp_IPA(Tl))

        dT_liquid = (
            dQ
            / (
                max(mdot, 1.0e-12)
                * cp_local
            )
        )

        T_next = np.clip(
            Tl - dT_liquid,
            -20.0,
            T_IPA_in
        )

        if mdot_next <= 1.0e-9:
            break

        T_liquid[i + 1] = T_next
        mdot_array[i + 1] = mdot_next
        wet[i + 1] = True

    # Evaluate quantities at the final wet point.
    wet_indices = np.where(wet)[0]

    if len(wet_indices) > 0:

        i = wet_indices[-1]
        Tl = T_liquid[i]
        mdot = mdot_array[i]

        if np.isfinite(Tl) and mdot > 1.0e-9:

            delta = film_thickness(Tl, mdot, r[i])
            h_lw = h_liquid_wafer(Tl, mdot, r[i])

            delta_array[i] = delta
            h_lw_array[i] = h_lw

            q_lw_array[i] = h_lw * (Tl - T_wafer[i])
            q_gas_array[i] = h_gas * (Tl - T_N2)

            m_evap = evaporation_mass_flux(Tl)
            m_evap_array[i] = m_evap
            q_latent_array[i] = m_evap * float(h_fg_IPA(Tl))

    return (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw_array,
        q_gas_array,
        q_latent_array,
        m_evap_array,
        wet
    )


# ============================================================
# 2. coupled_step wrapper
# ============================================================

def coupled_step_modified(
    T_wafer,
    r_source,
    apply_evap_to_mdot=True
):

    (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw,
        q_liquid_gas,
        q_latent,
        m_evap,
        wet
    ) = liquid_film_profile_modified(
        T_wafer,
        r_source,
        apply_evap_to_mdot=apply_evap_to_mdot
    )

    # Dry wafer region is exposed directly to N2.
    q_dry_N2 = np.zeros(N)

    q_dry_N2[~wet] = (
        h_gas
        * (
            T_wafer[~wet]
            - T_N2
        )
    )

    # Positive q_surface means net heat flux into the wafer.
    q_surface = q_lw - q_dry_N2

    rhs = (
        T_wafer
        + dt
        * q_surface
        / C_w_area
    )

    T_new = solve_fdm(rhs)

    return (
        T_new,
        T_liquid,
        mdot_array,
        delta_array,
        q_lw,
        q_latent,
        wet
    )


# ============================================================
# 3. Run 60 s preheat
# ============================================================

def run_preheat_case(
    apply_evap_to_mdot=True
):
    T_case = np.ones(N) * T_initial

    for _ in range(n_preheat):
        (
            T_case,
            _,
            _,
            _,
            _,
            _,
            _
        ) = coupled_step_modified(
            T_case,
            0.0,
            apply_evap_to_mdot=apply_evap_to_mdot
        )

    (
        T_liquid,
        mdot_array,
        delta_array,
        h_lw_array,
        q_lw,
        q_gas,
        q_latent,
        m_evap,
        wet
    ) = liquid_film_profile_modified(
        T_case,
        0.0,
        apply_evap_to_mdot=apply_evap_to_mdot
    )

    return {
        "T_wafer": T_case,
        "T_liquid": T_liquid,
        "mdot": mdot_array,
        "delta": delta_array,
        "h_lw": h_lw_array,
        "q_lw": q_lw,
        "q_gas": q_gas,
        "q_latent": q_latent,
        "m_evap": m_evap,
        "wet": wet
    }


# ============================================================
# 4. Two cases
# ============================================================

# Case A: original
case_full = run_preheat_case(
    apply_evap_to_mdot=True
)

# Case B: latent kept, but no mdot reduction
case_no_mdot_loss = run_preheat_case(
    apply_evap_to_mdot=False
)


# ============================================================
# 5. Print comparison
# ============================================================

print("\n====================================================")
print("Comparison: Full evaporation coupling vs No mdot loss")
print("====================================================")
print("r[mm] | Tw_full | Tw_no_mdot_loss | qlw_full | qlw_no_mdot_loss")

for pos in [0, 25, 50, 75, 100, 125, 150]:
    idx = np.argmin(np.abs(r_mm - pos))
    print(
        f"{r_mm[idx]:5.0f} | "
        f"{case_full['T_wafer'][idx]:8.3f} | "
        f"{case_no_mdot_loss['T_wafer'][idx]:15.3f} | "
        f"{case_full['q_lw'][idx]:8.2f} | "
        f"{case_no_mdot_loss['q_lw'][idx]:15.2f}"
    )


# ============================================================
# 6. Graph 1: wafer temperature
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    r_mm,
    case_full["T_wafer"],
    linewidth=2.5,
    label="Full evaporation coupling"
)

plt.plot(
    r_mm,
    case_no_mdot_loss["T_wafer"],
    linestyle="--",
    linewidth=2.5,
    label="Latent ON, mdot reduction OFF"
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("Wafer temperature [°C]")
plt.title("Wafer temperature after preheat")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. Graph 2: q_lw comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    r_mm,
    case_full["q_lw"],
    linewidth=2.5,
    label="q_lw : full coupling"
)

plt.plot(
    r_mm,
    case_no_mdot_loss["q_lw"],
    linestyle="--",
    linewidth=2.5,
    label="q_lw : latent ON, no mdot reduction"
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("q_lw [W/m²]")
plt.title("Effect of removing only mdot reduction term")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. Graph 3: mdot comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    r_mm,
    case_full["mdot"],
    linewidth=2.5,
    label="mdot : full coupling"
)

plt.plot(
    r_mm,
    case_no_mdot_loss["mdot"],
    linestyle="--",
    linewidth=2.5,
    label="mdot : no mdot reduction"
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("Liquid mass flow rate [kg/s]")
plt.title("Liquid mass flow comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 9. Graph 4: film thickness
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    r_mm,
    case_full["delta"] * 1e6,
    linewidth=2.5,
    label="delta : full coupling"
)

plt.plot(
    r_mm,
    case_no_mdot_loss["delta"] * 1e6,
    linestyle="--",
    linewidth=2.5,
    label="delta : no mdot reduction"
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("Film thickness [µm]")
plt.title("Film thickness comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. Graph 5: h_lw comparison
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    r_mm,
    case_full["h_lw"],
    linewidth=2.5,
    label="h_lw : full coupling"
)

plt.plot(
    r_mm,
    case_no_mdot_loss["h_lw"],
    linestyle="--",
    linewidth=2.5,
    label="h_lw : no mdot reduction"
)

plt.xlabel("Radial position r [mm]")
plt.ylabel("h_lw [W/m²/K]")
plt.title("IPA-to-wafer heat transfer coefficient")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
