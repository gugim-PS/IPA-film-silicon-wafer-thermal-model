# Coupled IPA Film + Silicon Wafer Thermal Model by FDM

Solution with Python.

This model calculates the transient radial temperature of a rotating silicon wafer heated by hot IPA while exchanging heat and mass with N2.

## Main Operating Conditions

Wafer radius: 150 mm

Wafer rotation: 500 rpm

IPA temperature: 110 °C

IPA flow rate: 500 cc/min

N2 temperature: 25 °C

N2 pressure assumption: 4 bar absolute

N2 flow rate: 500 LPM

Radial grid size: 1 mm

Time step: 0.01 s

## Wafer Heat-Conduction Equation

The wafer is modeled with one-dimensional axisymmetric radial heat conduction:

$$ {\partial T_w\over\partial t}=\alpha_w\left({\partial^2T_w\over\partial r^2}+{1\over r}{\partial T_w\over\partial r}\right)+{q''{\mathrm{surface}}\over C{w,\mathrm{area}}} $$

where:

$$ \alpha_w={k_w\over\rho_wc_{p,w}} $$

and:

$$ C_{w,\mathrm{area}}=\rho_wc_{p,w}t_w $$

The radial conduction term is solved using an implicit finite-difference method (FDM).

## Rotating IPA Film

The liquid-film thickness is estimated from centrifugal thin-film flow:

$$ \delta=\left({3\mu Q\over2\pi\rho\omega^2r^2}\right)^{1/3} $$

The liquid-to-wafer heat-transfer coefficient is approximated by conduction through the film:

$$ h_{lw}={k_{\mathrm{IPA}}\over\delta} $$

The corresponding heat flux is:

$$ q''{lw}=h{lw}(T_l-T_w) $$

## IPA Evaporation

The evaporation mass flux is estimated from the gas-side mass-transfer coefficient:

$$ \dot m''{\mathrm{evap}}=h_m\rho{v,\mathrm{sat}} $$

The latent heat loss is:

$$ q''{\mathrm{latent}}=\dot m''{\mathrm{evap}}h_{fg} $$

The IPA temperature and remaining liquid mass flow are marched radially from the nozzle position toward the wafer edge.

## Simulation Stages

### Stage 1: Preheat

The nozzle is fixed at the wafer center for:

$$ t=60\ \mathrm{s} $$

### Stage 2: No-Swing vs Swing

Both cases start from the same 60 s preheat temperature profile.

The swing case moves the nozzle:

$$ 0\rightarrow150\rightarrow0\ \mathrm{mm} $$

at:

$$ v=20\ \mathrm{mm/s} $$

so one complete swing cycle is:

$$ t_{\mathrm{cycle}}=15\ \mathrm{s} $$

## Evaporation Study

The script also compares:

evaporation ON

evaporation OFF

full evaporation coupling

latent heat retained while evaporation-induced mass-flow reduction is disabled

The last case is a sensitivity test and is not intended to be a fully mass-conservative physical model.

## Important Model Assumptions

The wafer temperature is axisymmetric: T = T(r,t).

A nozzle located at radius r_source is represented as an axisymmetric radial source.

The swing model therefore represents an axisymmetric / circumferentially averaged approximation rather than a fully 2D moving point nozzle.

The wafer outer radial edge is adiabatic.

N2 heat-transfer and mass-transfer coefficients use engineering correlations.

IPA properties are temperature-dependent approximations.

The effective center/nozzle radius is set to 5 mm to avoid the r = 0 singularity in the thin-film relation.

## Python Code

The checked code is provided in:

wafer_ipa_coupled_fdm_checked.py

Required packages:

pip install numpy matplotlib scipy

Run with:

python wafer_ipa_coupled_fdm_checked.py

## Main Outputs

The code produces radial profiles and comparisons for:

wafer temperature

IPA film temperature

center and edge temperature history

nozzle swing position

IPA film thickness

IPA-to-wafer heat flux

evaporation contribution

liquid mass-flow reduction

heat-transfer coefficient

<div align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/>
</div>
