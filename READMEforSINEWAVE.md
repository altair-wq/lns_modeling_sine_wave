=============================================================================
DOCUMENTATION
=============================================================================

Script: verify_sine_wave.py

This module validates the Linearized Navier-Stokes (LNS) solver by simulating 
a standing viscous gravity-capillary wave and comparing the numerical results 
against the analytical solution for finite depth.

-----------------------------------------------------------------------------
1. PHYSICAL CONFIGURATION
-----------------------------------------------------------------------------
The simulation parameters are derived from Galeano-Rios et al. (2017), 
specifically modeling the Silicone Oil (20 cSt) experiments described in 
Section 4.2 and Appendix C.

* Fluid: Silicone Oil
* Density (rho): 0.949 g/cm^3
* Surface Tension (sigma): 20.6 dyne/cm
* Viscosity Correction: The script applies the effective viscosity correction 
  factor nu* = 0.8025 * nu to account for the Faraday threshold matching used 
  in the reference paper.

-----------------------------------------------------------------------------
2. ANALYTICAL FORMULAS USED
-----------------------------------------------------------------------------
The numerical results (Blue Line) are compared against a theoretical decay 
curve (Red Dashed Line) calculated using the following relations.

A. Dispersion Relation (Frequency)
The oscillation frequency 'omega' is determined by the gravity-capillary 
dispersion relation for a fluid of finite depth D:

    omega^2 = (g*k + (sigma*k^3)/rho) * tanh(k*D)

    Where:
    - k = (2*pi*n)/L  is the wavenumber.
    - D is the depth of the fluid.

B. Decay Rate (Viscous Dissipation)
The amplitude A(t) is modeled as:
    
    A(t) = A_0 * exp(-gamma_total * t)

The total decay rate 'gamma_total' is the sum of bulk viscous dissipation 
and bottom boundary layer friction.

    gamma_total = gamma_bulk + gamma_bottom

    1. Bulk Dissipation:
       gamma_bulk = 2 * nu * k^2

    2. Bottom Boundary Layer Dissipation:
       Because the simulation imposes a no-slip condition (d_phi/dz + w3 = 0)
       at the bottom (z=-D), a Stokes boundary layer forms. The analytical 
       approximation for this boundary layer dissipation is:

       gamma_bottom = (k / (2 * sinh(2*k*D))) * sqrt(nu * omega / 2)

-----------------------------------------------------------------------------
3. NUMERICAL IMPLEMENTATION & BOUNDARY CONDITIONS
-----------------------------------------------------------------------------
The solver utilizes a finite difference grid to calculate the fluid dynamics, 
incorporating several advanced numerical techniques for stability and accuracy:

* Time Stepping (BDF2): 
  The temporal evolution of the surface and potential is handled using a 
  Second-Order Backward Differentiation Formula (BDF2) rather than a basic 
  Forward Euler loop, ensuring stability over long simulation periods.

* Surface Boundary (Free Surface):
  - Zero-Shear Stress: Horizontal friction (w1) is actively calculated and 
    coupled to the vertical friction (w3) via the condition w1_z + w3_x = 0, 
    allowing accurate surface deformation.
  - External Pressure (P_ext): The dynamic boundary condition includes an 
    external pressure array. While zero during wave verification, this serves 
    as the integration point for upcoming solid-fluid collision modeling.

* Bottom Boundary (Solid Floor):
  - Ghost Nodes: The Neumann boundary condition for the potential (phi_z = -w3) 
    is computed using a second-order accurate, 3-point backward difference 
    stencil (Ghost Nodes) to minimize energy leakage at the floor.

-----------------------------------------------------------------------------
4. REFERENCES
-----------------------------------------------------------------------------
* Simulation Model: 
  Galeano-Rios, C. A., Milewski, P. A., & Vanden-Broeck, J.-M. (2017). 
  "Non-wetting impact of a sphere onto a bath and its application to 
  bouncing droplets". J. Fluid Mech., 826, 97-127.

* Decay Formulas: 
  Lamb, H. (1932). "Hydrodynamics". Cambridge University Press.
