# OTI Interview — 1-Page Technical Equation Cheat-Sheet

> [!IMPORTANT]
> Grounded in your exact published notation (**SnowTek**, **WINICE**, **AIChE Journal 2016**) mapped directly onto **ICICLE / OTI aeroengine icing physics**. Keep open on desk/second screen for instant recall during Q&A.

---

## 1. Coupled ICICLE Multi-Physics System (Slide 03)

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \mathbf{S}(\mathbf{U}, T, \boldsymbol{\sigma}, \alpha)$$

* $\mathbf{U}$: conserved fluid/thermal/phase state vector; $\mathbf{F}(\mathbf{U})$: convective & diffusive fluxes.
* $\mathbf{S}(\mathbf{U}, T, \boldsymbol{\sigma}, \alpha)$: two-way coupling sources — Lagrangian momentum/energy exchange, phase-change latent heat, stress $\boldsymbol{\sigma}$, and ice/liquid volume fraction $\alpha$.

---

## 2. Eulerian–Lagrangian Transport & Non-Spherical Drag (Prof. di Mare)

### Discrete Particle Momentum (Lagrangian)
$$m_p \frac{d\mathbf{u}_p}{dt} = \mathbf{F}_D + \mathbf{F}_g + \mathbf{F}_{\text{lift}} + \mathbf{F}_{\text{frame}}, \qquad \mathbf{F}_D = \frac{1}{2} \rho_g C_D A_p |\mathbf{u}_g - \mathbf{u}_p|(\mathbf{u}_g - \mathbf{u}_p)$$

Two-way Eulerian momentum source per cell volume $V_{\text{cell}}$:
$$\mathbf{S}_M = -\frac{1}{V_{\text{cell}}} \sum_{p \in \text{cell}} \mathbf{F}_{D,p}$$

### Your SnowTek Non-Spherical Disc Model & RHSV
* **Disc geometry:** $A_i = \pi (D_i/2)^2$, $V_i = A_i H_i$, aspect ratio $\gamma = H_i/D_i \in (0,1]$.
* **Single-particle terminal velocity:**
  $$u_t = \sqrt{\frac{2g H_i (\rho_s - \rho_i)}{C_D \rho_s}}$$
* **Reversed Hindered Settling Velocity (RHSV) — swarm/voidage correction:**
  $$u = u_t \cdot (\epsilon^2 \Psi_p), \qquad \epsilon = \frac{f_f/\rho_f}{f_f/\rho_f + f_s/\rho_s}, \qquad \Psi_p \approx 0.79$$
* **Regime drag correlations ($C_D$):**
  * Stokes ($Re \le 0.1$): $C_D = 24/Re$
  * Intermediate ($0.1 \le Re \le 1000$): $C_D = \frac{24}{Re}(1 + 0.14 Re^{0.70})$
  * Newton ($10^3 \le Re \le 3.5\times 10^5$): $C_D = 0.445$
  * *For non-spherical crystals in aeroengines:* extend via **Ganser / Haider–Levenspiel** using sphericity $\Phi$ and orientation-dependent projected area.
* **Key SnowTek numbers:** $D_i = 3\text{ mm}$, $H/D = 0.35$, $Re = 32.2$, $C_D = 1.93$, $\text{RHSV} = 0.0233\text{ m/s}$ (measured $u_{tx} = 0.0335\text{–}0.05\text{ m/s}$).

---

## 3. Surface Mass–Energy Balance & Accretion Thermodynamics (Prof. Gillespie)

### Messinger Control-Volume Mass Balance
$$\dot{m}_{\text{imp}} + \dot{m}_{\text{in}} = \dot{m}_{\text{ice}} + \dot{m}_{\text{evap}} + \dot{m}_{\text{out}}$$

* **Freezing fraction:**
  $$n = \frac{\dot{m}_{\text{ice}}}{\dot{m}_{\text{imp}} + \dot{m}_{\text{in}}}$$
  * $n = 1$: **Rime ice** (dry, instantaneous freezing, zero runback).
  * $0 < n < 1$: **Glaze / mixed-phase ice** (wet surface + liquid runback film).
  * $n = 0$: **Fully wet / running-wet** (zero net accretion).

### Surface Energy Balance
$$\dot{Q}_{\text{latent}} + \dot{Q}_{\text{kinetic}} + \dot{Q}_{\text{sens,imp}} = \dot{Q}_{\text{conv}} + \dot{Q}_{\text{evap}} + \dot{Q}_{\text{cond,wall}}$$

* **Stefan / moving-boundary heat flux** through growing ice thickness $\delta(t)$:
  $$\rho_i L_f \frac{d\delta}{dt} = k_{\text{eff}}(\phi)\left.\frac{\partial T}{\partial y}\right|_{\text{ice}} - h_c(T_\infty - T_s)$$
* **Your SnowTek & WINICE Anchors:**
  * **Thermal fouling (SnowTek):** cold-wall ice scaling cuts heat transfer coefficient (HTC) by **up to 50%**.
  * **Constitutional supercooling (WINICE, *CEJ* 2017):** when bulk-to-interface heat transfer outpaces molecular mass diffusion, the planar front destabilizes into a **sutured, dendritic morphology** with occluded liquid pockets $\rightarrow$ macroscopic volume defects + soft, mechanically weakened ice.

---

## 4. 3D Mechanical Shedding & Fracture Mechanics (Slide 04)

### Energy-Release-Rate Failure Criteria
* **Interfacial (Adhesive) Failure** — debonding at ice–substrate interface:
  $$G_{\text{interface}} \ge G_a \qquad (\text{or stress form: } \tau_{\text{interface}} \ge \tau_{\text{adh}}(T, \alpha_{\text{liq}}))$$
  * *Physical trigger:* interfacial melt/lubricating liquid film drops $G_a$ and $\tau_{\text{adh}}$ sharply under boundary-layer shear & centrifugal load.
* **Cohesive (Bulk) Failure** — brittle fracture inside 3D ice mass:
  $$G_{\text{cohesive}} \ge G_c \qquad \Longleftrightarrow \qquad K_I \ge K_{Ic} = \sqrt{E' G_c}$$
  * Plane strain modulus: $E' = \frac{E}{1 - \nu^2}$; stress intensity factor $K_I = Y \sigma \sqrt{\pi a}$ (where flaw size $a$ scales with WINICE-type pore/liquid inclusions).
* **Mixed-mode / Cohesive Zone Model (CZM):** traction–separation law $\mathbf{t}(\boldsymbol{\delta})$ integrating to fracture energy $G = \int_0^{\delta_f} t(\delta)\,d\delta$ for 3D crack initiation and propagation.

---

## 5. Variable-Mass Kinetics & Population Balance Equations (Your *AIChE J.* 2016)

### Variable-Mass Momentum & Hold-Up (Your *AIChE J.* 2016)
$$\frac{dU}{dt} = \frac{U}{u}\frac{dv}{dt} + \frac{F}{u}, \qquad E(t) = U_0 - \big(U(t) + D(t)\big)$$
* Achieved **$< 3\%$ relative error** vs. 72 experimental runs ($0.75\text{–}3\text{ mm}$ particles; $95\text{–}100\%$ separation efficiency). Hold-up term $E(t)$ directly parallels surface sticking/retention.

### General Breakage PBE (for Shed Ice Fragmentation & Impact Shatter)
$$\frac{\partial n(v,t)}{\partial t} + \nabla \cdot \big(\mathbf{u}_p n(v,t)\big) = \int_v^\infty b(v|v')\, g(v')\, n(v',t)\,dv' - g(v)\,n(v,t)$$
* $n(v,t)$: fragment number density of volume $v$; $g(v)$: breakage rate kernel (driven by Weber/impact energy); $b(v|v')$: daughter size distribution.
* Sub-critical rebound uses Hertzian normal/tangential restitution $(e_n, e_t)$; super-critical impact triggers PBE shatter.

---

## 6. Reduced-Order Modelling (Slide 07) & Bayesian UQ (Slide 05)

* **POD (Proper Orthogonal Decomposition):** snapshot matrix $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$; project state onto top $r$ modes $\mathbf{u}(\mathbf{x},t) \approx \bar{\mathbf{u}}(\mathbf{x}) + \sum_{k=1}^r a_k(t)\boldsymbol{\phi}_k(\mathbf{x})$ capturing energy $\sum_{k=1}^r \sigma_k^2 / \sum \sigma_i^2 \ge 99\%$.
* **DMD (Dynamic Mode Decomposition):** linear best-fit operator $\mathbf{X}' \approx \mathbf{A}\mathbf{X}$, eigendecomposition $\mathbf{A}\boldsymbol{\psi}_k = \lambda_k \boldsymbol{\psi}_k$ gives growth rates & oscillation frequencies of shedding/accretion cycles.
* **Bayesian Calibration (UQ):**
  $$p(\boldsymbol{\theta} \mid \mathbf{y}_{\text{cal}}) \propto p(\mathbf{y}_{\text{cal}} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})$$
  * Calibrate $\boldsymbol{\theta}$ (e.g., $G_a$, $G_c$, $e_n$, $k_{\text{eff}}$) on **calibration set only**; evaluate posterior predictive intervals on **blind validation set**; rank parameter influence via **Sobol sensitivity indices**.

---

## Quick-Recall Credential Numbers
* **8 peer-reviewed journal articles** (7 multiphase/chemical engineering: *AIChE J.*, *CEJ*, *Powder Tech.*, *Minerals Eng.*, *Adv. Powder Tech.*; 1 theoretical physics) + **11+ total scholarly works**.
* **7+ years** production numerical software (Python, C++, Linux HPC, Docker, CI/CD).
* **SnowTek:** 50% HTC drop; $Re = 32.2$; $C_D = 1.93$; $\Psi_p = 0.79$; drainage $6.14\text{ g/(m}^2\cdot\text{s)}$.
* **AIChE 2016:** $< 3\%$ relative error across 72 experimental batches.
* **WINICE 2017:** $4\text{–}5\text{ wt\% Na}_2\text{SO}_4$ eutectic plateau at $10{,}000\text{ s}$; dendritic/sutured defects from constitutional supercooling.
