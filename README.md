# AIBMD
Artificial intelligence boosted molecular dynamics (AIBMD)

AIBMD is developed based on Gaussian accelerated molecular dynamics (GaMD), an enhanced sampling technique that works by adding harmonic boost potentials to reduce the system energy barriers. In AIBMD, however, the boost potentials are optimized to follow Gaussian distributions with minimized anharmonicities by probabilistic Bayesian neural network models.

In AIBMD, a short conventional molecular dynamics (cMD) was first performed on the biological system of interest. The potential statistics (Vmin and Vmax) were then collected as parameters for the pre-equilibration stage. During the pre-equilibration stage, the effective harmonic force constants (k0P and k0D) were kept fixed at (1.0, 1.0) for explicit-solvent simulations and (0.05, 1.0) for implicit-solvent simulations. The boost potentials were calculated as following:

E = Vmin + (Vmax - Vmin) / k0
if E > (1 + facE)*Vmax

, and the potential statistics (Vmin and Vmax) were updated during pre-equilibration. The system total and dihedral potential energies from the pre-equilibration stage were then collected, which served as inputs for the probabilistic Bayesian deep learning (DL) models. Moreover, reference boost potentials were randomly generated from the potential energies collected as following:




Representative distributions of these randomly generated boost potentials are shown in Supplementary Figure 1, which showed unreasonably high anharmonicities γ. DL was carried out in multiple iterations until the output boost potentials followed an exact Gaussian distribution with γ<0.01 for sufficient sampling and accurate reweighting. Based on the potential statistics learnt for the last frame of the pre-equilibration stage (Vmin, Vmax, V, and ∆V), the effective harmonic force constants were calculated as following:

and used as inputs alongside Vmin and Vmax to equilibrate the simulation system. The equilibration stage usually consisted of multiple rounds, with the effective harmonic force constants (k0P and k0D) kept fixed and potential statistics (Vmin and Vmax) updated in each round. DL was carried out at the end of each round using the potential energies collected within the round as inputs, with the DL protocol being exact same as in the end of the pre-equilibration stage. Finally, the effective harmonic force constants (k0P and k0D) and potential statistics (Vmin and Vmax) taken from the last round of the equilibration were used as input parameters for production simulations, during which the effective harmonic force constants and potential statistics were kept fixed, and boost potentials were calculated based on (4).
