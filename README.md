# SciML GSoC 2026
GalerkinNeuralODE for DiffEqFlux.jl is a project that extends Neural ODEs by allowing model parameters to vary continuously over depth or time through a basis expansion, instead of staying fixed during the ODE solve. The goal is to bring a production-quality implementation of this idea to DiffEqFlux.jl, making basis-parameterized continuous-depth models easier to use, test, and extend.

This repository will contains the development of a GalerkinNeuralODE layer, including the basis abstraction, parameter lifting and reconstruction utilities, compatibility with Lux models and SciML training workflows, and experiments comparing standard NeuralODEs with Galerkin-based variants. The project also focuses on correctness, automatic differentiation safety, benchmarking, and documentation so the implementation can serve as a solid upstream contribution to the SciML ecosystem.

**Personal Blog**: [arkdong.com](https://arkdong.com/posts/galerkin-neural-ode/)

## Recent PR
### [DiffEqFlux.jl #1010](https://github.com/SciML/DiffEqFlux.jl/pull/1010)
This PR is an initial draft implementation of a GalerkinNeuralODE #290 for DiffEqFlux.jl, inspired by the paper "Dissecting Neural ODEs", mainly opened for design feedback. The current implementation works on the example benchmark and the preliminary results look encouraging, but the code is not yet aligned with SciML style, still contains many learning-oriented comments, and does not yet include unit tests. Before polishing the implementation further, I would greatly appreciate feedback on the overall design, API, parameter lifting/reconstruction approach, and any obvious AD, sensitivity, or performance issues. After incorporating feedback, I plan to clean up the code, remove unnecessary comments, add unit tests, and improve the documentation.

#### Key Idea from the paper "Dissecting Neural ODEs"
- Vanilla Neural ODEs cannot be fully considered the deep limit of ResNets. The first attempt to pursue the true deep limit of ResNets is the hypernetwork approach of (Zhang et al., 2019b) where another neural network parametrizes the dynamics of $\theta(s)$.
- However, this approach is not backed by any theoretical argument and it exhibits a considerable parameter inefficiency, as it generally scales polynomially in $n_{\theta}$. This paper approach to the problem by uncovering an optimization problem in functional space, solved by a direct application of the adjoint sensitivity method in infinite-dimensions.

- Galerkin Neural ODEs is the spectral discretization verison. The idea is to expand $\theta (s)$ on complete orthogonal basis of a predetermined subspace $\mathbb{L}_{2}(\mathcal{S}\to \mathbb{R}^{n _\theta})$ and truncate the series to the $m$-th term, where $\psi_j(s)$ are basis functions and the trainable objects are the coefficients $\alpha_j$:

$$
\theta (s)=\sum_{j=1}^{m}\alpha_{j}\odot\psi_{j}(s)
$$


- This turns an infinite-dimensional optimization over functions $\theta(s)$ into an ordinary finite-dimensional optimization over coefficient vectors $\alpha=(\alpha_{1}, \dots\alpha_{m})\in \mathbb{R}^{mn_{\theta}}$, whose gradient can be computed as follows
	- **Corollary 1** (Spectral Gradients). Under the assumptions of Theorem 1 (Infinite-Dimensional Gradients), if $\theta (s)=\sum_{j=1} ^{m}\alpha_{j} \odot\psi_{j}(s)$, then:

$$
\frac{d \ell}{d\alpha}=\int_{\mathcal{S}}\vec{a}^{\top }(\tau) \frac{ \partial f_{\theta(s)} }{ \partial \theta(s) } \psi(\tau)d\tau, \quad \psi=(\psi_{1}, \dots \psi_{m}) 
$$

- At solver time $s$, evaluate the basis functions $\psi_j(s)$, reconstruct the current parameter set $\theta(s)$, and then use that parameter set inside the vector field. So the system you solve is still an ODE, but the ODE’s neural-network parameters now evolve with depth according to the learned basis expansion.

#### Preliminary Result
- Testing and graphs are generated using the example code from the doc [Neural Ordinary Differential Equations](https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode/). These results are preliminary and mainly intended as a sanity check for the implementation.
- In the untrained-state plots, both the standard NeuralODE and the Galerkin NeuralODE start far from the ground-truth trajectories, confirming that neither model matches the data before optimization. 
![neural_vs_galerkin_training](https://github.com/user-attachments/assets/87f91e4e-27a4-4c86-a36e-25da7a1c3fe2)


- During training, however, the loss curves show two important patterns: 
	- First, the Galerkin model with $M=1$ closely follows the standard NeuralODE, which is a key sanity check because the constant-only Galerkin case should reduce to the vanilla NeuralODE; 
	- Second, the Galerkin model with $M=5$ converges substantially faster and reaches a noticeably lower training loss, indicating that the additional basis modes provide extra expressive power. <img width="500" height="400" alt="training_loss" src="https://github.com/user-attachments/assets/4577aab8-cf38-4e21-8907-e6e8b0fcce8b" />


- Finally, the trained trajectory plots show that all three models recover the target dynamics well, with the $M=1$ Galerkin model nearly overlapping the NeuralODE baseline and the $M=5$ model achieving the best overall fit. <img width="500" height="600" alt="trajectories" src="https://github.com/user-attachments/assets/ec521a28-a1ff-4dfb-b06f-60add82e77e2"  />

- Taken together, these preliminary experiments suggest that the implementation is behaving in the expected direction:
	- the $M=1$ case behaves similarly to vanilla NeuralODE
	- richer basis expansion can improve the fit, although part of the improvement for $M=5$ may also come from its larger effective parameterization.

