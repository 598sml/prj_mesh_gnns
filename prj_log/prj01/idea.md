## 1. What is your project question?

My project question is: can a MeshGraphNet-type graph neural network learn to predict the time evolution of vortex shedding flow fields on a CFD mesh, and can this idea eventually be extended to immersed boundary simulations with coupled Eulerian and Lagrangian discretizations?

For the class project, I will focus on the benchmark problem of 2D cylinder-flow vortex shedding, where the model takes the current flow state on the mesh and predicts the next-step evolution. The broader motivation is to use this as a stepping stone toward immersed boundary method simulations, where the fluid is represented on a structured Eulerian grid and the immersed geometry is represented by a separate Lagrangian mesh.

## 2. Why is this a good SciML project?

This is a strong SciML project because it combines scientific simulation, physical structure, and machine learning in a natural way. The underlying problem comes from computational fluid dynamics, where vortex shedding is governed by the Navier–Stokes equations and is typically computed using expensive numerical solvers. The data is not generic tabular data: it lives on a mesh, evolves in time, and represents physical quantities like velocity and pressure.

It is also especially interesting for SciML because my longer-term interest is in immersed boundary methods, where the problem naturally involves multiple discretizations: an Eulerian grid for the fluid and a Lagrangian representation for the immersed body. This makes graph-based learning especially appealing, since graphs offer a flexible way to represent and couple different spatial structures.

## 3. What paper or benchmark will guide your work?

The main paper guiding my work is the MeshGraphNets paper by DeepMind, which introduced graph neural networks for learning simulation on meshes. I will use the cylinder-flow / vortex shedding benchmark from that ecosystem as the main reference problem. I am also looking at related implementations in PhysicsNeMo and PyTorch-based reimplementations to guide architecture and dataset handling.

## 4. What data or simulator will you use?

For the class project, I will use the existing cylinder-flow vortex shedding dataset from the MeshGraphNets benchmark. The original dataset is stored in TFRecord format, but I also have access to a preprocessed PyTorch Geometric version, which makes it manageable for a course project.

The reason I am starting from this benchmark is that it gives me a realistic but still tractable dataset for testing the approach. If the project goes well, the natural next step would be to move toward my own immersed boundary method simulations, where the data would come from a structured fluid solver coupled to a Lagrangian boundary representation.

## 5. What exactly will you implement yourself?

I plan to implement the model training and evaluation pipeline myself in PyTorch, and likely also a PyTorch version of the MeshGraphNet architecture rather than relying entirely on an external framework. I will also implement the data loading / inspection workflow, rollout evaluation, and postprocessing of results such as predicted velocity fields and possibly lift/drag comparisons.

The more interesting conceptual part for me is to think about how this framework could later be extended to an immersed boundary method setting, where I would need to represent and couple:
- the structured Eulerian grid of the fluid, and
- the Lagrangian mesh of the immersed boundary.

So even if the class project stays within the benchmark dataset, I want to keep the implementation flexible enough that it can serve as a foundation for that extension.

## 6. What are the main risks or bottlenecks?

The main risks are:

- Scope: even the standard MeshGraphNet vortex shedding benchmark already involves nontrivial data handling, graph construction, training, and rollout evaluation.
- Training cost: graph neural networks on CFD data can be expensive, especially for long autoregressive rollouts.
- Evaluation: it is easier to measure one-step error than to determine whether the model preserves meaningful vortex shedding dynamics over time.
- Extension to immersed boundary methods: this is the most interesting direction, but it is probably too large to fully complete within the class project timeline.

So the biggest challenge is balancing a realistic benchmark project with the more ambitious longer-term goal of handling coupled Eulerian–Lagrangian representations.

## 7. What is your minimum viable project?

My minimum viable project is: train a PyTorch MeshGraphNet-style model on a reduced version of the cylinder-flow dataset and demonstrate that it can predict one-step flow evolution on the mesh better than a simple baseline. If time allows, I will extend this to autoregressive rollout and comparison of derived quantities such as lift/drag.
