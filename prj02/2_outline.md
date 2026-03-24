## 1. Project Goals
Main question: can a MeshGraphNet learn the time evolution of 2D vortex shedding on a mesh, and can this serve as a stepping stone toward immersed boundary settings with Eulerian–Lagrangian coupling?  
Figure: simple schematic of cylinder wake / mesh / rollout idea.  
Sources: DeepMind MeshGraphNets, PhysicsNeMo, PyTorch Geometric.  
Math: very light.

## 2. Why this problem matters
Vortex shedding is a classic CFD problem, but full simulations are expensive. A learned surrogate could accelerate repeated prediction while still respecting the mesh-based structure of the flow.  
Figure: wake visualization behind a cylinder with alternating vortices.  
Sources: MeshGraphNets paper and benchmark description.  
Math: none.

## 3. Benchmark and data
Introduce the cylinder-flow dataset: trajectories, timesteps, mesh-based state, and graph construction. Mention that each trajectory has a fixed mesh, while different trajectories may use different meshes.  
Figure: a clean table showing `x`, `edge_index`, `edge_attr`, `y`, `mesh_pos`, and `cells`.  
Sources: benchmark preprocessing and dataset inspection.  
Math: none.

## 4. Graph representation of the CFD state
Explain how the unstructured CFD mesh is turned into a graph:
- nodes = mesh points
- edges = mesh connectivity
- node features = velocity + node type
- edge features = relative position and distance
- target = next-step velocity update  
Figure: annotated mesh-to-graph diagram.  
Sources: MeshGraphNets dataset formulation.  
Math: maybe one line showing \(x_t \rightarrow x_{t+1}\).

## 5. Model architecture
Describe the MeshGraphNet pipeline:
encoder → message-passing processor → decoder.  
Mention that the model is autoregressive and free-running during rollout.  
Figure: simple block diagram of node/edge updates.  
Sources: DeepMind MeshGraphNets, PhysicsNeMo.  
Math: light; maybe one line noting that the model predicts \((v_{t+1}-v_t)/\Delta t\).

## 6. Training setup and loss
Explain the implementation:
- PyTorch / PyG pipeline
- one-step supervised training
- rollout during testing  
State the loss clearly, such as nodewise MSE on the velocity target.  
Figure: training pipeline or one loss equation.  
Sources: my implementation.  
Math:
\[
\mathcal{L} = \frac{1}{N}\sum_i \|\hat{y}_i - y_i\|^2
\]

## 7. Results
Show one or two main results:
- predicted vs true velocity field
- short rollout comparison over time
- possibly one error visualization  
Figure: side-by-side CFD truth vs MeshGraphNet prediction.  
Sources: my experiments.  
Math: none.

## 8. What the results mean
Interpret the results with respect to the project goal:
- does the model capture short-term flow evolution?
- where does rollout error grow?
- does it transfer across different meshes?  
Figure: small metric plot or rollout error over time.  
Sources: my experiments.  
Math: none.

## 9. Reflection and next steps
Summarize what worked, what was challenging, and what I would improve. Then connect to the more interesting long-term direction: extending this to immersed boundary methods with
- a structured Eulerian fluid grid, and
- a Lagrangian mesh for the immersed boundary.  
Figure: conceptual immersed boundary schematic with Eulerian and Lagrangian discretizations.  
Sources: project motivation and future work.  
Math: none.
