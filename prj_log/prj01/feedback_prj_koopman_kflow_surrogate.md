**1. My understanding**  
I understand this project to be a comparison of DMD, an autoencoder with nonlinear latent dynamics, and a Koopman autoencoder for predicting 2D Kolmogorov flow.

**2. One strength**  
A strong point is that the project has a clear comparison setup with shared data, shared loss, and concrete evaluation criteria.

**3. One concern**  
My main concern is that training and fairly comparing three different models, while also studying rollout stability and latent dimension choices, may still be a bit ambitious.

**4. One suggestion**  
I would suggest making the main goal a clean comparison of the three models on one dataset and one rollout setting, and treating latent-dimension sweeps or extra experiments as optional.
