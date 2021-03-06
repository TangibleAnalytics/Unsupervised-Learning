# Unsupervised-Learning
All things unsupervised

The objective for any unsupervised learning model is to best capture the underlying relationships between variables within a feature matrix. At the core of every unsupervised learning model is an attempt to autoencode the original feature matrix into a latent representation. For an optimal unsupervised learning model, the latent representation capture only signal and discard the noise. With a small latent space, an unsupervised learning model cannot capture all of the signal. On the other hand, with a large latent space, an unsupervised learning model will capture undesirable noise. Therefore, the main problem in unsupervised learning is how to choose the optimal latent space size to best represent the original feature matrix.

This repository provides a working solution to this common problem in the "Validation" directory. Within the Latent_Validation module, the Latent_Validator class has the ability to effictively cross-validate any unsupervised learning model. Have you ever wondered what was the optimal number of PCA principal components to use for your data science project? The Latent_Validator class provides a way to convert this question into a convex optimization problem.

Along with Latent_Validation is a custom K_Means() clustering class, created specifically to be compatible with the Latent_Validator() class methods.
