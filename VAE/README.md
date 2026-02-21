## Data

The VAE notebook uses the [nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces) dataset from Hugging Face.

## Why I got curious

When I read about VAEs online, most articles say these autoencoders learn to represent data in a continuous latent space. And I saw an architecture diagram that looked like a rattle drum (narrow in the middle). Simply put, I had more questions than answers.

So, I read the [VAE](https://arxiv.org/pdf/1312.6114) paper :) It's actually a neat concept.

## The code

I don't want to leave my README just explaining the code. You can check the jupyter notebook for that. In the rest of this document, I'm going to give my interpretation of VAEs and how I built them.

Again, my code ain't perfect. The beautiful part (in my humble opinion) are the assumptions. I guess if I spend more time I can get better results? Hope you have fun reading this!

## The intuition behind VAEs

Imagine I have a magic machine that generates face images. Every time I ask it to generate an image, it produces a face that is different.  The mechanism that this magic machine uses is a random process. Which means, when I ask for a face, the machine will sample from a random variable `z` with a distribution `p(z)`.

What does this mean? Every time I ask this magical random process to give me a face at time step `i`, it samples from the random variable $z^{(i)}$ and gives me an image $x^{(i)}$.

What if I can learn information about `z` so that I can generate my own images with a likelihood of $p(x | z)$? That is what VAEs do. The `z` encodes the information of the latent space.

This means everything I deal with is conditioned on something. You might ask — why not use Bayes' theorem and solve it? The problem is intractability. We cannot compute $p(z|x)$ (the posterior) because computing it requires $p(x) = \int p(x|z)p(z)dz$, which involves integrating over all possible values of `z` — and that is intractable.

## Things get interesting

Let's look at what we're dealing with:
- $p(z)$ is the prior — we **assume** this is $\mathcal{N}(0, I)$. This is a design choice, not something we compute.
- $p(x)$ is the evidence — the probability of an image over all possible `z`. This is the intractable integral mentioned above.
- $p(x|z)$ is the likelihood — given a latent vector, how likely is this image? This is what the decoder learns.
- $p(z|x)$ is the posterior — given an image, what latent vector produced it? This is intractable because it depends on $p(x)$. But we will estimate it with our encoder.

To make our lives simple, we break this into 2 sub-problems and make assumptions. The questions we ask are: "Can I estimate the posterior probability? And can I use that to learn to encode information in `z` and later estimate the likelihood?" MLPs say "just do it!" And so did we.

## How we build VAEs with Neural Networks

### Sub-problem 1: Estimating the posterior

We assume that $q_\phi(z|x)$ follows a Gaussian distribution. We build a neural network (the encoder) with parameters $\phi$ to estimate the posterior. For each input image $x$, the encoder outputs the mean $\mu$ and log-variance $\log \sigma^2$ of this Gaussian. Each dimension of `z` gets its own mean and variance — so if `z` has 200 dimensions, the encoder outputs 400 values.

For this to be successful, the learnt distribution $q_\phi(z|x)$ should resemble $p(z|x)$. This is why there is a KL-Divergence term — it measures how different the learnt distribution is from the prior $p(z) = \mathcal{N}(0,I)$. The goal is to minimize this difference.

### Sub-problem 2: Estimating the likelihood

We assume we have a latent vector `z` (sampled from the encoder's distribution), and we want to maximize $p(x|z)$ — the probability of reconstructing the original image from `z`. This is what the decoder does. In practice, maximizing the log-likelihood simplifies to minimizing a reconstruction loss (like MSE between the original and reconstructed image). If you've done Maximum Likelihood Estimation before, it's the same optimizer + loss function combo.

### The reparameterization trick

There's a subtle but critical problem: to train the encoder, we need to backpropagate through the sampling of `z`. But sampling is a stochastic operation — you can't compute gradients through randomness.

The trick: instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly, we reparameterize as $z = \mu + \sigma \cdot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$. Now the randomness is in $\epsilon$ (a constant with respect to model parameters), and gradients flow cleanly through $\mu$ and $\sigma$.

### Combining pieces of the puzzle

Our goal is to learn the best estimates of posterior and likelihood. So we define our loss function to include both. The KLD term ensures the model learns a well-structured latent distribution. The reconstruction loss ensures the model learns to use the latent features to produce good images. These two terms compete with each other — KLD wants a simple, organized latent space while reconstruction wants an information-rich one. When we hit the right balance, you get your magic machine!

### In the paper, there is L in eq. 10. What is L?

In the original paper, the authors compute the log-likelihood by summing $p(x^{(i)}|z^{(i,l)})$ for all $l \in [1,L]$. This means for each input image, you can sample multiple `z` vectors and average the reconstruction loss across them for a better estimate. Since we are training on a large dataset and averaging over many samples anyway, we have the luxury of choosing L = 1 (one `z` sample per image per training step). This is standard practice.

### What is beta?

The KLD and reconstruction terms don't always cooperate. You might want to control the balance between them. A $\beta$-VAE weights the KLD term: $\mathcal{L} = \beta \cdot D_{KL} + \text{reconstruction loss}$. 

- **β > 1** pushes for a more disentangled latent space (individual dimensions capture separate concepts), at the cost of blurrier reconstructions.
- **β < 1** prioritizes reconstruction quality, but the latent space becomes messy and entangled.

I chose beta = 2 (for fun. never really paid attention to this one)

## Note

This is not a perfect implementation. VAEs inherently produce blurry outputs due to the MSE loss over pixel uncertainties. If I have time, I might explore some means to fix this. If I do, this README will be updated.