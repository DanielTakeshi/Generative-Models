# Generative-Models

Code for various Generative Models.  Unless otherwise stated, you can assume the
following plus whatever dependencies they may require:

- Python 3.5
- TensorFlow 1.3

Though I will try to keep the individual project READMEs more up to date.  To
run the code go inside a directory (e.g. `gan`) and run `main.py`. Every
directory should have its own `main.py` file which contains the command line
arguments and/or configuration stuff. Maybe it would be useful to get a few bash
scripts set up.

Currently implemented and "essentially" done, meaning that it's working but I
should still double check things.

- [**Generative Adversarial Networks (GANs)**](https://github.com/DanielTakeshi/Generative-Models/tree/master/gan).
- [**Variational Autoencoders (VAEs)**](https://github.com/DanielTakeshi/Generative-Models/tree/master/vae).

Here's what I'd like to implement:

- **Conditional Generative Adversarial Networks (GANs)**.
- **Deep Convolutional Generative Adversarial Networks (DCGANs)**.
- **Wasserstein Generative Adversarial Networks (WGANs)**.
- **Conditional Variational Autoencoders (CVAEs)**.

Optional challenge: do these *without* looking at other code online.

I'll try to push substantive updates only, and not to keep pushing every trivial
change at once.
