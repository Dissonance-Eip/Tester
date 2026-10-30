"""
Adversarial perturbation generator.

Where the Tester *measures* whether a perturbation fools AI models, this
package *creates* such perturbations by gradient descent against the models.

Phase 0 (this code): per-song attack against MERT (the music encoder behind
the genre/encoder confusion gate). Proves that an inaudible, gradient-crafted
perturbation can move the AI-confusion metric — the thing random DSP noise
cannot do.
"""
