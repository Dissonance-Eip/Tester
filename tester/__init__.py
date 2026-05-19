"""
Music Perturbation Tester
=========================

Evaluates whether an audio perturbation effectively poisons AI models
while remaining imperceptible to human listeners.

Public entry point:
    >>> from tester.runner import run
    >>> results = run("original.wav", "poisoned.wav")
"""

from .runner import run  # noqa: F401

__all__ = ["run"]
