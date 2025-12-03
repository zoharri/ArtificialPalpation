"""
Non-Learnable Models

This directory contains models that don't require training or learning.
These models typically use hand-crafted algorithms or statistical methods
to generate representations from tactile data.
"""

from .force_map import ForceMapModel

__all__ = [
    'ForceMapModel',
]
