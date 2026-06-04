"""Vtrue-thePlane — a multimodal (vector + vision + relational graph) CAD primitive
labeler for the Vitruvian drafting agent.

Pipeline:  parse → (raster + kNN relation graph) → multimodal transformer →
per-primitive semantic label → geometric grouping → GPT refinement.

Deliberately light: ~3M params, pure PyTorch, single GPU, no custom CUDA.
"""
__version__ = "0.1.0"
