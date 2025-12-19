# SPDX-License-Identifier: GPL-3.0-or-later
"""
ComfyUI-Cadrille - Multi-modal CAD Reconstruction

Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra

Includes:
- Cadrille: Multi-modal CAD from point clouds, images, or text
- CAD-Recode: Point cloud to CadQuery code generation

Paper: "Cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning"
"""

import sys

if 'pytest' not in sys.modules:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Set web directory for JavaScript extensions
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
