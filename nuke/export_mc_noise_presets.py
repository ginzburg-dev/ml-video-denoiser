# -*- coding: utf-8 -*-
"""Export selected MCNoise nodes from Nuke as a weighted JSON preset bank.

Usage -- paste into the Nuke Script Editor and run, or attach to a button knob:

    import sys
    sys.path.insert(0, "/path/to/ml-video-denoiser/nuke")
    from export_mc_noise_presets import export_selected
    export_selected("/path/to/output/mc_noise_presets.json")

Or inline on a button knob (Python tab):

    from export_mc_noise_presets import export_selected
    export_selected(nuke.getFilename("Save MC Noise presets", "*.json"))

Each selected node must have knobs matching MCNoise.blink param names:
    samples, intensity, chromaSpread, noiseDarkFade, noiseFadeFalloff,
    fireflyThresh, fireflyProb, fireflyChroma, fireflyDarkFade, fireflyFadeFalloff
    mc_weight  (user knob for training sampling weight, default 1.0)
    seed       (optional, ignored during training -- PyTorch RNG is used instead)

The node name becomes the preset "name" in the JSON.
JSON keys are written as snake_case so training.py can load them directly.

Compatible with Python 2.7+ (Nuke 11+) and Python 3.x.
"""

from __future__ import absolute_import, print_function

import json
import os

import nuke

_DEFAULT_PRESETS = (
    {
        "name": "MCNoise_light",
        "intensity": 0.5, "samples": 32, "chromaSpread": 0.08,
        "noiseDarkFade": 0.0, "noiseFadeFalloff": 1.0,
        "fireflyThresh": 6.0, "fireflyProb": 0.001, "fireflyChroma": 0.1,
        "fireflyDarkFade": 0.0, "fireflyFadeFalloff": 1.0,
        "mc_weight": 1.0, "xpos": -250, "ypos": 0,
    },
    {
        "name": "MCNoise_medium",
        "intensity": 1.0, "samples": 16, "chromaSpread": 0.12,
        "noiseDarkFade": 0.0, "noiseFadeFalloff": 1.0,
        "fireflyThresh": 6.0, "fireflyProb": 0.003, "fireflyChroma": 0.1,
        "fireflyDarkFade": 0.0, "fireflyFadeFalloff": 1.0,
        "mc_weight": 3.0, "xpos": 0, "ypos": 0,
    },
    {
        "name": "MCNoise_heavy",
        "intensity": 2.0, "samples": 8, "chromaSpread": 0.18,
        "noiseDarkFade": 0.0, "noiseFadeFalloff": 1.0,
        "fireflyThresh": 7.0, "fireflyProb": 0.005, "fireflyChroma": 0.15,
        "fireflyDarkFade": 0.0, "fireflyFadeFalloff": 1.0,
        "mc_weight": 1.0, "xpos": 250, "ypos": 0,
    },
)

# Nuke BlinkScript knob name -> JSON / Python key
_KNOB_MAP = (
    ("intensity",          "intensity"),
    ("samples",            "samples"),
    ("chromaSpread",       "chroma_spread"),
    ("noiseDarkFade",      "noise_dark_fade"),
    ("noiseFadeFalloff",   "noise_fade_falloff"),
    ("fireflyThresh",      "firefly_thresh"),
    ("fireflyProb",        "firefly_prob"),
    ("fireflyChroma",      "firefly_chroma"),
    ("fireflyDarkFade",    "firefly_dark_fade"),
    ("fireflyFadeFalloff", "firefly_fade_falloff"),
)


def _read_knob(node, knob_name, json_key):
    knob = node.knob(knob_name)
    if knob is None:
        return None
    val = knob.value()
    return int(val) if json_key == "samples" else float(val)


def create_default_presets(blink_dir=None):
    """Create light / medium / heavy MCNoise BlinkScript nodes in the current script."""
    if blink_dir is None:
        blink_dir = os.path.dirname(os.path.abspath(__file__))
    blink_path = os.path.join(blink_dir, "MCNoise.blink")
    if not os.path.exists(blink_path):
        nuke.message("MCNoise.blink not found at:\n%s" % blink_path)
        return

    with open(blink_path, "r") as f:
        kernel_source = f.read()

    try:
        ref = nuke.thisNode()
        ref_x = int(ref["xpos"].value())
        ref_y = int(ref["ypos"].value())
    except Exception:
        ref_x, ref_y = 0, 0

    # Deselect all so new nodes don't auto-connect
    for existing in nuke.allNodes():
        existing.setSelected(False)

    _SKIP = ("name", "xpos", "ypos")
    created = []
    for p in _DEFAULT_PRESETS:
        # nuke.nodes.BlinkScript() creates without auto-connecting
        n = nuke.nodes.BlinkScript()
        n["kernelSource"].setValue(kernel_source)
        n["xpos"].setValue(ref_x + p["xpos"])
        n["ypos"].setValue(ref_y - 160)
        n["name"].setValue(p["name"])

        # Build param assignments to apply after compile
        assignments = {}
        for key, val in p.items():
            if key in _SKIP:
                continue
            assignments[key] = val

        # Try setting immediately (works if kernelSource setValue triggered compile)
        remaining = {}
        for key, val in assignments.items():
            k = n.knob(key)
            if k is not None:
                k.setValue(val)
            else:
                remaining[key] = val

        # For any params not yet available, install a one-shot knobChanged
        # that applies them on first compile (xpos fires on placement)
        if remaining:
            lines = ["n = nuke.thisNode()"]
            lines.append("if nuke.thisKnob().name() == 'xpos':")
            for key, val in remaining.items():
                lines.append("    k = n.knob('%s')" % key)
                lines.append("    if k: k.setValue(%r)" % val)
            lines.append("    n['knobChanged'].setValue('')")  # remove self after first run
            n["knobChanged"].setValue("\\n".join(lines))

        created.append(p["name"])

    nuke.message("Created %d MCNoise nodes:\n%s" % (len(created), "\n".join(created)))


def export_all(output_path):
    """Export every node in the script that looks like an MCNoise node."""
    candidates = [
        n for n in nuke.allNodes()
        if n.knob("intensity") and n.knob("chromaSpread") and n.knob("mc_weight")
    ]
    if not candidates:
        nuke.message("No MCNoise nodes found in the script.\n"
                     "Nodes need intensity, chromaSpread, and mc_weight knobs.")
        return
    orig_selected = nuke.selectedNodes()
    for n in nuke.allNodes():
        n.setSelected(False)
    for n in candidates:
        n.setSelected(True)
    export_selected(output_path)
    for n in nuke.allNodes():
        n.setSelected(False)
    for n in orig_selected:
        n.setSelected(True)


def export_selected(output_path):
    """Export currently selected nodes to a JSON preset bank."""
    selected = nuke.selectedNodes()
    if not selected:
        nuke.message("No nodes selected.\nSelect one or more MCNoise nodes and try again.")
        return

    presets = []
    skipped = []
    for node in selected:
        entry = {"name": node.name()}

        weight_knob = node.knob("mc_weight")
        entry["weight"] = float(weight_knob.value()) if weight_knob else 1.0

        missing = []
        for knob_name, json_key in _KNOB_MAP:
            val = _read_knob(node, knob_name, json_key)
            if val is None:
                missing.append(knob_name)
            else:
                entry[json_key] = val

        if missing:
            skipped.append("%s (missing: %s)" % (node.name(), ", ".join(missing)))
            continue

        presets.append(entry)

    if not presets:
        msg = "No valid MCNoise nodes found."
        if skipped:
            msg += "\n\nSkipped:\n" + "\n".join(skipped)
        nuke.message(msg)
        return

    content = json.dumps(presets, indent=2)
    if not isinstance(content, bytes):
        content = content.encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(content)

    msg = "Exported %d preset(s) to:\n%s" % (len(presets), output_path)
    if skipped:
        msg += "\n\nSkipped %d node(s):\n%s" % (len(skipped), "\n".join(skipped))
    nuke.message(msg)
