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

import nuke

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


def _try_recompile(node):
    """Force BlinkScript recompilation so parameter knobs are created."""
    if node.Class() != "BlinkScript":
        return
    ks = node.knob("kernelSource")
    if ks is not None:
        ks.setValue(ks.value())


def _read_knob(node, knob_name, json_key):
    knob = node.knob(knob_name)
    if knob is None:
        return None
    val = knob.value()
    return int(val) if json_key == "samples" else float(val)


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
        _try_recompile(n)
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

    for node in selected:
        _try_recompile(node)

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
