# -*- coding: utf-8 -*-
"""Export selected MCNoise nodes from Nuke as a weighted JSON preset bank.

Usage -- paste into the Nuke Script Editor and run:

    import sys
    sys.path.insert(0, "/path/to/ml-video-denoiser/nuke")
    from export_mc_noise_presets import export_selected
    export_selected("/path/to/output/mc_noise_presets.json")

Each selected node must be a compiled MCNoise BlinkScript with knobs:
    intensity, samples, chromaSpread, noiseDarkFade, noiseFadeFalloff,
    fireflyThresh, fireflyProb, fireflyChroma, fireflyDarkFade, fireflyFadeFalloff,
    mc_weight (Training Weight kernel param)

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


def _is_mcnoise_node(node):
    ks = node.knob("kernelSource")
    if ks is not None and "kernel MCNoise" in ks.value():
        return True

    return all(node.knob(knob_name) is not None for knob_name, _ in _KNOB_MAP)


def _compile(node):
    recompile = node.knob("recompile")
    if recompile is not None:
        try:
            recompile.execute()
            return
        except Exception:
            pass

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
    """Export every MCNoise node in the script."""
    candidates = [n for n in nuke.allNodes() if _is_mcnoise_node(n)]
    if not candidates:
        nuke.message("No MCNoise nodes found in the script.")
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
    """Export currently selected MCNoise nodes to a JSON preset bank."""
    selected = nuke.selectedNodes()
    if not selected:
        nuke.message("No nodes selected.\nSelect one or more MCNoise nodes and try again.")
        return

    for node in selected:
        _compile(node)

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
