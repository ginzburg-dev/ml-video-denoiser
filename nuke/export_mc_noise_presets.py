# -*- coding: utf-8 -*-
"""Export selected MCNoise nodes from Nuke as a weighted JSON preset bank.

Usage -- paste into the Nuke Script Editor and run:

    import sys
    sys.path.insert(0, "/path/to/ml-video-denoiser/nuke")
    from export_mc_noise_presets import export_selected
    export_selected("/path/to/output/mc_noise_presets.json")

Each selected node must be a compiled MCNoise BlinkScript with knobs:
    intensity, samples, chromaSpreadR, chromaSpreadG, chromaSpreadB,
    noiseDarkFade, noiseFadeFalloff, fireflyThresh, fireflyProb, fireflyChroma,
    fireflyDarkFade, fireflyFadeFalloff, mc_weight (Training Weight kernel param)

The node name becomes the preset "name" in the JSON.
JSON keys are written as snake_case so training.py can load them directly.

Compatible with Python 2.7+ (Nuke 11+) and Python 3.x.
"""

from __future__ import absolute_import, print_function

import json

import nuke

# Raw kernel param name, JSON / Python key, Blink display label
_KNOB_MAP = (
    ("intensity",          "intensity",           "Intensity"),
    ("samples",            "samples",             "Samples"),
    ("chromaSpreadR",      "chroma_spread_r",     "Chroma Spread R"),
    ("chromaSpreadG",      "chroma_spread_g",     "Chroma Spread G"),
    ("chromaSpreadB",      "chroma_spread_b",     "Chroma Spread B"),
    ("noiseDarkFade",      "noise_dark_fade",     "Noise Dark Fade"),
    ("noiseFadeFalloff",   "noise_fade_falloff",  "Noise Fade Falloff"),
    ("fireflyThresh",      "firefly_thresh",      "Firefly Thresh"),
    ("fireflyProb",        "firefly_prob",        "Firefly Prob"),
    ("fireflyChroma",      "firefly_chroma",      "Firefly Chroma"),
    ("fireflyDarkFade",    "firefly_dark_fade",   "Firefly Dark Fade"),
    ("fireflyFadeFalloff", "firefly_fade_falloff","Firefly Fade Falloff"),
)


def _blink_generated_name(kernel_name, display_label):
    return "%s_%s" % (kernel_name, "".join(display_label.split()))


def _find_knob(node, knob_name, display_label):
    for candidate in (
        knob_name,
        _blink_generated_name("MCNoise", display_label),
    ):
        knob = node.knob(candidate)
        if knob is not None:
            return knob

    suffixes = (
        "_" + display_label,
        "_" + "".join(display_label.split()),
    )
    for existing_name, knob in node.knobs().items():
        if existing_name.endswith(suffixes):
            return knob
        try:
            if knob.label() == display_label:
                return knob
        except Exception:
            pass

    return None


def _is_mcnoise_node(node):
    ks = node.knob("kernelSource")
    if ks is not None and "kernel MCNoise" in ks.value():
        return True

    required = ("Intensity", "Samples", "Chroma Spread R")
    return all(_find_knob(node, "", label) is not None for label in required)


def _read_knob(node, knob_name, json_key, display_label):
    knob = _find_knob(node, knob_name, display_label)
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

    presets = []
    skipped = []
    for node in selected:
        entry = {"name": node.name()}

        weight_knob = _find_knob(node, "mc_weight", "Training Weight")
        entry["weight"] = float(weight_knob.value()) if weight_knob else 1.0

        missing = []
        for knob_name, json_key, display_label in _KNOB_MAP:
            val = _read_knob(node, knob_name, json_key, display_label)
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
