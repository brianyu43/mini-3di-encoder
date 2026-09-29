"""Paths and deterministic ordering for the follow-up experiments."""

import hashlib

from experiments.common import ROOT

OUT = ROOT / "artifacts/followup-v2"
ORIGINAL = ROOT / "artifacts/sessions-11-15"


def order(text):
    return hashlib.sha256(("mini3di-followup-v2-20260929:" + text).encode()).hexdigest()
