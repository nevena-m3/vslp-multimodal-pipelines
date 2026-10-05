"""Generate deterministic engineering QA for the F02 replication filters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vslp.acoustic.features.family02_dsp import (
    FILTER_PARAMETER_SET_ID, butterworth_sos, filter_response,
)


def generate_filter_qa(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("pvi_9_14hz", 3, (9., 14.), 200, "bandpass", "source order/band"),
        ("gne_0_1000hz", 4, 1000., 8000, "lowpass", "VSLP filter order/family"),
        ("gne_1000_2000hz", 4, (1000., 2000.), 8000,
         "bandpass", "VSLP filter order/family"),
        ("gne_2000_3000hz", 4, (2000., 3000.), 8000,
         "bandpass", "VSLP filter order/family"),
    ]
    metadata = {"parameter_set_id": FILTER_PARAMETER_SET_ID, "filters": []}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, order, cutoff, sr, kind, provenance in specifications:
        sos = butterworth_sos(order, cutoff, sr, kind)
        frequencies, db = filter_response(sos, sr)
        axis = axes[0 if name.startswith("pvi") else 1]
        axis.plot(frequencies, db, label=name)
        edges = (cutoff,) if isinstance(cutoff, float) else cutoff
        for edge in edges:
            axis.axvline(edge, color="gray", linestyle="--", linewidth=.7)
        metadata["filters"].append({
            "name": name, "order": order, "sample_rate_hz": sr,
            "kind": kind, "cutoff_hz": cutoff, "parameter_provenance": provenance,
            "sos": sos.tolist(), "maximum_pole_radius": float(max(
                np.abs(np.concatenate([np.roots(section[3:]) for section in sos])))),
        })
    for axis, title, maximum in ((axes[0], "PVI F0 modulation filter", 30),
                                 (axes[1], "GNE excitation filter bank", 3500)):
        axis.set(title=title, xlabel="Frequency (Hz)", ylabel="Magnitude (dB)",
                 xlim=(0, maximum), ylim=(-100, 5))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    image_path = output_dir / "family02_filter_responses.png"
    metadata_path = output_dir / "family02_filter_metadata.json"
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return [image_path, metadata_path]


if __name__ == "__main__":
    generate_filter_qa(Path("docs/qa/family02"))
