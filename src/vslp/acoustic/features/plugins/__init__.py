"""Acoustic feature plugin registry."""

from __future__ import annotations

from vslp.acoustic.features.plugins.base import AcousticFeaturePlugin
from vslp.acoustic.features.plugins.coordination import CoordinationPlugin
from vslp.acoustic.features.plugins.articulatory import ArticulatoryPlugin
from vslp.acoustic.features.plugins.phonatory import PhonatoryPlugin
from vslp.acoustic.features.plugins.resonatory import ResonatoryPlugin
from vslp.acoustic.features.plugins.rhythm import RhythmPlugin
from vslp.acoustic.features.plugins.timing import TimingPlugin


def build_default_plugins() -> list[AcousticFeaturePlugin]:
    return [
        TimingPlugin(),
        PhonatoryPlugin(),
        ArticulatoryPlugin(),
        RhythmPlugin(),
        ResonatoryPlugin(),
        CoordinationPlugin(),
    ]


def implemented_feature_names() -> set[str]:
    names: set[str] = set()
    for plugin in build_default_plugins():
        names.update(plugin.feature_names)
    return names
