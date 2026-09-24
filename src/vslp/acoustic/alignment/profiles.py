"""Packaged engineering alignment profiles."""

from pathlib import Path


def default_profile_path() -> str:
    return str(Path(__file__).parent / "profiles" /
               "mfa_vslp_en_us_arpa_334_engineering_v1.json")
