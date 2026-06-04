from vslp.core.project import initialize_project


def test_initialize_project(tmp_path):
    paths = initialize_project(tmp_path / "out", "demo")
    assert paths.root.exists()
    assert paths.manifest.exists()
    assert (paths.root / "acoustic").exists()
    assert (paths.root / "ml").exists()
