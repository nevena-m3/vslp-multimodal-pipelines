from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "vslp" / "gui" / "features" / "app.py"
PLOTS = ROOT / "src" / "vslp" / "analysis" / "features" / "plots.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_version_v109():
    assert 'APP_VERSION = "v0.109.0"' in read(APP)


def test_relationships_have_dedicated_regenerator_not_overview():
    s = read(APP)
    page = s[s.index('    def _relationships_page'):s.index('    def update_relationships_dashboard')]
    assert 'regen.clicked.connect(self.regenerate_relationships)' in page
    assert 'regen.clicked.connect(self.regenerate_overview_plots)' not in page
    assert 'Correlation structure' in page
    assert 'Redundancy triage' in page
    assert 'Feature-family structure' in page
    assert 'Dimensionality profile' in page
    assert 'Recording similarity map' in page


def test_relationships_are_task_scoped_and_use_feature_roles():
    s = read(APP)
    block = s[s.index('    def _relationship_current_task'):s.index('    def update_relationship_interpretation')]
    assert 'def _relationship_scoped_frame' in block
    assert 'def _relationship_feature_cols' in block
    assert 'roles.get("Feature", [])' in block
    assert 'feature_correlation_long_table(df, feature_cols, max_features=180)' in block
    assert 'redundant_feature_pairs(outputs["feature_correlation_long"], registry)' in block
    assert 'feature_family_correlation_matrix(outputs["feature_correlation_long"], registry)' in block
    assert 'plot_relationship_dimensionality_profile' in block


def test_preview_regenerates_relationships_not_overview():
    s = read(APP)
    block = s[s.index('    def preview_relationship_plot'):s.index('    def update_relationship_interpretation')]
    assert 'self.regenerate_relationships(silent=True)' in block
    assert 'self.regenerate_overview_plots()' not in block
    assert 'self._display_plot_image(self.relationship_plot_preview, path)' in block


def test_dimensionality_profile_plot_exists():
    s = read(PLOTS)
    assert 'def plot_relationship_dimensionality_profile' in s
    assert 'Variance concentration' in s
    assert 'Largest early-component drivers' in s
