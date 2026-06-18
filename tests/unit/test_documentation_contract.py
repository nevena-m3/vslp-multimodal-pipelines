import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
AUTHORITATIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "INSTALLATION.md",
    ROOT / "docs" / "USER_GUIDE.md",
    ROOT / "docs" / "ACOUSTIC_PIPELINE_SOP.md",
    ROOT / "docs" / "KINEMATICS_PIPELINE_SOP.md",
    ROOT / "docs" / "FEATURE_ANALYSIS_GUI_SOP.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "data_dictionary.md",
    ROOT / "docs" / "DEVELOPMENT.md",
]


def test_authoritative_documentation_files_exist_and_are_nonempty():
    for path in AUTHORITATIVE_DOCS:
        assert path.exists(), path
        assert len(path.read_text(encoding="utf-8").strip()) > 100, path


def test_authoritative_markdown_local_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    missing = []
    for document in AUTHORITATIVE_DOCS:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken local documentation links:\n" + "\n".join(missing)


def test_root_readme_documents_all_completed_gui_launch_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "vslp gui acoustic" in readme
    assert "vslp gui kinematics" in readme
    assert "vslp gui features" in readme
    assert "ML GUI | In development" in readme
