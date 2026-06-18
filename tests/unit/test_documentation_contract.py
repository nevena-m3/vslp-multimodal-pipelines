import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
AUTHORITATIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "getting-started" / "INSTALLATION.md",
    ROOT / "docs" / "getting-started" / "USER_GUIDE.md",
    ROOT / "docs" / "sops" / "ACOUSTIC_PIPELINE_SOP.md",
    ROOT / "docs" / "sops" / "KINEMATICS_PIPELINE_SOP.md",
    ROOT / "docs" / "sops" / "FEATURE_ANALYSIS_GUI_SOP.md",
    ROOT / "docs" / "development" / "architecture.md",
    ROOT / "docs" / "reference" / "data_dictionary.md",
    ROOT / "docs" / "development" / "DEVELOPMENT.md",
    ROOT / "docs" / "history" / "README.md",
]
MAINTAINED_DOCS = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    *(
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "history" not in path.relative_to(ROOT / "docs").parts
    ),
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


def test_all_maintained_markdown_local_links_resolve():
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    missing = []
    for document in MAINTAINED_DOCS:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken maintained documentation links:\n" + "\n".join(missing)


def test_root_readme_documents_all_completed_gui_launch_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "vslp gui acoustic" in readme
    assert "vslp gui kinematics" in readme
    assert "vslp gui features" in readme
    assert "ML GUI | In development" in readme
