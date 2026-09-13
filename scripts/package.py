"""Package source and recorded results; never include environment, Git or personal input HTML."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "RecAgent-prototype.zip"
    output.parent.mkdir(exist_ok=True)
    folders = ["recagent", "scripts", "tests", "docs", "notebooks", "report", "data", ".streamlit"]
    files = ["README.md", "requirements.txt", "requirements-lock.txt", "app.py", "Dockerfile", "compose.yaml", "start-demo.ps1", ".gitignore", ".dockerignore"]
    paths = [root / name for name in files]
    for folder in folders:
        paths.extend(path for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name != "secrets.toml")
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(root).as_posix())
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        print(f"Packaged {len(archive.namelist())} files: {output}")


if __name__ == "__main__":
    main()
