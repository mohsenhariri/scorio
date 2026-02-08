import re
import sys
from pathlib import Path


def main():
    """Read VERSION file and update all version references."""
    root = Path(__file__).parent.parent
    version_file = root / "VERSION"
    
    if not version_file.exists():
        print("ERROR: VERSION file not found")
        sys.exit(1)
    
    version = version_file.read_text().strip()
    
    if not version:
        print("ERROR: VERSION file is empty")
        sys.exit(1)
    
    print(f"Syncing to version: {version}")
    
    # Update Python __init__.py
    init_file = root / "scorio" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        new_content = re.sub(
            r'__version__ = "[^"]*"',
            f'__version__ = "{version}"',
            content
        )
        init_file.write_text(new_content)
        print(f"Updated {init_file.relative_to(root)}")
    else:
        print(f"WARNING: {init_file.relative_to(root)} not found")
    
    # Update Julia Project.toml
    toml_file = root / "julia" / "Scorio.jl" / "Project.toml"
    if toml_file.exists():
        content = toml_file.read_text()
        new_content = re.sub(
            r'version = "[^"]*"',
            f'version = "{version}"',
            content
        )
        toml_file.write_text(new_content)
        print(f"Updated {toml_file.relative_to(root)}")
    else:
        print(f"WARNING: {toml_file.relative_to(root)} not found")
    
    print(f"\nAll versions synced to {version}")


if __name__ == "__main__":
    main()
