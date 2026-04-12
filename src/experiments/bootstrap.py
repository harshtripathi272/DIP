from pathlib import Path
import sys


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.paths import ensure_project_directories


def main() -> None:
    created_paths = ensure_project_directories()
    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
