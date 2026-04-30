from pathlib import Path
import os
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_source_path(project_root: Path, relative_path: Path) -> Path:
    """Resolve source either from project root or its parent."""
    primary = project_root / relative_path
    if primary.exists():
        return primary

    fallback = project_root.parent / relative_path
    if fallback.exists():
        return fallback

    return primary


def collect_images(source_dir: Path) -> list[Path]:
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    images: list[Path] = []
    for root, _, files in os.walk(source_dir):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(file_path)

    images.sort()
    return images


def copy_and_rename(images: list[Path], destination_dir: Path, prefix: str) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    next_index = 1

    for source_path in images:
        extension = source_path.suffix.lower() if source_path.suffix else ".jpg"

        while True:
            new_name = f"{prefix}_{next_index:03d}{extension}"
            target_path = destination_dir / new_name
            next_index += 1
            if not target_path.exists():
                break

        shutil.copy2(source_path, target_path)
        copied_count += 1

    return copied_count


def main() -> None:
    project_root = Path.cwd()

    seed_source = resolve_source_path(project_root, Path("Anjasmoro"))
    footprint_source = resolve_source_path(project_root, Path("footprint-database") / "Scanned")

    s1_destination = project_root / "data" / "raw" / "S1" / "JPEG" / "200" / "loc1"
    s2_destination = project_root / "data" / "raw" / "S2" / "JPEG" / "200" / "loc1"

    seed_images = collect_images(seed_source)
    footprint_images = collect_images(footprint_source)

    seed_copied = copy_and_rename(seed_images, s1_destination, "seed")
    footprint_copied = copy_and_rename(footprint_images, s2_destination, "footprint")

    print(f"S1 copied files: {seed_copied}")
    print(f"S2 copied files: {footprint_copied}")


if __name__ == "__main__":
    main()