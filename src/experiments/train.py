from pathlib import Path
import sys
from typing import Callable


if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.experiments import phase2_dataset, phase3_features, phase4_classification


def _run_phase(phase_name: str, phase_main: Callable[[], None]) -> None:
	print(f"\n[START] {phase_name}")
	phase_main()
	print(f"[DONE]  {phase_name}")


def main() -> None:
	print("Running full training pipeline from project root...")
	print("Step 1/3: Phase 2 dataset indexing")
	try:
		_run_phase("Phase 2: dataset indexing", phase2_dataset.main)

		print("Step 2/3: Phase 3 feature extraction")
		_run_phase("Phase 3: feature extraction", phase3_features.main)

		print("Step 3/3: Phase 4 classification/training")
		_run_phase("Phase 4: classification/training", phase4_classification.main)
	except Exception as exc:
		print(f"\n[FAILED] Pipeline stopped due to error: {exc}")
		raise SystemExit(1) from exc

	print("\n[SUCCESS] Full pipeline completed.")


if __name__ == "__main__":
	main()
