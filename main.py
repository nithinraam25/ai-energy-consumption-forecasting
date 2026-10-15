"""Main entry point for AI-Powered Energy Consumption Forecasting.

Orchestrates runtime environment setup and executes the end-to-end
experiment workflow using real or staged energy time-series datasets.
"""

import sys
from src.config import ensure_directories, PROJECT_ROOT
from src.experiment import (
    run_pjm_experiment,
    run_phase5a_experiment,
    run_phase6_experiment,
    run_phase7_experiments,
    run_phase8_experiments,
    run_phase9_experiments,
    run_phase10_experiments,
)


def main() -> None:
    """Initialize project environment and execute the experiment pipeline."""
    ensure_directories()
    print(f"Project initialized. Root directory: {PROJECT_ROOT}")
    if len(sys.argv) > 1 and sys.argv[1] == "--phase10":
        run_phase10_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase9":
        run_phase9_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase8":
        run_phase8_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase7":
        run_phase7_experiments()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase6":
        run_phase6_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "--phase5a":
        run_phase5a_experiment()
    elif len(sys.argv) > 1 and sys.argv[1] == "--pjm-baseline":
        run_pjm_experiment()
    else:
        run_phase10_experiments()


if __name__ == "__main__":
    main()
