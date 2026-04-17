from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'

SCRIPTS = [
    'prepare_data.py',
    'eda.py',
    'aspect_generation.py',
    'sentiment_model.py',
]


def main() -> None:
    for script in SCRIPTS:
        script_path = SRC_DIR / script
        print(f'Running {script}...')
        subprocess.run([sys.executable, str(script_path)], check=True)
    print('Pipeline complete.')


if __name__ == '__main__':
    main()
