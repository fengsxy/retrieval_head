from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mmengine.config.config as mmcfg

_ORIG_GET_INSTALLED_PATH = mmcfg.get_installed_path


def _patched_get_installed_path(package: str) -> str:
    if package == "opencompass":
        return str(ROOT_DIR / "opencompass")
    return _ORIG_GET_INSTALLED_PATH(package)


mmcfg.get_installed_path = _patched_get_installed_path  # type: ignore

from opencompass.cli.main import main

if __name__ == '__main__':
    main()
