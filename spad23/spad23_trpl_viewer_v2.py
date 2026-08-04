import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from spad23.gui_v2.app import main  # noqa: E402

if __name__ == "__main__":
    main()
