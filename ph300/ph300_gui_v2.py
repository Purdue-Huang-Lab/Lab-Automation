import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ph300.gui_v2.app import main, MainWindow
from ph300.gui_v2.widget import PH300Widget

__all__ = ["PH300Widget", "MainWindow", "main"]

if __name__ == "__main__":
    main()
