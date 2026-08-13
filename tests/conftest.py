"""Put the repo root on sys.path so `import src...` works under pytest.

Tradingbot does this with a sys.path.insert at the top of every test file; a
single conftest is equivalent and less repetitive.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
