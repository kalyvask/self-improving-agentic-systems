"""Back-compat shim: the serve logic now lives in the package as `wdp.cli`
(installed as the `wdp-decide` console command). This script keeps the old
invocation working:

    python scripts/serve_policy.py --policy artifacts/policies/sql_dpo_policy.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.cli import main

if __name__ == "__main__":
    main()
