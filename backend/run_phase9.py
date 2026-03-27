"""
Phase 9 — API Layer

Launches the FastAPI server on http://localhost:8000
Interactive docs: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path

import uvicorn

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

# Load environment variables from .env if present
_env_file = BACKEND_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

HOST = "0.0.0.0"
PORT = 8000

if __name__ == "__main__":
    print(f"\nPhase 9: Portfolio Stress Testing API")
    print(f"Backend root : {BACKEND_ROOT}")
    print(f"Server       : http://localhost:{PORT}")
    print(f"API docs     : http://localhost:{PORT}/docs")
    print(f"Redoc        : http://localhost:{PORT}/redoc\n")

    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        reload_dirs=[str(BACKEND_ROOT)],
    )
