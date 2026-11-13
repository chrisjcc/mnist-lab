"""Streamlit entrypoint wrapper delegating to `app.app.run_app`."""

from __future__ import annotations

from app import run_app

if __name__ == "__main__":
    run_app()
