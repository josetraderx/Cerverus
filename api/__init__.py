"""
Top-level package marker for the API used during tests and local imports.

Adding an explicit __init__ makes `import api.app.main` reliable when pytest
collects tests from the project root (avoids ModuleNotFoundError on some setups).
"""

__all__ = ["app"]
