from __future__ import annotations

from pathlib import Path
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # Permite testar a física sem instalar a interface.
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit não instalado")
class AppSmokeTest(unittest.TestCase):
    def test_initial_page_renders_without_exception(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
