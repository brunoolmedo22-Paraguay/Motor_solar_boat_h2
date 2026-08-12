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
        self.assertEqual(app.session_state["current_page"], "Visão geral")
        self.assertEqual(len(app.get("image")), 1)

    def test_navigation_and_complete_synthetic_run(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        next(button for button in app.button if button.label == "Entrada").click().run()
        self.assertEqual(app.session_state["current_page"], "Entrada")
        self.assertEqual(app.button[0].label, "▶ RODAR MODELOS")

        next(radio for radio in app.radio if radio.label == "Fonte").set_value(
            "Perfil sintético"
        ).run()
        duration = next(selectbox for selectbox in app.selectbox if selectbox.label == "Janela sintética")
        condition = next(selectbox for selectbox in app.selectbox if selectbox.label == "Condição solar")
        self.assertEqual(len(duration.options), 2)
        self.assertIn("Irradiância perfeita · curva suave", condition.options)
        duration.set_value(1440).run()
        next(button for button in app.button if button.label == "▶ RODAR MODELOS").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["profile"]), 1440)
        self.assertEqual(
            set(app.session_state["results_by_model"]),
            {"irradiancia", "noct_eficiencia", "sdm"},
        )
        self.assertEqual(app.session_state["last_run_notice"]["kind"], "success")

        next(button for button in app.button if button.label == "Modelos").click().run()
        self.assertEqual(app.session_state["current_page"], "Modelos")
        selector = next(
            selectbox for selectbox in app.selectbox if selectbox.label == "Modelo analisado"
        )
        self.assertEqual(selector.value, "sdm")
        self.assertEqual(len(selector.options), 3)
        self.assertEqual(len(app.tabs), 0)
        for model_id in ("irradiancia", "noct_eficiencia", "sdm"):
            selector = next(
                selectbox for selectbox in app.selectbox if selectbox.label == "Modelo analisado"
            )
            selector.set_value(model_id).run()
            self.assertEqual(app.session_state["selected_result_model"], model_id)
            self.assertEqual(len(app.exception), 0)

        for page in ("Comparação", "Exportação", "Visão geral"):
            next(button for button in app.button if button.label == page).click().run()
            self.assertEqual(app.session_state["current_page"], page)
            self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
