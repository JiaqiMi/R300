import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LargeImageSourceContractTests(unittest.TestCase):
    def test_large_load_does_not_retain_full_rgb(self):
        source = (ROOT / "gui" / "layer_manager.py").read_text(encoding="utf-8")
        self.assertIn("self._image_rgb_full = None", source)
        self.assertIn("reader.read_preview", source)
        self.assertIn("update_layer_preview_region", source)

    def test_large_postprocess_routes_away_from_sync_dialog(self):
        source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        marker = "def _on_mask_postprocess(self):"
        body = source[source.index(marker):source.index("def _on_skeleton_from_mask", source.index(marker))]
        self.assertIn("if self._layer_manager.is_large_image_mode", body)
        self.assertIn("self._on_large_mask_postprocess()", body)

    def test_pipeline_copy_happens_in_worker_run(self):
        source = (ROOT / "roadnet" / "pipeline_worker.py").read_text(encoding="utf-8")
        constructor = source[source.index("def __init__"):source.index("def cancel")]
        run_body = source[source.index("def run(self)"):]
        self.assertNotIn("np.asarray(mask, dtype=np.uint8).copy()", constructor)
        self.assertIn("self.mask = np.asarray(self.mask, dtype=np.uint8).copy()", run_body)

    def test_large_menu_exposes_complete_workflow(self):
        source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        for label in (
            "\u6253\u5f00\u5927\u56fe\u9879\u76ee", "\u751f\u6210/\u5237\u65b0\u5927\u56fe\u9884\u89c8", "Tile Index",
            "Tile SAM-RoadPlus", "\u5927\u56fe Mask", "\u5c40\u90e8\u533a\u57df", "\u5168\u5c40\u8def\u7f51",
            "\u5bfc\u5165\u4efb\u52a1\u70b9", "\u89c4\u5212\u8def\u5f84", "\u5bfc\u51fa\u6bd4\u8d5b\u6570\u636e",
        ):
            self.assertIn(label, source)


if __name__ == "__main__":
    unittest.main()
