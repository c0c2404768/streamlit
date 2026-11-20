"""トップページ（apps/main.py）のテスト。小さいCSVで、外部ネットワークなしで実行できる。"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "tests"))

from test_music_app import COMEDY, fixture


class HomePageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "songs.csv"
        fixture().to_csv(self.path, index=False)
        self.environment = patch.dict(os.environ, {"SPOTIFY_DATASET_PATH": str(self.path)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "apps/main.py"), default_timeout=60)

    def test_unselected_home_shows_guide_and_data(self):
        app = self.app().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("まだ曲が選ばれていません" in x.value for x in app.info))
        self.assertEqual(app.metric[0].value, "59")

    def test_selected_song_is_shown(self):
        app = self.app()
        app.session_state["selected_track_id"] = COMEDY
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Comedy" in x.value for x in app.markdown))
        self.assertFalse(any("まだ曲が選ばれていません" in x.value for x in app.info))


if __name__ == "__main__":
    unittest.main()
