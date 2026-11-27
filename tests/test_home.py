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
        self.assertEqual(next(m for m in app.metric if m.label == "曲の数").value, "59")
        self.assertTrue(any("なぜ人気度を使わないのか" in x.value for x in app.subheader))
        self.assertEqual([m.label for m in app.metric][:3], ["音の特徴4つ", "音の特徴14項目", "ジャンル"])

    def test_search_button_opens_search_page(self):
        app = self.app().run()
        app.button(key="_home_search").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("曲の検索・選択" in x.value for x in app.title))

    def test_popularity_page_still_opens(self):
        app = self.app().run()
        app.switch_page("pages/人気度との関係.py").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("人気度との関係" in x.value for x in app.title))

    def test_selected_song_is_shown(self):
        app = self.app()
        app.session_state["selected_track_id"] = COMEDY
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Comedy" in x.value for x in app.markdown))
        self.assertFalse(any("まだ曲が選ばれていません" in x.value for x in app.info))

    def test_similar_page_uses_shared_filters(self):
        app = self.app()
        app.session_state["selected_track_id"] = COMEDY
        app.run()
        app.switch_page("pages/似ている楽曲.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio(key="_filter_method").value, "コサイン類似度")
        app.sidebar.radio(key="_filter_method").set_value("ユークリッド距離").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("0に近いほど似ています" in x.value for x in app.caption))
        app.sidebar.slider(key="_filter_pop_max").set_value(0).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("条件：人気度0以下" in x.value for x in app.caption))
        self.assertTrue(app.dataframe[0].value["popularity"].le(0).all())

    def test_grouping_page_uses_shared_data(self):
        app = self.app().run()
        app.switch_page("pages/グループ分け.py").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("該当: 59曲" in x.value for x in app.subheader))


if __name__ == "__main__":
    unittest.main()
