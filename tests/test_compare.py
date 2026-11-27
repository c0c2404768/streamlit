"""似ている曲の人気度ページ（小林担当）のテスト。小さいCSVで、外部ネットワークなしで実行できる。"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "tests"))

from music_app.compare import find_similar_in, radar_values
from music_app.data import prepare_catalog
from test_music_app import COMEDY, fixture

PAGE = "pages/似ている曲の人気度.py"


class CompareLogicTests(unittest.TestCase):
    def setUp(self):
        self.catalog = prepare_catalog(fixture())

    def test_three_similar_songs_without_the_song_itself(self):
        result = find_similar_in(self.catalog, COMEDY)
        self.assertEqual(len(result), 3)
        self.assertNotIn(COMEDY, result.track_id.tolist())
        self.assertTrue(result.similarity.is_monotonic_decreasing)

    def test_genre_limits_the_candidates(self):
        genre = self.catalog.genre_rows.track_genre.value_counts().index[0]
        ids = set(self.catalog.genre_rows.loc[self.catalog.genre_rows.track_genre.eq(genre), "track_id"])
        result = find_similar_in(self.catalog, COMEDY, genre)
        self.assertTrue(len(result) > 0)
        self.assertTrue(set(result.track_id) <= ids)

    def test_radar_values_are_between_0_and_1(self):
        song = self.catalog.tracks.iloc[0]
        values = radar_values(self.catalog.tracks, song)
        self.assertEqual(len(values), 4)
        self.assertTrue(all(0 <= v <= 1 for v in values))


class ComparePageTests(unittest.TestCase):
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

    def test_without_selection_shows_guide(self):
        app = self.app().run()
        app.switch_page(PAGE).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("まだ曲が選ばれていません" in x.value for x in app.info))

    def test_selected_song_and_genre_switch(self):
        app = self.app()
        app.session_state["selected_track_id"] = COMEDY
        app.run()
        app.switch_page(PAGE).run()
        self.assertFalse(app.exception)
        self.assertEqual(sum("image/svg+xml" in x.proto.body for x in app.get("html")), 4)  # 似ている3曲＋選んだ曲
        options = app.selectbox(key="_compare_genre").options
        app.selectbox(key="_compare_genre").set_value(app.selectbox(key="_compare_genre").options[1]).run()
        self.assertFalse(app.exception)
        self.assertGreater(len(options), 1)
        self.assertTrue(any("（" in o and "）" in o for o in options[1:]))  # 「日本語（英語）」の表示


if __name__ == "__main__":
    unittest.main()
