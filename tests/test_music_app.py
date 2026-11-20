from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

from music_app.data import FEATURES, dataset_source, prepare_catalog, read_csv_bytes
from music_app.genre import genre_statistics, heatmap_values, pairwise_statistics
from music_app.search import search_tracks
from music_app.selection import select_track, selected_track

COMEDY = "5SuOikwiRyPMVoIQDJUgSV"
GHOST = "4qPNDBW1i3p13qLCt0Ki3A"


def fixture():
    def song(track_id, title, artist, genre, tempo, popularity=30, album="Album"):
        return dict(track_id=track_id, track_name=title, artists=artist, album_name=album,
                    track_genre=genre, popularity=popularity, tempo=tempo,
                    energy=.5, valence=.4, danceability=.6, acousticness=.2,
                    instrumentalness=.1, speechiness=.05, duration_ms=180000)
    rows = [song(COMEDY, "Comedy", "Gen Hoshino", "pop", 100, 73),
            song(COMEDY, "Comedy", "Gen Hoshino", "rock", 100, 73),
            song(GHOST, "Ghost - Acoustic", "Ben Woodward", "pop", 120, 55),
            song("same-name", "Comedy", "Gen Hoshino", "pop", 140, album="Other album"),
            song("full-width", "Ｈｅｌｌｏ", "ＡＢＣ", "rock", 160)]
    rows += [song(f"extra-{i:03d}", f"Extra {i:03d}", "Artist", "jazz", 100+i, i % 100) for i in range(55)]
    return pd.DataFrame(rows)


class DataAndSearchTests(unittest.TestCase):
    def setUp(self):
        self.catalog = prepare_catalog(fixture())

    def test_memberships_and_optional_features(self):
        self.assertEqual(len(self.catalog.tracks), 59)
        self.assertEqual(len(self.catalog.genre_rows), 60)
        self.assertEqual(self.catalog.features, FEATURES)
        track = self.catalog.tracks.set_index("track_id").loc[COMEDY]
        self.assertEqual(track.genres, "pop / rock")
        self.assertEqual(track.duration_min, 3)
        duplicate = pd.concat([fixture(), fixture().head(1)], ignore_index=True)
        self.assertEqual(len(prepare_catalog(duplicate).genre_rows), 60)

    def test_numeric_validation_and_missing_columns(self):
        raw = fixture()
        raw.loc[0, "tempo"] = 0
        raw["energy"] = raw.energy.astype(object)
        raw.loc[1, "energy"] = "bad"
        cleaned = prepare_catalog(raw)
        self.assertFalse(cleaned.tracks.track_id.eq(COMEDY).any())
        with self.assertRaisesRegex(ValueError, "track_id"):
            prepare_catalog(pd.DataFrame({"tempo": [100]}))
        minimal = fixture().drop(columns=["acousticness", "instrumentalness", "speechiness", "duration_ms"])
        self.assertEqual(len(prepare_catalog(minimal).features), 4)

    def test_utf8_and_cp932(self):
        raw = fixture().head(1).copy()
        raw["track_name"] = "日本語の曲"
        for encoding in ("utf-8-sig", "cp932"):
            result = read_csv_bytes(raw.to_csv(index=False).encode(encoding))
            self.assertEqual(result.iloc[0].track_name, "日本語の曲")

    def test_normalized_multiword_search_and_literal_symbols(self):
        tracks = self.catalog.tracks
        self.assertEqual(search_tracks(tracks, "  ＣＯＭＥＤＹ　gen   ").track_id.tolist(), [COMEDY, "same-name"])
        self.assertEqual(search_tracks(tracks, "hello abc").track_id.tolist(), ["full-width"])
        self.assertTrue(search_tracks(tracks, "[").empty)
        self.assertTrue(search_tracks(tracks, "missing_song").empty)
        self.assertEqual(search_tracks(tracks, "Comedy", "人気度順").iloc[0].track_id, COMEDY)

    def test_same_titles_and_selection_validation(self):
        results = search_tracks(self.catalog.tracks, "Comedy")
        self.assertEqual(len(results), 2)
        state = {}
        select_track(self.catalog.tracks, COMEDY, state)
        self.assertEqual(state["selected_track_id"], COMEDY)
        self.assertNotIn("_search", state["selected_track"])
        self.assertEqual(selected_track(self.catalog.tracks, state)["genres"], "pop / rock")
        self.assertIsNone(selected_track(self.catalog.tracks.iloc[0:0], state))
        self.assertNotIn("selected_track_id", state)

    def test_quartiles_pairs_and_heatmap(self):
        rows = self.catalog.genre_rows[self.catalog.genre_rows.track_genre.eq("pop")].copy()
        rows.loc[:, "tempo"] = [100, 120, 140]
        summary = genre_statistics(rows, ("tempo",))
        row = summary.iloc[0]
        self.assertEqual((row["median"], row.q1, row.q3, row.iqr), (120, 110, 130, 20))
        full = genre_statistics(self.catalog.genre_rows, self.catalog.features)
        pairs = pairwise_statistics(full, ["pop", "rock", "jazz"])
        self.assertEqual(len(pairs), 24)
        self.assertTrue((pairs["中央値の差"] >= 0).all())
        heat = heatmap_values(full)
        self.assertTrue(heat.z.notna().all())

    def test_missing_explicit_path_does_not_fall_back(self):
        with patch.dict(os.environ, {"SPOTIFY_DATASET_PATH": str(ROOT / "not_present.csv")}):
            with self.assertRaises(FileNotFoundError):
                dataset_source()


class PageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "tracks.csv"
        fixture().to_csv(self.path, index=False)
        self.environment = patch.dict(os.environ, {"SPOTIFY_DATASET_PATH": str(self.path)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def app(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "apps/main.py"), default_timeout=60).run()
        app.switch_page("pages/曲の検索・選択.py").run()
        self.assertFalse(app.exception)
        return app

    def test_search_pagination_selection_and_page_handoff(self):
        app = self.app()
        self.assertEqual(app.number_input(key="_search_page").max, 3)
        app.number_input(key="_search_page").set_value(2).run()
        self.assertFalse(app.exception)
        app.text_input(key="_search_query").set_value("comedy gen").run()
        self.assertEqual(app.number_input(key="_search_page").value, 1)
        app.selectbox(key="_search_pick").set_value(COMEDY).run()
        app.button(key="_search_confirm").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_track_id"], COMEDY)
        app.text_input(key="_search_query").set_value("missing_song").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("該当する曲がありません" in x.value for x in app.warning))
        self.assertEqual(app.session_state["selected_track_id"], COMEDY)
        app.switch_page("pages/ジャンル比較.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox(key="_genre_feature").options), 8)
        self.assertTrue(any("Comedy" in x.value for x in app.info))
        app.button(key="_genre_use_selected").click().run()
        self.assertEqual(app.multiselect(key="genre_choices").value, ["pop", "rock"])
        app.multiselect(key="genre_choices").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("1つ以上" in x.value for x in app.info))
        app.switch_page("pages/曲の検索・選択.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_track_id"], COMEDY)
        app.button(key="_search_clear").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("selected_track_id", app.session_state)

    def test_unselected_and_bad_csv_are_explained(self):
        app = self.app()
        self.assertTrue(app.button(key="_search_confirm").disabled)
        self.assertTrue(any("検索結果から曲を選び" in x.value for x in app.info))
        self.path.write_text("wrong_column\n1\n", encoding="utf-8")
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("読み込めません" in x.value for x in app.error))


if __name__ == "__main__":
    unittest.main()
