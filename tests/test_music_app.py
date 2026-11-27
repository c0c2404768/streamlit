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

import music_app.data as music_data
from music_app.data import (
    FEATURES, TUNEBAT_CSV, add_tunebat_tracks, dataset_source, load_catalog,
    prepare_catalog, read_csv_bytes, read_table_bytes,
)
from music_app.genre import comparison_takeaway, distribution_bins, genre_statistics, pairwise_statistics, summary_for_display
from music_app.search import UNREGISTERED, search_tracks, tracks_in_genre
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

    def test_parquet_bytes_round_trip(self):
        raw = fixture().head(1).copy()
        raw["track_name"] = "日本語の曲"
        result = read_table_bytes(raw.to_parquet(index=False), "songs.parquet")
        self.assertEqual(result.iloc[0].track_name, "日本語の曲")

    def test_load_catalog_from_parquet(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tracks.parquet"
            fixture().to_parquet(path, index=False)
            catalog = load_catalog(str(path), path.stat().st_mtime_ns)
        self.assertEqual(len(catalog.tracks), 59)
        self.assertEqual(catalog.tracks.iloc[0].track_name, "Comedy")

    def test_normalized_multiword_search_and_literal_symbols(self):
        tracks = self.catalog.tracks
        self.assertEqual(search_tracks(tracks, "  ＣＯＭＥＤＹ　gen   ").track_id.tolist(), [COMEDY, "same-name"])
        self.assertEqual(search_tracks(tracks, "hello abc").track_id.tolist(), ["full-width"])
        self.assertTrue(search_tracks(tracks, "[").empty)
        self.assertTrue(search_tracks(tracks, "missing_song").empty)
        self.assertEqual(search_tracks(tracks, "Comedy", "人気度順").iloc[0].track_id, COMEDY)
        self.assertEqual(search_tracks(tracks, "星野源 Comedy").iloc[0].track_id, COMEDY)

    def test_additional_tracks_preserve_missing_values_and_original_genres(self):
        extra = fixture().iloc[:2].copy()
        extra.loc[extra.index[0], ["track_name", "artists", "tempo"]] = ["喜劇", "星野源", 190]
        extra.loc[extra.index[1], ["track_id", "track_name", "artists"]] = ["new-track", "元気を出して", "Mariya Takeuchi"]
        extra["track_genre"] = pd.NA
        extra["acousticness"] = float("nan")
        catalog = add_tunebat_tracks(self.catalog, extra)
        self.assertEqual(len(catalog.tracks), 60)
        pd.testing.assert_frame_equal(catalog.genre_rows, self.catalog.genre_rows)
        lookup = catalog.tracks.set_index("track_id")
        self.assertEqual(lookup.loc[COMEDY, "tempo"], 100)
        self.assertEqual(lookup.loc[COMEDY, "genres"], "pop / rock")
        self.assertEqual(lookup.loc[COMEDY, "data_sources"], "CSV / Tunebat")
        self.assertEqual(search_tracks(catalog.tracks, "Comedy Gen").iloc[0].track_id, COMEDY)
        self.assertEqual(search_tracks(catalog.tracks, "喜劇 星野源").iloc[0].track_id, COMEDY)
        self.assertEqual(search_tracks(catalog.tracks, "竹内まりや").iloc[0].track_id, "new-track")
        self.assertEqual(tracks_in_genre(catalog, UNREGISTERED).track_id.tolist(), ["new-track"])
        self.assertEqual(set(tracks_in_genre(catalog, "rock").track_id), {COMEDY, "full-width"})
        state = {}
        song = select_track(catalog.tracks, "new-track", state)
        self.assertEqual(song["genres"], "未登録")
        self.assertIsNone(song["acousticness"])

    def test_bundled_ten_thousand_tracks_have_correct_units_and_coverage(self):
        extra = read_csv_bytes(TUNEBAT_CSV.read_bytes())
        self.assertEqual(len(extra), 10000)
        self.assertEqual(extra.track_id.nunique(), 10000)
        self.assertTrue(extra.track_genre.isna().all())
        first = extra.set_index("track_id").loc["3igzzZE41AOz8aHFMGRsqi"]
        self.assertEqual(first.track_name, "元気を出して")
        self.assertEqual(first.energy, .6)
        self.assertEqual(first.tempo, 183)
        for feature in ("energy", "valence", "danceability"):
            self.assertTrue(extra[feature].between(0, 1).all())
        for feature in ("duration_ms", "acousticness", "instrumentalness", "speechiness"):
            self.assertEqual(extra[feature].notna().sum(), 155)
        catalog = prepare_catalog(extra)
        self.assertEqual(len(catalog.tracks), 10000)
        self.assertTrue(catalog.genre_rows.empty)

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

    def test_quartiles_and_pairs(self):
        rows = self.catalog.genre_rows[self.catalog.genre_rows.track_genre.eq("pop")].copy()
        rows.loc[:, "tempo"] = [100, 120, 140]
        summary = genre_statistics(rows, ("tempo",))
        row = summary.iloc[0]
        self.assertEqual((row["median"], row.q1, row.q3, row.iqr), (120, 110, 130, 20))
        full = genre_statistics(self.catalog.genre_rows, self.catalog.features)
        pairs = pairwise_statistics(full, ["pop", "rock", "jazz"])
        self.assertEqual(len(pairs), 24)
        self.assertTrue((pairs["中央値の差"] >= 0).all())
        display = summary_for_display(summary)
        self.assertEqual(display.columns[:3].tolist(), ["特徴量", "ジャンル", "曲数"])
        self.assertEqual(display.iloc[0]["特徴量"], "テンポ（BPM）")

    def test_tempo_takeaway_groups_nearly_equal_genres(self):
        typical = pd.Series({"j-pop": 123.85, "rock": 123.87, "jazz": 109.19})
        self.assertIn("**j-pop・rock** はほぼ同じ速さ", comparison_takeaway(typical, "tempo"))
        self.assertIn("**jazz** は「ゆっくり」寄り", comparison_takeaway(typical, "tempo"))
        self.assertEqual(comparison_takeaway(typical.drop("jazz"), "tempo"),
                         "比べたジャンル同士は、ほぼ同じ速さです。")

    def test_distribution_counts_include_maximum_and_use_each_genre_as_denominator(self):
        rows = pd.DataFrame({"track_genre": ["pop", "pop", "rock"], "energy": [0, 1, 1]})
        points, domain = distribution_bins(rows, "energy", None)
        self.assertEqual(domain, [0, 1])
        self.assertEqual(points.groupby("genre")["count"].sum().to_dict(), {"pop": 2, "rock": 1})
        self.assertEqual(points.groupby("genre").share.sum().to_dict(), {"pop": 1.0, "rock": 1.0})
        self.assertEqual(points.query("genre == 'rock' and share == 1").shape[0], 1)

    def test_missing_explicit_path_does_not_fall_back(self):
        with patch.dict(os.environ, {"SPOTIFY_DATASET_PATH": str(ROOT / "not_present.csv")}):
            with self.assertRaises(FileNotFoundError):
                dataset_source()

    def test_dataset_source_prefers_parquet_over_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "dataset.parquet"
            csv_path = data_dir / "dataset.csv"
            parquet_path.touch()
            csv_path.touch()
            with patch.object(music_data, "REPOSITORY_ROOT", Path(temporary)), patch.dict(os.environ, {"SPOTIFY_DATASET_PATH": ""}):
                self.assertEqual(Path(dataset_source()[0]), parquet_path)
                parquet_path.unlink()
                self.assertEqual(Path(dataset_source()[0]), csv_path)


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
        self.assertTrue(any("大きな点がある場所ほど" in x.value for x in app.caption))
        self.assertTrue(any("Comedy" in x.value for x in app.info))
        app.button(key="_genre_use_selected").click().run()
        self.assertEqual(app.multiselect(key="genre_choices").value, ["pop", "rock"])
        app.multiselect(key="genre_choices").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("1つ以上" in x.value for x in app.info))
        app.multiselect(key="genre_choices").set_value(["pop", "rock"]).run()
        app.selectbox(key="_genre_explore").set_value("rock").run()
        app.button(key="_genre_open_tracks").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="_search_genre").value, "rock")
        self.assertEqual(app.text_input(key="_search_query").value, "")
        self.assertEqual(len(app.dataframe[0].value), 2)
        # AppTestの次回実行先も、switch_pageで描画されたページに合わせる。
        app.switch_page("pages/曲の検索・選択.py").run()
        self.assertEqual(app.session_state["selected_track_id"], COMEDY)
        app.button(key="_search_clear").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("selected_track_id", app.session_state)

    def test_group_page_uses_selected_song_and_limits_search_results(self):
        raw = fixture()
        extras = []
        for i in range(60):
            row = raw.iloc[i % len(raw)].copy()
            row["track_id"] = f"group-{i:03d}"
            row["track_name"] = f"Group track {i:03d}"
            row["tempo"] = 60 + i
            row["energy"] = (i % 10) / 10
            row["valence"] = ((i * 3) % 10) / 10
            row["danceability"] = ((i * 7) % 10) / 10
            extras.append(row)
        pd.concat([raw, pd.DataFrame(extras)], ignore_index=True).to_csv(self.path, index=False)

        app = self.app()
        app.text_input(key="_search_query").set_value("Comedy").run()
        app.selectbox(key="_search_pick").set_value(COMEDY).run()
        app.button(key="_search_confirm").click().run()
        app.switch_page("pages/グループ分け.py").run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["selected_track_id"], COMEDY)
        self.assertTrue(any("選択中の曲: Comedy / Gen Hoshino" in item.value for item in app.text))
        self.assertEqual(app.text_input(key="_group_keyword").value, "Comedy / Gen Hoshino")
        similarities = app.dataframe[0].value
        self.assertGreater(len(similarities), 0)
        self.assertNotIn("Comedy", similarities["曲名"].tolist())
        self.assertIn("特徴距離", similarities.columns)
        self.assertEqual(similarities["特徴距離"].tolist(), sorted(similarities["特徴距離"].tolist()))
        selected_mood_caption = next(item.value.split(":", 1)[1].strip()
                                     for item in app.caption if item.value.startswith("ムード:"))
        self.assertEqual(app.selectbox(key="_group_mood_filter").value, selected_mood_caption)
        app.selectbox(key="_group_mood_filter").set_value("すべて表示").run()
        app.text_input(key="_group_keyword").set_value("").run()
        self.assertEqual(len(app.dataframe[0].value), 100)

        app.text_input(key="_group_keyword").set_value("Comedy").run()
        self.assertEqual(len(app.dataframe[0].value), 2)

        app.switch_page("pages/曲の検索・選択.py").run()
        app.text_input(key="_search_query").set_value("Ghost").run()
        app.selectbox(key="_search_pick").set_value(GHOST).run()
        app.button(key="_search_confirm").click().run()
        app.switch_page("pages/グループ分け.py").run()
        selected_mood_caption = next(item.value.split(":", 1)[1].strip()
                                     for item in app.caption if item.value.startswith("ムード:"))
        self.assertEqual(app.session_state["selected_track_id"], GHOST)
        self.assertEqual(app.text_input(key="_group_keyword").value, "Ghost - Acoustic / Ben Woodward")
        self.assertEqual(app.selectbox(key="_group_mood_filter").value, selected_mood_caption)
        self.assertIn("特徴距離", app.dataframe[0].value.columns)
        app.button(key="_group_clear_selected_song").click().run()
        self.assertNotIn("selected_track_id", app.session_state)
        self.assertEqual(app.text_input(key="_group_keyword").value, "")
        self.assertEqual(app.selectbox(key="_group_mood_filter").value, "すべて表示")

    def test_unknown_genre_song_can_be_compared_without_invented_values(self):
        raw = fixture()
        extra = raw.iloc[:1].copy()
        extra["track_id"] = "new-track"
        extra["track_name"] = "元気を出して"
        extra["artists"] = "Mariya Takeuchi"
        extra["track_genre"] = pd.NA
        extra["data_sources"] = "Tunebat"
        extra["duration_ms"] = float("nan")
        pd.concat([raw, extra], ignore_index=True).to_csv(self.path, index=False)
        app = self.app()
        app.checkbox(key="_search_only_added").check().run()
        app.text_input(key="_search_query").set_value("竹内まりや").run()
        self.assertEqual(len(app.dataframe[0].value), 1)
        app.selectbox(key="_search_pick").set_value("new-track").run()
        app.button(key="_search_confirm").click().run()
        app.switch_page("pages/ジャンル比較.py").run()
        self.assertFalse(app.exception)
        self.assertFalse(any(x.key == "_genre_use_selected" for x in app.button))
        self.assertTrue(any("ジャンルは未登録" in x.value for x in app.caption))
        self.assertTrue(any("オレンジの破線" in x.value for x in app.caption))
        app.selectbox(key="_genre_feature").set_value("duration_min").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("この特徴の値が収録されていません" in x.value for x in app.caption))
        app.button(key="_genre_open_tracks").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.checkbox(key="_search_only_added").value)
        self.assertEqual(app.session_state["selected_track_id"], "new-track")

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
