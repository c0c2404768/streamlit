"""追加曲（Tunebat）のジャンル補完のテスト。小さいCSVで、外部ネットワークなしで実行できる。"""
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
sys.path.insert(0, str(ROOT / "tests"))

from music_app.data import (FILLED_FROM_ARTIST, TUNEBAT_GENRES_CSV, add_tunebat_tracks, artist_genres,
                            load_artist_genre_table, prepare_catalog)
from music_app.search import UNREGISTERED, tracks_in_genre
from test_music_app import COMEDY, fixture


class TunebatGenreTests(unittest.TestCase):
    def setUp(self):
        self.catalog = prepare_catalog(fixture())
        self.empty_table = load_artist_genre_table(TUNEBAT_GENRES_CSV).iloc[0:0]

    def test_most_common_genre_per_artist(self):
        self.assertEqual(artist_genres(self.catalog.genre_rows)["gen hoshino"], "pop")  # pop 2曲・rock 1曲

    def test_added_tracks_get_genre_from_same_artist(self):
        extra = fixture().iloc[:1].copy()
        extra["track_id"], extra["track_name"], extra["track_genre"] = "hoshino-new", "新曲", pd.NA
        other = extra.copy()
        other["track_id"], other["artists"] = "unknown-new", "Nobody Known"
        catalog = add_tunebat_tracks(self.catalog, pd.concat([extra, other], ignore_index=True), self.empty_table)
        lookup = catalog.tracks.set_index("track_id")
        self.assertEqual(lookup.loc["hoshino-new", "genres"], "pop")
        self.assertEqual(lookup.loc["hoshino-new", "genre_source"], FILLED_FROM_ARTIST)
        self.assertEqual(lookup.loc["unknown-new", "genres"], "未登録")  # 元データにいないアーティストは推測しない
        self.assertEqual(lookup.loc[COMEDY, "genre_source"], "データセット")
        self.assertEqual(catalog.report["アーティストからジャンルを補完した曲数"], 1)
        pd.testing.assert_frame_equal(catalog.genre_rows, self.catalog.genre_rows)  # ジャンル比較の集計は変えない
        self.assertIn("hoshino-new", tracks_in_genre(catalog, "pop").track_id.tolist())
        self.assertNotIn("hoshino-new", tracks_in_genre(catalog, "rock").track_id.tolist())
        self.assertEqual(tracks_in_genre(catalog, UNREGISTERED).track_id.tolist(), ["unknown-new"])

    def test_artist_table_fills_the_rest_with_its_source(self):
        extra = fixture().iloc[:2].copy()
        extra["track_id"] = ["wd-new", "judged-new"]
        extra["artists"] = ["Anri", "Unknown Band"]
        extra["track_genre"] = pd.NA
        table = pd.DataFrame({"artist": ["ANRI", "unknown band"], "genre": ["j-pop", "j-rock"],
                              "genre_source": ["Wikidataから付与", "アーティストの調査から付与"], "wikidata": ["Q2761128", ""]})
        table["_main"] = table.artist.str.casefold()
        catalog = add_tunebat_tracks(self.catalog, extra, table)
        lookup = catalog.tracks.set_index("track_id")
        self.assertEqual(lookup.loc["wd-new", ["genres", "genre_source"]].tolist(), ["j-pop", "Wikidataから付与"])
        self.assertEqual(lookup.loc["judged-new", ["genres", "genre_source"]].tolist(), ["j-rock", "アーティストの調査から付与"])
        self.assertEqual(catalog.report["ジャンル表で付けた曲数"], 2)
        self.assertIn("judged-new", tracks_in_genre(catalog, "j-rock").track_id.tolist())

    def test_bundled_table_uses_known_genres_only(self):
        table = load_artist_genre_table(TUNEBAT_GENRES_CSV)
        genres = set(prepare_catalog(fixture()).genre_rows.track_genre) | {
            "j-pop", "j-rock", "j-idol", "jazz", "hip-hop", "r-n-b", "mandopop", "funk", "electronic", "reggae",
            "house", "synth-pop", "anime", "soul", "ska", "cantopop", "disco", "brazil", "classical", "blues"}
        self.assertGreater(len(table), 1600)
        self.assertTrue(table.genre.isin(genres).all())
        self.assertTrue(table.genre_source.isin(["Wikidataから付与", "アーティストの調査から付与"]).all())


if __name__ == "__main__":
    unittest.main()
