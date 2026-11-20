"""小林担当（人気度との関係・絞り込み条件）のテスト。小さいCSVで、外部ネットワークなしで実行できる。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

from music_app.data import prepare_catalog
from music_app.filters import DEFAULTS, apply_filters, describe_filters, get_filters
from music_app.popularity import (bin_table, correlation_ratio, correlation_table,
                                  multiple_correlation, numeric_features, strength_summary)


def fixture():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        genre = ["pop", "jazz", "rock"][i % 3]
        rows.append(dict(track_id=f"id-{i:03d}", track_name=f"Song {i:03d}", artists=f"Artist {i % 7}",
                         album_name="Album", track_genre=genre,
                         popularity={"pop": 60, "jazz": 20, "rock": 40}[genre] + int(rng.integers(-5, 6)),
                         explicit=bool(i % 2), danceability=rng.random(), energy=rng.random(), key=i % 12,
                         loudness=-rng.random() * 20, mode=i % 2, speechiness=rng.random(),
                         acousticness=rng.random(), instrumentalness=rng.random(), liveness=rng.random(),
                         valence=rng.random(), tempo=80 + rng.random() * 100, time_signature=4,
                         duration_ms=150000 + i * 100))
    rows[0]["popularity"] = 0
    rows.append(dict(rows[1], track_id="id-copy"))          # 同じ曲の別バージョン
    rows.append(dict(rows[2], track_genre="jazz"))          # 同じ曲が2つのジャンルに所属
    return pd.DataFrame(rows)


class PopularityTests(unittest.TestCase):
    def setUp(self):
        self.catalog = prepare_catalog(fixture())

    def test_explicit_is_numeric_and_input_unchanged(self):
        before = self.catalog.tracks.explicit.copy()
        table = numeric_features(self.catalog.tracks)
        self.assertEqual(set(table.explicit.dropna().unique()), {0, 1})
        pd.testing.assert_series_equal(self.catalog.tracks.explicit, before)

    def test_correlation_sorted_by_strength(self):
        corr = correlation_table(self.catalog.tracks)
        self.assertNotIn("time_signature", corr.index)      # 値が1種類の項目は除く
        self.assertTrue((corr.abs().diff().dropna() <= 1e-12).all())

    def test_genre_is_stronger_than_features(self):
        s = strength_summary(self.catalog.tracks, self.catalog.genre_rows)
        self.assertGreater(s["genre"], 0.9)
        self.assertLess(s["core"], s["genre"])
        self.assertTrue(0 <= s["all"] <= 1)

    def test_multiple_correlation_matches_perfect_fit(self):
        data = pd.DataFrame({"x": range(10), "popularity": [2 * v + 1 for v in range(10)]})
        self.assertAlmostEqual(multiple_correlation(data, ["x"]), 1.0)
        self.assertTrue(np.isnan(multiple_correlation(data.head(2), ["x"])))

    def test_correlation_ratio_constant_popularity(self):
        rows = pd.DataFrame({"track_genre": ["a", "b"], "popularity": [10, 10]})
        self.assertTrue(np.isnan(correlation_ratio(rows)))

    def test_bins_drop_small_groups(self):
        table = bin_table(self.catalog.tracks, "energy", bins=10, min_count=10)
        self.assertTrue((table["曲数"] >= 10).all())
        self.assertGreater(len(table), 0)
        self.assertTrue(bin_table(self.catalog.tracks, "energy", min_count=10_000).empty)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.tracks = prepare_catalog(fixture()).tracks

    def test_defaults_and_description(self):
        state = {}
        self.assertEqual(get_filters(state), {"pop_max": 100, "other_genre": False, "method": "cos"})
        self.assertEqual(set(state), set(DEFAULTS))
        self.assertEqual(describe_filters({"pop_max": 30, "other_genre": True, "method": "euc"}),
                         "条件：人気度30以下・ほかのジャンルだけ")

    def test_excludes_self_and_other_versions(self):
        cond = {"pop_max": 100, "other_genre": False, "method": "cos"}
        result = apply_filters(self.tracks, "id-001", cond)
        self.assertFalse(result.track_id.isin(["id-001", "id-copy"]).any())
        self.assertEqual(len(result), len(self.tracks) - 2)

    def test_popularity_and_all_member_genres(self):
        cond = {"pop_max": 45, "other_genre": True, "method": "cos"}
        result = apply_filters(self.tracks, "id-002", cond)    # id-002 は rock と jazz に所属
        self.assertTrue((result.popularity <= 45).all())
        self.assertFalse(result.genres.str.contains("rock|jazz").any())

    def test_unknown_or_missing_selection(self):
        cond = {"pop_max": 100, "other_genre": False, "method": "cos"}
        self.assertTrue(apply_filters(self.tracks, "no-such-id", cond).empty)
        self.assertTrue(apply_filters(self.tracks, None, cond).empty)
        self.assertTrue(apply_filters(self.tracks, "id-001", {**cond, "pop_max": -1}).empty)


if __name__ == "__main__":
    unittest.main()
