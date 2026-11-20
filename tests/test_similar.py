from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

from music_app.data import prepare_catalog
from music_app.similar import closest_features, find_similar, radar_points, similarity_matrix, spotify_url


def fixture():
    def song(track_id, title, artist, genre, tempo, energy, valence, dance, popularity=50):
        return dict(track_id=track_id, track_name=title, artists=artist, track_genre=genre,
                    popularity=popularity, tempo=tempo, energy=energy, valence=valence, danceability=dance)
    return pd.DataFrame([
        song("base", "Base", "A", "pop", 120, .60, .70, .65, 80),
        song("base", "Base", "A", "rock", 120, .60, .70, .65, 80),
        song("copy", "Base", "A", "pop", 120, .60, .70, .65, 70),        # 同じ曲の別ID
        song("near-pop", "Near Pop", "B", "pop", 122, .62, .71, .66, 75),
        song("near-jazz", "Near Jazz", "C", "jazz", 118, .58, .69, .64, 20),
        song("mid", "Mid", "D", "jazz", 100, .40, .40, .50, 10),
        song("far", "Far", "E", "metal", 180, .95, .10, .20, 60),
    ])


class SimilarTests(unittest.TestCase):
    def setUp(self):
        self.tracks = prepare_catalog(fixture()).tracks
        self.matrix = similarity_matrix(self.tracks)

    def test_matrix_rows_are_unit_vectors(self):
        self.assertTrue(np.allclose(np.linalg.norm(self.matrix, axis=1), 1))

    def test_ranking_excludes_itself_and_copies(self):
        result = find_similar(self.tracks, self.matrix, "base", top_n=10)
        self.assertNotIn("base", result.track_id.tolist())
        self.assertNotIn("copy", result.track_id.tolist())
        self.assertEqual(result.track_id.iloc[-1], "far")
        self.assertEqual(result["rank"].tolist(), list(range(1, len(result) + 1)))
        self.assertTrue(result.similarity.is_monotonic_decreasing)
        self.assertEqual(result.url.iloc[0], spotify_url(result.track_id.iloc[0]))

    def test_filters(self):
        low = find_similar(self.tracks, self.matrix, "base", max_popularity=30)
        self.assertTrue(low.popularity.le(30).all())
        other = find_similar(self.tracks, self.matrix, "base", other_genre=True)
        self.assertNotIn("near-pop", other.track_id.tolist())
        self.assertEqual(other.track_id.iloc[0], "near-jazz")
        self.assertTrue(find_similar(self.tracks, self.matrix, "base", max_popularity=0).empty)
        with self.assertRaisesRegex(ValueError, "楽曲ID"):
            find_similar(self.tracks, self.matrix, "missing")

    def test_reason_and_radar(self):
        base = self.tracks.set_index("track_id").loc["base"]
        row = self.tracks.set_index("track_id").loc["near-jazz"]
        self.assertEqual(len(closest_features(base, row)), 2)
        result = find_similar(self.tracks, self.matrix, "base", top_n=3)
        points = radar_points(self.tracks, base, result)
        self.assertEqual(len(points), 4 * 5)  # 4曲 × (4特徴 + 閉じる点)
        self.assertTrue(points["値"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
