"""及川担当：似ている楽曲の計算・表示、比較グラフ、試聴リンク。"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from music_app.data import CORE_FEATURES, Catalog, LABELS
from music_app.selection import selected_track

FEATURES = CORE_FEATURES  # テンポ・力強さ・明るさ・ノリの良さ（班で共通の4特徴）
COLORS = ["#C55A11", "#2E8B6A", "#1B4D3E", "#8FB8A8"]


@st.cache_data(show_spinner=False)
def similarity_matrix(tracks: pd.DataFrame) -> np.ndarray:
    """標準化 → L2正規化した行列。行どうしの内積がコサイン類似度になる。"""
    values = tracks[list(FEATURES)].to_numpy(dtype=float)
    std = values.std(axis=0)
    z = (values - values.mean(axis=0)) / np.where(std == 0, 1, std)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.where(norm == 0, 1, norm)


def find_similar(tracks: pd.DataFrame, matrix: np.ndarray, track_id: str, top_n=10,
                 max_popularity=None, other_genre=False) -> pd.DataFrame:
    """選んだ曲にコサイン類似度が高い順で top_n 曲を返す。"""
    positions = np.flatnonzero(tracks.track_id.to_numpy() == track_id)
    if len(positions) == 0:
        raise ValueError("指定された楽曲IDがデータにありません。")
    base = tracks.iloc[positions[0]]
    result = tracks.copy()
    result["similarity"] = matrix @ matrix[positions[0]]
    # 同じ曲が別のIDで登録されている場合も除く（自分自身のコピーが1位に出ないように）
    same_song = result._title.eq(base._title) & result._artist.eq(base._artist)
    result = result.loc[~same_song]
    if max_popularity is not None:
        result = result.loc[result.popularity.le(max_popularity)]
    if other_genre:
        own = set(str(base.genres).split(" / "))
        result = result.loc[[not (set(str(g).split(" / ")) & own) for g in result.genres]]
    result = (result.sort_values(["similarity", "popularity"], ascending=[False, False], kind="stable")
              .drop_duplicates(["_title", "_artist"]).head(top_n).copy())
    result.insert(0, "rank", range(1, len(result) + 1))
    result["url"] = result.track_id.map(spotify_url)
    return result


def closest_features(base, row, n=2) -> list[str]:
    """基準の曲と特に近い特徴（似ている理由）を返す。"""
    diffs = {f: abs(row[f] - base[f]) / (200 if f == "tempo" else 1) for f in FEATURES}
    return [LABELS[f] for f in sorted(diffs, key=diffs.get)[:n]]


def radar_points(tracks: pd.DataFrame, base, rows: pd.DataFrame) -> pd.DataFrame:
    """レーダーチャート用に、各曲の4特徴を円周上のx, y座標に変換する。"""
    tempo_max = tracks.tempo.max()
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(FEATURES)) / len(FEATURES)
    points = []
    songs = [("選んだ曲", base)] + [(f"{r['rank']}位 {r.track_name}", r) for _, r in rows.iterrows()]
    for order, (name, song) in enumerate(songs):
        values = [song.tempo / tempo_max] + [song[f] for f in FEATURES[1:]]
        for k in range(len(FEATURES) + 1):  # 最初の点に戻って図形を閉じる
            i = k % len(FEATURES)
            points.append({"曲": name, "順": order, "k": k, "特徴": LABELS[FEATURES[i]],
                           "値": round(float(values[i]), 3),
                           "x": values[i] * np.cos(angles[i]), "y": values[i] * np.sin(angles[i])})
    return pd.DataFrame(points)


def radar_spec(points: pd.DataFrame, names: list[str]) -> dict:
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(FEATURES)) / len(FEATURES)
    grid = [{"x": r * np.cos(a), "y": r * np.sin(a), "r": r, "k": k % len(FEATURES)}
            for r in (0.25, 0.5, 0.75, 1.0) for k, a in enumerate([*angles, angles[0]])]
    labels = [{"x": 1.18 * np.cos(a), "y": 1.12 * np.sin(a), "label": LABELS[f]}
              for f, a in zip(FEATURES, angles)]
    axis = {"axis": None, "scale": {"domain": [-1.3, 1.3]}}
    return {
        "height": 380,
        "layer": [
            {"data": {"values": grid}, "mark": {"type": "line", "color": "#D9D9D9", "strokeWidth": 1},
             "encoding": {"x": {"field": "x", "type": "quantitative", **axis},
                          "y": {"field": "y", "type": "quantitative", **axis},
                          "detail": {"field": "r"}, "order": {"field": "k"}}},
            {"data": {"values": labels}, "mark": {"type": "text", "fontSize": 13},
             "encoding": {"x": {"field": "x", "type": "quantitative", **axis},
                          "y": {"field": "y", "type": "quantitative", **axis},
                          "text": {"field": "label"}}},
            {"data": {"values": points.to_dict("records")},
             "mark": {"type": "line", "point": True, "strokeWidth": 2},
             "encoding": {"x": {"field": "x", "type": "quantitative", **axis},
                          "y": {"field": "y", "type": "quantitative", **axis},
                          "color": {"field": "曲", "type": "nominal", "sort": names,
                                    "scale": {"domain": names, "range": COLORS[:len(names)]},
                                    "legend": {"orient": "bottom", "columns": 2, "title": None}},
                          "order": {"field": "k"},
                          "tooltip": [{"field": "曲"}, {"field": "特徴"}, {"field": "値"}]}},
        ],
    }


def spotify_url(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


def spotify_embed_url(track_id: str) -> str:
    return f"https://open.spotify.com/embed/track/{track_id}"


def render_similar(catalog: Catalog):
    st.title("似ている楽曲")
    st.caption("選んだ曲と、テンポ・力強さ・明るさ・ノリの良さのバランスが近い曲を探します。")
    tracks = catalog.tracks
    song = selected_track(tracks)
    if song is None:
        st.info("まだ曲が選ばれていません。「曲の検索・選択」で基準にする曲を選んでください。")
        st.page_link("pages/曲の検索・選択.py", label="曲を検索して選ぶ", icon="🔍")
        st.stop()
    base = tracks.loc[tracks.track_id.eq(song["track_id"])].iloc[0]
    st.info(f"選択中の曲：{base.track_name} / {base.artists}　｜　{base.genres}")

    with st.sidebar:
        st.header("表示の設定")
        top_n = st.slider("表示する曲数", 5, 20, 10, key="_similar_top_n")
        use_pop = st.checkbox("人気度で絞り込む（まだ知らない曲を探す）", key="_similar_use_pop")
        max_pop = st.slider("人気度の上限", 0, 100, 30, key="_similar_max_pop") if use_pop else None
        other_genre = st.checkbox("選んだ曲と別のジャンルだけにする", key="_similar_other_genre")

    result = find_similar(tracks, similarity_matrix(tracks), base.track_id, top_n, max_pop, other_genre)
    if result.empty:
        st.warning("条件に合う曲がありません。絞り込み条件をゆるめてください。")
        st.stop()
    result["reason"] = [" と ".join(closest_features(base, row)) + "が近い" for _, row in result.iterrows()]

    st.subheader("似ている曲のランキング")
    table = result[["rank", "track_name", "artists", "genres", "popularity", "similarity", "reason", "url"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "rank": "順位", "track_name": "曲名", "artists": "アーティスト", "genres": "ジャンル",
        "popularity": "人気度", "similarity": st.column_config.NumberColumn("類似度", format="%.3f"),
        "reason": "似ている理由", "url": st.column_config.LinkColumn("Spotify", display_text="開く"),
    })
    st.caption("類似度は−1〜1で、1に近いほど似ています。4つの特徴を標準化し、ベクトルの向き（特徴のバランス）で比べています。")

    left, right = st.columns(2)
    with left:
        st.subheader("特徴の比較（上位3曲）")
        top3 = result.head(3)
        names = ["選んだ曲"] + [f"{r['rank']}位 {r.track_name}" for _, r in top3.iterrows()]
        st.vega_lite_chart(spec=radar_spec(radar_points(tracks, base, top3), names), width="stretch")
        st.caption("テンポは最大値で割って0〜1にそろえています。点にカーソルを合わせると値を確認できます。")
    with right:
        st.subheader("試聴")
        pick = st.selectbox("試聴する曲", result.track_id.tolist(), key="_similar_listen",
                            format_func=lambda i: f"{int(result.loc[result.track_id.eq(i), 'rank'].iloc[0])}位 "
                                                  f"{result.loc[result.track_id.eq(i), 'track_name'].iloc[0]}")
        components.iframe(spotify_embed_url(pick), height=152)
        st.caption("選んだ曲")
        components.iframe(spotify_embed_url(base.track_id), height=152)
        st.caption("Spotifyの埋め込みプレーヤーです。表示にはインターネット接続が必要です。")
