"""及川担当：似ている楽曲の計算・表示、比較グラフ、試聴リンク。"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from music_app.data import CORE_FEATURES, FEATURES as ALL_FEATURES, Catalog, LABELS
from music_app.filters import describe_filters, render_filters, show_no_result
from music_app.selection import selected_track

FEATURES = CORE_FEATURES  # テンポ・力強さ・明るさ・ノリの良さ（班で共通の4特徴）
COLORS = ["#C55A11", "#2E8B6A", "#1B4D3E", "#8FB8A8"]
RADAR_FEATURES = ALL_FEATURES  # 比較グラフに出せる特徴（4特徴＋アコースティックらしさなど4つ）


@st.cache_data(show_spinner=False)
def standardized_matrix(tracks: pd.DataFrame) -> np.ndarray:
    """4特徴を標準化（平均0・標準偏差1）した行列。行どうしの距離がユークリッド距離になる。"""
    values = tracks[list(FEATURES)].to_numpy(dtype=float)
    std = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(std == 0, 1, std)


@st.cache_data(show_spinner=False)
def similarity_matrix(tracks: pd.DataFrame) -> np.ndarray:
    """標準化 → L2正規化した行列。行どうしの内積がコサイン類似度になる。"""
    z = standardized_matrix(tracks)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.where(norm == 0, 1, norm)


def find_similar(tracks: pd.DataFrame, matrix: np.ndarray, track_id: str, top_n=10,
                 max_popularity=None, other_genre=False, method="cos") -> pd.DataFrame:
    """選んだ曲に似ている順で top_n 曲を返す。
    method="cos"：matrix はL2正規化した行列で、コサイン類似度（similarity）が高い順。
    method="euc"：matrix は標準化した行列で、ユークリッド距離（distance）が小さい順。"""
    positions = np.flatnonzero(tracks.track_id.to_numpy() == track_id)
    if len(positions) == 0:
        raise ValueError("指定された楽曲IDがデータにありません。")
    base = tracks.iloc[positions[0]]
    result = tracks.copy()
    if method == "euc":
        result["distance"] = np.linalg.norm(matrix - matrix[positions[0]], axis=1)
    else:
        result["similarity"] = matrix @ matrix[positions[0]]
    # 同じ曲が別のIDで登録されている場合も除く（自分自身のコピーが1位に出ないように）
    same_song = result._title.eq(base._title) & result._artist.eq(base._artist)
    result = result.loc[~same_song]
    if max_popularity is not None:
        result = result.loc[result.popularity.le(max_popularity)]
    if other_genre:
        own = set(str(base.genres).split(" / "))
        result = result.loc[[not (set(str(g).split(" / ")) & own) for g in result.genres]]
    key, closer_first = ("distance", True) if method == "euc" else ("similarity", False)
    result = (result.sort_values([key, "popularity"], ascending=[closer_first, False], kind="stable")
              .drop_duplicates(["_title", "_artist"]).head(top_n).copy())
    result.insert(0, "rank", range(1, len(result) + 1))
    result["url"] = result.track_id.map(spotify_url)
    return result


def closest_features(base, row, n=2) -> list[str]:
    """基準の曲と特に近い特徴（似ている理由）を返す。"""
    diffs = {f: abs(row[f] - base[f]) / (200 if f == "tempo" else 1) for f in FEATURES}
    return [LABELS[f] for f in sorted(diffs, key=diffs.get)[:n]]


def radar_features(tracks: pd.DataFrame, songs: list) -> list[str]:
    """比較グラフに出せる特徴。列がない特徴や、表示する曲のどれかで値が空（Tunebatの追加曲など）の特徴は除く。"""
    usable = []
    for f in RADAR_FEATURES:
        if f in tracks.columns and all(pd.notna(pd.to_numeric(song.get(f), errors="coerce")) for song in songs):
            usable.append(f)
    return usable


def _scale(tracks: pd.DataFrame, feature: str) -> float:
    """0〜1にそろえるための割る数。テンポは最大値、曲の長さは極端に長い曲に引っぱられないよう上位1%の値。"""
    values = pd.to_numeric(tracks[feature], errors="coerce")
    if feature == "tempo":
        top = values.max()
    elif feature == "duration_min":
        top = values.quantile(0.99)
    else:
        return 1.0
    return float(top) if pd.notna(top) and top > 0 else 1.0


def radar_points(tracks: pd.DataFrame, base, rows: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """レーダーチャート用に、各曲の特徴を円周上のx, y座標に変換する（値は0〜1）。"""
    features = list(features)
    scales = {f: _scale(tracks, f) for f in features}
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(features)) / len(features)
    points = []
    songs = [("選んだ曲", base)] + [(f"{r['rank']}位 {r.track_name}", r) for _, r in rows.iterrows()]
    for order, (name, song) in enumerate(songs):
        values = [min(max(float(song[f]) / scales[f], 0.0), 1.0) for f in features]
        for k in range(len(features) + 1):  # 最初の点に戻って図形を閉じる
            i = k % len(features)
            raw = float(song[features[i]])
            points.append({"曲": name, "順": order, "k": k, "特徴": LABELS[features[i]],
                           "値": round(values[i], 3), "元の値": round(raw, 3),
                           "x": values[i] * np.cos(angles[i]), "y": values[i] * np.sin(angles[i])})
    return pd.DataFrame(points)


def radar_spec(points: pd.DataFrame, names: list[str], features=FEATURES) -> dict:
    features = list(features)
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(features)) / len(features)
    grid = [{"x": r * np.cos(a), "y": r * np.sin(a), "r": r, "k": k}
            for r in (0.25, 0.5, 0.75, 1.0) for k, a in enumerate([*angles, angles[0]])]
    spokes = [{"x": v * np.cos(a), "y": v * np.sin(a), "s": i, "k": v}
              for i, a in enumerate(angles) for v in (0, 1)]
    labels = []
    for f, a in zip(features, angles):
        c = np.cos(a)
        # 左右の文字は外側に向けて寄せ、上下は中央ぞろえにして、文字が図の外にはみ出さないようにする
        align = "center" if abs(c) < 0.2 else ("left" if c > 0 else "right")
        labels.append({"x": 1.1 * c, "y": 1.1 * np.sin(a), "label": LABELS[f], "align": align})
    axis = {"axis": None, "scale": {"domain": [-1.6, 1.6], "nice": False}}
    xy = {"x": {"field": "x", "type": "quantitative", **axis},
          "y": {"field": "y", "type": "quantitative", **axis}}
    label_layer = lambda align: {
        "data": {"values": [l for l in labels if l["align"] == align]},
        "mark": {"type": "text", "fontSize": 12, "align": align, "baseline": "middle"},
        "encoding": {**xy, "text": {"field": "label"}}}
    return {
        "height": 460,
        "padding": {"top": 10, "bottom": 10, "left": 10, "right": 10},
        "layer": [
            {"data": {"values": grid}, "mark": {"type": "line", "color": "#D9D9D9", "strokeWidth": 1},
             "encoding": {**xy, "detail": {"field": "r"}, "order": {"field": "k"}}},
            {"data": {"values": spokes}, "mark": {"type": "line", "color": "#E6E6E6", "strokeWidth": 1},
             "encoding": {**xy, "detail": {"field": "s"}, "order": {"field": "k"}}},
            *[label_layer(a) for a in ("center", "left", "right")],
            {"data": {"values": points.to_dict("records")},
             "mark": {"type": "line", "point": True, "strokeWidth": 2},
             "encoding": {**xy,
                          "color": {"field": "曲", "type": "nominal", "sort": names,
                                    "scale": {"domain": names, "range": COLORS[:len(names)]},
                                    "legend": {"orient": "bottom", "columns": 1, "title": None,
                                               "labelLimit": 360, "labelFontSize": 12}},
                          "order": {"field": "k"},
                          "tooltip": [{"field": "曲"}, {"field": "特徴"},
                                      {"field": "値", "title": "グラフ上の値（0〜1）"},
                                      {"field": "元の値", "title": "元の値"}]}},
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

    # 絞り込み条件と表示の設定はサイドバーではなくページ内に置く（共通の filters を where 指定で使う）
    with st.container(border=True):
        set_left, set_right = st.columns([1, 2])
        with set_left:
            st.subheader("表示の設定")
            top_n = st.slider("表示する曲数", 5, 20, 10, key="_similar_top_n")
        with set_right:
            cond = render_filters(st.container())
    max_pop = cond["pop_max"] if cond["pop_max"] < 100 else None
    euclid = cond["method"] == "euc"
    matrix = standardized_matrix(tracks) if euclid else similarity_matrix(tracks)
    result = find_similar(tracks, matrix, base.track_id, top_n, max_pop, cond["other_genre"], cond["method"])
    if result.empty:
        show_no_result()
        st.stop()
    result["reason"] = [" と ".join(closest_features(base, row)) + "が近い" for _, row in result.iterrows()]

    st.subheader("似ている曲のランキング")
    st.caption(describe_filters(cond))
    score = "distance" if euclid else "similarity"
    # スクロールや並べ替えができない固定の表にする（順位を左端の見出しにする）
    table = (result[["rank", "track_name", "artists", "genres", "popularity", score, "reason"]]
             .rename(columns={"rank": "順位", "track_name": "曲名", "artists": "アーティスト",
                              "genres": "ジャンル", "popularity": "人気度", "similarity": "類似度",
                              "distance": "距離", "reason": "似ている理由"})
             .set_index("順位"))
    table[table.columns[4]] = table[table.columns[4]].map("{:.3f}".format)
    table["人気度"] = table["人気度"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")  # 23.0000 ではなく 23 と出す
    st.table(table)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("特徴の比較（上位3曲）")
        top3 = result.head(3)
        names = ["選んだ曲"] + [f"{r['rank']}位 {r.track_name}" for _, r in top3.iterrows()]
        usable = radar_features(tracks, [base] + [r for _, r in top3.iterrows()])
        # 曲が変わって出せる特徴が減ったとき、前に選んでいた特徴が残ってエラーにならないようにする
        kept = st.session_state.get("_similar_radar_features")
        if kept is not None:
            st.session_state["_similar_radar_features"] = [f for f in kept if f in usable] or usable
        chosen = st.multiselect("比べる特徴", usable, default=None if kept is not None else usable,
                                key="_similar_radar_features",
                                format_func=LABELS.get, help="3つ以上選ぶとグラフになります")
        if len(chosen) < 3:
            st.warning("比べる特徴を3つ以上選んでください。")
        else:
            st.vega_lite_chart(spec=radar_spec(radar_points(tracks, base, top3, chosen), names, chosen),
                               width="stretch")
        missing = [LABELS[f] for f in RADAR_FEATURES if f not in usable]
        if missing:
            st.caption(f"{'・'.join(missing)}は値がない曲があるため外しています。")
    with right:
        st.subheader("試聴")
        pick = st.selectbox("試聴する曲", result.track_id.tolist(), key="_similar_listen",
                            format_func=lambda i: f"{int(result.loc[result.track_id.eq(i), 'rank'].iloc[0])}位 "
                                                  f"{result.loc[result.track_id.eq(i), 'track_name'].iloc[0]}")
        components.iframe(spotify_embed_url(pick), height=152)
        st.caption("選んだ曲")
        components.iframe(spotify_embed_url(base.track_id), height=152)
