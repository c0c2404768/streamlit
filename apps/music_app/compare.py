"""小林担当：似ている曲の人気度くらべ。

選んだ曲と音が似ている3曲を、レーダーチャートと人気度で並べる。
「人気度との関係」の分析（音の特徴から人気度はほとんど説明できない）を、選んだ曲で確かめるためのページ。
レーダーチャートは、似ている曲の計算に使っている4つの特徴（テンポ・力強さ・明るさ・ノリの良さ）にそろえる。
この4つは追加曲（Tunebat）も含めて全曲に値があり、残りの特徴は追加曲の98%で値がないため使わない。
"""
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from music_app.data import CORE_FEATURES, Catalog, LABELS
from music_app.popularity import strength_summary
from music_app.search import tracks_in_genre
from music_app.selection import selected_track
from music_app.similar import similarity_matrix

RADAR = CORE_FEATURES
ALL_GENRES = "__all__"
SHOW = 3
SONG_COLOR, SIMILAR_COLOR = "#C55A11", "#2E8B6A"


def find_similar_in(catalog: Catalog, track_id: str, genre: str | None = None, n: int = SHOW) -> pd.DataFrame:
    """選んだ曲にコサイン類似度が高い順で n 曲。genre を指定すると、そのジャンルの曲だけから選ぶ。
    選んだ曲そのものと、曲名・アーティストが同じ別バージョンは除く。"""
    tracks = catalog.tracks
    positions = np.flatnonzero(tracks.track_id.to_numpy() == track_id)
    if len(positions) == 0:
        return tracks.iloc[0:0]
    matrix = similarity_matrix(tracks)
    base = tracks.iloc[positions[0]]
    scores = pd.Series(matrix @ matrix[positions[0]], index=tracks.index)
    keep = ~(tracks._title.eq(base._title) & tracks._artist.eq(base._artist))
    if genre is not None:
        keep &= tracks.index.isin(tracks_in_genre(catalog, genre).index)
    candidates = tracks.loc[keep].assign(similarity=scores[keep])
    return (candidates.sort_values(["similarity", "popularity"], ascending=[False, False], kind="stable")
            .drop_duplicates(["_title", "_artist"]).head(n))


def radar_values(tracks: pd.DataFrame, song) -> list[float]:
    """4特徴を0〜1にそろえる（テンポだけは全曲の最大値で割る）。"""
    tempo_max = float(tracks.tempo.max()) or 1.0
    return [min(max(float(song[f]) / (tempo_max if f == "tempo" else 1.0), 0.0), 1.0) for f in RADAR]


def radar_chart(tracks: pd.DataFrame, song, color: str, height: int, base=None) -> dict:
    """1曲分のレーダーチャート（vega-lite）。base を渡すと、選んだ曲の形を灰色の点線で重ねる。"""
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(RADAR)) / len(RADAR)

    def shape(values, name):
        return [{"x": v * np.cos(a), "y": v * np.sin(a), "k": k, "曲": name,
                 "特徴": LABELS[RADAR[k % len(RADAR)]], "値": round(v, 2)}
                for k, (v, a) in enumerate(zip([*values, values[0]], [*angles, angles[0]]))]

    grid = [{"x": r * np.cos(a), "y": r * np.sin(a), "r": r, "k": k}
            for r in (0.5, 1.0) for k, a in enumerate([*angles, angles[0]])]
    spokes = [{"x": v * np.cos(a), "y": v * np.sin(a), "s": i, "k": v} for i, a in enumerate(angles) for v in (0, 1)]
    small = height < 250
    # 左右の文字は外側へ寄せ、上下は中央ぞろえにして、小さい図でも文字が切れないようにする
    labels = [{"x": 1.12 * np.cos(a), "y": 1.18 * np.sin(a), "label": LABELS[f],
               "align": "center" if abs(np.cos(a)) < 0.2 else ("left" if np.cos(a) > 0 else "right")}
              for f, a in zip(RADAR, angles)]
    width = 1.85 if small else 1.7
    axis = {"axis": None, "scale": {"domain": [-width, width], "nice": False}}
    xy = {"x": {"field": "x", "type": "quantitative", **axis}, "y": {"field": "y", "type": "quantitative", **axis}}
    layers = [
        {"data": {"values": grid}, "mark": {"type": "line", "color": "#D9D9D9", "strokeWidth": 1},
         "encoding": {**xy, "detail": {"field": "r"}, "order": {"field": "k"}}},
        {"data": {"values": spokes}, "mark": {"type": "line", "color": "#E6E6E6", "strokeWidth": 1},
         "encoding": {**xy, "detail": {"field": "s"}, "order": {"field": "k"}}},
        *[{"data": {"values": [l for l in labels if l["align"] == align]},
           "mark": {"type": "text", "fontSize": 9 if small else 13, "color": "#555", "align": align, "baseline": "middle"},
           "encoding": {**xy, "text": {"field": "label"}}} for align in ("center", "left", "right")],
    ]
    tooltip = [{"field": "曲"}, {"field": "特徴"}, {"field": "値", "title": "値（0〜1）"}]
    if base is not None:
        layers.append({"data": {"values": shape(radar_values(tracks, base), "選んだ曲")},
                       "mark": {"type": "line", "color": "#9A9A9A", "strokeDash": [3, 3], "strokeWidth": 1.5},
                       "encoding": {**xy, "order": {"field": "k"}, "tooltip": tooltip}})
    layers.append({"data": {"values": shape(radar_values(tracks, song), str(song["track_name"]))},
                   "mark": {"type": "line", "color": color, "strokeWidth": 2.5, "point": {"filled": True, "size": 30, "color": color}},
                   "encoding": {**xy, "order": {"field": "k"}, "tooltip": tooltip}})
    return {"height": height, "width": height, "padding": {"top": 4, "bottom": 4, "left": 4, "right": 4}, "layer": layers}


def popularity_text(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{int(value)}"


def big_number(label: str, value: str, color: str, size: str = "3.2rem", note: str = ""):
    note_html = f'<div style="font-size:.85rem;color:#666">{escape(note)}</div>' if note else ""
    st.html(f'<div style="text-align:center;line-height:1.15">'
            f'<div style="font-size:.95rem;color:#555">{escape(label)}</div>'
            f'<div style="font-size:{size};font-weight:700;color:{color}">{escape(value)}</div>{note_html}</div>')


def song_header(song):
    st.markdown(f"**{escape(str(song['track_name']))}**")
    st.caption(f"{song['artists']}　／　{song['genres']}")


def render_compare(catalog: Catalog):
    st.title("似ている曲の人気度")
    st.caption("音が似ている曲どうしでも、人気度は大きく違うことがあります。選んだ曲と、音が近い3曲の人気度を比べます。")
    tracks = catalog.tracks
    song = selected_track(tracks)
    if song is None:
        st.info("まだ曲が選ばれていません。「曲の検索・選択」で基準にする曲を選んでください。")
        if st.button("曲を検索して選ぶ", icon=":material/search:", type="primary", key="_compare_search"):
            st.switch_page("pages/曲の検索・選択.py")
        st.stop()

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        genres = sorted(catalog.genre_rows.track_genre.unique())
        picker, _ = st.columns([2, 3])
        genre = picker.selectbox("ジャンル", [ALL_GENRES, *genres], key="_compare_genre", label_visibility="collapsed",
                                 format_func=lambda g: "すべてのジャンル" if g == ALL_GENRES else g)
        similar = find_similar_in(catalog, song["track_id"], None if genre == ALL_GENRES else genre)
        if similar.empty:
            st.info("このジャンルには比べられる曲がありません。別のジャンルを選んでください。")
        else:
            for column, (_, row) in zip(st.columns(SHOW), similar.iterrows()):
                with column, st.container(border=True):
                    song_header(row)
                    st.vega_lite_chart(radar_chart(tracks, row, SIMILAR_COLOR, 128, base=song), width="content")
                    big_number("人気度", popularity_text(row.popularity), SIMILAR_COLOR, "2rem",
                               f"類似度 {row.similarity:.2f}")
            average = similar.popularity.mean()
            gap = "" if pd.isna(average) or pd.isna(song.get("popularity")) else (
                f"選んだ曲より {abs(song['popularity'] - average):.0f} {'低い' if average < song['popularity'] else '高い'}"
                if round(abs(song["popularity"] - average)) else "選んだ曲とほぼ同じ")
            where = "すべてのジャンル" if genre == ALL_GENRES else genre
            big_number(f"似ている{len(similar)}曲の平均人気度（{where}）",
                       "—" if pd.isna(average) else f"{average:.0f}", SIMILAR_COLOR, "4rem", gap)
            st.caption("灰色の点線は選んだ曲の形です。形が重なるほど音が似ています。")

    with right, st.container(border=True):
        st.markdown("##### あなたが選んだ曲")
        song_header(song)
        st.vega_lite_chart(radar_chart(tracks, song, SONG_COLOR, 300), width="content")
        big_number("人気度", popularity_text(song.get("popularity")), SONG_COLOR, "4rem")

    if not similar.empty and tracks.popularity.notna().sum() >= 3:
        r = strength_summary(tracks, catalog.genre_rows)["core"]
        spread = similar.popularity.max() - similar.popularity.min()
        st.info(f"音がよく似た{len(similar)}曲でも、人気度には最大 {spread:.0f} の差があります。"
                + ("" if np.isnan(r) else f"全曲で調べても、この4つの特徴から人気度はほとんど説明できません"
                   f"（重相関係数 {r:.2f}、説明できる割合 {r ** 2 * 100:.0f}%）。")
                + "だからこのアプリは、人気度ではなく音の特徴で曲を探しています。")
        st.page_link("pages/人気度との関係.py", label="分析の詳細を見る（人気度との関係）", icon="📈")
