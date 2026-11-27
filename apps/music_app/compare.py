"""小林担当：似ている曲との人気度比較。

選んだ曲と音が似ている3曲を、レーダーチャートと人気度で並べる。
「人気度との関係」の分析（音の特徴から人気度はほとんど説明できない）を、選んだ曲で確かめるためのページ。
レーダーチャートは、似ている曲の計算に使っている4つの特徴（テンポ・力強さ・明るさ・ノリの良さ）にそろえる。
この4つは追加曲（Tunebat）も含めて全曲に値があり、残りの特徴は追加曲の98%で値がないため使わない。
"""
import base64
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from music_app.data import CORE_FEATURES, Catalog, LABELS
from music_app.genre_names import genre_label
from music_app.popularity import strength_summary
from music_app.search import tracks_in_genre
from music_app.selection import selected_track
from music_app.similar import similarity_matrix

RADAR = CORE_FEATURES
ALL_GENRES = "__all__"
SHOW = 3
SONG_COLOR, SIMILAR_COLOR = "#C55A11", "#2E8B6A"
SHORT_LABELS = {"danceability": "ノリ"}  # 小さい図では横幅が足りないので短くする


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


def radar_svg(tracks: pd.DataFrame, song, color: str, base=None, max_width: int = 320, small: bool = False) -> str:
    """1曲分のレーダーチャートをSVGで作る。viewBox で縦横比を保ったまま枠の幅に合わせて縮むので、
    サイドバーを開いて列が狭くなっても図がはみ出さない。base を渡すと選んだ曲の形を灰色の点線で重ねる。"""
    # 小さい図は枠が狭いので、図形を大きめ・文字を相対的に大きくして、縮んでも項目名が読めるようにする
    # 小さい図は左の「ノリ」より右の「力強さ」が長いので、中心を少し左に寄せて余白を減らす
    width, height, radius, font, cx = (290, 206, 80, 17, 142) if small else (300, 200, 62, 14, 150)
    cy = height / 2
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(RADAR)) / len(RADAR)

    def point(value, angle):
        return cx + value * radius * np.cos(angle), cy - value * radius * np.sin(angle)

    def polygon(values):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(v, a) for v, a in zip(values, angles)))

    parts = [f'<polygon points="{polygon([r] * len(RADAR))}" fill="none" stroke="#D9D9D9" stroke-width="1"/>'
             for r in (0.5, 1.0)]
    parts += [f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#E6E6E6" stroke-width="1"/>'
              for x, y in (point(1.0, a) for a in angles)]
    for feature, angle in zip(RADAR, angles):
        x, y = point(1.16, angle)
        c = np.cos(angle)
        anchor = "middle" if abs(c) < 0.2 else ("start" if c > 0 else "end")
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
                     f'font-size="{font}" fill="#555">{escape(SHORT_LABELS.get(feature, LABELS[feature]) if small else LABELS[feature])}</text>')
    if base is not None:
        parts.append(f'<polygon points="{polygon(radar_values(tracks, base))}" fill="none" stroke="#9A9A9A" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    values = radar_values(tracks, song)
    parts.append(f'<polygon points="{polygon(values)}" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="2.5"/>')
    for feature, value, angle in zip(RADAR, values, angles):
        x, y = point(value, angle)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
                     f'<title>{escape(LABELS[feature])}：{value:.2f}</title></circle>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif">{"".join(parts)}</svg>')
    # 画像として埋め込む（st.html はSVGを取り除くため）。幅100%で縦横比を保ったまま縮む
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{data}" alt="レーダーチャート" '
            f'style="width:100%;max-width:{max_width}px;height:auto;display:block;margin:0 auto">')


def popularity_text(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{int(value)}"


def big_number(label: str, value: str, color: str, size: str = "3.2rem", note: str = ""):
    note_html = f'<div style="font-size:.85rem;color:#666">{escape(note)}</div>' if note else ""
    st.html(f'<div style="text-align:center;line-height:1.15">'
            f'<div style="font-size:.95rem;color:#555">{escape(label)}</div>'
            f'<div style="font-size:{size};font-weight:700;color:{color}">{escape(value)}</div>{note_html}</div>')


def song_header(song, fixed: bool = False):
    """曲名とアーティスト・ジャンル。fixed=True のときは高さをそろえ（長い名前は「…」で省略）、3つのカードの下端をそろえる。"""
    if not fixed:
        st.markdown(f"**{escape(str(song['track_name']))}**")
        st.caption(f"{song['artists']}　／　{song['genres']}")
        return
    clamp = "display:-webkit-box;-webkit-box-orient:vertical;overflow:hidden;"
    st.html(f'<div style="height:6.4em;line-height:1.45" title="{escape(str(song["track_name"]))} / {escape(str(song["artists"]))}">'
            f'<div style="{clamp}-webkit-line-clamp:2;font-weight:700">{escape(str(song["track_name"]))}</div>'
            f'<div style="{clamp}-webkit-line-clamp:2;font-size:.85rem;color:#777;margin-top:.2em">'
            f'{escape(str(song["artists"]))}　／　{escape(str(song["genres"]))}</div></div>')


def render_compare(catalog: Catalog):
    st.title("似ている曲との人気度比較")
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
        picker, _ = st.columns([3, 2])
        genre = picker.selectbox("ジャンル", [ALL_GENRES, *genres], key="_compare_genre", label_visibility="collapsed",
                                 format_func=lambda g: "すべてのジャンル" if g == ALL_GENRES else genre_label(g))
        similar = find_similar_in(catalog, song["track_id"], None if genre == ALL_GENRES else genre)
        if similar.empty:
            st.info("このジャンルには比べられる曲がありません。別のジャンルを選んでください。")
        else:
            for column, (_, row) in zip(st.columns(SHOW), similar.iterrows()):
                with column, st.container(border=True):
                    song_header(row, fixed=True)
                    st.html(radar_svg(tracks, row, SIMILAR_COLOR, base=song, max_width=240, small=True))
                    big_number("人気度", popularity_text(row.popularity), SIMILAR_COLOR, "2rem",
                               f"類似度 {row.similarity:.2f}")
            average = similar.popularity.mean()
            gap = "" if pd.isna(average) or pd.isna(song.get("popularity")) else (
                f"選んだ曲より {abs(song['popularity'] - average):.0f} {'低い' if average < song['popularity'] else '高い'}"
                if round(abs(song["popularity"] - average)) else "選んだ曲とほぼ同じ")
            where = "すべてのジャンル" if genre == ALL_GENRES else genre_label(genre)
            big_number(f"似ている{len(similar)}曲の平均人気度：{where}",
                       "—" if pd.isna(average) else f"{average:.0f}", SIMILAR_COLOR, "4rem", gap)
            st.caption("灰色の点線は選んだ曲の形です。形が重なるほど音が似ています。")

    with right, st.container(border=True):
        st.markdown("##### あなたが選んだ曲")
        song_header(song)
        st.html(radar_svg(tracks, song, SONG_COLOR))
        big_number("人気度", popularity_text(song.get("popularity")), SONG_COLOR, "4rem")

    if not similar.empty and tracks.popularity.notna().sum() >= 3:
        r = strength_summary(tracks, catalog.genre_rows)["core"]
        spread = similar.popularity.max() - similar.popularity.min()
        st.info(f"音がよく似た{len(similar)}曲でも、人気度には最大 {spread:.0f} の差があります。"
                + ("" if np.isnan(r) else f"全曲で調べても、この4つの特徴から人気度はほとんど説明できません"
                   f"（重相関係数 {r:.2f}、説明できる割合 {r ** 2 * 100:.0f}%）。")
                + "だからこのアプリは、人気度ではなく音の特徴で曲を探しています。")
