"""入江担当：音の特徴からジャンルを見比べ、曲を探す画面。"""
from itertools import combinations
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from music_app.data import Catalog, LABELS, UNITS
from music_app.selection import selected_track

DEFAULT_GENRES = ("j-pop", "rock", "jazz")
GENRE_COLORS = {"j-pop": "#5AB2FF", "rock": "#F77991", "jazz": "#E9CB62"}
OTHER_COLORS = ("#37CDB2", "#B4A3F2", "#B2D775", "#EBA8D7", "#4FD6EF",
                "#CE98FF", "#7DBDE1", "#A9B88C", "#E2BA99", "#D5CFE6", "#ABB8CB", "#77B18A")
NEAR_DIFFERENCE = {"tempo": 3.0, "duration_min": 0.15}


@st.cache_data(show_spinner=False)
def genre_statistics(rows: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for feature in features:
        groups = rows.groupby("track_genre", sort=True)[feature]
        table = groups.agg(count="count", median="median")
        table["q1"] = groups.quantile(0.25)
        table["q3"] = groups.quantile(0.75)
        table["iqr"] = table.q3 - table.q1
        table["feature"] = feature
        parts.append(table.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pairwise_statistics(summary: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    results = []
    for feature, group in summary.groupby("feature", sort=False):
        lookup = group.set_index("track_genre")
        for genre_a, genre_b in combinations(genres, 2):
            a, b = lookup.loc[genre_a], lookup.loc[genre_b]
            results.append({
                "特徴量": LABELS[feature], "単位": UNITS[feature],
                "ジャンルA": genre_a, "ジャンルB": genre_b,
                "中央値A": a["median"], "中央値B": b["median"],
                "中央値の差": abs(a["median"] - b["median"]),
                "IQR_A": a.iqr, "IQR_B": b.iqr, "IQRの差": abs(a.iqr - b.iqr),
            })
    return pd.DataFrame(results)


def summary_for_display(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["feature"] = table.feature.map(lambda feature: f"{LABELS[feature]}（{UNITS[feature]}）")
    return table[["feature", "track_genre", "count", "median", "q1", "q3", "iqr"]].rename(columns={
        "track_genre": "ジャンル", "feature": "特徴量", "count": "曲数",
        "median": "中央値", "q1": "第1四分位数", "q3": "第3四分位数", "iqr": "IQR",
    })


FEATURE_ENDS = {
    "tempo": ("ゆっくり", "速い"),
    "energy": ("穏やか", "力強い"),
    "valence": ("落ち着いた響き", "明るい響き"),
    "danceability": ("控えめなノリ", "踊りやすい"),
    "acousticness": ("アコースティック感が少ない", "アコースティック感が強い"),
    "instrumentalness": ("歌・声が入る傾向", "楽器中心の傾向"),
    "speechiness": ("話し声が少ない", "話し声が多い"),
    "duration_min": ("短い", "長い"),
}


def format_value(value: float, feature: str) -> str:
    if feature == "tempo":
        return f"{value:.0f} BPM"
    if feature == "duration_min":
        return f"{value:.1f}分"
    return f"{value:.0%}"


def distribution_bins(rows: pd.DataFrame, feature: str, song_value: float | None):
    """全ジャンルで区切りをそろえ、点の大きさをジャンル内の曲の割合にする。"""
    values = rows[feature]
    if UNITS[feature] == "0〜1":
        low, high = 0.0, 1.0
    else:
        low, high = float(values.min()), float(values.max())
        if song_value is not None:
            low, high = min(low, song_value), max(high, song_value)
        padding = max((high - low) * 0.04, 0.5)
        low, high = max(0.0, low - padding), high + padding
    edges = np.linspace(low, high, 13)
    points = []
    for genre, group in rows.groupby("track_genre", sort=False):
        counts, _ = np.histogram(group[feature], bins=edges)
        for index, count in enumerate(counts):
            start, end = edges[index:index + 2]
            points.append({
                "genre": genre,
                "mid": (start + end) / 2,
                "share": count / len(group),
                "count": int(count),
                "interval": f"{format_value(start, feature)} ～ {format_value(end, feature)}",
            })
    return pd.DataFrame(points), [low, high]


def comparison_colors(choices: list[str]) -> dict[str, str]:
    other = iter(OTHER_COLORS)
    return {genre: GENRE_COLORS[genre] if genre in GENRE_COLORS else next(other) for genre in choices}


def render_color_key(colors: dict[str, str]):
    keys = [
        f'<span style="white-space:nowrap"><span aria-hidden="true" style="color:{color}">●</span> '
        f'{escape(genre)}</span>'
        for genre, color in colors.items()
    ]
    st.html('<div aria-label="グラフの色" style="display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.85rem">'
            + "".join(keys) + "</div>")


def comparison_takeaway(typical: pd.Series, feature: str) -> str | None:
    if len(typical) < 2:
        return None

    tolerance = NEAR_DIFFERENCE.get(feature, 0.03)
    lowest, highest = typical.min(), typical.max()
    same = "ほぼ同じ速さ" if feature == "tempo" else f"{LABELS[feature]}はほぼ同じ"
    if highest - lowest <= tolerance:
        return f"比べたジャンル同士は、{same}です。"

    from_low = typical - lowest
    from_high = highest - typical
    low_genres = typical.index[(from_low <= tolerance) & (from_low <= from_high)].tolist()
    high_genres = typical.index[(from_high <= tolerance) & (from_high < from_low)].tolist()
    low, high = FEATURE_ENDS[feature]
    low_text = "・".join(low_genres)
    high_text = "・".join(high_genres)
    if len(high_genres) > 1 and len(low_genres) == 1:
        return f"**{high_text}** は{same}で、**{low_text}** は「{low}」寄りです。"
    if len(low_genres) > 1 and len(high_genres) == 1:
        return f"**{low_text}** は{same}で、**{high_text}** は「{high}」寄りです。"
    return f"比べた中では **{low_text}** が「{low}」寄り、**{high_text}** が「{high}」寄りです。"


def render_distribution(rows, colors, feature, song_value):
    points, domain = distribution_bins(rows, feature, song_value)
    tick_format = {"tempo": ".0f", "duration_min": ".1f"}.get(feature, ".0%")
    scale = {"domain": domain, "nice": False}
    x = {
        "field": "mid", "type": "quantitative", "scale": scale, "title": None,
        "axis": {"tickCount": 5, "format": tick_format},
    }
    y = {"field": "genre", "type": "nominal", "sort": list(colors), "title": None}
    layers = [
        {
            "mark": {"type": "point", "filled": True, "opacity": 0.88},
            "encoding": {
                "x": x,
                "y": y,
                "color": {
                    "field": "genre", "type": "nominal", "legend": None,
                    "scale": {"domain": list(colors), "range": list(colors.values())},
                },
                "size": {
                    "field": "share", "type": "quantitative", "legend": None,
                    "scale": {"domain": [0, 1], "range": [0, 1250]},
                },
                "tooltip": [
                    {"field": "genre", "title": "ジャンル"},
                    {"field": "interval", "title": LABELS[feature]},
                    {"field": "share", "type": "quantitative", "format": ".1%", "title": "このジャンルでの割合"},
                    {"field": "count", "type": "quantitative", "title": "曲数"},
                ],
            },
        },
    ]
    if song_value is not None:
        layers.insert(0, {
            "data": {"values": [{"song": song_value}]},
            "mark": {"type": "rule", "color": "#F69B55", "strokeDash": [4, 4], "strokeWidth": 2},
            "encoding": {"x": {**x, "field": "song"}},
        })
    left, right = FEATURE_ENDS[feature]
    axis_unit = {"tempo": " · BPM", "duration_min": " · 分"}.get(feature, "")
    st.caption(f"{left} ← {LABELS[feature]} → {right}{axis_unit}")
    st.vega_lite_chart(points, {"height": max(135, len(colors) * 46), "layer": layers}, width="stretch")


def song_position(values: pd.Series, value: float, feature: str) -> str:
    lower = int(values.lt(value).sum())
    higher = int(values.gt(value).sum())
    if lower > higher:
        return f"このジャンルの {lower:,} / {len(values):,} 曲より「{FEATURE_ENDS[feature][1]}」側です。"
    if higher > lower:
        return f"このジャンルの {higher:,} / {len(values):,} 曲より「{FEATURE_ENDS[feature][0]}」側です。"
    return f"このジャンルでは、これより値が小さい曲と大きい曲が同じ数です（各{lower:,}曲）。"


def render_details(catalog, choices, feature):
    details = st.expander("詳しいデータを見る", key="_genre_details", on_change="rerun")
    if not details.open:
        return

    with details:
        rows = catalog.genre_rows[catalog.genre_rows.track_genre.isin(choices)]
        summary = genre_statistics(rows, catalog.features)
        st.caption("中央値は曲を値順に並べたときの真ん中の値です。IQRは真ん中50%の幅です。")
        st.caption("画面の説明では、真ん中の値の差が3 BPM以内、曲の長さなら9秒以内、0〜1の指標なら0.03以内を「ほぼ同じ」の目安にしています。")
        show_all = st.toggle("すべての特徴を表示", key="_genre_show_all")
        visible = summary if show_all else summary.loc[summary.feature.eq(feature)]
        table = summary_for_display(visible)
        formats = {
            column: st.column_config.NumberColumn(format="%.2f")
            for column in ("中央値", "第1四分位数", "第3四分位数", "IQR")
        }
        st.dataframe(table, hide_index=True, width="stretch", column_config=formats)
        table_export, pairs_export = st.columns(2) if len(choices) > 1 else (st.container(), None)
        with table_export:
            st.download_button("表示中の表をCSVで保存", table.to_csv(index=False).encode("utf-8-sig"),
                               "genre_statistics.csv", "text/csv", key="_genre_download", width="stretch")
            st.caption("表に出ているジャンルと特徴を保存します。")
        if len(choices) > 1:
            pairs = pairwise_statistics(summary, choices)
            with pairs_export:
                st.download_button("ジャンル間の差をCSVで保存", pairs.to_csv(index=False).encode("utf-8-sig"),
                                   "genre_pairwise.csv", "text/csv", key="_genre_pairs_download", width="stretch")
                st.caption("選んだジャンルを2つずつ組み合わせ、全特徴の値と差を保存します。")
        st.markdown("出典：[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)")


def render_genre_comparison(catalog: Catalog):
    st.title("ジャンル比較")
    st.caption("どんな音が多いかを見比べて、気になるジャンルの曲を探そう。")
    genres = sorted(catalog.genre_rows.track_genre.unique())
    if not genres:
        st.info("このデータにはジャンル情報がありません。標準データに戻すと比較できます。")
        return
    if not catalog.features:
        st.info("このデータには、比較できる音の特徴がありません。")
        return

    song = selected_track(catalog.tracks)
    if song:
        st.info(f"選択中の曲　{song['track_name']} / {song['artists']}")
        own_genres = catalog.genre_rows.loc[
            catalog.genre_rows.track_id.eq(song["track_id"]), "track_genre"
        ].unique().tolist()
        if own_genres:
            if st.button("この曲が入っているジャンルを比べる", key="_genre_use_selected"):
                st.session_state["genre_choices"] = own_genres[:12]
        else:
            st.caption("ジャンルは未登録ですが、音の特徴はグラフで比べられます。")
    else:
        if st.button("曲の検索画面で基準曲を選ぶ", icon=":material/arrow_forward:"):
            st.switch_page("pages/曲の検索・選択.py")

    defaults = [genre for genre in DEFAULT_GENRES if genre in genres] or genres[:3]
    previous = st.session_state.get("genre_choices", defaults)
    st.session_state["genre_choices"] = [genre for genre in previous if genre in genres][:12]
    left, right = st.columns([3, 2])
    with left:
        choices = st.multiselect("比べたいジャンル", genres, key="genre_choices", max_selections=12)
    with right:
        feature = st.selectbox("どんな音を比べる？", catalog.features, format_func=LABELS.get, key="_genre_feature")
    if not choices:
        st.info("比較したいジャンルを1つ以上選んでください。")
        return

    colors = comparison_colors(choices)
    render_color_key(colors)
    rows = catalog.genre_rows.loc[catalog.genre_rows.track_genre.isin(choices),
                                  ["track_genre", feature]].dropna(subset=[feature])
    song_value = song.get(feature) if song else None
    if rows.empty:
        st.info("選んだジャンルには、この特徴のデータがありません。")
    else:
        available = set(rows.track_genre)
        missing = [genre for genre in choices if genre not in available]
        with st.container(border=True):
            st.subheader(f"{LABELS[feature]}の分布")
            st.caption("大きな点がある場所ほど、そのジャンルでよく見かける音です。")
            if song_value is not None:
                st.caption(f"オレンジの破線は選んだ曲の値（{format_value(song_value, feature)}）です。")
            elif song:
                st.caption("選んだ曲には、この特徴の値が収録されていません。")
            render_distribution(rows, colors, feature, song_value)
            if missing:
                st.caption("この特徴のデータがないジャンル：" + "、".join(missing))
            typical = rows.groupby("track_genre")[feature].median()
            takeaway = comparison_takeaway(typical, feature)
            if takeaway:
                st.write(takeaway)

    picker, action = st.columns([3, 2], vertical_alignment="bottom")
    with picker:
        explore = st.selectbox("気になるジャンル", choices, key="_genre_explore")
    with action:
        open_tracks = st.button(f"{explore} の曲一覧を開く", key="_genre_open_tracks", type="primary",
                                icon=":material/arrow_forward:", icon_position="right", width="stretch")
    if open_tracks:
        st.session_state["pending_search_genre"] = explore
        st.switch_page("pages/曲の検索・選択.py")
    st.caption("曲の検索画面に移動し、選んだジャンルの曲を表示します。")
    explore_values = rows.loc[rows.track_genre.eq(explore), feature]
    if song_value is not None and not explore_values.empty:
        st.write(song_position(explore_values, song_value, feature))

    render_details(catalog, choices, feature)
