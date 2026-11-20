"""入江担当：ジャンル別の8特徴、中央値・IQR・分布の比較。"""
from itertools import combinations

import pandas as pd
import streamlit as st

from music_app.data import Catalog, FEATURES, LABELS, UNITS
from music_app.selection import selected_track

DEFAULT_GENRES = ("j-pop", "k-pop", "rock", "jazz", "techno", "hip-hop", "classical")


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


def heatmap_values(summary: pd.DataFrame) -> pd.DataFrame:
    """全ジャンルを基準に標準化し、選択ジャンルに依存しない色の値を計算。"""
    table = summary.copy()
    group = table.groupby("feature")["median"]
    mean = group.transform("mean")
    std = group.transform(lambda values: values.std(ddof=0)).replace(0, 1)
    table["z"] = (table["median"] - mean) / std
    return table


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
    table["feature"] = table.feature.map(LABELS)
    return table.rename(columns={
        "track_genre": "ジャンル", "feature": "特徴量", "count": "曲数",
        "median": "中央値", "q1": "第1四分位数", "q3": "第3四分位数", "iqr": "IQR",
    })


def render_distribution(catalog: Catalog, summary: pd.DataFrame, genres: list[str], feature: str, song=None):
    title = f"{LABELS[feature]}（{UNITS[feature]}）"
    table = summary[summary.track_genre.isin(genres) & summary.feature.eq(feature)].dropna(subset=["median"])
    if table.empty:
        st.info("選択したジャンルには、この特徴の有効な値がありません。")
        return
    scale = {"domain": [0, 1]} if UNITS[feature] == "0〜1" else {"zero": False}
    st.subheader("中央値と真ん中50%の範囲")
    st.caption("点が中央値、線が第1〜第3四分位数です。IQRはこの線の幅を表します。")
    y_axis = {"field": "track_genre", "type": "nominal", "sort": genres, "title": None}
    layers = [
        {"mark": {"type": "rule", "strokeWidth": 9, "color": "#8bb9cd"},
         "encoding": {"y": y_axis, "x": {"field": "q1", "type": "quantitative", "scale": scale, "title": title}, "x2": {"field": "q3"}}},
        {"mark": {"type": "point", "filled": True, "size": 85, "color": "#126782"},
         "encoding": {"y": y_axis, "x": {"field": "median", "type": "quantitative", "scale": scale, "title": title}}},
    ]
    value = song.get(feature) if song else None
    if value is not None and pd.notna(value):
        layers.append({
            "data": {"values": [{"selected_value": float(value), "selected_song": song["track_name"]}]},
            "mark": {"type": "rule", "color": "#ef8a36", "strokeDash": [6, 4], "strokeWidth": 2},
            "encoding": {"x": {"field": "selected_value", "type": "quantitative", "scale": scale, "title": title},
                         "tooltip": [{"field": "selected_song", "title": "選択曲"}, {"field": "selected_value", "title": title, "format": ".3f"}]},
        })
        st.caption(f"オレンジの破線は選択曲「{song['track_name']}」の値：{value:.3f} {UNITS[feature]}です。")
    elif song:
        st.caption("選択曲にはこの特徴の値が収録されていません。値のある特徴を選ぶと、グラフに重ねて比較できます。")
    st.vega_lite_chart(table, {
        "height": max(180, len(genres) * 35),
        "encoding": {
            "tooltip": [
                {"field": "track_genre", "title": "ジャンル"}, {"field": "count", "title": "曲数"},
                {"field": "median", "title": "中央値", "format": ".3f"},
                {"field": "q1", "title": "第1四分位数", "format": ".3f"},
                {"field": "q3", "title": "第3四分位数", "format": ".3f"},
                {"field": "iqr", "title": "IQR", "format": ".3f"},
            ],
        },
        "layer": layers,
    }, width="stretch")
    if value is not None and pd.notna(value):
        closest = table.loc[(table["median"] - value).abs().idxmin()]
        position = "真ん中50%の範囲内" if closest.q1 <= value <= closest.q3 else "真ん中50%の範囲より低め" if value < closest.q1 else "真ん中50%の範囲より高め"
        st.write(f"比較中では、{LABELS[feature]}の中央値が選択曲に最も近いのは **{closest.track_genre}** です。選択曲の値は、そのジャンルの{position}にあります。")
        st.caption("いま選んでいる1つの特徴での比較です。好みや曲全体の類似度を示すものではありません。")
    with st.expander("箱ひげ図で分布を見る"):
        values = catalog.genre_rows.loc[catalog.genre_rows.track_genre.isin(genres), ["track_genre", feature]].dropna()
        st.vega_lite_chart(values, {
            "mark": {"type": "boxplot", "extent": 1.5, "outliers": False},
            "height": max(180, len(genres) * 35),
            "encoding": {
                "y": {"field": "track_genre", "type": "nominal", "sort": genres, "title": None},
                "x": {"field": feature, "type": "quantitative", "scale": scale, "title": title},
            },
        }, width="stretch")
        st.caption("箱は真ん中50%。ひげはIQRの1.5倍以内にある実測値の範囲です。外れ値の点は非表示ですが、集計データからは除外していません。")


def render_genre_comparison(catalog: Catalog):
    st.title("ジャンル比較")
    st.caption("音の特徴の中央値とばらつきから、ジャンルの違いを比べます。")
    song = selected_track(catalog.tracks)
    genres = sorted(catalog.genre_rows.track_genre.unique())
    if not genres:
        st.info("このデータにはジャンル情報がありません。ジャンルが登録された標準データを使うと比較できます。")
        return
    defaults = [genre for genre in DEFAULT_GENRES if genre in genres] or genres[:7]
    if song:
        st.info(f"選択中の曲：{song['track_name']} / {song['artists']}　｜　{song['genres']}")
        own_genres = catalog.genre_rows.loc[catalog.genre_rows.track_id.eq(song["track_id"]), "track_genre"].tolist()
        if own_genres:
            if st.button("選択曲のジャンルで比較する", key="_genre_use_selected"):
                st.session_state["genre_choices"] = list(dict.fromkeys(own_genres))[:12]
        else:
            st.caption("選択曲のジャンルは未登録です。下でジャンルを選び、この曲の特徴と各ジャンルの代表値を比べられます。")
    previous = st.session_state.get("genre_choices", defaults)
    st.session_state["genre_choices"] = [genre for genre in previous if genre in genres][:12]
    choices = st.multiselect("比較するジャンル（最大12個）", genres, key="genre_choices", max_selections=12)
    if not choices:
        st.info("比較したいジャンルを1つ以上選んでください。")
        return
    if not catalog.features:
        st.warning("比較に使える数値の特徴がありません。CSVの列を確認してください。")
        return
    missing = [LABELS[feature] for feature in FEATURES if feature not in catalog.features]
    if missing:
        st.caption("このCSVに含まれていない特徴：" + "、".join(missing))
    summary = genre_statistics(catalog.genre_rows, catalog.features)
    feature = st.selectbox("詳しく見る特徴", catalog.features, format_func=lambda f: f"{LABELS[f]}（{UNITS[f]}）", key="_genre_feature")
    render_distribution(catalog, summary, choices, feature, song)
    explore = st.selectbox("気になるジャンルの曲を探す", choices, key="_genre_explore")
    if st.button(f"{explore} の曲を見る", key="_genre_open_tracks", type="primary"):
        st.session_state["pending_search_genre"] = explore
        st.switch_page("pages/曲の検索・選択.py")
    st.subheader(f"{len(catalog.features)}項目をまとめて比較")
    heat = heatmap_values(summary)
    # 極端なジャンルだけで色が薄まらないよう、色の範囲を固定する。
    limit = 3.0
    heat = heat[heat.track_genre.isin(choices)].copy()
    heat["特徴"] = heat.feature.map(LABELS)
    heat["単位"] = heat.feature.map(UNITS)
    st.vega_lite_chart(heat, {
        "height": max(210, len(choices) * 35),
        "encoding": {
            "x": {"field": "特徴", "type": "nominal", "sort": [LABELS[f] for f in catalog.features], "axis": {"labelAngle": -50, "labelOverlap": False, "labelLimit": 180}, "title": None},
            "y": {"field": "track_genre", "type": "nominal", "sort": choices, "axis": {"labelOverlap": False}, "title": None},
            "tooltip": [
                {"field": "track_genre", "title": "ジャンル"}, {"field": "特徴"},
                {"field": "median", "title": "中央値", "format": ".3f"}, {"field": "単位"},
                {"field": "iqr", "title": "IQR", "format": ".3f"}, {"field": "count", "title": "曲数"},
                {"field": "z", "title": "標準化値", "format": ".2f"},
            ],
        },
        "layer": [
            {"mark": "rect", "encoding": {"color": {"field": "z", "type": "quantitative", "scale": {"domain": [-limit, limit], "scheme": "redblue", "reverse": True, "clamp": True, "nice": False}, "title": "標準化値"}}},
            {"mark": "text", "encoding": {"text": {"field": "z", "type": "quantitative", "format": ".1f"}, "color": {"condition": {"test": f"abs(datum.z) > {limit * 0.6}", "value": "white"}, "value": "black"}}},
        ],
    }, width="stretch")
    st.caption("各特徴を、全ジャンルの中央値の平均・標準偏差で標準化しています。赤は高め、青は低め。色の範囲は常に−3〜3で、それを超える値は同じ端の色になります。実際の数値はカーソルを合わせて確認できます。")
    selected_summary = summary[summary.track_genre.isin(choices)]
    table = summary_for_display(selected_summary)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("中央値・IQRの表をCSVで保存", table.to_csv(index=False).encode("utf-8-sig"), "genre_statistics.csv", "text/csv", key="_genre_download")
    if len(choices) > 1:
        with st.expander("ジャンルの組み合わせごとの差を見る"):
            pairwise = pairwise_statistics(summary, choices)
            st.dataframe(pairwise, hide_index=True, width="stretch")
            st.download_button("全組み合わせの比較表を保存", pairwise.to_csv(index=False).encode("utf-8-sig"), "genre_pairwise.csv", "text/csv", key="_genre_pairs_download")
            st.caption("中央値やIQRの差の大きさは、統計的有意差を示すものではありません。")
    with st.expander("データと特徴の説明"):
        st.write("テンポは速さ、Energyは力強さ、Valenceは明るさ、Danceabilityはノリの良さの指標です。Acousticnessはアコースティックらしさ、Instrumentalnessはインストらしさ、Speechinessは話し声の多さ、Durationは曲の長さです。")
        st.caption("同じ楽曲が複数ジャンルに登録されている場合は、それぞれのジャンルに含めます。同一曲・同一ジャンルの重複は除いています。欠損値がある場合の曲数は特徴ごとに異なります。")
        if catalog.report.get("追加ファイルの曲数"):
            st.caption("Tunebatの追加曲は検索・選択に使えます。ジャンル未登録の追加曲をジャンル別の集計には含めず、既存曲と同じIDの音響値は元データの値を維持しています。")
        st.dataframe(pd.DataFrame(catalog.report.items(), columns=["項目", "件数"]), hide_index=True)
        st.markdown("出典：[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)")
