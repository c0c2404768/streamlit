"""小林担当：楽曲の特徴と人気度の関係（相関係数・区間ごとの平均人気度・重相関係数と相関比）。"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from music_app.data import CORE_FEATURES, Catalog, LABELS

# 人気度と比べる全14項目（曲の長さは共通データの duration_min を使う）
ALL_FEATURES = ("duration_min", "explicit", "danceability", "energy", "key", "loudness", "mode",
                "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
                "tempo", "time_signature")
EXTRA_LABELS = {
    "explicit": "過激な歌詞（Eマーク）", "key": "キー", "loudness": "音量", "mode": "長調・短調",
    "liveness": "ライブ感", "time_signature": "拍子",
}
# 区間に分けて見るのは、値が連続している項目だけ（キー・長調短調・拍子・Eマークは除く）
BIN_FEATURES = ("instrumentalness", "danceability", "speechiness", "loudness", "energy", "valence",
                "acousticness", "liveness", "tempo", "duration_min")
MIN_COUNT = 30   # 曲数がこれより少ない区間は、平均がぶれやすいので出さない
BLUE, RED = "#1f77b4", "#d62728"


def label(feature: str) -> str:
    return LABELS.get(feature) or EXTRA_LABELS.get(feature, feature)


def numeric_features(tracks: pd.DataFrame) -> pd.DataFrame:
    """人気度と14項目を数値にそろえる（explicit の True/False は 1/0 に）。元のデータは変更しない。"""
    columns = [c for c in ("popularity", *ALL_FEATURES) if c in tracks]
    table = tracks[columns].copy()
    if "explicit" in table:
        text = table["explicit"].astype("string").str.strip().str.lower()
        table["explicit"] = text.map({"true": 1, "false": 0, "1": 1, "0": 0})
    return table.apply(pd.to_numeric, errors="coerce")


@st.cache_data(show_spinner=False)
def correlation_table(tracks: pd.DataFrame) -> pd.Series:
    """各項目と人気度の相関係数。関係の強い順（絶対値の大きい順）に並べる。"""
    table = numeric_features(tracks)
    usable = [c for c in ALL_FEATURES if c in table and table[c].nunique(dropna=True) > 1]
    if "popularity" not in table or table["popularity"].nunique(dropna=True) < 2 or not usable:
        return pd.Series(dtype=float)
    corr = table[usable].corrwith(table["popularity"]).dropna()
    return corr[corr.abs().sort_values(ascending=False).index]


def multiple_correlation(table: pd.DataFrame, columns) -> float:
    """columns をまとめて使って人気度を予測したときの重相関係数 R（最小二乗法）。"""
    data = table[[*columns, "popularity"]].dropna()
    if len(data) <= len(columns) + 1:
        return float("nan")
    X = np.column_stack([np.ones(len(data)), data[list(columns)].to_numpy(float)])
    y = data["popularity"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = ((y - X @ coef) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(np.sqrt(max(0.0, 1 - residual / total))) if total > 0 else float("nan")


def correlation_ratio(genre_rows: pd.DataFrame) -> float:
    """ジャンルの相関比 η（ジャンルごとのデータを使う。同じ曲でもジャンルが違えば別の行）。"""
    rows = genre_rows.dropna(subset=["popularity"])
    total = ((rows.popularity - rows.popularity.mean()) ** 2).sum()
    if total == 0:
        return float("nan")
    groups = rows.groupby("track_genre").popularity
    between = ((groups.mean() - rows.popularity.mean()) ** 2 * groups.count()).sum()
    return float(np.sqrt(between / total))


@st.cache_data(show_spinner=False)
def strength_summary(tracks: pd.DataFrame, genre_rows: pd.DataFrame) -> dict[str, float]:
    table = numeric_features(tracks)
    usable = [c for c in ALL_FEATURES if c in table and table[c].nunique(dropna=True) > 1]
    core = [c for c in CORE_FEATURES if c in usable]
    return {
        "core": multiple_correlation(table, core),
        "all": multiple_correlation(table, usable),
        "all_no0": multiple_correlation(table[table.popularity > 0], usable),
        "genre": correlation_ratio(genre_rows),
    }


@st.cache_data(show_spinner=False)
def bin_table(tracks: pd.DataFrame, feature: str, bins: int = 10, min_count: int = MIN_COUNT) -> pd.DataFrame:
    """feature を等間隔の区間に分け、区間ごとの平均人気度を出す（全曲と、人気度0を除いた場合）。"""
    table = numeric_features(tracks)[["popularity", feature]].dropna()
    if table[feature].nunique() < 2:
        return pd.DataFrame(columns=["区間の中央", "全曲", "人気度0を除く", "曲数"])
    edges = pd.cut(table[feature], bins=bins, retbins=True)[1]
    cut = pd.cut(table[feature], bins=edges, include_lowest=True)
    all_songs = table.groupby(cut, observed=True).popularity.agg(["count", "mean"])
    no0 = table[table.popularity > 0]
    no0_mean = no0.groupby(pd.cut(no0[feature], bins=edges, include_lowest=True), observed=True).popularity.mean()
    all_songs = all_songs[all_songs["count"] >= min_count]
    return pd.DataFrame({
        "区間の中央": [round(interval.mid, 3) for interval in all_songs.index],
        "全曲": all_songs["mean"].round(1).to_numpy(),
        "人気度0を除く": no0_mean.reindex(all_songs.index).round(1).to_numpy(),
        "曲数": all_songs["count"].to_numpy(),
    })


def _metric(column, title, value, help_text):
    column.metric(title, "—" if np.isnan(value) else f"{value:.3f}", help=help_text)
    if not np.isnan(value):
        column.caption(f"説明できる割合 {value ** 2 * 100:.1f}%")


def render_popularity(catalog: Catalog):
    tracks, genre_rows = catalog.tracks, catalog.genre_rows
    st.title("人気度との関係")
    st.caption("曲の音の特徴と人気度（0〜100、最近よく再生されている曲ほど高い）に関係があるかを調べた結果です。"
               f"曲ごとのデータ {len(tracks):,} 曲を使っています。")
    if tracks["popularity"].notna().sum() < 3:
        st.warning("人気度が入っている曲が少ないため、関係を計算できません。人気度（popularity）の列があるCSVを使ってください。")
        return

    st.subheader("① まとめて見たときの関係の強さ")
    summary = strength_summary(tracks, genre_rows)
    c1, c2, c3 = st.columns(3)
    _metric(c1, "音の特徴4つ", summary["core"], "テンポ・力強さ・明るさ・ノリの良さで人気度を予測したときの重相関係数")
    _metric(c2, "音の特徴14項目すべて", summary["all"], "全14項目で人気度を予測したときの重相関係数")
    _metric(c3, "ジャンル", summary["genre"], "ジャンルごとの平均人気度の違いから求めた相関比（ジャンル集計用のデータを使用）")
    if not np.isnan(summary["all_no0"]):
        st.caption(f"人気度0の曲を除くと、14項目の重相関係数は {summary['all_no0']:.3f}（{summary['all_no0'] ** 2 * 100:.1f}%）。"
                   "0〜1で、1に近いほど関係が強い。（ ）内は人気度のばらつきを説明できる割合。")

    st.subheader("② 項目ごとの相関係数")
    corr = correlation_table(tracks)
    if corr.empty:
        st.info("相関係数を計算できる項目がありません。")
    else:
        chart_data = pd.DataFrame({"項目": [label(c) for c in corr.index], "相関係数": corr.round(3).to_numpy()})
        chart_data["向き"] = np.where(chart_data["相関係数"] >= 0, "プラス", "マイナス")
        order = chart_data["項目"].tolist()
        bars = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("相関係数:Q", title="相関係数（青：プラス　赤：マイナス）"),
            y=alt.Y("項目:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("向き:N", scale=alt.Scale(domain=["プラス", "マイナス"], range=[BLUE, RED]), legend=None),
            tooltip=["項目", alt.Tooltip("相関係数:Q", format="+.3f")],
        )
        st.altair_chart(bars.properties(height=34 * len(order)), width="stretch")
        st.caption(f"上ほど人気度との関係が強い項目。一番強い「{label(corr.index[0])}」でも相関係数は {corr.iloc[0]:+.3f} で、関係はとても弱い。")

    st.subheader("③ 区間ごとの平均人気度")
    choices = [f for f in BIN_FEATURES if f in tracks]
    left, right = st.columns([2, 1])
    feature = left.selectbox("項目を選ぶ", choices, format_func=label, key="pop_bin_feature")
    show_no0 = right.checkbox("人気度0の曲を除いた線も出す", value=True, key="pop_bin_no0")
    table = bin_table(tracks, feature)
    if table.empty:
        st.info("この項目は区間に分けられませんでした。")
    else:
        lines = ["全曲", "人気度0を除く"] if show_no0 else ["全曲"]
        st.line_chart(table.set_index("区間の中央")[lines], x_label=label(feature), y_label="平均人気度")
        st.caption(f"{label(feature)}を10の区間に分け、区間ごとに人気度の平均を出したもの。曲数が{MIN_COUNT}曲より少ない区間は出していない。")
        with st.expander("数値の表を見る"):
            st.dataframe(table, hide_index=True, width="stretch")

    st.info("音の特徴と人気度の関係は弱く、人気度はジャンルとの関係のほうが強い。"
            "そのため、このアプリでは人気度を使わず、音の特徴だけで似ている曲を探している。")
