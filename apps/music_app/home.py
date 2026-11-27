"""アプリ全体の入口（トップページ）。使い方、選択中の曲、各ページへの移動、データの概要を表示する。"""
import numpy as np
import pandas as pd
import streamlit as st

from music_app.data import Catalog
from music_app.popularity import strength_summary
from music_app.selection import selected_track

SEARCH = "pages/曲の検索・選択.py"
SIMILAR = "pages/似ている楽曲.py"
ANALYSES = (
    ("pages/ジャンル比較.py", "ジャンル比較", "📊", "ジャンルごとに、テンポや明るさなどの特徴の分布を比べる"),
    ("pages/グループ分け.py", "グループ分け", "🧩", "音の特徴が近い曲どうしを、いくつかのグループに分ける"),
    ("pages/人気度との関係.py", "人気度との関係", "📈", "音の特徴やジャンルと人気度の関係の強さを調べる"),
)


def _text(value, empty="—") -> str:
    return empty if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA else str(value)


def search_button(label: str, key: str, primary: bool = True):
    """曲の検索画面へ移動するボタン。文字だけのリンクより目立たせる。"""
    if st.button(label, key=key, icon=":material/search:", type="primary" if primary else "secondary", width="stretch"):
        st.switch_page(SEARCH)


def render_selected(song: dict | None):
    with st.container(border=True):
        st.markdown("**選択中の曲**")
        if song is None:
            st.info("まだ曲が選ばれていません。まずは下のボタンから、好きな曲を1曲選んでください。")
            search_button("曲を検索して選ぶ", key="_home_search")
            st.caption("曲名・アーティスト名（日本語・英語どちらでも）で探せます。")
            return
        st.markdown(f"### {_text(song.get('track_name'))}")
        genres = _text(song.get("genres") or song.get("track_genre"), "未登録")
        popularity = song.get("popularity")
        pop = "—" if popularity is None or pd.isna(popularity) else f"{int(popularity)}"
        st.caption(f"{_text(song.get('artists'))}　／　ジャンル：{genres}　／　人気度：{pop}")
        left, right = st.columns(2)
        with left:
            st.page_link(SIMILAR, label="この曲に似ている曲を見る", icon="🎧")
        with right:
            search_button("別の曲を選び直す", key="_home_research", primary=False)


def render_basis(catalog: Catalog):
    """人気度との関係ページの結果を短くまとめ、「人気度を使わずに探す」根拠として示す。"""
    st.subheader("このアプリの根拠：なぜ人気度を使わないのか")
    tracks = catalog.tracks
    if "popularity" not in tracks or tracks["popularity"].notna().sum() < 3:
        st.caption("このデータには人気度がないため、根拠の数値は表示できません。")
        return
    summary = strength_summary(tracks, catalog.genre_rows)
    with st.container(border=True):
        st.write("音の特徴やジャンルから、曲の人気度をどれくらい説明できるかを調べました。"
                 "数値は0〜1で、1に近いほど関係が強いことを表します。")
        items = (("音の特徴4つ", "core", "テンポ・力強さ・明るさ・ノリの良さから人気度を予測したときの重相関係数"),
                 ("音の特徴14項目", "all", "Spotifyの14項目すべてから人気度を予測したときの重相関係数"),
                 ("ジャンル", "genre", "ジャンルごとの平均人気度の違いから求めた相関比"))
        for column, (title, key, help_text) in zip(st.columns(3), items):
            value = summary[key]
            column.metric(title, "—" if np.isnan(value) else f"{value:.2f}", help=help_text)
            if not np.isnan(value):
                column.caption(f"人気度のばらつきの {value ** 2 * 100:.0f}% を説明")
        st.markdown("**音の特徴と人気度の関係はとても弱い**ことが分かりました。"
                    "人気度で絞ると、音は好みに近いのに人気が低いだけの曲を見落としてしまいます。"
                    "そのため、このアプリは**音の特徴だけ**で似ている曲を探し、人気度は好みで絞り込む条件にとどめています。")
        st.page_link("pages/人気度との関係.py", label="詳しい分析を見る（人気度との関係）", icon="📈")


def render_home(catalog: Catalog, show_title: bool = True):
    """show_title=False のときは、ページのタイトルを main.py 側に任せて副題から表示する。"""
    if show_title:
        st.title("似ている曲から、まだ知らない好きな曲へ")
    else:
        st.subheader("似ている曲から、まだ知らない好きな曲へ")
    st.write("好きな曲を1曲選ぶと、テンポ・力強さ・明るさ・ノリの良さといった**音の特徴**が近い曲を探します。"
             "人気度やジャンルに関係なく探せるので、ランキングには出てこない曲とも出会えます。")

    render_selected(selected_track(catalog.tracks))

    st.subheader("使い方")
    steps = st.columns(3)
    with steps[0]:
        st.markdown("**1. 曲を選ぶ**")
        st.caption("曲名やアーティスト名で検索して、基準にする曲を1曲決めます。")
        search_button("曲の検索・選択へ", key="_home_step_search", primary=False)
    with steps[1]:
        st.markdown("**2. 似ている曲を見る**")
        st.caption("音の特徴が近い順にランキングで表示します。人気度やジャンルで絞り込めます。")
        st.page_link(SIMILAR, label="似ている楽曲", icon="🎧")
    with steps[2]:
        st.markdown("**3. 分析を見る**")
        st.caption("ジャンルや人気度と、音の特徴の関係をグラフで確かめます。")
        st.page_link(ANALYSES[0][0], label=ANALYSES[0][1], icon=ANALYSES[0][2])

    st.subheader("分析のページ")
    for path, label, icon, description in ANALYSES:
        st.page_link(path, label=f"{label}：{description}", icon=icon)

    render_basis(catalog)

    st.subheader("使っているデータ")
    report = catalog.report
    c1, c2, c3 = st.columns(3)
    c1.metric("曲の数", f"{report.get('楽曲数', len(catalog.tracks)):,}")
    c2.metric("ジャンルの数", f"{report.get('ジャンル数', 0):,}")
    c3.metric("ジャンル集計用の行数", f"{report.get('ジャンル集計用の行数', len(catalog.genre_rows)):,}")
    st.caption("主な出典：[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)"
               "（Hugging Face、BSDライセンス）。左の「データの設定」から別のCSVも使えます。")
    st.caption("東京工科大学 社会情報プロジェクト実習II　G1（入江 諒・長島 清華・小林 琉永・及川 拓海）")
