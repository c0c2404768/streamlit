"""アプリ全体の入口（トップページ）。使い方、選択中の曲、各ページへの移動、データの概要を表示する。"""
import pandas as pd
import streamlit as st

from music_app.data import Catalog
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


def render_selected(song: dict | None):
    with st.container(border=True):
        st.markdown("**選択中の曲**")
        if song is None:
            st.info("まだ曲が選ばれていません。まずは好きな曲を1曲選んでください。")
            st.page_link(SEARCH, label="曲を検索して選ぶ", icon="🔍")
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
            st.page_link(SEARCH, label="別の曲を選び直す", icon="🔍")


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
        st.page_link(SEARCH, label="曲の検索・選択", icon="🔍")
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

    st.subheader("使っているデータ")
    report = catalog.report
    c1, c2, c3 = st.columns(3)
    c1.metric("曲の数", f"{report.get('楽曲数', len(catalog.tracks)):,}")
    c2.metric("ジャンルの数", f"{report.get('ジャンル数', 0):,}")
    c3.metric("ジャンル集計用の行数", f"{report.get('ジャンル集計用の行数', len(catalog.genre_rows)):,}")
    st.caption("主な出典：[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)"
               "（Hugging Face、BSDライセンス）。左の「データの設定」から別のCSVも使えます。")
    st.caption("東京工科大学 社会情報プロジェクト実習II　G1（入江 諒・長島 清華・小林 琉永・及川 拓海）")
