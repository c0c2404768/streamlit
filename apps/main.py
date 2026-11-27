import streamlit as st

st.set_page_config(
    page_title="楽曲ムード分析",
    page_icon="🎵",
    layout="wide",
)

from music_app.data import catalog_for_page
from music_app.home import render_home


def home():
    st.title("🎵 楽曲ムード分析")
    render_home(catalog_for_page(), show_title=False)


# 左のメニューの表示名と並び順（使う順に並べる）
page = st.navigation([
    st.Page(home, title="ホーム", icon="🏠", default=True),
    st.Page("pages/曲の検索・選択.py", title="曲の検索・選択", icon="🔍"),
    st.Page("pages/似ている楽曲.py", title="似ている楽曲", icon="🎧"),
    st.Page("pages/似ている曲の人気度.py", title="似ている曲の人気度", icon="⚖️"),
    st.Page("pages/ジャンル比較.py", title="ジャンル比較", icon="📊"),
    st.Page("pages/グループ分け.py", title="グループ分け", icon="🧩"),
    # 人気度との関係はホームの「このアプリの根拠」から開く詳細ページなので、左のメニューには出さない
    st.Page("pages/人気度との関係.py", title="人気度との関係", icon="📈", visibility="hidden"),
])
page.run()
