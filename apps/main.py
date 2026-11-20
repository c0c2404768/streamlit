import streamlit as st

st.set_page_config(
    page_title="楽曲ムード分析",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 楽曲ムード分析")

from music_app.data import catalog_for_page
from music_app.home import render_home

render_home(catalog_for_page(), show_title=False)
