import streamlit as st

from music_app.data import catalog_for_page
from music_app.genre import render_genre_comparison

st.sidebar.caption("設定せずに始められます。画面の「比べたいジャンル」から選んでください。")
render_genre_comparison(catalog_for_page())
