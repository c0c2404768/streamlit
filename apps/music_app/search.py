"""曲名・アーティスト検索、ページ分け、共通の選択状態への受け渡し。"""
import math

import pandas as pd
import streamlit as st

from music_app.data import Catalog, normalize_text
from music_app.selection import clear_selection, select_track, selected_track

PAGE_SIZE = 25


def search_tracks(tracks: pd.DataFrame, query: str, order="一致順") -> pd.DataFrame:
    normalized = normalize_text(query)
    mask = pd.Series(True, index=tracks.index)
    for word in normalized.split():
        mask &= tracks._search.str.contains(word, regex=False, na=False)
    results = tracks.loc[mask].copy()
    if order == "曲名順":
        keys, ascending = ["_title", "_artist", "track_id"], [True, True, True]
    elif order == "人気度順" or not normalized:
        keys, ascending = ["popularity", "_title", "track_id"], [False, True, True]
    else:
        results["_rank"] = 2
        prefix = results._title.str.startswith(normalized) | results._artist.str.startswith(normalized)
        exact = results._title.eq(normalized) | results._artist.eq(normalized)
        results.loc[prefix, "_rank"] = 1
        results.loc[exact, "_rank"] = 0
        keys, ascending = ["_rank", "popularity", "_title", "track_id"], [True, False, True, True]
    return results.sort_values(keys, ascending=ascending, kind="stable", na_position="last")


def reset_page():
    st.session_state["_search_page"] = 1


def render_search(catalog: Catalog):
    st.title("曲の検索・選択")
    st.caption("好きな曲を選んで、ジャンルの特徴や似ている楽曲を探す基準にします。")
    query = st.text_input("曲名・アーティストを検索", placeholder="例：Comedy / 星野源", key="_search_query", on_change=reset_page)
    st.caption("大文字・小文字、全角・半角を区別しません。空白で区切ると、すべての語を含む曲を検索します。")
    order = st.selectbox("表示順", ["一致順", "人気度順", "曲名順"], key="_search_order", on_change=reset_page)
    results = search_tracks(catalog.tracks, query, order)
    st.write(f"検索結果 **{len(results):,}曲** ／ 全{len(catalog.tracks):,}曲")
    if results.empty:
        st.warning("該当する曲がありません。検索語を短くするか、別の曲名・アーティスト名を試してください。")
    else:
        pages = math.ceil(len(results) / PAGE_SIZE)
        if st.session_state.get("_search_page", 1) > pages:
            st.session_state["_search_page"] = 1
        page = st.number_input("ページ", min_value=1, max_value=pages, step=1, key="_search_page")
        start = (page - 1) * PAGE_SIZE
        visible = results.iloc[start:start + PAGE_SIZE].set_index("track_id", drop=False)
        st.caption(f"{start + 1:,}〜{min(start + PAGE_SIZE, len(results)):,}曲目 ／ {pages:,}ページ")
        table = visible[["track_name", "artists", "album_name", "genres", "popularity"]].rename(columns={
            "track_name": "曲名", "artists": "アーティスト", "album_name": "アルバム", "genres": "ジャンル", "popularity": "人気度",
        })
        st.dataframe(table, hide_index=True, width="stretch")
        option_labels = {i: f"{row.track_name} / {row.artists} — {row.album_name} [{i[-6:]}]"
                         for i, row in visible.iterrows()}
        chosen = st.selectbox(
            "このページから曲を選ぶ", visible.index.tolist(), index=None,
            placeholder="曲を選んでください", key="_search_pick",
            format_func=lambda i: option_labels.get(i, str(i)),
        )
        if st.button("この曲を基準にする", disabled=chosen is None, key="_search_confirm", type="primary"):
            select_track(catalog.tracks, chosen)
            st.success("基準にする曲を選択しました。ほかの画面でもこの曲を使えます。")
    song = selected_track(catalog.tracks)
    st.divider()
    st.subheader("現在選択している曲")
    if song is None:
        st.info("検索結果から曲を選び、「この曲を基準にする」を押してください。")
    else:
        st.write(f"**{song['track_name']}**")
        st.write(song["artists"])
        st.caption(f"アルバム：{song['album_name']}　／　ジャンル：{song['genres']}")
        if song.get("popularity") is not None:
            st.caption(f"人気度：{song['popularity']:g} / 100")
        if st.button("曲の選択を解除", key="_search_clear"):
            clear_selection()
            st.rerun()
        st.page_link("pages/ジャンル比較.py", label="この曲のジャンルを比較する", icon="📊")
