"""曲名・アーティスト検索、ページ分け、共通の選択状態への受け渡し。"""
import math

import pandas as pd
import streamlit as st

from music_app.data import ADDED_GENRE_SOURCES, Catalog, normalize_text
from music_app.genre_names import genre_label
from music_app.selection import clear_selection, select_track, selected_track

PAGE_SIZE = 25
ALL_GENRES = "__all__"
UNREGISTERED = "__unregistered__"


def tracks_in_genre(catalog: Catalog, genre: str | None) -> pd.DataFrame:
    if genre is None:
        return catalog.tracks
    if genre == UNREGISTERED:
        return catalog.tracks.loc[catalog.tracks.genres.eq("未登録")]
    ids = catalog.genre_rows.loc[catalog.genre_rows.track_genre.eq(genre), "track_id"]
    hit = catalog.tracks.track_id.isin(ids)
    if "genre_source" in catalog.tracks:  # アーティストからジャンルを補った追加曲も、そのジャンルで探せるようにする
        hit |= catalog.tracks.genre_source.isin(ADDED_GENRE_SOURCES) & catalog.tracks.genres.eq(genre)
    return catalog.tracks.loc[hit]


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
    genres = sorted(catalog.genre_rows.track_genre.unique())
    genre_options = [ALL_GENRES, *genres]
    if catalog.tracks.genres.eq("未登録").any():
        genre_options.append(UNREGISTERED)
    if "pending_search_genre" in st.session_state:
        requested = st.session_state.pop("pending_search_genre")
        st.session_state["_search_genre"] = requested if requested in genre_options else ALL_GENRES
        st.session_state["_search_query"] = ""
        st.session_state["_search_only_added"] = False
        reset_page()
    if "_search_genre" not in st.session_state:
        st.session_state["_search_genre"] = st.session_state.get("search_genre", ALL_GENRES)
    if st.session_state["_search_genre"] not in genre_options:
        st.session_state["_search_genre"] = ALL_GENRES
    genre = st.selectbox("ジャンルで絞る", genre_options, key="_search_genre", on_change=reset_page,
                         format_func=lambda value: "すべてのジャンル" if value == ALL_GENRES else "ジャンル未登録" if value == UNREGISTERED else genre_label(value))
    st.session_state["search_genre"] = genre
    candidates = tracks_in_genre(catalog, None if genre == ALL_GENRES else genre)
    if catalog.tracks.data_sources.str.contains("Tunebat", regex=False, na=False).any():
        only_added = st.checkbox("追加したTunebatの曲だけ表示", key="_search_only_added", on_change=reset_page)
        if only_added:
            candidates = candidates.loc[candidates.data_sources.str.contains("Tunebat", regex=False, na=False)]
            st.caption("追加曲のジャンルはアーティスト単位で付けています（元データの同じアーティスト → Wikidata → アーティスト単位の判断の順）。"
                       "ジャンル比較の集計には含めていません。")
    query = st.text_input("曲名・アーティストを検索", placeholder="例：元気を出して / 星野源 / Gen Hoshino", key="_search_query", on_change=reset_page)
    st.caption("大文字・小文字、全角・半角を区別しません。空白で区切ると、すべての語を含む曲を検索します。")
    order = st.selectbox("表示順", ["一致順", "人気度順", "曲名順"], key="_search_order", on_change=reset_page)
    results = search_tracks(candidates, query, order)
    st.write(f"検索結果 **{len(results):,}曲** ／ 全{len(catalog.tracks):,}曲")
    if results.empty:
        st.warning("該当する曲がありません。検索語を短くするか、ジャンルや追加曲の絞り込み条件を変えてください。")
    else:
        pages = math.ceil(len(results) / PAGE_SIZE)
        if st.session_state.get("_search_page", 1) > pages:
            st.session_state["_search_page"] = 1
        page = st.number_input("ページ", min_value=1, max_value=pages, step=1, key="_search_page")
        start = (page - 1) * PAGE_SIZE
        visible = results.iloc[start:start + PAGE_SIZE].set_index("track_id", drop=False)
        st.caption(f"{start + 1:,}〜{min(start + PAGE_SIZE, len(results)):,}曲目 ／ {pages:,}ページ")
        table = visible[["track_name", "artists", "album_name", "genres", "popularity", "data_sources"]].rename(columns={
            "track_name": "曲名", "artists": "アーティスト", "album_name": "アルバム", "genres": "ジャンル", "popularity": "人気度", "data_sources": "出典",
        })
        # 表の行をクリックすると、その曲をそのまま基準の曲にする
        # 検索結果やページが変わったら表を作り直し、前の選択が別の曲に当たらないようにする
        table_key = "_search_table_" + str(abs(hash(tuple(visible.index))))
        event = st.dataframe(table, hide_index=True, width="stretch", key=table_key,
                             on_select="rerun", selection_mode="single-row")
        rows = event.selection.rows
        picked = visible.index[rows[0]] if rows else None
        # クリックが変わったときだけ選び直す（下の選択欄で別の曲を選んだあとに、表の古い選択で上書きしないため）
        if picked is not None and st.session_state.get("_search_table_done") != (table_key, picked):
            st.session_state["_search_table_done"] = (table_key, picked)
            select_track(catalog.tracks, picked)
            st.toast("基準にする曲を選択しました。ほかの画面でもこの曲を使えます。")
        option_labels = {i: f"{row.track_name} / {row.artists} — {row.album_name} [{i[-6:]}]"
                         for i, row in visible.iterrows()}
        chosen = st.selectbox(
            "曲名から選ぶ（表の行をクリックしても選べます）", visible.index.tolist(), index=None,
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
        st.info("検索結果から曲を選び、表の行をクリックするか「この曲を基準にする」を押してください。")
    else:
        st.write(f"**{song['track_name']}**")
        st.write(song["artists"])
        st.caption(f"アルバム：{song['album_name']}　／　ジャンル：{song['genres']}")
        st.caption("出典：" + song["data_sources"])
        if song.get("popularity") is not None:
            st.caption(f"人気度：{song['popularity']:g} / 100")
        if st.button("曲の選択を解除", key="_search_clear"):
            clear_selection()
            st.rerun()
        st.page_link("pages/ジャンル比較.py", label="この曲とジャンルの特徴を比較する", icon="📊")
