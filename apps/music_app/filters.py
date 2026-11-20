"""小林担当：似ている曲の絞り込み条件（人気度の上限・同じジャンルを除く・似ている度合いの測り方）。

推薦担当のページでの使い方：
    from music_app.filters import apply_filters, describe_filters, render_filters, show_no_result
    cond = render_filters()                                  # サイドバーに条件を表示
    candidates = apply_filters(catalog.tracks, song["track_id"], cond)
    if candidates.empty:
        show_no_result(); st.stop()
    # cond["method"] が "cos" ならコサイン類似度、"euc" ならユークリッド距離
"""
import pandas as pd
import streamlit as st

from music_app.data import normalize_text

METHODS = {"コサイン類似度": "cos", "ユークリッド距離": "euc"}
DEFAULTS = {"filter_pop_max": 100, "filter_other_genre": False, "filter_method": "コサイン類似度"}


def init_filters(state=None):
    state = st.session_state if state is None else state
    for key, value in DEFAULTS.items():
        state.setdefault(key, value)


def _save(key):
    # 部品のキー（_filter_〜）は、その部品がないページに移ると消える。共通キーに写して残す
    st.session_state[key] = st.session_state["_" + key]


def render_filters(where=None) -> dict:
    """条件の部品を表示して、今の条件を返す。where を省略するとサイドバーに出す。"""
    init_filters()
    area = st.sidebar if where is None else where
    with area:
        st.subheader("絞り込み条件")
        st.slider("人気度の上限", 0, 100, step=5, value=st.session_state["filter_pop_max"],
                  key="_filter_pop_max", on_change=_save, args=("filter_pop_max",),
                  help="下げるほど、あまり知られていない曲だけが出ます（人気度は0〜100）")
        st.checkbox("同じジャンルの曲は出さない", value=st.session_state["filter_other_genre"],
                    key="_filter_other_genre", on_change=_save, args=("filter_other_genre",))
        names = list(METHODS)
        st.radio("似ている度合いの測り方", names, index=names.index(st.session_state["filter_method"]),
                 key="_filter_method", on_change=_save, args=("filter_method",),
                 help="コサイン類似度：特徴のバランスが近い曲（1に近いほど似ている）\n\n"
                      "ユークリッド距離：特徴の値そのものが近い曲（0に近いほど似ている）")
    return get_filters()


def get_filters(state=None) -> dict:
    """今の条件。部品を表示していないページからも読める。"""
    state = st.session_state if state is None else state
    init_filters(state)
    return {"pop_max": int(state["filter_pop_max"]), "other_genre": bool(state["filter_other_genre"]),
            "method": METHODS[state["filter_method"]]}


def _genre_set(row) -> set[str]:
    text = row.get("genres") if isinstance(row.get("genres"), str) else row.get("track_genre")
    return {g.strip() for g in str(text).split("/") if g.strip()}


def apply_filters(tracks: pd.DataFrame, track_id, cond: dict | None = None) -> pd.DataFrame:
    """選んだ曲と比べる候補を、条件で絞り込んで返す（index は tracks のまま）。
    ・選んだ曲そのものと、曲名・アーティストが同じ別バージョンを除く
    ・人気度が上限より大きい曲と、人気度が空の曲を除く
    ・「同じジャンルの曲は出さない」なら、選んだ曲の所属ジャンルのどれかに入る曲を除く"""
    cond = get_filters() if cond is None else cond
    rows = tracks.loc[tracks.track_id.eq(track_id)] if track_id else tracks.iloc[0:0]
    if rows.empty:
        return tracks.iloc[0:0]
    song = rows.iloc[0]
    title = tracks["_title"] if "_title" in tracks else tracks.track_name.map(normalize_text)
    artist = tracks["_artist"] if "_artist" in tracks else tracks.artists.map(normalize_text)
    keep = tracks.track_id.ne(track_id)
    keep &= ~(title.eq(normalize_text(song.track_name)) & artist.eq(normalize_text(song.artists)))
    keep &= tracks.popularity.le(cond["pop_max"])
    if cond["other_genre"]:
        genres = _genre_set(song)
        column = tracks["genres"] if "genres" in tracks else tracks["track_genre"]
        each = column.astype("string").str.split("/").explode().str.strip()
        same = each.isin(genres).groupby(level=0).any().reindex(tracks.index, fill_value=False)
        keep &= ~same
    return tracks[keep]


def describe_filters(cond: dict | None = None) -> str:
    cond = get_filters() if cond is None else cond
    parts = []
    if cond["pop_max"] < 100:
        parts.append(f"人気度{cond['pop_max']}以下")
    if cond["other_genre"]:
        parts.append("ほかのジャンルだけ")
    return "条件：" + "・".join(parts) if parts else "条件なし"


def show_no_result():
    st.warning("条件に合う曲がありません。人気度の上限を上げるか、「同じジャンルの曲は出さない」を外してください。")
