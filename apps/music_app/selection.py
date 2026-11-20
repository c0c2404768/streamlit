"""ページを切り替えても残る、担当者共通の選択曲。"""
import pandas as pd
import streamlit as st


def select_track(tracks: pd.DataFrame, track_id: str, state=None) -> dict:
    state = st.session_state if state is None else state
    rows = tracks.loc[tracks.track_id.eq(track_id)]
    if rows.empty:
        raise ValueError("指定された楽曲IDがデータにありません。")
    row = rows.iloc[0]
    snapshot = {key: (None if pd.isna(value) else value)
                for key, value in row.items() if not key.startswith("_")}
    state["selected_track_id"] = str(track_id)
    state["selected_track"] = snapshot
    return snapshot


def selected_track(tracks: pd.DataFrame, state=None) -> dict | None:
    state = st.session_state if state is None else state
    track_id = state.get("selected_track_id")
    if not track_id:
        return None
    if not tracks.track_id.eq(track_id).any():
        clear_selection(state)
        return None
    return select_track(tracks, track_id, state)


def clear_selection(state=None):
    state = st.session_state if state is None else state
    state.pop("selected_track_id", None)
    state.pop("selected_track", None)
