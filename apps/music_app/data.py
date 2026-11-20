"""CSVの受け取り口。小林担当の前処理に差し替えられる共通形式。"""
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import unicodedata
from urllib.request import urlopen

import numpy as np
import pandas as pd
import streamlit as st

DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORE_FEATURES = ("tempo", "energy", "valence", "danceability")
FEATURES = (*CORE_FEATURES, "acousticness", "instrumentalness", "speechiness", "duration_min")
LABELS = {
    "tempo": "テンポ", "energy": "力強さ", "valence": "明るさ",
    "danceability": "ノリの良さ", "acousticness": "アコースティックらしさ",
    "instrumentalness": "インストらしさ", "speechiness": "話し声の多さ",
    "duration_min": "曲の長さ",
}
UNITS = {feature: "0〜1" for feature in FEATURES}
UNITS.update(tempo="BPM", duration_min="分")
META = ("track_id", "track_name", "artists", "track_genre")


@dataclass
class Catalog:
    genre_rows: pd.DataFrame
    tracks: pd.DataFrame
    features: tuple[str, ...]
    report: dict[str, int]


def normalize_text(text) -> str:
    """全半角・英字の大文字小文字・連続空白をそろえる。"""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(content), encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding="cp932", low_memory=False)


def prepare_catalog(raw: pd.DataFrame) -> Catalog:
    """genre_rowsは曲×ジャンル、tracksは曲IDごとに1行。元データは変更しない。"""
    missing = sorted(set(META) - set(raw.columns))
    if missing:
        raise ValueError("CSVに必要な列がありません: " + ", ".join(missing))
    df = raw.copy()
    for column in META:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    for column in ("popularity", "duration_ms", *FEATURES):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "duration_ms" in df:
        df["duration_min"] = df["duration_ms"] / 60000
    for feature in FEATURES:
        if feature in df:
            valid = df[feature].gt(0) if feature in ("tempo", "duration_min") else df[feature].between(0, 1)
            df.loc[~valid, feature] = np.nan
    if "popularity" not in df:
        df["popularity"] = np.nan
    df.loc[~df.popularity.between(0, 100), "popularity"] = np.nan
    if "album_name" not in df:
        df["album_name"] = ""
    else:
        df["album_name"] = df.album_name.fillna("").astype(str)
    # メタデータと収録済みの推薦用4特徴を確認。追加特徴の欠損は項目別に扱う。
    required = [*META, *[f for f in CORE_FEATURES if f in df]]
    valid = df.dropna(subset=required)
    genre_rows = valid.drop_duplicates(["track_id", "track_genre"]).reset_index(drop=True)
    tracks = genre_rows.drop_duplicates("track_id").reset_index(drop=True).copy()
    genres = genre_rows.groupby("track_id", sort=False).track_genre.agg(lambda values: " / ".join(sorted(set(values))))
    tracks["genres"] = tracks.track_id.map(genres)
    tracks["_title"] = tracks.track_name.map(normalize_text)
    tracks["_artist"] = tracks.artists.map(normalize_text)
    tracks["_search"] = tracks._title + " " + tracks._artist
    available = tuple(f for f in FEATURES if f in genre_rows and genre_rows[f].notna().any())
    report = {
        "入力行数": len(raw), "欠損・不正値による除外": len(df) - len(valid),
        "同一曲・同一ジャンルの重複除外": len(valid) - len(genre_rows),
        "ジャンル集計用の行数": len(genre_rows), "楽曲数": len(tracks),
        "ジャンル数": genre_rows.track_genre.nunique(),
    }
    return Catalog(genre_rows, tracks, available, report)


def dataset_source() -> tuple[str, int | None]:
    """実行時の作業フォルダではなく、このファイルの場所を基準に探す。"""
    configured = os.environ.get("SPOTIFY_DATASET_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return str(path), path.stat().st_mtime_ns
    for path in (REPOSITORY_ROOT / "data/dataset.csv", REPOSITORY_ROOT / "apps/data/dataset.csv"):
        if path.is_file():
            return str(path), path.stat().st_mtime_ns
    return DATASET_URL, None


@st.cache_data(show_spinner=False)
def load_catalog(source: str, modified: int | None = None) -> Catalog:
    if source == DATASET_URL:
        with urlopen(source, timeout=45) as response:
            content = response.read()
    else:
        content = Path(source).read_bytes()
    return prepare_catalog(read_csv_bytes(content))


@st.cache_data(show_spinner=False)
def load_uploaded_catalog(content: bytes) -> Catalog:
    return prepare_catalog(read_csv_bytes(content))


def catalog_for_page() -> Catalog:
    """各ページで同じデータを取得。公開CSVか手元のCSVを使える。"""
    with st.sidebar.expander("データの設定"):
        upload_version = st.session_state.get("catalog_upload_version", 0)
        uploaded = st.file_uploader("別の楽曲CSVを使う", type=["csv"], key=f"_catalog_upload_{upload_version}")
        if uploaded is not None:
            st.session_state["catalog_csv_bytes"] = uploaded.getvalue()
        if st.session_state.get("catalog_csv_bytes") is not None:
            st.caption("アップロードしたCSVを使用中")
            if st.button("標準データに戻す", key="_catalog_reset"):
                st.session_state.pop("catalog_csv_bytes", None)
                st.session_state["catalog_upload_version"] = upload_version + 1
                st.rerun()
        else:
            st.caption("標準データ：Spotify Tracks Dataset")
    try:
        with st.spinner("楽曲データを読み込んでいます…"):
            if st.session_state.get("catalog_csv_bytes") is not None:
                catalog = load_uploaded_catalog(st.session_state["catalog_csv_bytes"])
            else:
                catalog = load_catalog(*dataset_source())
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        st.error("楽曲データを読み込めませんでした。ネットワーク接続を確認するか、左の「データの設定」からCSVを選んでください。")
        with st.expander("エラーの詳細"):
            st.code(str(exc), language=None)
        st.stop()
    if catalog.tracks.empty:
        st.warning("利用できる楽曲が0件です。CSVの内容を確認してください。")
        st.stop()
    return catalog
