"""CSVの受け取り口。小林担当の前処理に差し替えられる共通形式。"""
from dataclasses import dataclass
from io import BytesIO
import os
import traceback
from pathlib import Path
import unicodedata
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st

DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
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
META = ("track_id", "track_name", "artists")
TUNEBAT_CSV = Path(__file__).with_name("assets") / "tunebat_tracks.csv"
TUNEBAT_PARQUET = TUNEBAT_CSV.with_suffix(".parquet")
ARTIST_ALIASES = {"gen hoshino": ("星野源", "ほしのげん"), "mariya takeuchi": ("竹内まりや", "たけうちまりや")}


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


def read_table_bytes(content: bytes, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.casefold()
    if suffix == ".parquet":
        return pd.read_parquet(BytesIO(content))
    if suffix == ".csv":
        return read_csv_bytes(content)
    raise ValueError("CSVまたはParquetファイルを指定してください。")


def index_tracks(tracks: pd.DataFrame, genre_rows: pd.DataFrame) -> pd.DataFrame:
    tracks = tracks.copy().reset_index(drop=True)
    genres = genre_rows.groupby("track_id", sort=False).track_genre.agg(lambda values: " / ".join(sorted(set(values))))
    # 環境によっては map の結果が数値型（pyarrow の double）になり、fillna で文字を入れると
    # 「Could not convert '未登録' ... to double」になるため、文字の列として作り直す
    mapped = tracks.track_id.map(genres)
    tracks["genres"] = pd.Series(np.where(mapped.isna(), "未登録", mapped.astype(object)),
                                 index=tracks.index, dtype=object)
    tracks["_title"] = tracks.track_name.map(normalize_text)
    tracks["_artist"] = tracks.artists.map(normalize_text)
    if "_aliases" not in tracks:
        tracks["_aliases"] = ""
    aliases = tracks._aliases.fillna("").map(normalize_text)
    japanese = tracks._artist.map(lambda value: " ".join(alias for artist in value.split(";") for alias in ARTIST_ALIASES.get(artist.strip(), ())))
    tracks["_search"] = tracks._title + " " + tracks._artist + " " + aliases + " " + japanese
    return tracks


def prepare_catalog(raw: pd.DataFrame, source_label="CSV") -> Catalog:
    """genre_rowsは曲×ジャンル、tracksは曲IDごとに1行。元データは変更しない。"""
    missing = sorted(set(META) - set(raw.columns))
    if missing:
        raise ValueError("楽曲データに必要な列がありません: " + ", ".join(missing))
    df = raw.copy()
    if "track_genre" not in df:
        df["track_genre"] = pd.NA
    for column in (*META, "track_genre"):
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    if "data_sources" not in df:
        df["data_sources"] = source_label
    else:
        df["data_sources"] = df.data_sources.fillna(source_label)
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
    known_genres = valid.dropna(subset=["track_genre"])
    genre_rows = known_genres.drop_duplicates(["track_id", "track_genre"]).reset_index(drop=True)
    tracks = index_tracks(valid.drop_duplicates("track_id"), genre_rows)
    available = tuple(f for f in FEATURES if f in genre_rows and genre_rows[f].notna().any())
    report = {
        "入力行数": len(raw), "欠損・不正値による除外": len(df) - len(valid),
        "同一曲・同一ジャンルの重複除外": len(known_genres) - len(genre_rows),
        "ジャンル集計用の行数": len(genre_rows), "楽曲数": len(tracks),
        "ジャンル数": genre_rows.track_genre.nunique(),
    }
    return Catalog(genre_rows, tracks, available, report)


def add_tunebat_tracks(catalog: Catalog, additional: pd.DataFrame) -> Catalog:
    """同じSpotify IDは統合。既存曲の音響値と既知ジャンルは変更しない。"""
    extra = prepare_catalog(additional, source_label="Tunebat").tracks
    tracks = catalog.tracks.set_index("track_id", drop=False).copy()
    overlap = set(tracks.index) & set(extra.track_id)
    for _, row in extra.loc[extra.track_id.isin(overlap)].iterrows():
        track_id = row.track_id
        previous = tracks.loc[track_id]
        tracks.at[track_id, "_aliases"] = " ".join(str(value) for value in (
            previous.get("_aliases", ""), previous.track_name, previous.artists, row.track_name, row.artists,
        ) if pd.notna(value))
        tracks.at[track_id, "track_name"] = row.track_name
        tracks.at[track_id, "artists"] = row.artists
        sources = str(previous.data_sources).split(" / ") + ["Tunebat"]
        tracks.at[track_id, "data_sources"] = " / ".join(dict.fromkeys(sources))
    new_tracks = extra.loc[~extra.track_id.isin(overlap)]
    tracks = pd.concat([tracks.reset_index(drop=True), new_tracks], ignore_index=True)
    tracks = index_tracks(tracks, catalog.genre_rows)
    report = dict(catalog.report)
    report.update({
        "元データの楽曲数": len(catalog.tracks), "追加ファイルの曲数": len(extra),
        "既存の楽曲IDと一致": len(overlap), "新規追加曲数": len(new_tracks),
        "楽曲数": len(tracks), "ジャンル未登録の曲数": int(tracks.genres.eq("未登録").sum()),
    })
    return Catalog(catalog.genre_rows, tracks, catalog.features, report)


def dataset_source() -> tuple[str, int | None]:
    """実行時の作業フォルダではなく、このファイルの場所を基準に探す。"""
    configured = os.environ.get("SPOTIFY_DATASET_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        return str(path), path.stat().st_mtime_ns
    for path in (
        REPOSITORY_ROOT / "data/dataset.parquet",
        REPOSITORY_ROOT / "apps/data/dataset.parquet",
        REPOSITORY_ROOT / "data/dataset.csv",
        REPOSITORY_ROOT / "apps/data/dataset.csv",
    ):
        if path.is_file():
            return str(path), path.stat().st_mtime_ns
    return DATASET_URL, None


@st.cache_data(show_spinner=False)
def load_catalog(source: str, modified: int | None = None, supplement_stamp: int | None = None,
                 source_label: str = "Spotify Tracks Dataset") -> Catalog:
    if source == DATASET_URL:
        with urlopen(source, timeout=45) as response:
            content = response.read()
        raw = read_table_bytes(content, "dataset.parquet")
    else:
        source_path = Path(source)
        raw = read_table_bytes(source_path.read_bytes(), source_path.name)
    catalog = prepare_catalog(raw, source_label=source_label)
    if supplement_stamp is not None:
        supplement_path = TUNEBAT_PARQUET if TUNEBAT_PARQUET.is_file() else TUNEBAT_CSV
        supplement = read_table_bytes(supplement_path.read_bytes(), supplement_path.name)
        catalog = add_tunebat_tracks(catalog, supplement)
    return catalog


@st.cache_data(show_spinner=False)
def load_uploaded_catalog(content: bytes, file_name: str = "upload.csv") -> Catalog:
    return prepare_catalog(read_table_bytes(content, file_name))


def catalog_for_page() -> Catalog:
    """各ページで同じデータを取得。公開データか手元のCSV・Parquetを使える。"""
    if "catalog_upload_bytes" not in st.session_state and "catalog_csv_bytes" in st.session_state:
        st.session_state["catalog_upload_bytes"] = st.session_state.pop("catalog_csv_bytes")
        st.session_state["catalog_upload_name"] = "upload.csv"
    with st.sidebar.expander("データの設定"):
        upload_version = st.session_state.get("catalog_upload_version", 0)
        uploaded = st.file_uploader("別の楽曲ファイルを使う", type=["csv", "parquet"], key=f"_catalog_upload_{upload_version}")
        if uploaded is not None:
            st.session_state["catalog_upload_bytes"] = uploaded.getvalue()
            st.session_state["catalog_upload_name"] = uploaded.name
        if st.session_state.get("catalog_upload_bytes") is not None:
            st.caption("アップロードしたデータを使用中")
            if st.button("標準データに戻す", key="_catalog_reset"):
                st.session_state.pop("catalog_upload_bytes", None)
                st.session_state.pop("catalog_upload_name", None)
                st.session_state["catalog_upload_version"] = upload_version + 1
                st.rerun()
        elif os.environ.get("SPOTIFY_DATASET_PATH"):
            st.caption("環境変数で指定したデータを使用中")
        else:
            st.caption("標準データ：Spotify Tracks Dataset ＋ Tunebat追加曲")
    try:
        with st.spinner("楽曲データを読み込んでいます…"):
            if st.session_state.get("catalog_upload_bytes") is not None:
                catalog = load_uploaded_catalog(
                    st.session_state["catalog_upload_bytes"],
                    st.session_state.get("catalog_upload_name", "upload.csv"),
                )
            else:
                supplement_path = TUNEBAT_PARQUET if TUNEBAT_PARQUET.is_file() else TUNEBAT_CSV
                stamp = supplement_path.stat().st_mtime_ns if supplement_path.is_file() and not os.environ.get("SPOTIFY_DATASET_PATH") else None
                source_label = "指定したデータ" if os.environ.get("SPOTIFY_DATASET_PATH") else "Spotify Tracks Dataset"
                catalog = load_catalog(*dataset_source(), supplement_stamp=stamp, source_label=source_label)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError, pa.ArrowException) as exc:
        st.error("楽曲データを読み込めませんでした。ネットワーク接続を確認するか、左の「データの設定」からCSVまたはParquetを選んでください。")
        with st.expander("エラーの詳細"):
            st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", language=None)
        st.stop()
    if catalog.tracks.empty:
        st.warning("利用できる楽曲が0件です。データの内容を確認してください。")
        st.stop()
    return catalog
