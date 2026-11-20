"""既存データセットと同じ列名のTunebat追加Excelを読み取る。"""
from io import BytesIO
import posixpath
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REQUIRED = ("track_id", "artists", "track_name", "tempo", "popularity", "energy", "danceability", "valence")
UNIT_FEATURES = ("energy", "danceability", "valence", "speechiness", "acousticness", "instrumentalness", "liveness")


def read_tunebat_xlsx(content: bytes) -> tuple[pd.DataFrame, dict]:
    """0〜1に変換済みの数値を保持する。ジャンル・欠損値は推測しない。"""
    with ZipFile(BytesIO(content)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]

        def text(cell):
            value = cell.find("s:v", NS)
            inline = cell.find("s:is", NS)
            value = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
            return strings[int(value)] if cell.get("t") == "s" and value else value or ""

        book = ET.fromstring(archive.read("xl/workbook.xml"))
        book_rels = {node.get("Id"): node.get("Target") for node in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = {}
        for sheet in book.findall("s:sheets/s:sheet", NS):
            target = book_rels[sheet.get(f"{{{NS['r']}}}id")]
            sheets[sheet.get("name")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        if "tracks" not in sheets:
            raise ValueError("Excelに「tracks」シートがありません。データセットと同じ列名の形式を指定してください。")
        rows = ET.fromstring(archive.read(sheets["tracks"])).findall("s:sheetData/s:row", NS)
        if not rows:
            raise ValueError("tracksシートにデータがありません。")
        headers = {re.sub(r"\d+", "", cell.get("r")): text(cell) for cell in rows[0].findall("s:c", NS)}
        missing = sorted(set(REQUIRED) - set(headers.values()))
        if missing:
            raise ValueError("追加Excelに必要な列がありません: " + ", ".join(missing))
        records = []
        for row in rows[1:]:
            values = {headers.get(re.sub(r"\d+", "", cell.get("r"))): text(cell) for cell in row.findall("s:c", NS)}
            if any(values.values()):
                records.append(values)
        notes = {}
        if "メモ" in sheets:
            for row in ET.fromstring(archive.read(sheets["メモ"])).findall("s:sheetData/s:row", NS):
                values = [text(cell) for cell in row.findall("s:c", NS)]
                if len(values) >= 2:
                    notes[values[0]] = values[1]
    frame = pd.DataFrame(records).replace("", pd.NA)
    if frame.empty:
        raise ValueError("Excelに取り込める楽曲がありません。")
    for column in ("track_id", "artists", "track_name"):
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"追加Excelの「{column}」に空欄があります。")
    if not frame.track_id.str.fullmatch(r"[A-Za-z0-9]{22}").all():
        raise ValueError("追加Excelに無効なSpotify楽曲IDがあります。")
    if frame.track_id.duplicated().any():
        raise ValueError("追加Excel内に同じSpotify楽曲IDが重複しています。")
    for column in ("tempo", "popularity", "duration_ms", "key", "loudness", "mode", "time_signature", *UNIT_FEATURES):
        if column not in frame:
            continue
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError):
            raise ValueError(f"追加Excelの「{column}」に数値以外が入っています。") from None
        values = frame[column].dropna()
        if column in UNIT_FEATURES and not values.between(0, 1).all():
            raise ValueError(f"追加Excelの「{column}」は変換済みの0〜1で指定してください。")
        if column == "popularity" and not values.between(0, 100).all():
            raise ValueError("追加Excelの人気度は0〜100で指定してください。")
        if column in ("tempo", "duration_ms") and not (values.gt(0) & values.lt(float("inf"))).all():
            raise ValueError(f"追加Excelの「{column}」は0より大きい有限の値が必要です。")
    if frame[["tempo", "energy", "danceability", "valence"]].isna().any().any():
        raise ValueError("追加Excelの基本4特徴に欠損があります。")
    frame["data_sources"] = "Tunebat"
    frame["retrieved_on"] = notes.get("取得日", "")
    frame["spotify_url"] = "https://open.spotify.com/track/" + frame.track_id
    return frame, notes
