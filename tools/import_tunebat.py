"""元Excelを変更せず、追加データのCSVと出典メモを更新する。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))
from music_app.tunebat import read_tunebat_xlsx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    args = parser.parse_args()
    content = args.workbook.read_bytes()
    tracks, notes = read_tunebat_xlsx(content)
    directory = ROOT / "apps/music_app/assets"
    directory.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(directory / "tunebat_tracks.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_file": args.workbook.name, "source_sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(tracks), "notes": notes,
        "units": "入力済みの0〜1をそのまま保持。再度100で割らない。Popularityは0〜100、tempoはBPM、duration_msはミリ秒。",
        "missing": "ジャンルや追加特徴の空欄を保持。補完・推測はしない。特徴ごとの収録件数を確認する。",
        "feature_counts": {key: int(tracks[key].notna().sum()) for key in ("tempo", "energy", "danceability", "valence", "duration_ms", "acousticness", "instrumentalness", "speechiness") if key in tracks},
    }
    (directory / "tunebat_source.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(tracks):,}曲を取り込みました。元のExcelは変更していません。")


if __name__ == "__main__":
    main()
