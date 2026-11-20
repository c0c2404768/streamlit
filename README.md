# 音楽分析アプリ

長嶋担当の `apps/main.py` と既存の `apps/pages/` 構成に、入江担当の「曲の検索・選択」「ジャンル比較」を接続しています。メイン、推薦、グループ分け、人気度の各画面の中身は担当者側で実装します。

## 起動

Python 3.10以降で、リポジトリのフォルダから実行します。

```powershell
python -m pip install -r requirements.txt
python -m streamlit run apps/main.py
```

左のページ一覧から「曲の検索・選択」「ジャンル比較」を開いてください。入江担当の画面は `apps/pages/` に置いているので、メインのページ構成を作り直さず表示されます。

## データ

初回は [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) の公開CSVを取得します。取得済みのデータはStreamlitでキャッシュします。GoogleログインやAPIキーは不要です。配布元のライセンス表記はBSDです。データの権利・ライセンスは配布元の案内に従ってください。

手元のCSVを使う場合は、次のいずれかを選べます。

- リポジトリ直下の `data/dataset.csv` に置く（Git管理からは除外）。
- 左の「データの設定」からCSVを選ぶ。アップロードしたCSVはページを移動しても利用できます。「標準データに戻す」で解除します。
- 環境変数 `SPOTIFY_DATASET_PATH` にCSVの絶対パスを設定する。

読込エラー時は案内を表示します。指定したローカルCSVが見つからない場合に、別のデータへ自動で切り替えることはありません。

必要な列は `track_id`, `track_name`, `artists`, `track_genre` です。推薦用の4特徴（`tempo`, `energy`, `valence`, `danceability`）がある場合は欠損・不正値を除外します。ジャンル比較の追加4項目は `acousticness`, `instrumentalness`, `speechiness`, `duration_ms`（分に変換）です。追加項目がないCSVでも、利用できる特徴で表示します。

公開元の原本で確認した件数は、入力114,000行、ジャンル集計113,392行、楽曲89,583曲、114ジャンルです。以前のCP932版CSVには列ずれや文字化けがあったため、検証には公開元のUTF-8原本を使用しています。入力ファイルは変更しません。

## 担当画面

「曲の検索・選択」は曲名・アーティスト名の部分一致検索です。大文字小文字、全角半角、連続空白を正規化し、複数語はすべての語を含む曲を検索します。正規表現は使用しません。日本語名のローマ字変換や別名辞書による検索は含みません。

一致順・人気度順・曲名順で並べ替え、25曲ごとにページを切り替えます。曲IDで重複を除き、複数ジャンルの所属はまとめて表示します。同名曲はアーティスト、アルバム、ID末尾で区別できます。「この曲を基準にする」で選択を確定し、検索やページを変えても保持します。

「ジャンル比較」は8特徴の中央値・第1四分位数・第3四分位数・IQR・曲数、箱ひげ図、ヒートマップ、ジャンルの全組み合わせの差を表示します。初期値は元ノートブックと同じ7ジャンルです。7ジャンル×8特徴なら組み合わせ比較は168行です。CSV保存にも対応します。検索で選択した曲の所属ジャンルを比較対象にできます。

## 他の担当者との接続

選択曲の共有キーは、ウィジェット専用キーとは別に保存しています。

```python
import streamlit as st

selected_id = st.session_state.get("selected_track_id")  # 未選択ならNone
selected_song = st.session_state.get("selected_track")  # 楽曲情報のdict。未選択ならNone
```

共通データを使うページでは、次のように取得します。

```python
from music_app.data import catalog_for_page
from music_app.selection import selected_track

catalog = catalog_for_page()
tracks = catalog.tracks  # track_idごとに1曲。曲名、アーティスト、genres等
genre_rows = catalog.genre_rows  # 曲×ジャンルの行を保持
song = selected_track(tracks)  # データ変更で選択IDが消えた場合も扱える
```

小林担当の前処理済みDataFrameを使う場合は `prepare_catalog(df)` で共通形式へ変換できます。ジャンル比較に渡すデータは、track_idだけの重複除去を行う前の、複数ジャンルへの所属が残ったデータを使ってください。

推薦担当は `selected_track_id` を基準に推薦計算を行えます。推薦や試聴リンクの計算・実装はこの変更に含めません。選択状態は利用者のセッション内だけで保持し、ブラウザを再読み込みした後の永続保存は行いません。

## テスト

```powershell
python -m unittest discover -s tests -v
```

テストは小さいCSVを使い、外部ネットワークなしで実行できます。数値集計、重複とジャンル所属、検索の表記ゆれ、同名曲、ページ分け、選択の保持、メインからのページ移動、未選択・検索0件・CSV不備を確認します。
