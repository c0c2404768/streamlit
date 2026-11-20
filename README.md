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

必要な列は `track_id`, `track_name`, `artists` です。`track_genre` が空欄・未収録の曲も検索できますが、ジャンル別の集計には含めません。推薦用の4特徴（`tempo`, `energy`, `valence`, `danceability`）がある場合は欠損・不正値を除外します。ジャンル比較の追加4項目は `acousticness`, `instrumentalness`, `speechiness`, `duration_ms`（分に変換）です。追加項目がないCSVでも、利用できる特徴で表示します。

公開元の原本で確認した件数は、入力114,000行、ジャンル集計113,392行、楽曲89,583曲、114ジャンルです。以前のCP932版CSVには列ずれや文字化けがあったため、検証には公開元のUTF-8原本を使用しています。入力ファイルは変更しません。

### Tunebatの追加1万曲

標準データには `apps/music_app/assets/tunebat_tracks.csv` の10,000曲を追加します。元ファイルは `tunebat_データセット外_10000曲.xlsx`（`tracks` シート）です。1,000曲版は使いません。取得日・出典・元ExcelのSHA-256・特徴ごとの件数を `tunebat_source.json` に記録しています。元Excelは変更しません。

- 標準データとの同一Spotify IDは0件で、合計99,583曲。ジャンル集計は元の113,392行・114ジャンルを維持します。
- 表記の正規化後に曲名・アーティストが一致する追加曲は2曲ありますが、Spotify IDは異なります。別IDを曲名だけで統合せず、音源の違いを確認できる形で保持します。
- テンポ、Energy、Valence、Danceabilityは追加10,000曲すべてに収録されています。Energy等はExcelで既に0〜1なので、再び100で割りません。
- 曲の長さ、Acousticness、Instrumentalness、Speechinessは155曲だけに収録され、残りは欠損のままです。未収録を0に置き換えません。選択曲に値がない特徴はグラフに線を描かず、その旨を表示します。
- 追加曲のジャンルはすべて未登録です。推測でジャンルを付けず、既存ジャンルの集計に混ぜません。既知ジャンルの中央値と追加曲の特徴を重ねて比較することはできます。
- 提供ファイルは竹内まりや「元気を出して」からTunebatのおすすめをたどった曲群です。日本の音楽全体などを代表する標本ではありません。取得元の指標と既存データの取得時期・測定方法の差があり得るため、比較は探索の目安です。
- 同じIDを含むファイルへ将来更新した場合は1曲に統合し、既存の音響値と既知ジャンルを維持します。提供側の曲名・アーティスト名を表示し、旧表記も検索に残します。

同じ列構成のExcelから更新する場合：

```powershell
python tools/import_tunebat.py "追加データのExcelの絶対パス.xlsx"
```

アップロードCSVと `SPOTIFY_DATASET_PATH` は指定したデータへの差し替えとして扱い、追加1万曲は自動で混ぜません。各担当も同じ追加データを使うには下記の `catalog_for_page()` を使ってください。`load_catalog()` を直接呼ぶ場合は追加CSVの更新時刻を `supplement_stamp` に渡す必要があります。

## 担当画面

「曲の検索・選択」は曲名・アーティスト名の部分一致検索です。大文字小文字、全角半角、連続空白を正規化し、複数語はすべての語を含む曲を検索します。正規表現は使用しません。星野源／Gen Hoshino、竹内まりや／Mariya Takeuchiには明示的な別名を登録しています。一般的な自動翻訳・ローマ字変換は行いません。ジャンルの絞り込み（未登録も選択可能）とTunebat追加曲だけの表示に対応します。

一致順・人気度順・曲名順で並べ替え、25曲ごとにページを切り替えます。曲IDで重複を除き、複数ジャンルの所属はまとめて表示します。同名曲はアーティスト、アルバム、ID末尾で区別できます。「この曲を基準にする」で選択を確定し、検索やページを変えても保持します。

「ジャンル比較」は8つの特徴から1つ選び、どんな音が多いかを点の大きさで見比べます。各ジャンル内の曲の割合を使うので、収録曲数の違うジャンルも同じ目安で見られます。「詳しいデータを見る」を開くと、集計表とCSVが表示されます。

選択曲の値はオレンジの破線で重ねます。気になるジャンルを選ぶと、そのジャンルの何曲より速い・明るいなど、今見ている特徴での位置を文章で確認できます。「曲を探す」ボタンでそのジャンルの検索結果を開けます。基準曲は保持されます。
「人気度との関係」（小林担当）は、人気度と14項目（曲の長さ・Eマーク・ノリの良さ・力強さ・キー・音量・長調短調・話し声の多さ・アコースティックらしさ・インストらしさ・ライブ感・明るさ・テンポ・拍子）の相関係数を関係の強い順の横棒グラフで表示します。項目を選ぶと、10区間ごとの平均人気度を折れ線グラフで表示し、人気度0の曲を除いた線も重ねられます（曲数30未満の区間は非表示）。4特徴・14項目をまとめて使った重相関係数と、ジャンルの相関比（ジャンル集計用の行を使用）も表示します。原本では順に0.085、0.186、0.502です。重相関係数は scikit-learn を使わず、NumPy の最小二乗法で計算しています。

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

推薦の絞り込み条件（小林担当）は `music_app.filters` にあります。人気度の上限、同じジャンルの除外、似ている度合いの測り方（コサイン類似度・ユークリッド距離）をサイドバーに表示し、条件は `filter_` で始まる共通キーに保存するため、ページを移動しても残ります。

```python
from music_app.filters import apply_filters, describe_filters, render_filters, show_no_result

cond = render_filters()  # {"pop_max": 100, "other_genre": False, "method": "cos"}
candidates = apply_filters(catalog.tracks, st.session_state.get("selected_track_id"), cond)
if candidates.empty:
    show_no_result()
    st.stop()
st.caption(describe_filters(cond))  # 例：条件：人気度30以下・ほかのジャンルだけ
```

`apply_filters` は、選択曲そのもの、曲名とアーティストが同じ別バージョン、人気度が上限を超える曲を除きます。「同じジャンルの曲は出さない」では、選択曲が所属するどのジャンルにも入らない曲だけを残します。返す表の行番号は `catalog.tracks` と同じです。

## テスト

```powershell
python -m unittest discover -s tests -v
```

テストは小さいCSVと同梱の追加CSVを使い、外部ネットワークなしで実行できます。数値集計、重複とジャンル所属、検索の表記ゆれ、同名曲、ページ分け、選択の保持、メインからのページ移動、未選択・検索0件・CSV不備に加え、追加曲の単位・欠損、同一IDの統合、ジャンル比較から絞り込み検索への移動を確認します。
