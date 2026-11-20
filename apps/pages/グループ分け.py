import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="グループ分け - 楽曲ムード分析",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 楽曲グループ分け（ムードマップ）")
st.markdown(
    "楽曲の音響特徴量をもとに自動分類された3つのグループから、気分や条件に合う曲を検索できます。"
)


# ---------------------------------------------------------
# 1. データの読み込みとKMeansモデルの学習
# ---------------------------------------------------------
@st.cache_data
def load_and_cluster_data():
  # ※実際のCSVパスに合わせて変更してください
  # df = pd.read_csv("dataset (1).csv", encoding="cp932")

  # デモ用サンプルデータ（実データ読み込み時は削除）
  np.random.seed(42)
  n_samples = 150
  df = pd.DataFrame({
      "track_name": [f"Sample Song {i+1}" for i in range(n_samples)],
      "artist_name": [
          f"Artist {np.random.randint(1, 30)}" for _ in range(n_samples)
      ],
      "tempo": np.random.uniform(70, 175, n_samples),
      "energy": np.random.uniform(0.1, 0.98, n_samples),
      "valence": np.random.uniform(0.05, 0.95, n_samples),
      "danceability": np.random.uniform(0.2, 0.95, n_samples),
  })

  features = ["tempo", "energy", "valence", "danceability"]
  X = df[features].copy()

  for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

  valid_condition = (
      (X["tempo"] >= 40)
      & (X["tempo"] <= 250)
      & (X["energy"] >= 0)
      & (X["energy"] <= 1)
      & (X["valence"] >= 0)
      & (X["valence"] <= 1)
      & (X["danceability"] >= 0)
      & (X["danceability"] <= 1)
  )
  X_clean = X[valid_condition].dropna()
  df_clean = df.loc[X_clean.index].copy()

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_clean)

  kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
  df_clean["cluster"] = kmeans.fit_predict(X_scaled)

  # クラスタの特性からムード名を自動割り当て
  cluster_means = df_clean.groupby("cluster")[["energy", "valence"]].mean()
  chill_cluster = cluster_means["energy"].idxmin()
  remaining = [c for c in [0, 1, 2] if c != chill_cluster]
  pop_cluster = cluster_means.loc[remaining, "valence"].idxmax()
  rock_cluster = [c for c in remaining if c != pop_cluster][0]

  mood_mapping = {
      chill_cluster: "🌙 バラード・チル系（静か・落ち着き）",
      pop_cluster: "☀️ ポップ・ハッピー系（明るい・気分が上がる）",
      rock_cluster: "🔥 ロック・エナジー系（激しい・スピード感）",
  }
  df_clean["mood"] = df_clean["cluster"].map(mood_mapping)

  return df_clean


df = load_and_cluster_data()

# ---------------------------------------------------------
# 2. ページ上部：検索・フィルターエリア（横並び3カラム）
# ---------------------------------------------------------
with st.container(border=True):
  st.subheader("🔍 楽曲検索・絞り込み")

  search_col1, search_col2, search_col3 = st.columns([1.2, 1.2, 1])

  with search_col1:
    search_query = st.text_input(
        "キーワード検索（曲名・アーティスト名）",
        placeholder="例: Sample Song / Artist...",
    )

  with search_col2:
    mood_options = ["すべて表示"] + sorted(df["mood"].unique().tolist())
    selected_mood = st.selectbox("ムード（グループ）を選択", mood_options)

  with search_col3:
    min_tempo, max_tempo = int(df["tempo"].min()), int(df["tempo"].max())
    tempo_range = st.slider(
        "BPM（テンポ）の範囲",
        min_value=min_tempo,
        max_value=max_tempo,
        value=(min_tempo, max_tempo),
    )

# ---------------------------------------------------------
# 3. 条件によるフィルタリング処理
# ---------------------------------------------------------
filtered_df = df[
    (df["tempo"] >= tempo_range[0]) & (df["tempo"] <= tempo_range[1])
]

# ムード選択での絞り込み
if selected_mood != "すべて表示":
  filtered_df = filtered_df[filtered_df["mood"] == selected_mood]

# キーワード検索での絞り込み（曲名またはアーティスト名に部分一致・大文字小文字区別なし）
if search_query:
  filtered_df = filtered_df[
      filtered_df["track_name"].str.contains(search_query, case=False, na=False)
      | filtered_df["artist_name"].str.contains(
          search_query, case=False, na=False
      )
  ]

st.divider()

# ---------------------------------------------------------
# 4. メイン画面：散布図 と 検索結果リスト
# ---------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
  st.subheader("📊 楽曲ムードマップ（Valence × Energy）")
  fig = px.scatter(
      filtered_df,
      x="valence",
      y="energy",
      color="mood",
      hover_data={
          "track_name": True,
          "artist_name": True,
          "tempo": ":.1f",
          "danceability": ":.2f",
          "valence": ":.2f",
          "energy": ":.2f",
          "mood": False,
      },
      labels={
          "valence": "Valence（明るさ）",
          "energy": "Energy（激しさ）",
          "mood": "グループ",
      },
      color_discrete_map={
          "🌙 バラード・チル系（静か・落ち着き）": "#3b82f6",
          "☀️ ポップ・ハッピー系（明るい・気分が上がる）": "#f59e0b",
          "🔥 ロック・エナジー系（激しい・スピード感）": "#ef4444",
      },
      height=480,
  )
  fig.update_layout(
      xaxis=dict(range=[0, 1]),
      yaxis=dict(range=[0, 1]),
      legend=dict(
          orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
      ),
  )
  st.plotly_chart(fig, use_container_width=True)

with col2:
  st.subheader(f"🎶 検索結果（該当: {len(filtered_df)}曲）")

  if len(filtered_df) > 0:
    display_cols = [
        "track_name",
        "artist_name",
        "mood",
        "tempo",
        "energy",
        "valence",
    ]
    display_df = filtered_df[display_cols].copy()
    display_df.columns = [
        "曲名",
        "アーティスト",
        "グループ",
        "BPM",
        "Energy",
        "Valence",
    ]

    st.dataframe(
        display_df.style.format(
            {"BPM": "{:.1f}", "Energy": "{:.2f}", "Valence": "{:.2f}"}
        ),
        use_container_width=True,
        height=440,
        hide_index=True,
    )
  else:
    st.info("条件に一致する楽曲が見つかりませんでした。検索条件を変更してください。")

# =========================================================
# 5. ページ下部：クラスタリング分析の解説・各グループの特徴
# =========================================================
st.divider()
st.subheader("📖 各グループ（ムード）の特徴と分析の仕組み")
st.markdown(
    "このアプリでは、機械学習（KMeans法）を用いて楽曲の4つの音響特徴量（`tempo`, `energy`, `valence`, `danceability`）から全楽曲を3つのグループに自動分類しています。"
)

# 3つのグループの特徴をカード形式で横並びに解説
exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
  with st.container(border=True):
    st.markdown("#### 🌙 バラード・チル系")
    st.markdown(
        "**静かで落ち着いた雰囲気の楽曲**\n"
        "- **特徴:** Energy（激しさ）・Valence（明るさ）ともに低く、テンポもゆったりめ。\n"
        "- **おすすめシーン:** 夜のリラックスタイム、作業用BGM、静かに聴き入りたい時。"
    )

with exp_col2:
  with st.container(border=True):
    st.markdown("#### ☀️ ポップ・ハッピー系")
    st.markdown(
        "**メロディーが明るく気分が上がる楽曲**\n"
        "- **特徴:** Valence（明るさ）と Danceability（踊りやすさ）が最も高く、軽快なリズム。\n"
        "- **おすすめシーン:** ドライブ、朝の通勤・通学、楽しい気分になりたい時。"
    )

with exp_col3:
  with st.container(border=True):
    st.markdown("#### 🔥 ロック・エナジー系")
    st.markdown(
        "**スピード感や激しさが前面に出ている楽曲**\n"
        "- **特徴:** Energy（激しさ）と BPM（テンポ）が非常に高く、力強いサウンド。\n"
        "- **おすすめシーン:** 運動・ワークアウト、気分を奮い立たせたい時。"
    )

# 折りたたみメニュー（expander）で詳細データや分析の根拠を表示
with st.expander("📊 グループごとの平均値データと分析の根拠を見る"):
  st.markdown("#### 各グループの音響特徴量（平均値）")

  # 現在のデータから各グループの平均値を自動計算してテーブル表示
  summary_df = (
      df.groupby("mood")[["tempo", "energy", "valence", "danceability"]]
      .mean()
      .reset_index()
  )
  summary_df.columns = [
      "グループ",
      "平均BPM",
      "Energy（激しさ）",
      "Valence（明るさ）",
      "Danceability（踊りやすさ）",
  ]

  st.dataframe(
      summary_df.style.format({
          "平均BPM": "{:.1f}",
          "Energy（激しさ）": "{:.2f}",
          "Valence（明るさ）": "{:.2f}",
          "Danceability（踊りやすさ）": "{:.2f}",
      }),
      use_container_width=True,
      hide_index=True,
  )

  st.info(
      "**💡 分析のポイント（なぜこの散布図なのか？）**\n"
      "- 4つの指標のうち、特に **Energy（激しさ）** と **Valence（明るさ）** の2軸を組み合わせることで、3つのグループが最も明確に分離されることが確認されています。\n"
      "- `tempo` や `danceability` は単体ではグループ間の重なりが大きいため、絞り込みの補助指標として活用しています。"
  )