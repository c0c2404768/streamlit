import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from music_app.data import catalog_for_page
from music_app.similar import find_similar, standardized_matrix
from music_app.selection import clear_selection, selected_track

st.set_page_config(
    page_title="グループ分け - 楽曲ムード分析",
    page_icon="🎵",
    layout="wide"
)


def clear_group_selection():
    clear_selection()
    st.session_state["_group_filter_source_track_id"] = None
    st.session_state["_group_seed_keyword"] = ""
    st.session_state["_group_keyword"] = ""
    st.session_state["_group_mood_filter"] = "すべて表示"


# ---------------------------------------------------------
# 1. 共通関数でデータ取得 ＋ KMeans法による3グループ自動分類
# ---------------------------------------------------------
@st.cache_data(show_spinner="グループ分けを計算しています…")
def get_clustered_catalog(tracks: pd.DataFrame):
    # catalog_for_page() は画面の部品（データの設定）を出すので、キャッシュする関数の外で呼ぶ
    df = tracks.copy()

    # 列名の自動判別（music_app.data のカラム名に柔軟に対応）
    track_col = next((c for c in ["track_name", "name", "title", "song_name", "曲名"] if c in df.columns), df.columns[0])
    artist_col = next((c for c in ["artist_name", "artists", "artist", "アーティスト"] if c in df.columns), df.columns[1])

    features = ["tempo", "energy", "valence", "danceability"]
    df = df.dropna(subset=features).copy()

    # 4つの音響特徴量を標準化してKMeans法（3グループ）で分類
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # クラスタごとの特徴（Valence・Energyの平均値）から自動でラベルを割り当て
    cluster_means = df.groupby("cluster")[["valence", "energy"]].mean()
    pop_cluster = cluster_means["valence"].idxmax()
    remaining = cluster_means.drop(index=pop_cluster)
    rock_cluster = remaining["energy"].idxmax()
    chill_cluster = remaining.drop(index=rock_cluster).index[0]

    label_map = {
        pop_cluster: "☀️ ポップ・ハッピー系（明るい・気分が上がる）",
        chill_cluster: "🌙 バラード・チル系（静か・落ち着き）",
        rock_cluster: "🔥 ロック・エナジー系（激しい・スピード感）",
    }
    df["ムード（グループ）"] = df["cluster"].map(label_map)
    return df, track_col, artist_col


@st.cache_data(show_spinner="近い楽曲を検索しています…")
def get_similar_tracks(tracks: pd.DataFrame, track_id: str) -> pd.DataFrame:
    return find_similar(
        tracks,
        standardized_matrix(tracks),
        track_id,
        top_n=len(tracks),
        method="euc",
    )


df, TRACK_COL, ARTIST_COL = get_clustered_catalog(catalog_for_page().tracks)
selected_song = selected_track(df)

# ---------------------------------------------------------
# 2. タイトル・導入文
# ---------------------------------------------------------
st.title("🎵 楽曲グループ分け（ムードマップ）")
st.write("楽曲の音響特徴量をもとに自動分類された3つのグループから、気分や条件に合う曲を検索できます。")

# ---------------------------------------------------------
# 3. 楽曲検索・絞り込みUI
# ---------------------------------------------------------
with st.container(border=True):
    st.subheader("🔍 楽曲検索・絞り込み")
    col1, col2, col3 = st.columns(3)
    mood_options = [
        "すべて表示",
        "☀️ ポップ・ハッピー系（明るい・気分が上がる）",
        "🌙 バラード・チル系（静か・落ち着き）",
        "🔥 ロック・エナジー系（激しい・スピード感）"
    ]
    keyword_key = "_group_keyword"
    seed_keyword_key = "_group_seed_keyword"
    mood_filter_key = "_group_mood_filter"
    source_track_key = "_group_filter_source_track_id"
    selected_track_id = str(selected_song["track_id"]) if selected_song is not None else None
    if source_track_key not in st.session_state or st.session_state[source_track_key] != selected_track_id:
        seed_keyword = (
            f"{selected_song['track_name']} / {selected_song['artists']}"
            if selected_song is not None else ""
        )
        st.session_state[source_track_key] = selected_track_id
        st.session_state[seed_keyword_key] = seed_keyword
        st.session_state[keyword_key] = seed_keyword
        st.session_state[mood_filter_key] = (
            selected_song["ムード（グループ）"] if selected_song is not None else mood_options[0]
        )
    else:
        st.session_state.setdefault(keyword_key, "")
        st.session_state.setdefault(seed_keyword_key, "")
        if st.session_state.get(mood_filter_key) not in mood_options:
            st.session_state[mood_filter_key] = mood_options[0]

    with col1:
        keyword = st.text_input(
            "キーワード検索（曲名・アーティスト名）",
            placeholder="例: Lemon / Kenshi Yonezu",
            key=keyword_key
        )
        exact_selected_track = (
            selected_track_id is not None
            and keyword == st.session_state[seed_keyword_key]
        )
        if exact_selected_track:
            st.caption("選択曲のIDで厳密に絞り込み中です。キーワードを編集すると通常検索になります。")
        elif selected_song is not None:
            st.caption("選択曲を変更すると、キーワードとムードの初期値が更新されます。")

    with col2:
        selected_mood = st.selectbox(
            "ムード（グループ）を選択",
            mood_options,
            key=mood_filter_key
        )
        if selected_song is not None:
            st.caption("選択曲のムードを初期値に設定しました。必要に応じて変更できます。")
        else:
            st.caption("曲を選ぶと、その曲のムードが初期選択されます。")

    with col3:
        min_bpm = int(df["tempo"].min())
        max_bpm = int(df["tempo"].max())
        bpm_range = st.slider(
            "BPM（テンポ）の範囲",
            min_value=min_bpm,
            max_value=max_bpm,
            value=(min_bpm, max_bpm)
        )

    if selected_song is not None:
        st.button(
            "選択曲をクリア",
            key="_group_clear_selected_song",
            help="共有している選択曲を解除し、キーワードとムードを初期状態に戻します。",
            on_click=clear_group_selection,
        )

# ---------------------------------------------------------
# 4. 絞り込み処理（フィルタリング）
# ---------------------------------------------------------
if exact_selected_track:
    filtered_df = get_similar_tracks(df, selected_track_id)
else:
    filtered_df = df.copy()
    if keyword:
        filtered_df = filtered_df[
            filtered_df[TRACK_COL].astype(str).str.contains(keyword, case=False, na=False, regex=False) |
            filtered_df[ARTIST_COL].astype(str).str.contains(keyword, case=False, na=False, regex=False)
        ]

if selected_mood != "すべて表示":
    filtered_df = filtered_df[filtered_df["ムード（グループ）"] == selected_mood]

filtered_df = filtered_df[
    (filtered_df["tempo"] >= bpm_range[0]) &
    (filtered_df["tempo"] <= bpm_range[1])
]
SEARCH_RESULT_LIMIT = 100
search_results = filtered_df.head(SEARCH_RESULT_LIMIT)

st.divider()

# ---------------------------------------------------------
# 5. 楽曲ムードマップ（散布図）と検索結果の表示
# ---------------------------------------------------------
col_chart, col_list = st.columns([1.2, 1])

with col_chart:
    st.subheader("📊 楽曲ムードマップ（Valence × Energy）")
    color_map = {
        "☀️ ポップ・ハッピー系（明るい・気分が上がる）": "#FFD166",
        "🌙 バラード・チル系（静か・落ち着き）": "#4EA8DE",
        "🔥 ロック・エナジー系（激しい・スピード感）": "#EF476F",
    }
    MAP_LIMIT = 5000  # 点が多すぎると表示が重くなるため、地図に出す曲数の上限
    if exact_selected_track:
        map_df = search_results
    else:
        map_df = filtered_df.sample(n=MAP_LIMIT, random_state=0) if len(filtered_df) > MAP_LIMIT else filtered_df
    fig = px.scatter(
        map_df,
        x="valence",
        y="energy",
        color="ムード（グループ）",
        color_discrete_map=color_map,
        hover_data=[TRACK_COL, ARTIST_COL, "tempo"],
        labels={
            "valence": "Valence（明るさ）",
            "energy": "Energy（激しさ）",
            "ムード（グループ）": "グループ"
        }
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    if selected_song is not None:
        fig.add_scatter(
            x=[selected_song["valence"]],
            y=[selected_song["energy"]],
            mode="markers",
            marker={"size": 17, "symbol": "star", "color": "#111827", "line": {"width": 2, "color": "white"}},
            name="選択中の曲",
            text=[f"{selected_song['track_name']} / {selected_song['artists']}"],
            hovertemplate="%{text}<br>Valence=%{x}<br>Energy=%{y}<extra></extra>",
        )
    st.plotly_chart(fig, width="stretch")
    if exact_selected_track:
        st.caption("選択曲との4つの音響特徴量を標準化し、距離が近い上位100曲をマップにも表示しています。基準曲と同じ曲名・アーティストの曲は除外しています。")
    elif len(filtered_df) > MAP_LIMIT:
        st.caption(f"該当する{len(filtered_df):,}曲のうち、{MAP_LIMIT:,}曲をランダムに選んで表示しています。")
    if selected_song is not None:
        st.caption("★は「曲の検索・選択」で選んだ曲です。")

with col_list:
    result_title = "選択曲に値が近い楽曲" if exact_selected_track else "検索結果"
    st.subheader(f"🎶 {result_title}（該当: {len(filtered_df):,}曲）")
    if selected_song is not None:
        st.text(f"選択中の曲: {selected_song['track_name']} / {selected_song['artists']}")
        st.caption(f"ムード: {selected_song['ムード（グループ）']}")
    else:
        st.page_link("pages/曲の検索・選択.py", label="曲を検索・選択してムードマップに表示", icon="🔍")
    display_df = search_results[[TRACK_COL, ARTIST_COL, "ムード（グループ）", "tempo"]].rename(columns={
        TRACK_COL: "曲名",
        ARTIST_COL: "アーティスト",
        "tempo": "BPM"
    })
    if exact_selected_track:
        display_df["特徴距離"] = search_results["distance"].round(3).to_numpy()
    st.dataframe(display_df, width="stretch", hide_index=True)
    if exact_selected_track and len(filtered_df) > SEARCH_RESULT_LIMIT:
        st.caption(f"値が近い順に上位{SEARCH_RESULT_LIMIT}曲を表示しています。")
    elif len(filtered_df) > SEARCH_RESULT_LIMIT:
        st.caption(f"検索結果一覧は先頭{SEARCH_RESULT_LIMIT}曲まで表示しています。条件を追加すると絞り込めます。")

st.divider()

# ---------------------------------------------------------
# 6. 各グループ（ムード）の特徴と分析の仕組み
# ---------------------------------------------------------
st.subheader("📖 各グループ（ムード）の特徴と分析の仕組み")
st.write("このアプリでは、機械学習（KMeans法）を用いて楽曲の4つの音響特徴量（ `tempo` , `energy` , `valence` , `danceability` ）から全楽曲を3つのグループに自動分類しています。")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🌙 バラード・チル系")
    st.caption("静かで落ち着いた雰囲気の楽曲")
    st.markdown("""
- **特徴:** Energy（激しさ）・Valence（明るさ）ともに低く、テンポもゆったりめ。
- **おすすめシーン:** 夜のリラックスタイム、作業用BGM、静かに聴き入りたい時。
""")

with c2:
    st.markdown("### ☀️ ポップ・ハッピー系")
    st.caption("メロディーが明るく気分が上がる楽曲")
    st.markdown("""
- **特徴:** Valence（明るさ）と Danceability（踊りやすさ）が最も高く、軽快なリズム。
- **おすすめシーン:** ドライブ、朝の通勤・通学、楽しい気分になりたい時。
""")

with c3:
    st.markdown("### 🔥 ロック・エナジー系")
    st.caption("スピード感や激しさが前面に出ている楽曲")
    st.markdown("""
- **特徴:** Energy（激しさ）と BPM（テンポ）が非常に高く、力強いサウンド。
- **おすすめシーン:** 運動・ワークアウト、気分を奮い立たせたい時。
""")

with st.expander("📊 グループごとの平均値データと分析の根拠を見る"):
    st.subheader("各グループの音響特徴量（平均値）")
    mean_df = df.groupby("ムード（グループ）")[["tempo", "energy", "valence", "danceability"]].mean().round(2)
    st.dataframe(mean_df, width="stretch")
    st.markdown("""
    *💡 分析のポイント（なぜこの散布図なのか？）*
    - 4つの指標のうち、特に **Energy（激しさ）** と **Valence（明るさ）** の2軸を組み合わせることで、3つのグループが最も明確に分離されることが確認されています。
    - `tempo` や `danceability` は単体ではグループ間の重なりが大きいため、絞り込みの補助指標として活用しています。
    """)