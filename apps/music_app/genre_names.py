"""ジャンル名（元データの英語表記）の日本語訳。画面に出すときだけ使い、データ側の値は英語のまま。"""

GENRE_JA = {
    "acoustic": "アコースティック", "afrobeat": "アフロビート", "alt-rock": "オルタナティブ・ロック",
    "alternative": "オルタナティブ", "ambient": "アンビエント", "anime": "アニメ",
    "black-metal": "ブラックメタル", "bluegrass": "ブルーグラス", "blues": "ブルース",
    "brazil": "ブラジル音楽", "breakbeat": "ブレイクビーツ", "british": "ブリティッシュ",
    "cantopop": "広東ポップ", "chicago-house": "シカゴ・ハウス", "children": "子ども向け",
    "chill": "チル", "classical": "クラシック", "club": "クラブ", "comedy": "コメディ",
    "country": "カントリー", "dance": "ダンス", "dancehall": "ダンスホール",
    "death-metal": "デスメタル", "deep-house": "ディープ・ハウス", "detroit-techno": "デトロイト・テクノ",
    "disco": "ディスコ", "disney": "ディズニー", "drum-and-bass": "ドラムンベース", "dub": "ダブ",
    "dubstep": "ダブステップ", "edm": "EDM", "electro": "エレクトロ", "electronic": "エレクトロニック",
    "emo": "エモ", "folk": "フォーク", "forro": "フォホー", "french": "フランス音楽", "funk": "ファンク",
    "garage": "ガレージ", "german": "ドイツ音楽", "gospel": "ゴスペル", "goth": "ゴス",
    "grindcore": "グラインドコア", "groove": "グルーヴ", "grunge": "グランジ", "guitar": "ギター",
    "happy": "ハッピー", "hard-rock": "ハードロック", "hardcore": "ハードコア", "hardstyle": "ハードスタイル",
    "heavy-metal": "ヘヴィメタル", "hip-hop": "ヒップホップ", "honky-tonk": "ホンキートンク",
    "house": "ハウス", "idm": "IDM（エレクトロニカ）", "indian": "インド音楽", "indie": "インディー",
    "indie-pop": "インディー・ポップ", "industrial": "インダストリアル", "iranian": "イラン音楽",
    "j-dance": "J-ダンス", "j-idol": "アイドル（日本）", "j-pop": "J-POP", "j-rock": "J-ROCK",
    "jazz": "ジャズ", "k-pop": "K-POP", "kids": "キッズ", "latin": "ラテン", "latino": "ラティーノ",
    "malay": "マレー音楽", "mandopop": "北京語ポップ", "metal": "メタル", "metalcore": "メタルコア",
    "minimal-techno": "ミニマル・テクノ", "mpb": "MPB（ブラジル大衆音楽）", "new-age": "ニューエイジ",
    "opera": "オペラ", "pagode": "パゴージ", "party": "パーティー", "piano": "ピアノ", "pop": "ポップ",
    "pop-film": "映画音楽（インド）", "power-pop": "パワーポップ", "progressive-house": "プログレッシブ・ハウス",
    "psych-rock": "サイケデリック・ロック", "punk": "パンク", "punk-rock": "パンク・ロック",
    "r-n-b": "R&B", "reggae": "レゲエ", "reggaeton": "レゲトン", "rock": "ロック",
    "rock-n-roll": "ロックンロール", "rockabilly": "ロカビリー", "romance": "ロマンス", "sad": "サッド",
    "salsa": "サルサ", "samba": "サンバ", "sertanejo": "セルタネージョ", "show-tunes": "ミュージカル",
    "singer-songwriter": "シンガーソングライター", "ska": "スカ", "sleep": "睡眠用", "songwriter": "ソングライター",
    "soul": "ソウル", "spanish": "スペイン語の音楽", "study": "勉強用", "swedish": "スウェーデン音楽",
    "synth-pop": "シンセポップ", "tango": "タンゴ", "techno": "テクノ", "trance": "トランス",
    "trip-hop": "トリップホップ", "turkish": "トルコ音楽", "world-music": "ワールドミュージック",
}


def genre_label(genre: str) -> str:
    """「日本語（英語）」の形にする。英語も残すので、英語名で入力しても候補が見つかる。訳がなければ英語のまま。"""
    ja = GENRE_JA.get(genre)
    return f"{ja}（{genre}）" if ja else genre
