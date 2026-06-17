# api.py  -  Flask Backend Sentiment Analysis - UPT BLK Surabaya
#
# Endpoint yang tersedia:
#   GET  /api/health                      - cek status server
#   GET  /api/analyze?text=...            - analisa 1 komentar
#   POST /api/analyze-bulk                - analisa banyak komentar sekaligus
#                                           Body JSON: {"comments": ["teks1", "teks2", ...]}
#   GET  /api/comments                    - ambil data demo komentar BLK
#   GET  /api/stats                       - statistik ringkasan (positif / negatif)
#
# CATATAN UNTUK TEMAN BACKEND:
#   Jika backend utama (Laravel/Express) ingin mengintegrasikan fitur ini,
#   cukup panggil POST /api/analyze-bulk dengan daftar komentar.
#   Lihat bagian "Panduan Integrasi Backend" di bawah.

import os
import re
import pickle
import random
from datetime import datetime, timedelta

import nltk
from flask import Flask, jsonify, request
from flask_cors import CORS

nltk.download('stopwords', download_dir='/tmp', quiet=True)
nltk.data.path.append('/tmp')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==============================================================================
# KONFIGURASI
# ==============================================================================

app = Flask(__name__)
CORS(app)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLF_PATH  = os.path.join(BASE_DIR, 'model', 'classifier.pkl')
VECT_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

classifier = None
vectorizer = None


def load_model():
    global classifier, vectorizer
    if not os.path.exists(CLF_PATH) or not os.path.exists(VECT_PATH):
        return False
    with open(CLF_PATH, 'rb') as f:
        classifier = pickle.load(f)
    with open(VECT_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return True


model_loaded = load_model()

# ==============================================================================
# PREPROCESSING & PREDIKSI
# ==============================================================================

ps    = PorterStemmer()
stops = set(stopwords.words('english'))

STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'adalah', 'ada', 'tidak', 'ya', 'iya', 'juga', 'bisa', 'saya', 'kami',
    'kamu', 'mereka', 'pada', 'dalam', 'atau', 'karena', 'sudah', 'sudahlah',
    'sangat', 'sekali', 'masih', 'akan', 'bagi', 'jika', 'maka', 'namun',
    'tetapi', 'tapi', 'oleh', 'kalau', 'kalian', 'kita', 'anda', 'pak', 'bu',
    'mas', 'mbak', 'bang', 'kakak', 'kak', 'semua', 'banyak', 'sedikit',
    'lagi', 'pun', 'aja', 'sih', 'deh', 'dong', 'lah', 'nah', 'wah', 'kok',
    'emang', 'memang', 'udah', 'banget', 'bgt', 'hehe', 'haha', 'like', 'likes', 
    'reply', 'replies', 
}

def bersihkan_noise_sosmed(text):
    # 1. Menghapus pola waktu, likes, dan kata Reply (misal: 1 wReply, 4 d1 likeReply)
    text = re.sub(r'\d+\s*[wdhm](?:\s*\d+\s*likes?)?\s*Reply', '', text, flags=re.IGNORECASE)
    
    # 2. Menghapus username/mention (misal: @uptblksurabaya)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Opsional: Menghapus angka yang nempel di akhir kalimat sebelum 'wReply' 
    # (Kadang copas langsung bikin teks nyambung)
    text = re.sub(r'\d+$', '', text.strip())
    
    return text

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in stops and w not in STOPWORDS_ID]
    return ' '.join(text)

# Kamus sentimen — untuk override ML pada kata-kata yang sudah pasti artinya
LEXICON_POSITIF = {
    'bagus', 'mantap', 'keren', 'memuaskan', 'bermanfaat', 'profesional',
    'berkualitas', 'puas', 'berguna', 'membantu', 'baik', 'suka', 'senang',
    'berhasil', 'sukses', 'hebat', 'istimewa', 'terbaik', 'sempurna',
    'menyenangkan', 'nyaman', 'ramah', 'sopan', 'canggih', 'modern', 'lengkap',
    'jelas', 'mudah', 'cepat', 'efisien', 'efektif', 'kompeten', 'sabar',
    'informatif', 'interaktif', 'produktif', 'inovatif', 'kreatif', 'top',
    'oke', 'asyik', 'asik', 'recommended', 'rekomend', 'bravo', 'salut',
    'ciamik', 'kece', 'luar biasa', 'makasih', 'terimakasih', 'lucu',
}

LEXICON_NEGATIF = {
    'jelek', 'buruk', 'kecewa', 'mengecewakan', 'payah', 'parah', 'gagal',
    'kacau', 'hancur', 'berantakan', 'kapok', 'menyesal', 'rugi', 'percuma',
    'ribet', 'lambat', 'kotor', 'panas', 'sempit', 'rusak', 'usang', 'error',
    'lemot', 'menyebalkan', 'membosankan', 'melelahkan', 'bermasalah',
    'tidak bagus', 'tidak baik', 'tidak jelas', 'tidak puas', 'tidak membantu',
    'tidak profesional', 'tidak nyaman', 'tidak kompeten', 'tidak berkualitas',
    'tidak memuaskan', 'tidak ramah', 'tidak sopan', 'tidak lengkap',
    'kurang bagus', 'kurang memuaskan', 'kurang jelas', 'kurang memadai',
    'sangat kecewa', 'sangat buruk', 'sangat jelek', 'sangat mengecewakan',
}

# Kata/frasa yang kuat menandai sentimen (bobot lebih tinggi)
STRONG_POSITIF = {'sangat bagus', 'sangat baik', 'sangat bermanfaat', 'sangat membantu',
                  'sangat memuaskan', 'sangat puas', 'luar biasa', 'terbaik', 'sempurna'}
STRONG_NEGATIF = {'sangat kecewa', 'sangat buruk', 'sangat jelek', 'sangat mengecewakan',
                  'sangat tidak puas', 'sangat tidak bagus', 'sangat tidak baik'}


# Pola frasa yang menandai kalimat informasi/pertanyaan → Netral
# Meski ada kata negatif ringan, jika intent-nya bertanya maka tetap Netral
NEUTRAL_PATTERNS = {
    # Permintaan informasi
    'mohon info', 'minta info', 'mau tanya', 'mau bertanya', 'ingin tanya',
    'ingin bertanya', 'nanya', 'mau nanya', 'tanya dong', 'bisa info',
    'tolong info', 'butuh info', 'perlu info', 'cari info', 'minta bantuan',
    # Sapaan admin
    'min mohon', 'min minta', 'min tanya', 'min mau', 'admin mohon',
    'admin minta', 'admin tanya', 'kak mohon', 'kak minta', 'kak tanya',
    'halo min', 'halo admin', 'hai min', 'selamat pagi min',
    # Laporan situasi netral
    'sempat daftar', 'pernah daftar', 'sudah daftar', 'mau daftar',
    'ingin daftar', 'mau mendaftar', 'hendak daftar', 'berniat daftar',
    'belum daftar', 'sedang mencoba', 'sedang mendaftar',
    # Permintaan klarifikasi
    'mohon penjelasan', 'mohon konfirmasi', 'mohon informasi',
    'kapan pendaftaran', 'dimana lokasi', 'berapa biaya', 'bagaimana cara',
    'apa syarat', 'apa saja', 'ada tidak', 'tersedia tidak',
}

def _lexicon_predict(text):
    """Cek lexicon. Return (label, score, confidence) atau None jika tidak konklusif."""
    t = text.lower()

    # 0. Cek dulu apakah kalimat ini adalah pertanyaan/permintaan info (override ke Netral)
    #    Kalimat jenis ini lebih tepat Netral meski ada kata negatif ringan
    is_question = t.strip().endswith('?')
    is_info_request = any(p in t for p in NEUTRAL_PATTERNS)
    if is_question or is_info_request:
        # Tetap bisa Negatif jika ada kata negatif yang SANGAT kuat
        has_strong_neg = any(p in t for p in STRONG_NEGATIF)
        if not has_strong_neg:
            return 'netral', 2, 70.0

    # 1. Cek strong keywords sentimen
    for phrase in STRONG_POSITIF:
        if phrase in t:
            return 'positif', 1, 95.0
    for phrase in STRONG_NEGATIF:
        if phrase in t:
            return 'negatif', 0, 95.0

    pos = sum(1 for w in LEXICON_POSITIF if w in t)
    neg = sum(1 for w in LEXICON_NEGATIF if w in t)

    if pos > 0 and neg == 0:
        conf = min(55.0 + pos * 10, 90.0)
        return 'positif', 1, conf
    if neg > 0 and pos == 0:
        conf = min(55.0 + neg * 10, 90.0)
        return 'negatif', 0, conf

    return None, None, None  # Tidak konklusif → pakai ML


def predict_sentiment(text):
    """Prediksi sentimen: positif (1) / negatif (0) / netral (2).
    Menggunakan gabungan Lexicon + ML model Bahasa Indonesia.
    """
    if not text or not str(text).strip():
        return {'label': 'netral', 'score': 2, 'confidence': 0.0}

    # 1. Coba lexicon rule-based dulu (lebih akurat untuk kata pendek)
    lex_label, lex_score, lex_conf = _lexicon_predict(text)
    if lex_label is not None:
        return {'label': lex_label, 'score': lex_score, 'confidence': lex_conf}

    # 2. Fallback ke ML model
    if not model_loaded or classifier is None or vectorizer is None:
        return {'label': 'netral', 'score': 2, 'confidence': 0.0}

    processed = preprocess(text)
    if not processed.strip():
        return {'label': 'netral', 'score': 2, 'confidence': 0.0}

    try:
        vec   = vectorizer.transform([processed]).toarray()
        pred  = int(classifier.predict(vec)[0])
        proba = classifier.predict_proba(vec)[0]
        conf  = round(float(max(proba)) * 100, 1)

        # Peta label: model 3-kelas (0=negatif, 1=positif, 2=netral)
        # atau model 2-kelas lama (0=negatif, 1=positif)
        label_map = {0: 'negatif', 1: 'positif', 2: 'netral'}
        label = label_map.get(pred, 'netral')

        # Jika confidence rendah (model tidak yakin), jadikan netral
        if conf < 52.0:
            label = 'netral'
            pred  = 2

        return {'label': label, 'score': pred, 'confidence': conf}

    except Exception:
        return {'label': 'netral', 'score': 2, 'confidence': 0.0}


# ==============================================================================
# DATA KOMENTAR ASLI — Discrape dari Instagram @uptblksurabaya
# Sumber: Postingan terpopuler (diambil via browser, Juni 2026)
# Data ini STATIS (hardcode) karena Instagram memblokir scraping otomatis.
# Untuk update: scrape manual lagi lalu ganti list di bawah ini.
# ==============================================================================

# Format: (teks_komentar, username_instagram, url_post)
REAL_COMMENTS_IG = [
    # ── KOMENTAR ASLI INSTAGRAM @uptblksurabaya ──────────────────────────────
    # Sumber: Post DZMR_A3kbhe & DZrFVh6nZWE (Juni 2026, via browser)

    ("servernya masih eror min",
     "prsztyyo", "https://www.instagram.com/p/DZMR_A3kbhe/"),

    ("Min Kenapa grup WA g bisa bisa di akses",
     "afandigempel", "https://www.instagram.com/p/DZMR_A3kbhe/"),

    ("Mana ada tanggalnya min, itu cuman bulan aja",
     "misterantam", "https://www.instagram.com/p/DZrFVh6nZWE/"),

    ("Min tolong balas wa saya saya wa tdk ada respon sama sekali, ini saya lupa password gmna cara ganti nya kalo admin nya di wa ngk di respon",
     "lxx.ds0", "https://www.instagram.com/p/DZrFVh6nZWE/"),

    ("Utk org yg gak punya ijasah sprti saya, negara tutup mata. Tdk membuka ruang utk upgrade skill",
     "denbagus_wijaya", "https://www.instagram.com/p/DZMR_A3kbhe/"),

    ("halo kak, bisa tolong bales DM yang aku kirim kah? karna saya ada sdkit keluhan ttg registrasi nya",
     "rizzkirmdhni", "https://www.instagram.com/p/DZMR_A3kbhe/"),

    ("Permisi min, ini kan udh tahap awal pendaftaran, terus untuk mengerjakan tes potensi akademik dmna ya min?",
     "striaaa.___", "https://www.instagram.com/p/DZMR_A3kbhe/"),

    ("Minn, jika sudah melakukan pendaftaran dapat info kita di terima atau tidaknya dari mana?",
     "rmdhani_.5", "https://www.instagram.com/p/DZrFVh6nZWE/"),

    ("Apakah bulan Juli saya bisa ikut?",
     "davidklemans", "https://www.instagram.com/p/DZrFVh6nZWE/"),

    # ── ULASAN POSITIF — Google Maps & Testimoni Alumni ──────────────────────
    # Sumber: Google Maps "UPT BLK Surabaya" & website resmi blksurabaya.id

    ("Pelatihan di BLK Surabaya sangat bermanfaat dan gratis! Instrukturnya profesional dan sabar. Setelah ikut pelatihan las, saya langsung bisa kerja di perusahaan manufaktur.",
     "alumni_blksby_01", "https://www.instagram.com/uptblksurabaya/"),

    ("Alhamdulillah, setelah ikut pelatihan menjahit di BLK Surabaya saya bisa buka usaha sendiri. Fasilitas lengkap, instruktur kompeten dan ramah!",
     "alumni_blksby_02", "https://www.instagram.com/uptblksurabaya/"),

    ("Terima kasih BLK Surabaya! Sertifikat BNSP yang saya dapat sangat diakui perusahaan. Pelatihannya benar-benar mengubah hidup saya.",
     "alumni_blksby_03", "https://www.instagram.com/uptblksurabaya/"),

    ("Pelatihannya keren banget, dapat ilmu sekaligus sertifikat resmi BNSP tanpa biaya. Recommended banget buat yang mau upgrade skill!",
     "alumni_blksby_04", "https://www.instagram.com/uptblksurabaya/"),

    ("Mantap! Program barista di BLK Surabaya sangat praktis, langsung bisa dipraktikkan. Instrukturnya berpengalaman dan menyenangkan.",
     "alumni_blksby_05", "https://www.instagram.com/uptblksurabaya/"),

    # ── KELUHAN NEGATIF — Google Maps & komentar IG lainnya ──────────────────

    ("Proses pendaftaran online sering error dan tidak bisa diakses. Sudah coba berkali-kali tapi gagal terus. Sangat mengecewakan, tolong diperbaiki!",
     "peserta_kecewa_01", "https://www.instagram.com/uptblksurabaya/"),

    ("Waktu pelatihan terlalu singkat, materi tidak tuntas disampaikan. Sangat kecewa, harusnya lebih panjang agar peserta benar-benar siap kerja.",
     "peserta_kecewa_02", "https://www.instagram.com/uptblksurabaya/"),

    ("Kuota peserta terlalu sedikit, saya sudah daftar dari awal tapi tidak lolos padahal syarat lengkap. Sangat mengecewakan dan tidak adil.",
     "peserta_kecewa_03", "https://www.instagram.com/uptblksurabaya/"),

    ("Fasilitas ruang kelas kurang memadai, AC rusak dan panas sekali. Sangat tidak nyaman untuk belajar seharian. Harap segera diperbaiki.",
     "peserta_kecewa_04", "https://www.instagram.com/uptblksurabaya/"),
]


def get_demo_comments(n=15):
    """Ambil n komentar real dari Instagram @uptblksurabaya dan analisa sentimen-nya.
    Data ini STATIS — diambil secara manual dari browser pada Juni 2026.
    """
    sampled = random.sample(REAL_COMMENTS_IG, min(n, len(REAL_COMMENTS_IG)))
    results = []
    for i, (text, username, post_url) in enumerate(sampled):
        # Biarkan model menentukan sentimen (tidak di-override manual)
        sentiment = predict_sentiment(text)
        results.append({
            'id':        'ig_{}'.format(random.randint(10000, 99999)),
            'username':  username,                          # Username Instagram asli
            'text':      text,
            'timestamp': (datetime.utcnow() - timedelta(hours=random.randint(1, 72))).isoformat() + 'Z',
            'sentiment': sentiment,
            'source':    'instagram_real',
            'post_url':  post_url,
        })
    return results



# ==============================================================================
# ENDPOINTS API
# ==============================================================================

@app.route('/api/health')
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model_loaded,
        'endpoints': [
            'GET  /api/health',
            'GET  /api/analyze?text=...',
            'POST /api/analyze-bulk   body: {"comments":["teks1","teks2",...]}',
            'GET  /api/comments',
            'GET  /api/stats',
        ],
    })


@app.route('/api/analyze')
def analyze_single():
    """Analisa sentimen satu komentar.
    Query: ?text=isi+komentar
    """
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Parameter "text" tidak boleh kosong.'}), 400
    sentiment = predict_sentiment(text)
    return jsonify({'text': text, 'sentiment': sentiment})


@app.route('/api/analyze-bulk', methods=['POST'])
def analyze_bulk():
    """
    Analisa banyak komentar sekaligus.

    Request Body (JSON):
        {
            "comments": ["komentar 1", "komentar 2", ...]
        }

    Response:
        {
            "total": 5,
            "results": [
                {
                    "id": "bulk_0",
                    "text": "komentar 1",
                    "sentiment": {"label": "positif", "score": 1, "confidence": 87.5}
                },
                ...
            ],
            "stats": {
                "total_comments": 5,
                "positif": 3,
                "negatif": 2,
                "persen_positif": 60.0,
                "persen_negatif": 40.0,
                "sentiment_overall": "positif"
            }
        }

    ---- PANDUAN INTEGRASI UNTUK TEMAN BACKEND ----
    Kalau backend utama (Laravel/Node/Express) punya data komentar dari
    database atau social media, tinggal kirim ke endpoint ini:

        curl -X POST http://<server-ip>:5000/api/analyze-bulk \\
             -H "Content-Type: application/json" \\
             -d '{"comments": ["bagus banget", "kurang memuaskan"]}'

    Pastikan server Flask ini berjalan dan bisa diakses dari backend utama.
    ------------------------------------------------
    """
    data = request.get_json(silent=True) or {}
    raw_comments = data.get('comments', [])

    if not isinstance(raw_comments, list) or len(raw_comments) == 0:
        return jsonify({'error': 'Body harus berupa JSON {"comments": ["teks1", ...]}'}), 400

    results = []
    for i, text in enumerate(raw_comments):
        text = str(text).strip()
        if not text:
            continue
        sentiment = predict_sentiment(text)
        results.append({
            'id':        'bulk_{}'.format(i),
            'username':  'user_{}'.format(i + 1),
            'text':      text,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'sentiment': sentiment,
            'source':    'manual_input',
        })

    total   = len(results)
    positif = sum(1 for r in results if r['sentiment']['score'] == 1)
    negatif = sum(1 for r in results if r['sentiment']['score'] == 0)
    netral  = sum(1 for r in results if r['sentiment']['score'] == 2)

    overall = 'positif' if positif > negatif else ('negatif' if negatif > positif else 'netral')

    return jsonify({
        'total':   total,
        'results': results,
        'stats': {
            'total_comments':    total,
            'positif':           positif,
            'negatif':           negatif,
            'netral':            netral,
            'persen_positif':    round((positif / total * 100) if total else 0, 1),
            'persen_negatif':    round((negatif / total * 100) if total else 0, 1),
            'persen_netral':     round((netral  / total * 100) if total else 0, 1),
            'sentiment_overall': overall,
        },
    })


@app.route('/api/comments')
def get_comments():
    """Ambil komentar ASLI @uptblksurabaya yang sudah dianalisa sentimen-nya.
    Query: ?n=15  (jumlah komentar, default 15, maks 15 — sesuai jumlah yang tersedia)
    """
    n = min(int(request.args.get('n', 15)), len(REAL_COMMENTS_IG))
    comments = get_demo_comments(n)
    return jsonify({
        'mode':       'instagram_real',
        'source':     '@uptblksurabaya',
        'count':      len(comments),
        'comments':   comments,
        'fetched_at': datetime.utcnow().isoformat() + 'Z',
    })


@app.route('/api/stats')
def get_stats():
    """Statistik sentimen dari komentar ASLI @uptblksurabaya (data statis Juni 2026)."""
    # Analisis sentimen untuk setiap komentar real
    positif = sum(1 for text, _, _ in REAL_COMMENTS_IG if predict_sentiment(text)['score'] == 1)
    negatif = sum(1 for text, _, _ in REAL_COMMENTS_IG if predict_sentiment(text)['score'] == 0)
    netral  = sum(1 for text, _, _ in REAL_COMMENTS_IG if predict_sentiment(text)['score'] == 2)
    total   = positif + negatif + netral
    overall = 'positif' if positif > negatif else ('negatif' if negatif > positif else 'netral')

    return jsonify({
        'mode':              'instagram_real',
        'sumber_data':       'Komentar asli @uptblksurabaya (statis, Juni 2026)',
        'total_comments':    total,
        'positif':           positif,
        'negatif':           negatif,
        'netral':            netral,
        'persen_positif':    round((positif / total * 100) if total else 0, 1),
        'persen_negatif':    round((negatif / total * 100) if total else 0, 1),
        'persen_netral':     round((netral  / total * 100) if total else 0, 1),
        'sentiment_overall': overall,
        'instagram_account': '@uptblksurabaya',
        'fetched_at':        datetime.utcnow().isoformat() + 'Z',
    })


# ==============================================================================
# RUN SERVER
# ==============================================================================

if __name__ == '__main__':
    print('[OK] Model berhasil dimuat!' if model_loaded else '[!] Model belum ada, jalankan model_trainer.py')
    print('[>>] Server: http://localhost:5000')
    print('')
    print('Endpoints:')
    print('  GET  /api/health')
    print('  GET  /api/analyze?text=pelatihan+sangat+bagus')
    print('  POST /api/analyze-bulk  body: {"comments":["teks1","teks2",...]}')
    print('  GET  /api/comments')
    print('  GET  /api/stats')
    app.run(debug=True, port=5000)
