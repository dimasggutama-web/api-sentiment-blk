# model_trainer.py  -  Training Sentiment Analysis 3 Kelas (Bahasa Indonesia)
#
# Label:
#   0 = Negatif
#   1 = Positif
#   2 = Netral
#
# Dataset: dibuat khusus untuk konteks BLK / pelatihan kerja (Bahasa Indonesia)
# Model:   Logistic Regression + TF-IDF (lebih akurat dari Naive Bayes untuk B.Indonesia)

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ==============================================================================
# DATASET BAHASA INDONESIA (BLK / PELATIHAN KONTEKS)
# Format: (teks, label)   0=negatif  1=positif  2=netral
# ==============================================================================

DATA = [
    # ========== POSITIF (label=1) ==========
    # Kata tunggal / pendek positif
    ("bagus", 1),
    ("mantap", 1),
    ("keren", 1),
    ("memuaskan", 1),
    ("bermanfaat", 1),
    ("profesional", 1),
    ("berkualitas", 1),
    ("luar biasa", 1),
    ("recommended", 1),
    ("sangat bagus", 1),
    ("sangat baik", 1),
    ("sangat membantu", 1),
    ("sangat bermanfaat", 1),
    ("top banget", 1),
    ("keren banget", 1),
    ("bagus banget", 1),
    ("mantap banget", 1),
    ("puas banget", 1),
    ("bagus sekali", 1),
    ("baik sekali", 1),
    ("membantu sekali", 1),
    ("memuaskan sekali", 1),
    ("luar biasa sekali", 1),
    ("sangat puas", 1),
    ("puas", 1),
    ("memuaskan", 1),
    ("helpful", 1),
    ("bermanfaat", 1),
    ("berguna", 1),
    ("membantu", 1),
    ("baik", 1),
    ("suka", 1),
    ("senang", 1),
    ("berhasil", 1),
    ("sukses", 1),
    ("hebat", 1),
    ("istimewa", 1),
    ("memuaskan", 1),

    # Kalimat positif tentang pelatihan
    ("Pelatihannya sangat bermanfaat dan instrukturnya profesional sekali", 1),
    ("Instruktur sangat sabar dan jelas dalam menjelaskan materi pelatihan", 1),
    ("Pelatihan di BLK Surabaya sangat membantu saya mendapatkan pekerjaan baru", 1),
    ("Fasilitas lengkap dan modern, sangat mendukung proses belajar", 1),
    ("Terima kasih BLK Surabaya, saya sudah bisa kerja berkat pelatihan ini", 1),
    ("Program pelatihannya sangat bagus dan terstruktur dengan baik", 1),
    ("Sertifikat BNSP dari BLK sangat diakui perusahaan, terima kasih", 1),
    ("Alat dan mesin pelatihan sudah modern dan lengkap", 1),
    ("Instruktur pelatihan las sangat berpengalaman dan sabar", 1),
    ("Setelah ikut pelatihan barista di BLK langsung bisa kerja di kafe", 1),
    ("Pelatihan menjahit di BLK sangat bagus, sekarang saya buka usaha sendiri", 1),
    ("BLK Surabaya sangat membantu masyarakat yang ingin meningkatkan keterampilan", 1),
    ("Proses pendaftaran mudah dan petugasnya sangat ramah", 1),
    ("Materi pelatihan sangat relevan dengan kebutuhan industri saat ini", 1),
    ("Sangat puas dengan program pelatihan komputer di BLK", 1),
    ("Instruktur sangat kompeten dan berdedikasi tinggi", 1),
    ("Pelatihan gratis tapi kualitasnya setara dengan kursus berbayar", 1),
    ("Saya sangat merekomendasikan BLK Surabaya untuk yang ingin belajar keterampilan", 1),
    ("Fasilitasnya bersih dan nyaman untuk belajar", 1),
    ("Program pelatihan refrigerasi sangat praktis dan langsung bisa diterapkan", 1),
    ("Terima kasih BLK, pelatihan ini sangat mengubah hidup saya", 1),
    ("Instruktur TIK sangat sabar membimbing dari nol hingga bisa", 1),
    ("Pelatihan batik di sini sangat menyenangkan dan berkualitas tinggi", 1),
    ("Senang bisa dapat sertifikat BNSP melalui uji kompetensi yang transparan", 1),
    ("BLK Surabaya terbaik, sangat membantu meningkatkan keahlian kerja", 1),
    ("Pelatihan administrasi perkantoran sangat relevan dengan kebutuhan kantor", 1),
    ("Kualitas pengajaran sangat baik dan materi mudah dipahami", 1),
    ("Sangat puas dengan hasilnya, sekarang sudah bekerja di perusahaan", 1),
    ("Pelatihan otomotif di BLK sangat komprehensif dan praktikal", 1),
    ("Alhamdulillah berkat BLK saya sekarang punya usaha sendiri", 1),
    ("Instrukturnya berpengalaman dan sangat membantu peserta yang kesulitan", 1),
    ("Fasilitas lab komputer sangat lengkap dan mendukung pembelajaran", 1),
    ("Pelatihan di sini gratis tapi kualitasnya luar biasa bagus", 1),
    ("Sangat berguna, ilmu yang didapat langsung bisa dipraktikkan di dunia kerja", 1),
    ("BLK Surabaya membantu banyak warga untuk mendapatkan pekerjaan layak", 1),
    ("Pelatihan kecantikan di BLK sangat profesional dan up to date", 1),
    ("Semua instruktur sangat berdedikasi dan kompeten di bidangnya", 1),
    ("Program pelatihan ini sangat membantu saya naik jabatan di tempat kerja", 1),
    ("Saya sangat puas dengan fasilitas dan kualitas pelatihan di BLK Surabaya", 1),
    ("Instrukturnya baik hati, sabar, dan mudah diajak komunikasi", 1),
    ("Pelatihan las MIG di BLK top banget, langsung bisa kerja di pabrik", 1),
    ("Terimakasih BLK telah memberi kesempatan belajar bagi kami", 1),
    ("Program pelatihan sangat membantu pengembangan diri dan karir saya", 1),
    ("Fasilitas uji kompetensi sangat standar dan profesional", 1),
    ("Sertifikat kompetensi dari BLK sangat berguna untuk melamar kerja", 1),
    ("Pelatihan di BLK membantu saya menemukan passion di bidang kuliner", 1),
    ("Instruktur sangat komunikatif dan selalu siap membantu peserta", 1),
    ("Kelas pelatihan menjahit sangat menyenangkan dan ilmunya sangat berguna", 1),
    ("BLK Surabaya luar biasa, programnya sangat membantu masyarakat", 1),
    ("Sangat terkesan dengan kualitas pelatihan dan profesionalisme instruktur", 1),
    ("Pelatihan pemrograman di BLK membuka jalan karir saya di bidang IT", 1),
    ("Mantap banget program BLK, gratis tapi berkualitas tinggi", 1),
    ("Sangat merekomendasikan pelatihan barista di BLK untuk yang mau kerja di kafe", 1),
    ("Instruktur sabar sekali mengajari dari awal hingga mahir", 1),
    ("Fasilitas bengkel otomotif sangat lengkap dan terawat dengan baik", 1),
    ("Berkat BLK saya sekarang punya kemampuan las yang diakui perusahaan", 1),
    ("Pelatihan TIK di BLK sangat relevan dan langsung bisa diterapkan", 1),
    ("BLK Surabaya memberikan kesempatan yang luar biasa bagi masyarakat", 1),
    ("Saya sangat senang bisa ikut pelatihan di BLK dan hasilnya memuaskan", 1),
    ("Instruktur pelatihan kecantikan sangat berpengalaman dan inovatif", 1),
    ("Terima kasih BLK telah membantu saya memulai usaha jahit yang sukses", 1),
    ("Materi pelatihan sangat update dan sesuai perkembangan industri", 1),
    ("BLK Surabaya sangat bagus, programnya terstruktur dan mudah diikuti", 1),
    ("Fasilitas yang disediakan sangat memadai untuk kebutuhan pembelajaran", 1),
    ("Pelatihan barista di BLK sangat berkualitas, ilmunya bisa langsung dipraktikkan", 1),

    # ========== NEGATIF (label=0) ==========
    # Kata tunggal / pendek negatif
    ("jelek", 0),
    ("buruk", 0),
    ("kecewa", 0),
    ("mengecewakan", 0),
    ("tidak bagus", 0),
    ("tidak memuaskan", 0),
    ("tidak puas", 0),
    ("tidak bermanfaat", 0),
    ("payah", 0),
    ("sangat kecewa", 0),
    ("sangat buruk", 0),
    ("sangat mengecewakan", 0),
    ("kurang bagus", 0),
    ("kurang memuaskan", 0),
    ("tidak jelas", 0),
    ("tidak profesional", 0),
    ("buruk sekali", 0),
    ("sangat jelek", 0),
    ("parah", 0),
    ("gagal", 0),
    ("kacau", 0),
    ("hancur", 0),
    ("mengecewakan", 0),
    ("berantakan", 0),
    ("tidak berguna", 0),
    ("tidak membantu", 0),
    ("tidak baik", 0),
    ("tidak suka", 0),
    ("kapok", 0),
    ("menyesal", 0),
    ("rugi", 0),
    ("percuma", 0),

    # Kalimat negatif tentang pelatihan
    ("Waktu pelatihan terlalu singkat dan materi tidak tuntas disampaikan", 0),
    ("Instruktur sering tidak hadir tanpa pemberitahuan yang jelas", 0),
    ("Fasilitas sangat kurang dan banyak peralatan yang rusak", 0),
    ("Jadwal ujian sering berubah mendadak tanpa informasi yang jelas", 0),
    ("Pendaftaran sangat ribet dan prosesnya tidak transparan", 0),
    ("Kuota peserta terlalu sedikit sehingga banyak yang tidak bisa ikut", 0),
    ("Pelatihan ini sangat mengecewakan, tidak sesuai ekspektasi saya", 0),
    ("Alat dan mesin pelatihan sudah tua dan tidak layak pakai", 0),
    ("AC ruangan sering mati, panas sekali saat belajar", 0),
    ("Modul pelatihan sangat ketinggalan zaman dan tidak relevan", 0),
    ("Instruktur kurang kompeten dan penjelasannya membingungkan", 0),
    ("Pelayanan administrasi sangat lambat dan tidak ramah", 0),
    ("Fasilitas kamar mandi sangat kotor dan tidak terawat", 0),
    ("Parkir sangat sempit dan tidak aman untuk kendaraan peserta", 0),
    ("Website BLK sering error dan informasi tidak update", 0),
    ("Sangat kecewa dengan kualitas pelatihan yang jauh dari harapan", 0),
    ("Materi terlalu berat dan instruktur tidak sabar mengajari peserta", 0),
    ("Proses pendaftaran online sering bermasalah dan susah diakses", 0),
    ("Jadwal pelatihan terlalu padat sehingga peserta kelelahan", 0),
    ("Informasi pelatihan tidak jelas dan sulit didapatkan", 0),
    ("Sangat tidak puas dengan fasilitas dan pelayanan BLK", 0),
    ("Instruktur tidak menguasai materi dengan baik", 0),
    ("Pelatihan tidak bermanfaat dan membuang waktu saya", 0),
    ("Fasilitas jauh dari standar, alat banyak yang tidak berfungsi", 0),
    ("Pengajaran sangat membosankan dan tidak interaktif", 0),
    ("Kurikulum pelatihan tidak sesuai dengan kebutuhan industri sekarang", 0),
    ("Sangat kecewa, harapannya tinggi tapi kenyataan sangat buruk", 0),
    ("Pelayanan petugasnya tidak ramah dan tidak profesional", 0),
    ("Pelatihan terlalu singkat tapi biayanya terlalu mahal", 0),
    ("Alat peraga sudah rusak dan tidak diperbaiki-perbaiki", 0),
    ("Sangat tidak merekomendasikan BLK ini karena kualitasnya buruk", 0),
    ("Instruktur jarang masuk kelas tanpa alasan yang jelas", 0),
    ("Sistem pendaftaran sangat kacau dan membingungkan peserta", 0),
    ("Fasilitas komputer sudah sangat tua dan sering hang", 0),
    ("Materi pelatihan tidak update dan tidak relevan dengan industri", 0),
    ("Pelayanan sangat mengecewakan, tidak profesional sama sekali", 0),
    ("Tidak puas dengan hasil pelatihan, tidak ada dampak nyata untuk karir", 0),
    ("Ruang kelas sangat panas dan tidak nyaman untuk belajar", 0),
    ("Instruktur sering terlambat dan tidak menghargai waktu peserta", 0),
    ("Banyak fasilitas yang tidak berfungsi dan tidak diperbaiki", 0),
    ("Proses uji kompetensi tidak transparan dan tidak adil", 0),
    ("Sangat tidak puas, biaya mahal tapi kualitas sangat buruk", 0),
    ("Instruktur tidak sabar dan sering membentak peserta", 0),
    ("Pelatihan tidak terstruktur dan membingungkan peserta", 0),
    ("Fasilitas sangat tidak memadai untuk kebutuhan pelatihan", 0),
    ("Materi terlalu teoritis dan tidak ada praktek yang cukup", 0),
    ("Sangat kecewa dengan standar pengajaran yang sangat rendah", 0),
    ("Pelayanan administrasi sangat buruk dan tidak membantu peserta", 0),
    ("Sertifikat tidak diakui perusahaan manapun, percuma ikut pelatihan ini", 0),
    ("Pelatihan ini buang-buang waktu dan tidak ada manfaatnya", 0),
    ("Instruktur tidak kompeten dan tidak mampu menjawab pertanyaan peserta", 0),
    ("Sangat tidak nyaman, fasilitas kotor dan tidak terawat", 0),
    ("Jadwal sering berubah mendadak tanpa pemberitahuan sebelumnya", 0),
    ("Program pelatihan sangat tidak terorganisir dan kacau", 0),
    ("Kualitas pengajaran sangat buruk dan tidak profesional", 0),
    ("Tidak ada improvement setelah ikut pelatihan ini sama sekali", 0),

    # ========== NETRAL (label=2) ==========
    # Pertanyaan
    ("Kapan pendaftaran pelatihan dibuka?", 2),
    ("Dimana lokasi BLK Surabaya?", 2),
    ("Berapa biaya pelatihan di BLK?", 2),
    ("Ada pelatihan apa saja di BLK Surabaya?", 2),
    ("Bagaimana cara daftar pelatihan di BLK?", 2),
    ("Apakah ada pelatihan untuk ibu rumah tangga?", 2),
    ("Apa saja syarat pendaftaran pelatihan BLK?", 2),
    ("Berapa lama durasi pelatihan las?", 2),
    ("Kapan jadwal ujian kompetensi bulan ini?", 2),
    ("Ada pelatihan barista tidak di BLK Surabaya?", 2),
    ("Bagaimana cara mendapatkan sertifikat BNSP?", 2),
    ("Pelatihan komputer mulai jam berapa?", 2),
    ("Apakah ada kuota untuk pelatihan menjahit?", 2),
    ("Tolong info jadwal pelatihan bulan depan", 2),
    ("Mohon informasi tentang program pelatihan gratis", 2),
    ("Saya ingin mendaftar pelatihan, apa yang harus dilakukan?", 2),
    ("Ada program pelatihan untuk lulusan SMA tidak?", 2),
    ("Berapa orang yang bisa ikut satu angkatan pelatihan?", 2),
    ("Apakah pelatihan bisa diikuti dari luar Surabaya?", 2),
    ("Kapan pelatihan batch berikutnya dimulai?", 2),
    ("Mohon info lokasi TUK untuk uji kompetensi", 2),
    ("Bagaimana alur pendaftaran uji kompetensi BNSP?", 2),
    ("Apakah ada pelatihan online di BLK Surabaya?", 2),
    ("Saya mau tanya informasi program pelatihan 2025", 2),
    ("Tolong info nomor kontak yang bisa dihubungi", 2),
    ("Apa persyaratan untuk ikut pelatihan kecantikan?", 2),
    ("Apakah ada dorm atau mess untuk peserta pelatihan dari luar kota?", 2),
    ("Kapan BLK Surabaya buka untuk konsultasi?", 2),
    ("Pelatihan refrigerasi diadakan berapa kali setahun?", 2),
    ("Apakah pelatihan ini bersertifikat?", 2),

    # Pernyataan netral / faktual
    ("BLK Surabaya terletak di Jl Dukuh Menanggal", 2),
    ("Program pelatihan yang ada antara lain las, menjahit, komputer, dan barista", 2),
    ("Pendaftaran pelatihan bisa dilakukan secara online maupun offline", 2),
    ("Sertifikat yang dikeluarkan adalah sertifikat BNSP yang diakui secara nasional", 2),
    ("BLK Surabaya menyediakan berbagai program pelatihan gratis untuk masyarakat", 2),
    ("Pelatihan dilaksanakan setiap hari Senin sampai Jumat", 2),
    ("Kapasitas peserta per angkatan biasanya 16 orang", 2),
    ("Uji kompetensi dilakukan setelah pelatihan selesai", 2),
    ("Saya baru tahu tentang BLK Surabaya dari teman", 2),
    ("Sedang cari informasi tentang program pelatihan las di BLK", 2),
    ("Mau tanya tentang pelatihan TIK untuk pemula", 2),
    ("Pelatihan berlangsung selama 30 hari kerja", 2),
    ("Lokasi BLK mudah dijangkau dengan transportasi umum", 2),
    ("Program pelatihan gratis dibiayai oleh pemerintah daerah", 2),
    ("Ada beberapa pilihan program pelatihan yang bisa dipilih", 2),
    ("Saya baca informasi pelatihan dari website BLK Surabaya", 2),
    ("Program pelatihan dibuka dua kali dalam setahun", 2),
    ("Peserta yang lulus uji kompetensi akan mendapat sertifikat BNSP", 2),
    ("Pelatihan menjahit tersedia untuk laki-laki maupun perempuan", 2),
    ("BLK bekerja sama dengan beberapa perusahaan untuk penyaluran kerja", 2),
    ("Materi pelatihan mencakup teori dan praktik lapangan", 2),
    ("Ada kelas pagi dan kelas siang untuk beberapa program pelatihan", 2),
    ("Saya sedang mempertimbangkan untuk ikut pelatihan di BLK Surabaya", 2),
    ("Mencari info lebih lanjut tentang pelatihan barista", 2),
    ("Teman saya sudah ikut pelatihan di BLK dan sekarang bekerja", 2),
    ("Ingin tahu lebih banyak tentang uji kompetensi BNSP", 2),
    ("Pendaftaran bisa dilakukan lewat website atau datang langsung", 2),
    ("Instruktur BLK adalah tenaga ahli bersertifikat", 2),
    ("Program pelatihan otomotif tersedia untuk kendaraan roda dua dan empat", 2),
    ("BLK Surabaya adalah unit pelaksana teknis Disnakertrans Jatim", 2),
    ("Mau lihat-lihat dulu program apa yang tersedia di BLK", 2),
    ("Baru dengar tentang BLK Surabaya, ingin tahu lebih lanjut", 2),
    ("Apakah sudah ada testimoni dari alumni BLK Surabaya?", 2),
    ("Saya belum pernah ikut pelatihan, ini pertama kalinya saya mendaftar", 2),
    ("Kalau tidak lolos seleksi apakah bisa daftar lagi?", 2),
]

# ==============================================================================
# PREPROCESSING (Khusus Bahasa Indonesia)
# ==============================================================================

STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'adalah', 'ada', 'tidak', 'ya', 'iya', 'juga', 'bisa', 'saya', 'kami',
    'kamu', 'mereka', 'pada', 'dalam', 'atau', 'karena', 'sudah', 'akan',
    'bagi', 'jika', 'maka', 'namun', 'tetapi', 'tapi', 'oleh', 'kalau',
    'kalian', 'kita', 'anda', 'pak', 'bu', 'mas', 'mbak', 'bang', 'kak',
    'semua', 'lagi', 'pun', 'aja', 'sih', 'deh', 'dong', 'lah', 'nah',
    'wah', 'kok', 'emang', 'memang', 'udah', 'hehe', 'haha', 'nih', 'tuh',
    'gue', 'gw', 'lu', 'lo', 'nya', 'nya', 'ini', 'itu', 'si', 'pun',
    'telah', 'sedang', 'serta', 'bahwa', 'bahkan', 'namun', 'meski', 'walau',
    'selain', 'seperti', 'setelah', 'sebelum', 'antara', 'hingga', 'sampai',
    'lebih', 'kurang', 'paling', 'agar', 'supaya', 'sehingga', 'jadi',
    'menjadi', 'sebagai', 'bukan', 'belum', 'pernah', 'selalu', 'sering',
}

def preprocess_id(text):
    """Preprocessing untuk Bahasa Indonesia."""
    import re
    original = str(text).lower().strip()
    cleaned  = re.sub(r'[^a-z\s]', ' ', original)
    tokens   = cleaned.split()

    # Filter stopwords
    filtered = [w for w in tokens if w not in STOPWORDS_ID and len(w) > 1]

    # Jika tidak ada token tersisa (teks terlalu pendek atau semua stopwords),
    # kembalikan semua token tanpa filter agar model tetap bisa prediksi
    if not filtered:
        filtered = [w for w in tokens if len(w) > 1] or tokens

    return ' '.join(filtered)


# ==============================================================================
# TRAINING
# ==============================================================================

def main():
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 55)
    print("  TRAINING MODEL SENTIMEN BAHASA INDONESIA (3 Kelas)")
    print("=" * 55)
    print(f"Total data: {len(DATA)}")

    texts  = [preprocess_id(d[0]) for d in DATA]
    labels = [d[1] for d in DATA]

    label_names = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
    for lb, name in label_names.items():
        count = labels.count(lb)
        print(f"  {name:10}: {count} contoh")

    print()

    # TF-IDF Vectorizer — dioptimasi untuk Bahasa Indonesia
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),    # unigram + bigram + trigram
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        analyzer='word',
    )
    X = vectorizer.fit_transform(texts)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Model: Logistic Regression (lebih baik dari Naive Bayes untuk B.Indonesia)
    model = LogisticRegression(
        C=5.0,
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print("Hasil Evaluasi (Test Set):")
    print("-" * 45)
    print(classification_report(
        y_test, y_pred,
        target_names=['Negatif', 'Positif', 'Netral'],
        zero_division=0,
    ))

    # Cross-validation
    cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print()

    # Test kata/frasa manual
    test_cases = [
        ("bagus", 1),
        ("jelek", 0),
        ("mantap sekali", 1),
        ("sangat kecewa", 0),
        ("kapan pendaftaran", 2),
        ("tidak puas sama sekali", 0),
        ("bermanfaat banget", 1),
        ("dimana lokasi blk", 2),
        ("mengecewakan sekali", 0),
        ("pelatihannya sangat membantu saya", 1),
        ("instrukturnya profesional", 1),
        ("fasilitasnya kurang memadai", 0),
    ]
    print("Test Manual:")
    print("-" * 45)
    all_correct = 0
    for text, expected in test_cases:
        processed = preprocess_id(text)
        vec   = vectorizer.transform([processed])
        pred  = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        conf  = round(max(proba) * 100, 1)
        status = "✓" if pred == expected else "✗"
        print(f"  {status} '{text}' → {label_names[pred]} ({conf}%)")
        if pred == expected:
            all_correct += 1
    print(f"\nAkurasi manual: {all_correct}/{len(test_cases)}")

    # Simpan model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE_DIR, 'model')
    os.makedirs(model_dir, exist_ok=True)

    clf_path  = os.path.join(model_dir, 'classifier.pkl')
    vect_path = os.path.join(model_dir, 'vectorizer.pkl')

    with open(clf_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vect_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n[OK] Model disimpan di {model_dir}")
    print("  classifier.pkl  + vectorizer.pkl")
    print()
    print("Sekarang restart api.py untuk menggunakan model baru.")
    print("=" * 55)


if __name__ == '__main__':
    main()
