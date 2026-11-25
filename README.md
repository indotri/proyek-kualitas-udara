# Proyek Akhir Big Data: Klasifikasi Kualitas Udara (Air Quality Classification)

Proyek pipeline Big Data end-to-end untuk pemrosesan, analisis eksploratif, pemodelan Machine Learning (*Linear Support Vector Classifier / LinearSVC*), dan penyimpanan NoSQL menggunakan ekosistem terdistribusi modern: **Hadoop HDFS 3.2.1**, **Apache Spark 3.5.1**, **Spark MLlib**, dan **MongoDB 6.0** yang dikemas dalam **Docker Compose**.

---

## 1. Arsitektur Sistem & Alur Pipeline

```
+-----------------------------------------------------------------------------------+
|                                 DOCKER NETWORK                                    |
|                                                                                   |
|  +--------------------+         +-----------------------------------------------+ |
|  | Local CSV Dataset  |         | Hadoop Distributed File System (HDFS 3.2.1)   | |
|  | ./data/indonesia...| ------> | - namenode:9000 (/data/kualitas_udara/)       | |
|  +--------------------+  (put)  | - datanode:9870 (Web UI)                      | |
|                                 +-----------------------------------------------+ |
|                                                        |                          |
|                                                        v (Distributed Read)       |
|                                 +-----------------------------------------------+ |
|                                 | Apache Spark 3.5.1 Container                  | |
|                                 | [1] Ingestion & Schema Inspection             | |
|                                 | [2] Preprocessing & Binary Labeling           | |
|                                 | [3] Spark SQL / DataFrame API Analytics       | |
|                                 | [4] Spark MLlib: VectorAssembler + Scaler     | |
|                                 |     + LinearSVC (Classification & Scaling Exp)| |
|                                 +-----------------------------------------------+ |
|                                          |                                |       |
|                             (Write Analisis)                  (Write Prediksi)    |
|                                          |                                |       |
|                                          v                                v       |
|                                 +-----------------------------------------------+ |
|                                 | MongoDB 6.0 Database: db_kualitas_udara       | |
|                                 | - Collection: ringkasan_analisis              | |
|                                 | - Collection: hasil_prediksi                  | |
|                                 +-----------------------------------------------+ |
|                                                        |                          |
|                                                        v (Verification Read)      |
|                                         Spark baca kembali 5 sampel data          |
+-----------------------------------------------------------------------------------+
```

---

## 2. Struktur Direktori Proyek

```
proyek-kualitas-udara/
├── docker-compose.yml       # Konfigurasi multi-container Docker (HDFS, Spark, MongoDB)
├── data/                    # Volume lokal untuk dataset mentah
│   └── indonesia_kualitas_udara_ispu_enriched.csv
├── app/                     # Skrip aplikasi PySpark
│   └── pipeline.py          # Pipeline pemrosesan data, MLlib, dan integrasi MongoDB
└── README.md                # Dokumentasi & panduan eksekusi lengkap
```

---

## 3. Prasyarat Sistem

- **Docker Desktop** (dengan engine Docker dan Docker Compose aktif).
- RAM minimal 6 GB dialokasikan untuk Docker engine (disarankan 8 GB ke atas agar kontainer HDFS, Spark, dan MongoDB berjalan lancar).
- Ruang disk kosong minimal 10 GB.

---

## 4. Panduan Eksekusi Lengkap (Langkah demi Langkah)

### Langkah 1: Jalankan Ekosistem Container Docker

Buka terminal (PowerShell atau Bash) pada direktori root proyek ini, kemudian jalankan:

```powershell
docker compose up -d
```

Pastikan seluruh container telah berjalan dengan status `Up`:

```powershell
docker compose ps
```

*Daftar Port Layanan:*
- **HDFS NameNode Web UI**: [http://localhost:9870](http://localhost:9870)
- **HDFS IPC Port**: `localhost:9000`
- **MongoDB**: `localhost:27017`

---

### Langkah 2: Buat Direktori di HDFS dan Unggah Dataset

Mounting folder `./data` langsung terhubung ke `/data` di dalam container `namenode`. Jalankan perintah berikut untuk membuat direktori HDFS `/data/kualitas_udara` dan mengunggah dataset CSV:

```powershell
# Buat direktori di HDFS
docker exec -it namenode hdfs dfs -mkdir -p /data/kualitas_udara

# Unggah file CSV dari folder mount /data ke HDFS
docker exec -it namenode hdfs dfs -put /data/indonesia_kualitas_udara_ispu_enriched.csv /data/kualitas_udara/

# Verifikasi isi direktori HDFS
docker exec -it namenode hdfs dfs -ls -h /data/kualitas_udara
```

> **Verifikasi Visual:** Anda juga dapat membuka browser di [http://localhost:9870/explorer.html#/data/kualitas_udara](http://localhost:9870/explorer.html#/data/kualitas_udara) untuk melihat file yang berhasil diunggah.

---

### Langkah 3: Eksekusi Pipeline PySpark

Pasang library `numpy` (dibutuhkan oleh PySpark MLlib) dan jalankan skrip PySpark di dalam container `spark` menggunakan `spark-submit`. Paket `mongo-spark-connector` akan diunduh secara otomatis ke direktori cache `/tmp/.ivy2`:

```powershell
# Pastikan dependensi numpy terpasang di container spark
docker exec -it spark pip install numpy

# Jalankan pipeline PySpark
docker exec -it spark spark-submit \
  --master local[*] \
  --conf spark.jars.ivy=/tmp/.ivy2 \
  --packages org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 \
  /app/pipeline.py
```

---

## 5. Ringkasan Fitur & Tahapan Pemrosesan pada Skrip

1. **Ingestion dari HDFS**: Membaca dataset CSV berukuran 10.000+ baris langsung dari kluster terdistribusi HDFS (`hdfs://namenode:9000/data/kualitas_udara/...`).
2. **Pembersihan & Preprocessing**:
   - Menghapus duplikasi baris data (`dropDuplicates`).
   - Melakukan casting tipe data ke `DoubleType` untuk seluruh kolom numerik polutan dan meteorologi.
   - Menangani missing values pada parameter polutan utama.
   - Membuat label biner:
     - `1.0` (*Tidak Sehat*) jika konsentrasi **PM2.5 > 35.0 µg/m³** (standar baku mutu WHO / ISPU).
     - `0.0` (*Aman*) jika konsentrasi PM2.5 ≤ 35.0 µg/m³.
3. **Analisis Eksploratif (Spark SQL & DataFrame API)**:
   - **Analisis 1**: Rata-rata dan nilai puncak (maksimum) polutan (PM2.5, PM10, SO2, NO2, CO, O3) per stasiun pemantau / lokasi.
   - **Analisis 2**: Distribusi status kualitas udara (Aman vs Tidak Sehat) di berbagai provinsi.
   - **Analisis 3**: Karakteristik rata-rata polutan lain (PM10, SO2, NO2, CO, O3) pada kelompok kualitas udara Aman vs Tidak Sehat.
4. **Implementasi Spark MLlib**:
   - **Pencegahan Data Leakage**: Parameter pembentuk label (`pm25_concentration_ugm3`) dan indeks turunan (`ispu`) **dikeluarkan** dari fitur model. Fitur model murni menggunakan polutan lain: `PM10`, `SO2`, `NO2`, `CO`, dan `O3`.
   - **Feature Engineering**: `VectorAssembler` menggabungkan fitur polutan, dilanjutkan `StandardScaler` untuk normalisasi distribusi nilai.
   - **Klasifikasi**: Model **LinearSVC** (*Linear Support Vector Classifier*) dengan pembagian 80% data latih (*training*) dan 20% data uji (*testing*).
   - **Evaluasi Metrik**: Menghitung **Accuracy**, **Precision**, **Recall**, dan **F1-Score**.
   - **Eksperimen Skalabilitas**: Mengukur performa akurasi dan efisiensi waktu komputasi (*execution time*) pada skala data 25%, 50%, dan 100%.
5. **Penyimpanan & Pembacaan NoSQL (MongoDB)**:
   - Menyimpan agregasi ke koleksi `ringkasan_analisis`.
   - Menyimpan sampel hasil prediksi lengkap dengan metadata (`model="Linear SVC"`, `sumber="Spark MLlib"`) ke koleksi `hasil_prediksi`.
   - Melakukan pembacaan kembali (*read verification*) 5 dokumen teratas dari MongoDB ke terminal.

---

## 6. Verifikasi Data di MongoDB

Selain dari pembacaan otomatis skrip PySpark, Anda juga dapat mengecek isi koleksi MongoDB langsung menggunakan `mongosh`:

```powershell
# Masuk ke shell MongoDB
docker exec -it mongodb mongosh

# Di dalam mongosh, jalankan:
use db_kualitas_udara
show collections
db.ringkasan_analisis.find().pretty()
db.hasil_prediksi.find({}, {lokasi_stasiun: 1, kategori_prediksi: 1, model: 1, _id: 0}).limit(5).pretty()
exit
```

---

## 7. Menjalankan Portal Web Interaktif (Gaya BMKG + Peta Indonesia + Gemini API)

Proyek ini dilengkapi antarmuka web portal bergaya BMKG pada direktori `web/` yang menampilkan:
- **Peta Interaktif Indonesia (23 Provinsi)** lengkap dengan filter status dan zoom per pulau.
- **Perkiraan Kualitas Udara Wilayah** berbasis model `LinearSVC` Spark MLlib & data meteorologi dataset.
- **Perbandingan Kualitas Udara Hari Ini (Gemini API)**: Mengambil kondisi hari ini dan membandingkannya secara langsung terhadap data historis & ambang batas program (`PM2.5 > 35.0 µg/m³ = Tidak Sehat`).

Untuk menjalankan website di `http://localhost:8050`:

```powershell
# (Opsional) Salin .env.example menjadi .env lalu isi GEMINI_API_KEY
# atau masukkan API key langsung melalui kolom input di halaman web.
uvicorn web.server:app --host 127.0.0.1 --port 8050
```

Kemudian buka browser ke **[http://localhost:8050](http://localhost:8050)**.

---

## 8. Menghentikan Layanan

Jika eksperimen telah selesai dan Anda ingin mematikan semua container:

```powershell
docker compose down
```

Untuk mematikan sekaligus menghapus seluruh volume penyimpanan persistent (reset total):

```powershell
docker compose down -v
```
