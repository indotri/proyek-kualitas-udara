"""
================================================================================
PROYEK AKHIR BIG DATA: KLASIFIKASI KUALITAS UDARA
Teknologi: Apache Spark 3.5.1, Spark MLlib, Hadoop HDFS, MongoDB 6.0
================================================================================
"""

import time
import sys
import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, count, avg, max, min, round, desc, current_timestamp
)
from pyspark.sql.types import (
    DoubleType, StringType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80 + "\n")

def main():
    # --------------------------------------------------------------------------
    # 0. INISIALISASI SPARK SESSION
    # --------------------------------------------------------------------------
    print_banner("[0] INISIALISASI SPARK SESSION")
    
    spark = SparkSession.builder \
        .appName("Klasifikasi-Kualitas-Udara-SparkMLlib") \
        .config("spark.mongodb.read.connection.uri", "mongodb://mongodb:27017/db_kualitas_udara.hasil_prediksi") \
        .config("spark.mongodb.write.connection.uri", "mongodb://mongodb:27017/db_kualitas_udara.hasil_prediksi") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    # Minimalisir log verbose agar output terminal bersih dan rapi untuk laporan
    spark.sparkContext.setLogLevel("WARN")
    print(">> SparkSession berhasil dibuat!")
    print(f">> Spark Version : {spark.version}")
    print(f">> Spark Master  : {spark.sparkContext.master}")

    # --------------------------------------------------------------------------
    # 1. MEMBACA DATASET DARI HADOOP HDFS
    # --------------------------------------------------------------------------
    print_banner("[1] MEMBACA DATASET DARI HDFS")
    
    # Path dataset di HDFS (dengan fallback direktori)
    hdfs_primary_path = "hdfs://namenode:9000/data/kualitas_udara/indonesia_kualitas_udara_ispu_enriched.csv"
    hdfs_fallback_path = "hdfs://namenode:9000/data/kualitas_udara/*.csv"

    try:
        print(f">> Mencoba membaca dataset dari: {hdfs_primary_path}")
        df_raw = spark.read.csv(hdfs_primary_path, header=True, inferSchema=True)
    except Exception as e:
        print(f"[WARN] Gagal membaca path utama ({e}), mencoba fallback: {hdfs_fallback_path}")
        df_raw = spark.read.csv(hdfs_fallback_path, header=True, inferSchema=True)

    print(">> Berhasil memuat dataset dari HDFS!")
    total_raw_rows = df_raw.count()
    total_cols = len(df_raw.columns)
    print(f">> Jumlah Baris Awal   : {total_raw_rows:,} baris")
    print(f">> Jumlah Kolom        : {total_cols} kolom\n")

    print(">> Schema Dataset Awal:")
    df_raw.printSchema()

    print(">> Sampel 5 Baris Pertama Data Mentah:")
    df_raw.select(
        "station_record_id", "province", "monitoring_station_name",
        "pm25_concentration_ugm3", "pm10_concentration_ugm3",
        "sulfur_dioxide_so2_ugm3", "nitrogen_dioxide_no2_ugm3",
        "carbon_monoxide_co_ugm3", "ozone_o3_ugm3"
    ).show(5, truncate=True)

    # --------------------------------------------------------------------------
    # 2. DATA PREPROCESSING & GENERASI LABEL BINer
    # --------------------------------------------------------------------------
    print_banner("[2] PREPROCESSING & PEMBUATAN LABEL BINER")

    # 2.1 Tangani Duplikasi
    initial_count = df_raw.count()
    df_clean = df_raw.dropDuplicates()
    dedup_count = df_clean.count()
    print(f">> Baris duplikat dihapus: {initial_count - dedup_count:,}")

    # 2.2 Casting Tipe Data Numerik
    numeric_columns = [
        "pm25_concentration_ugm3", "pm10_concentration_ugm3",
        "sulfur_dioxide_so2_ugm3", "nitrogen_dioxide_no2_ugm3",
        "carbon_monoxide_co_ugm3", "ozone_o3_ugm3",
        "ambient_temperature_celsius", "relative_humidity_pct",
        "wind_speed_kmh", "rainfall_intensity_mm"
    ]
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            df_clean = df_clean.withColumn(col_name, col(col_name).cast(DoubleType()))

    # 2.3 Tangani Missing Values (Null)
    pollutant_cols = [
        "pm25_concentration_ugm3", "pm10_concentration_ugm3",
        "sulfur_dioxide_so2_ugm3", "nitrogen_dioxide_no2_ugm3",
        "carbon_monoxide_co_ugm3", "ozone_o3_ugm3"
    ]
    df_clean = df_clean.dropna(subset=pollutant_cols)
    clean_count = df_clean.count()
    print(f">> Baris setelah drop missing values polutan: {clean_count:,}")

    # 2.4 Pembentukan Label Kualitas Udara (Biner)
    # Ambang Batas: PM2.5 > 35.0 µg/m3 -> 1.0 ("Tidak Sehat"), <= 35.0 -> 0.0 ("Aman")
    # Standar mengacu pada ambang batas baku mutu harian WHO & ISPU KLHK
    df_clean = df_clean.withColumn(
        "label",
        when(col("pm25_concentration_ugm3") > 35.0, 1.0).otherwise(0.0)
    ).withColumn(
        "label_kategori",
        when(col("label") == 1.0, lit("Tidak Sehat")).otherwise(lit("Aman"))
    )

    print("\n>> Distribusi Label Kualitas Udara (Target Klasifikasi):")
    label_dist = df_clean.groupBy("label", "label_kategori").agg(
        count("*").alias("jumlah_baris"),
        round(count("*") * 100.0 / clean_count, 2).alias("persentase_pct")
    ).orderBy("label")
    label_dist.show(truncate=False)

    # --------------------------------------------------------------------------
    # 3. ANALISIS DATA DENGAN SPARK SQL / DATAFRAME API (MINIMAL 3 ANALISIS)
    # --------------------------------------------------------------------------
    print_banner("[3] ANALISIS DATA KUALITAS UDARA (SPARK SQL & DATAFRAME API)")

    # Analisis 1: Rata-rata dan Maksimum Polutan per Stasiun Pemantau (Top 10 PM2.5)
    print(">> [Analisis 1] Rata-rata & Nilai Maksimum Polutan per Stasiun Pemantau (Top 10 PM2.5):")
    analisis_stasiun = df_clean.groupBy("monitoring_station_name", "province").agg(
        round(avg("pm25_concentration_ugm3"), 2).alias("avg_pm25"),
        round(max("pm25_concentration_ugm3"), 2).alias("max_pm25"),
        round(avg("pm10_concentration_ugm3"), 2).alias("avg_pm10"),
        round(avg("nitrogen_dioxide_no2_ugm3"), 2).alias("avg_no2"),
        round(avg("sulfur_dioxide_so2_ugm3"), 2).alias("avg_so2"),
        round(avg("carbon_monoxide_co_ugm3"), 2).alias("avg_co"),
        round(avg("ozone_o3_ugm3"), 2).alias("avg_o3"),
        count("*").alias("total_rekaman")
    ).orderBy(desc("avg_pm25"))
    analisis_stasiun.show(10, truncate=False)

    # Analisis 2: Distribusi Status Kualitas Udara Berdasarkan Provinsi
    print(">> [Analisis 2] Distribusi Status Kualitas Udara (Aman vs Tidak Sehat) per Provinsi:")
    analisis_provinsi = df_clean.groupBy("province", "label_kategori").agg(
        count("*").alias("jumlah_observasi")
    ).orderBy("province", "label_kategori")
    analisis_provinsi.show(15, truncate=False)

    # Analisis 3: Karakteristik Rata-rata Polutan Lain Berdasarkan Kategori Kualitas Udara
    print(">> [Analisis 3] Perbandingan Rata-rata Polutan Lain Berdasarkan Kategori Udara:")
    analisis_polutan_kategori = df_clean.groupBy("label_kategori").agg(
        round(avg("pm10_concentration_ugm3"), 2).alias("avg_pm10"),
        round(avg("sulfur_dioxide_so2_ugm3"), 2).alias("avg_so2"),
        round(avg("nitrogen_dioxide_no2_ugm3"), 2).alias("avg_no2"),
        round(avg("carbon_monoxide_co_ugm3"), 2).alias("avg_co"),
        round(avg("ozone_o3_ugm3"), 2).alias("avg_o3"),
        count("*").alias("total_sampel")
    ).orderBy("label_kategori")
    analisis_polutan_kategori.show(truncate=False)

    # --------------------------------------------------------------------------
    # 4. IMPLEMENTASI SPARK MLLIB (LINEAR SVC)
    # --------------------------------------------------------------------------
    print_banner("[4] PEMODELAN MACHINE LEARNING (SPARK MLLIB - LINEAR SVC)")

    # CATATAN PENTING ANTI DATA LEAKAGE:
    # Karena label biner dibuat dari ambang batas PM2.5 (>35), kolom 'pm25_concentration_ugm3'
    # serta 'ispu_calculated_index' dan 'ispu_category' TIDAK dimasukkan ke dalam fitur.
    # Model ditantang untuk mendeteksi apakah udara "Tidak Sehat" murni berdasarkan
    # pola polutan lain (PM10, SO2, NO2, CO, O3) dan kondisi meteorologi!
    feature_cols = [
        "pm10_concentration_ugm3",
        "sulfur_dioxide_so2_ugm3",
        "nitrogen_dioxide_no2_ugm3",
        "carbon_monoxide_co_ugm3",
        "ozone_o3_ugm3"
    ]
    print(f">> Fitur Polutan yang Digunakan : {feature_cols}")
    print(">> Catatan Data Leakage        : 'pm25_concentration_ugm3' & ISPU dikeluarkan dari fitur!")

    # 4.1 VectorAssembler & StandardScaler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    df_assembled = assembler.transform(df_clean)

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)

    # 4.2 Split Data Train (80%) dan Data Test (20%)
    train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
    train_count = train_df.count()
    test_count = test_df.count()
    print(f">> Data Train (80%) : {train_count:,} sampel")
    print(f">> Data Test (20%)  : {test_count:,} sampel")

    # 4.3 Training Model LinearSVC
    print("\n>> Memulai pelatihan model LinearSVC...")
    lsvc = LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)
    t_start = time.time()
    lsvc_model = lsvc.fit(train_df)
    t_train = time.time() - t_start
    print(f">> Model LinearSVC berhasil dilatih dalam {t_train:.2f} detik.")

    # 4.4 Prediksi Data Uji
    predictions = lsvc_model.transform(test_df)
    predictions = predictions.withColumn(
        "kategori_prediksi",
        when(col("prediction") == 1.0, lit("Tidak Sehat")).otherwise(lit("Aman"))
    )

    # 4.5 Evaluasi Model
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    prec_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rec_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    accuracy = acc_evaluator.evaluate(predictions)
    precision = prec_evaluator.evaluate(predictions)
    recall = rec_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)

    print("\n" + "-" * 50)
    print(" HASIL EVALUASI MODEL LINEAR SVC (DATASET PENUH)")
    print("-" * 50)
    print(f"  Accuracy  : {accuracy * 100:.2f} %")
    print(f"  Precision : {precision * 100:.2f} %")
    print(f"  Recall    : {recall * 100:.2f} %")
    print(f"  F1-Score  : {f1 * 100:.2f} %")
    print("-" * 50)

    # 4.6 Eksperimen Skala Data (25%, 50%, 100%)
    print_banner("EKSPERIMEN SKALA DATASET (25%, 50%, 100%)")
    print("Mengukur akurasi, F1-Score, dan waktu training pada berbagai variasi skala data:")
    
    scale_results = []
    for scale in [0.25, 0.50, 1.00]:
        print(f"\n>> Menjalankan eksperimen pada skala {int(scale * 100)}% ...")
        # Sampling dataset
        sampled_df = df_scaled.sample(withReplacement=False, fraction=scale, seed=42)
        s_train, s_test = sampled_df.randomSplit([0.8, 0.2], seed=42)
        s_train_count = s_train.count()
        s_test_count = s_test.count()

        # Training LinearSVC
        exp_svc = LinearSVC(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)
        exp_start = time.time()
        exp_model = exp_svc.fit(s_train)
        exp_duration = time.time() - exp_start

        # Prediksi & Evaluasi
        exp_preds = exp_model.transform(s_test)
        exp_acc = acc_evaluator.evaluate(exp_preds)
        exp_f1 = f1_evaluator.evaluate(exp_preds)

        scale_results.append({
            "skala": f"{int(scale * 100)}%",
            "train_size": s_train_count,
            "test_size": s_test_count,
            "accuracy": builtins.round(exp_acc * 100, 2),
            "f1_score": builtins.round(exp_f1 * 100, 2),
            "waktu_detik": builtins.round(exp_duration, 3)
        })

    # Tampilkan tabel perbandingan eksperimen skala
    print("\n" + "=" * 80)
    print(f"{'Skala':<10} | {'Data Train':<12} | {'Data Test':<12} | {'Accuracy (%)':<14} | {'F1-Score (%)':<14} | {'Waktu (s)':<10}")
    print("-" * 80)
    for res in scale_results:
        print(f"{res['skala']:<10} | {res['train_size']:<12,d} | {res['test_size']:<12,d} | {res['accuracy']:<14.2f} | {res['f1_score']:<14.2f} | {res['waktu_detik']:<10.3f}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 5. INTEGRASI MONGODB (SIMPAN HASIL KE MONGODB & BACA KEMBALI)
    # --------------------------------------------------------------------------
    print_banner("[5] INTEGRASI MONGODB (SIMPAN ANALISIS & PREDIKSI)")

    mongo_uri = "mongodb://mongodb:27017"
    mongo_db = "db_kualitas_udara"

    # 5.1 Simpan Ringkasan Analisis ke MongoDB ('ringkasan_analisis')
    print(f">> Menyimpan ringkasan analisis ke MongoDB ({mongo_db}.ringkasan_analisis)...")
    try:
        analisis_polutan_kategori.write.format("mongodb") \
            .mode("overwrite") \
            .option("connection.uri", mongo_uri) \
            .option("database", mongo_db) \
            .option("collection", "ringkasan_analisis") \
            .save()
        print(">> [SUKSES] Ringkasan analisis berhasil disimpan ke MongoDB!")
    except Exception as e:
        print(f">> [ERROR] Gagal menyimpan ringkasan analisis: {e}")

    # 5.2 Simpan Sampel Hasil Prediksi ke MongoDB ('hasil_prediksi')
    print(f">> Menyiapkan sampel hasil prediksi untuk disimpan ke MongoDB ({mongo_db}.hasil_prediksi)...")
    # Siapkan dataframe bersih dari tipe Vector Spark ML (karena BSON MongoDB tidak mendukung VectorUDT secara langsung)
    pred_export = predictions.select(
        col("station_record_id"),
        col("province").alias("provinsi"),
        col("monitoring_station_name").alias("lokasi_stasiun"),
        col("pm25_concentration_ugm3").alias("pm25"),
        col("pm10_concentration_ugm3").alias("pm10"),
        col("sulfur_dioxide_so2_ugm3").alias("so2"),
        col("nitrogen_dioxide_no2_ugm3").alias("no2"),
        col("carbon_monoxide_co_ugm3").alias("co"),
        col("ozone_o3_ugm3").alias("o3"),
        col("label").alias("label_aktual_numerik"),
        col("label_kategori").alias("label_aktual"),
        col("prediction").alias("prediksi_numerik"),
        col("kategori_prediksi"),
        lit("Linear SVC").alias("model"),
        lit("Spark MLlib").alias("sumber"),
        current_timestamp().alias("waktu_prediksi")
    )

    # Ambil sampel representatif (misal 50 baris) untuk disimpan ke MongoDB
    sample_to_mongo = pred_export.limit(50)
    
    try:
        sample_to_mongo.write.format("mongodb") \
            .mode("append") \
            .option("connection.uri", mongo_uri) \
            .option("database", mongo_db) \
            .option("collection", "hasil_prediksi") \
            .save()
        print(f">> [SUKSES] Sebanyak 50 sampel hasil prediksi berhasil disimpan ke MongoDB!")
    except Exception as e:
        print(f">> [ERROR] Gagal menyimpan hasil prediksi ke MongoDB: {e}")

    # 5.3 Pembacaan Kembali (Read) Data dari MongoDB
    print("\n>> Melakukan pembacaan kembali (Read) dari MongoDB collection 'hasil_prediksi':")
    try:
        mongo_read_df = spark.read.format("mongodb") \
            .option("connection.uri", mongo_uri) \
            .option("database", mongo_db) \
            .option("collection", "hasil_prediksi") \
            .load()

        mongo_total_count = mongo_read_df.count()
        print(f">> [VERIFIKASI SUKSES] Data terbaca dari MongoDB: {mongo_total_count} dokumen.")
        print(">> Menampilkan 5 Dokumen Pertama yang Berhasil Dibaca Kembali dari MongoDB:")
        mongo_read_df.select(
            "lokasi_stasiun", "provinsi", "pm10", "no2", "label_aktual",
            "kategori_prediksi", "model", "sumber"
        ).show(5, truncate=False)
    except Exception as e:
        print(f">> [ERROR] Gagal membaca kembali dari MongoDB: {e}")

    print_banner("PIPELINE SELESAI DENGAN SUKSES!")
    spark.stop()

if __name__ == "__main__":
    main()
