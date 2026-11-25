"""
Backend API Portal Kualitas Udara Indonesia (Gaya BMKG)
Mengintegrasikan Dataset Historis CSV, MongoDB (hasil_prediksi & ringkasan_analisis),
Model Klasifikasi LinearSVC (konsisten dengan Spark MLlib app/pipeline.py),
dan Gemini API untuk Perbandingan Kualitas Udara Hari Ini.
"""

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CSV_PATH = BASE_DIR / "data" / "indonesia_kualitas_udara_ispu_enriched.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Portal Kualitas Udara Indonesia - Spark MLlib & Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Koordinat ibu kota / pusat pemantauan untuk 23 provinsi di dataset
PROVINCE_COORDS: Dict[str, Dict[str, Any]] = {
    "Aceh": {"lat": 5.5483, "lng": 95.3238, "hub": "Banda Aceh", "pulau": "Sumatera"},
    "Sumatera Utara": {"lat": 3.5952, "lng": 98.6722, "hub": "Medan", "pulau": "Sumatera"},
    "Sumatera Barat": {"lat": -0.9471, "lng": 100.4172, "hub": "Padang / Agam", "pulau": "Sumatera"},
    "Riau": {"lat": 0.5071, "lng": 101.4478, "hub": "Pekanbaru", "pulau": "Sumatera"},
    "Jambi": {"lat": -1.6101, "lng": 103.6131, "hub": "Jambi", "pulau": "Sumatera"},
    "Sumatera Selatan": {"lat": -2.9761, "lng": 104.7754, "hub": "Palembang", "pulau": "Sumatera"},
    "Lampung": {"lat": -5.4292, "lng": 105.2610, "hub": "Bandar Lampung", "pulau": "Sumatera"},
    "Banten": {"lat": -6.1200, "lng": 106.1503, "hub": "Serang / Tangerang Selatan", "pulau": "Jawa"},
    "DKI Jakarta": {"lat": -6.2088, "lng": 106.8456, "hub": "Jakarta Pusat", "pulau": "Jawa"},
    "Jawa Barat": {"lat": -6.9175, "lng": 107.6191, "hub": "Bandung", "pulau": "Jawa"},
    "Jawa Tengah": {"lat": -6.9932, "lng": 110.4203, "hub": "Semarang", "pulau": "Jawa"},
    "DI Yogyakarta": {"lat": -7.7956, "lng": 110.3695, "hub": "Yogyakarta", "pulau": "Jawa"},
    "Jawa Timur": {"lat": -7.2575, "lng": 112.7521, "hub": "Surabaya", "pulau": "Jawa"},
    "Bali": {"lat": -8.6705, "lng": 115.2126, "hub": "Denpasar", "pulau": "Bali & Nusa Tenggara"},
    "Nusa Tenggara Timur": {"lat": -8.4897, "lng": 119.8877, "hub": "Labuan Bajo / Kupang", "pulau": "Bali & Nusa Tenggara"},
    "Kalimantan Barat": {"lat": -0.0263, "lng": 109.3425, "hub": "Pontianak", "pulau": "Kalimantan"},
    "Kalimantan Tengah": {"lat": -2.2136, "lng": 113.9108, "hub": "Palangka Raya", "pulau": "Kalimantan"},
    "Kalimantan Selatan": {"lat": -3.4422, "lng": 114.8325, "hub": "Banjarbaru", "pulau": "Kalimantan"},
    "Kalimantan Timur": {"lat": -0.5022, "lng": 117.1536, "hub": "Samarinda", "pulau": "Kalimantan"},
    "Sulawesi Tengah": {"lat": -0.8917, "lng": 119.8707, "hub": "Palu", "pulau": "Sulawesi"},
    "Sulawesi Selatan": {"lat": -5.1477, "lng": 119.4327, "hub": "Makassar", "pulau": "Sulawesi"},
    "Maluku": {"lat": -3.6954, "lng": 128.1814, "hub": "Ambon", "pulau": "Maluku & Papua"},
    "Papua": {"lat": -2.5916, "lng": 140.6690, "hub": "Jayapura", "pulau": "Maluku & Papua"},
}

FEATURE_COLS = [
    "pm10_concentration_ugm3",
    "sulfur_dioxide_so2_ugm3",
    "nitrogen_dioxide_no2_ugm3",
    "carbon_monoxide_co_ugm3",
    "ozone_o3_ugm3",
]


class LinearSVCModel:
    """
    Implementasi LinearSVC (Hinge Loss + L2 Regularization) yang dioptimasi
    untuk menyamai keputusan klasifikasi model Spark MLlib di app/pipeline.py.
    """

    def __init__(self) -> None:
        self.means: np.ndarray = np.zeros(len(FEATURE_COLS))
        self.stds: np.ndarray = np.ones(len(FEATURE_COLS))
        self.weights: np.ndarray = np.zeros(len(FEATURE_COLS))
        self.bias: float = 0.0
        self.metrics: Dict[str, float] = {
            "accuracy": 90.84,
            "precision": 90.96,
            "recall": 90.84,
            "f1_score": 90.88,
        }
        self.trained: bool = False

    def fit(self, X: np.ndarray, y_binary: np.ndarray) -> None:
        # Normalisasi internal saat optimasi agar konvergen presisi seperti L-BFGS Spark MLlib
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=1)
        stds = np.where(stds == 0, 1.0, stds)
        self.means = means
        self.stds = stds
        X_scaled = (X - means) / stds

        # Split deterministik 80/20
        rng = np.random.default_rng(42)
        idx = np.arange(len(X_scaled))
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]

        X_tr, y_tr = X_scaled[train_idx], y_binary[train_idx]
        X_te, y_te = X_scaled[test_idx], y_binary[test_idx]

        # Target hinge loss {-1, +1} dengan Adam optimizer (regParam=0.01 efektif pada fitur ternormalisasi)
        y_hinge = np.where(y_tr == 1.0, 1.0, -1.0)
        w = np.zeros(X_tr.shape[1], dtype=np.float64)
        b = 0.0
        reg_param = 0.01
        n_samples = len(X_tr)

        m_w = np.zeros_like(w)
        v_w = np.zeros_like(w)
        m_b = 0.0
        v_b = 0.0
        beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.05

        for epoch in range(1, 501):
            margins = y_hinge * (X_tr @ w + b)
            active = margins < 1.0
            if np.any(active):
                grad_w = reg_param * w - (X_tr[active].T @ y_hinge[active]) / n_samples
                grad_b = -np.sum(y_hinge[active]) / n_samples
            else:
                grad_w = reg_param * w
                grad_b = 0.0

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

            m_w_hat = m_w / (1 - beta1 ** epoch)
            v_w_hat = v_w / (1 - beta2 ** epoch)
            m_b_hat = m_b / (1 - beta1 ** epoch)
            v_b_hat = v_b / (1 - beta2 ** epoch)

            w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
            b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        self.weights = w
        self.bias = float(b)
        self.trained = True

        # Evaluasi di data uji
        preds = (X_te @ self.weights + self.bias >= 0.0).astype(float)
        tp = float(np.sum((preds == 1.0) & (y_te == 1.0)))
        tn = float(np.sum((preds == 0.0) & (y_te == 0.0)))
        fp = float(np.sum((preds == 1.0) & (y_te == 0.0)))
        fn = float(np.sum((preds == 0.0) & (y_te == 1.0)))
        acc = (tp + tn) / max(len(y_te), 1)
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-9)

        self.metrics = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
        }

    def predict(self, features: List[float]) -> Dict[str, Any]:
        x = np.array(features, dtype=np.float64)
        x_scaled = (x - self.means) / self.stds
        raw_score = float(x_scaled @ self.weights + self.bias)
        prob = 1.0 / (1.0 + np.exp(-np.clip(raw_score * 1.5, -15, 15)))
        is_unhealthy = raw_score >= 0.0
        return {
            "prediction": 1.0 if is_unhealthy else 0.0,
            "kategori_prediksi": "Tidak Sehat" if is_unhealthy else "Aman",
            "raw_score": round(raw_score, 4),
            "probabilitas_tidak_sehat_pct": round(float(prob) * 100, 1),
        }


class AirQualityStore:
    """Cache in-memory untuk dataset CSV, model LinearSVC, dan koneksi opsional MongoDB."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.provinces: Dict[str, Dict[str, Any]] = {}
        self.province_recent: Dict[str, List[Dict[str, Any]]] = {}
        self.model = LinearSVCModel()
        self.overview: Dict[str, Any] = {}
        self.load_data()

    def load_data(self) -> None:
        if not DATA_CSV_PATH.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan di {DATA_CSV_PATH}")

        by_prov: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        feature_matrix: List[List[float]] = []
        labels: List[float] = []

        with open(DATA_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pm25 = float(row["pm25_concentration_ugm3"])
                    pm10 = float(row["pm10_concentration_ugm3"])
                    so2 = float(row["sulfur_dioxide_so2_ugm3"])
                    no2 = float(row["nitrogen_dioxide_no2_ugm3"])
                    co = float(row["carbon_monoxide_co_ugm3"])
                    o3 = float(row["ozone_o3_ugm3"])
                    temp = float(row["ambient_temperature_celsius"])
                    hum = float(row["relative_humidity_pct"])
                    wind = float(row["wind_speed_kmh"])
                    rain = float(row["rainfall_intensity_mm"])
                    ispu = float(row["ispu_calculated_index"])
                    stagnation = float(row["atmospheric_stagnation_index"])
                except (ValueError, KeyError):
                    continue

                parsed = {
                    "id": row["station_record_id"],
                    "datetime": row["observation_datetime"],
                    "province": row["province"],
                    "region_type": row["region_type"],
                    "station": row["monitoring_station_name"],
                    "pm25": pm25,
                    "pm10": pm10,
                    "so2": so2,
                    "no2": no2,
                    "co": co,
                    "o3": o3,
                    "temp": temp,
                    "humidity": hum,
                    "wind": wind,
                    "rain": rain,
                    "ispu": ispu,
                    "ispu_category": row["ispu_category"],
                    "stagnation": stagnation,
                    "smoke_flag": row["wildfire_smoke_plume_flag"] == "True",
                    "risk_tier": row["respiratory_health_risk_tier"],
                    "label": 1.0 if pm25 > 35.0 else 0.0,
                    "kategori": "Tidak Sehat" if pm25 > 35.0 else "Aman",
                }
                self.rows.append(parsed)
                by_prov[parsed["province"]].append(parsed)
                feature_matrix.append([pm10, so2, no2, co, o3])
                labels.append(parsed["label"])

        if feature_matrix:
            self.model.fit(
                np.array(feature_matrix, dtype=np.float64),
                np.array(labels, dtype=np.float64),
            )

        total_rows = len(self.rows)
        unhealthy_total = sum(1 for r in self.rows if r["label"] == 1.0)

        for prov, p_rows in sorted(by_prov.items()):
            p_rows_sorted = sorted(p_rows, key=lambda x: x["datetime"])
            recent = p_rows_sorted[-12:]
            latest = p_rows_sorted[-1]

            avg_pm25 = float(np.mean([r["pm25"] for r in p_rows]))
            max_pm25 = float(np.max([r["pm25"] for r in p_rows]))
            avg_pm10 = float(np.mean([r["pm10"] for r in p_rows]))
            avg_so2 = float(np.mean([r["so2"] for r in p_rows]))
            avg_no2 = float(np.mean([r["no2"] for r in p_rows]))
            avg_co = float(np.mean([r["co"] for r in p_rows]))
            avg_o3 = float(np.mean([r["o3"] for r in p_rows]))
            avg_ispu = float(np.mean([r["ispu"] for r in p_rows]))
            avg_temp = float(np.mean([r["temp"] for r in p_rows]))
            avg_hum = float(np.mean([r["humidity"] for r in p_rows]))
            avg_wind = float(np.mean([r["wind"] for r in p_rows]))
            avg_stagnation = float(np.mean([r["stagnation"] for r in p_rows]))

            unhealthy_cnt = sum(1 for r in p_rows if r["label"] == 1.0)
            unhealthy_pct = round(unhealthy_cnt * 100.0 / len(p_rows), 1)
            smoke_cnt = sum(1 for r in p_rows if r["smoke_flag"])

            # Perkiraan (forecast) periode berikutnya menggunakan tren observasi terbaru + LinearSVC
            recent_pm10 = float(np.mean([r["pm10"] for r in recent]))
            recent_so2 = float(np.mean([r["so2"] for r in recent]))
            recent_no2 = float(np.mean([r["no2"] for r in recent]))
            recent_co = float(np.mean([r["co"] for r in recent]))
            recent_o3 = float(np.mean([r["o3"] for r in recent]))
            recent_pm25 = float(np.mean([r["pm25"] for r in recent]))

            # Faktor penyesuaian dispersi meteorologi (angin kencang membantu dispersi, stagnasi tinggi menahan polutan)
            dispersion_factor = 1.0 + (avg_stagnation - 5.5) * 0.025 - (avg_wind - 10.0) * 0.01
            forecast_pm25 = round(max(5.0, (0.65 * recent_pm25 + 0.35 * avg_pm25) * dispersion_factor), 1)
            forecast_features = [
                recent_pm10 * dispersion_factor,
                recent_so2 * dispersion_factor,
                recent_no2 * dispersion_factor,
                recent_co * dispersion_factor,
                recent_o3,
            ]
            svc_forecast = self.model.predict(forecast_features)

            # Tentukan status perkiraan & tingkat keparahan BMKG-style secara konsisten
            is_forecast_unhealthy = (
                svc_forecast["kategori_prediksi"] == "Tidak Sehat" or forecast_pm25 > 35.0
            )
            perkiraan_status = "Tidak Sehat" if is_forecast_unhealthy else "Aman"
            if is_forecast_unhealthy:
                map_tier = "Tidak Sehat"
            elif forecast_pm25 > 15.0:
                map_tier = "Sedang"
            else:
                map_tier = "Baik"

            # Agregasi stasiun dalam provinsi
            by_station: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in p_rows:
                by_station[r["station"]].append(r)

            stations_summary = []
            for st_name, st_rows in by_station.items():
                st_pm25 = float(np.mean([r["pm25"] for r in st_rows]))
                stations_summary.append({
                    "station": st_name,
                    "count": len(st_rows),
                    "avg_pm25": round(st_pm25, 1),
                    "max_pm25": round(float(np.max([r["pm25"] for r in st_rows])), 1),
                    "avg_pm10": round(float(np.mean([r["pm10"] for r in st_rows])), 1),
                    "status": "Tidak Sehat" if st_pm25 > 35.0 else "Aman",
                })
            stations_summary.sort(key=lambda x: x["avg_pm25"], reverse=True)

            # Proyeksi 3 Hari Ke Depan berbasis tren observasi dataset + dispersi meteorologi + LinearSVC
            pm25_series = np.array([r["pm25"] for r in recent], dtype=np.float64)
            slope_pm25 = float(np.polyfit(np.arange(len(pm25_series)), pm25_series, 1)[0]) if len(pm25_series) >= 3 else 0.0
            # Redam slope agar proyeksi 3 hari tetap realistis terhadap mean historis
            slope_pm25 = float(np.clip(slope_pm25, -4.0, 4.0))
            base_date = datetime.now()
            day_names = ["Besok (H+1)", "Lusa (H+2)", "H+3"]
            forecast_3days_dataset = []
            for d_idx in range(1, 4):
                target_dt = base_date + timedelta(days=d_idx)
                damping = 0.65 ** (d_idx - 1)
                proj_pm25 = round(max(5.0, (forecast_pm25 + slope_pm25 * d_idx * damping * 0.8)), 1)
                ratio = proj_pm25 / max(recent_pm25, 1.0)
                ratio_clamped = float(np.clip(ratio, 0.75, 1.35))
                d_pred = self.model.predict([
                    recent_pm10 * ratio_clamped,
                    recent_so2 * ratio_clamped,
                    recent_no2 * ratio_clamped,
                    recent_co * ratio_clamped,
                    recent_o3,
                ])
                d_unhealthy = proj_pm25 > 35.0 or d_pred["kategori_prediksi"] == "Tidak Sehat"
                forecast_3days_dataset.append({
                    "day_label": day_names[d_idx - 1],
                    "date": target_dt.strftime("%d %b"),
                    "pm25": proj_pm25,
                    "status": "Tidak Sehat" if d_unhealthy else "Aman",
                    "prob_pct": d_pred["probabilitas_tidak_sehat_pct"],
                    "svc_status": d_pred["kategori_prediksi"],
                })

            coord = PROVINCE_COORDS.get(
                prov, {"lat": -2.5, "lng": 118.0, "hub": prov, "pulau": "Indonesia"}
            )

            self.provinces[prov] = {
                "province": prov,
                "hub": coord["hub"],
                "pulau": coord["pulau"],
                "lat": coord["lat"],
                "lng": coord["lng"],
                "region_type": latest["region_type"],
                "total_observations": len(p_rows),
                "unhealthy_pct": unhealthy_pct,
                "safe_pct": round(100.0 - unhealthy_pct, 1),
                "smoke_incidents": smoke_cnt,
                "avg_pm25": round(avg_pm25, 2),
                "max_pm25": round(max_pm25, 2),
                "avg_pm10": round(avg_pm10, 2),
                "avg_so2": round(avg_so2, 2),
                "avg_no2": round(avg_no2, 2),
                "avg_co": round(avg_co, 2),
                "avg_o3": round(avg_o3, 2),
                "avg_ispu": round(avg_ispu, 0),
                "meteorology": {
                    "temperature_c": round(avg_temp, 1),
                    "humidity_pct": round(avg_hum, 1),
                    "wind_speed_kmh": round(avg_wind, 1),
                    "stagnation_index": round(avg_stagnation, 2),
                },
                "forecast": {
                    "perkiraan_status": perkiraan_status,
                    "map_tier": map_tier,
                    "perkiraan_pm25": forecast_pm25,
                    "probabilitas_tidak_sehat_pct": svc_forecast["probabilitas_tidak_sehat_pct"],
                    "ambang_batas_pm25": 35.0,
                    "metode": "Spark MLlib LinearSVC + Tren Stasioneritas Atmosfer",
                    "catatan_bmkg": self._build_forecast_note(
                        prov, forecast_pm25, perkiraan_status, avg_wind, avg_stagnation
                    ),
                },
                "forecast_3days_dataset": forecast_3days_dataset,
                "stations": stations_summary[:8],
                "recent_series": [
                    {
                        "datetime": r["datetime"][:16],
                        "pm25": r["pm25"],
                        "pm10": r["pm10"],
                        "ispu": r["ispu"],
                        "kategori": r["kategori"],
                    }
                    for r in recent
                ],
            }
            self.province_recent[prov] = recent

        self.overview = {
            "total_observations": total_rows,
            "total_provinces": len(self.provinces),
            "total_stations": len({r["station"] for r in self.rows}),
            "unhealthy_count": unhealthy_total,
            "safe_count": total_rows - unhealthy_total,
            "unhealthy_pct": round(unhealthy_total * 100.0 / max(total_rows, 1), 2),
            "threshold_pm25": 35.0,
            "spark_mllib_metrics": {
                "model": "LinearSVC",
                "features": FEATURE_COLS,
                "anti_leakage": True,
                "accuracy": 90.84,
                "precision": 90.96,
                "recall": 90.84,
                "f1_score": 90.88,
                "local_eval": self.model.metrics,
            },
        }

    @staticmethod
    def _build_forecast_note(
        province: str, forecast_pm25: float, status: str, wind: float, stagnation: float
    ) -> str:
        if status == "Tidak Sehat":
            if stagnation >= 6.5:
                return (
                    f"Perkiraan wilayah {province} cenderung Tidak Sehat (PM2.5 ~{forecast_pm25} µg/m³). "
                    f"Indeks stagnasi atmosfer tinggi ({stagnation:.1f}/10) menahan akumulasi partikulat; "
                    f"kelompok sensitif disarankan membatasi aktivitas luar ruang."
                )
            return (
                f"Perkiraan wilayah {province} berada pada kategori Tidak Sehat (PM2.5 ~{forecast_pm25} µg/m³), "
                f"melewati ambang batas 35.0 µg/m³ model LinearSVC. Gunakan masker filtrasi saat beraktivitas di luar."
            )
        return (
            f"Perkiraan wilayah {province} dalam kategori Aman (PM2.5 ~{forecast_pm25} µg/m³) dengan kecepatan angin "
            f"rata-rata {wind:.1f} km/jam mendukung dispersi polutan secara baik."
        )

    def get_mongodb_status(self) -> Dict[str, Any]:
        if not PYMONGO_AVAILABLE:
            return {"connected": False, "source": "Dataset CSV Lokal", "docs_hasil_prediksi": 0}
        try:
            client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=800)
            db = client["db_kualitas_udara"]
            count_pred = db["hasil_prediksi"].count_documents({})
            count_ring = db["ringkasan_analisis"].count_documents({})
            sample = list(
                db["hasil_prediksi"].find(
                    {},
                    {
                        "_id": 0,
                        "lokasi_stasiun": 1,
                        "provinsi": 1,
                        "pm10": 1,
                        "no2": 1,
                        "label_aktual": 1,
                        "kategori_prediksi": 1,
                        "model": 1,
                    },
                ).limit(6)
            )
            client.close()
            return {
                "connected": True,
                "source": "MongoDB (db_kualitas_udara) + Dataset HDFS",
                "docs_hasil_prediksi": count_pred,
                "docs_ringkasan_analisis": count_ring,
                "sample_predictions": sample,
            }
        except Exception:
            return {"connected": False, "source": "Dataset CSV Lokal (MongoDB offline)", "docs_hasil_prediksi": 0}


store = AirQualityStore()


class PredictRequest(BaseModel):
    pm10: float
    so2: float
    no2: float
    co: float
    o3: float


class GeminiCompareRequest(BaseModel):
    province: str
    api_key: Optional[str] = None
    today_pm25_manual: Optional[float] = None
    today_note_manual: Optional[str] = None


@app.get("/")
def serve_home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/overview")
def api_overview() -> Dict[str, Any]:
    mongo_status = store.get_mongodb_status()
    return {
        **store.overview,
        "mongodb": mongo_status,
        "gemini_env_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "updated_at": datetime.now().strftime("%d %b %Y, %H:%M WIB"),
    }


_realtime_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


@app.get("/api/provinces")
def api_provinces() -> Dict[str, Any]:
    return {"provinces": list(store.provinces.values())}


@app.get("/api/provinces/realtime")
def api_provinces_realtime() -> Dict[str, Any]:
    """Ambil data kualitas udara real-time hari ini untuk seluruh 23 provinsi via Open-Meteo batch API."""
    now_ts = datetime.now().timestamp()
    if _realtime_cache["data"] is not None and (now_ts - _realtime_cache["ts"]) < 600:
        return _realtime_cache["data"]

    prov_list = list(store.provinces.values())
    lats = ",".join(str(p["lat"]) for p in prov_list)
    lngs = ",".join(str(p["lng"]) for p in prov_list)

    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lats,
                "longitude": lngs,
                "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,european_aqi",
            },
            timeout=12,
        )
        r.raise_for_status()
        raw_items = r.json()
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gagal mengambil data real-time Open-Meteo: {exc}")

    results: Dict[str, Dict[str, Any]] = {}
    for p, item in zip(prov_list, raw_items):
        cur = item.get("current") or {}
        pm25 = round(float(cur.get("pm2_5") if cur.get("pm2_5") is not None else p["forecast"]["perkiraan_pm25"]), 1)
        pm10 = round(float(cur.get("pm10") if cur.get("pm10") is not None else p["avg_pm10"]), 1)
        so2 = round(float(cur.get("sulphur_dioxide") if cur.get("sulphur_dioxide") is not None else p["avg_so2"]), 1)
        no2 = round(float(cur.get("nitrogen_dioxide") if cur.get("nitrogen_dioxide") is not None else p["avg_no2"]), 1)
        co = round(float(cur.get("carbon_monoxide") if cur.get("carbon_monoxide") is not None else p["avg_co"]), 1)
        o3 = round(float(cur.get("ozone") if cur.get("ozone") is not None else p["avg_o3"]), 1)
        aqi = int(round(float(cur["european_aqi"]))) if cur.get("european_aqi") is not None else None

        svc_pred = store.model.predict([pm10, so2, no2, co, o3])
        is_unhealthy = pm25 > 35.0
        status = "Tidak Sehat" if is_unhealthy else "Aman"
        if is_unhealthy:
            map_tier = "Tidak Sehat"
        elif pm25 > 15.0:
            map_tier = "Sedang"
        else:
            map_tier = "Baik"

        results[p["province"]] = {
            "province": p["province"],
            "pm25": pm25,
            "pm10": pm10,
            "so2": so2,
            "no2": no2,
            "co": co,
            "o3": o3,
            "aqi": aqi,
            "status": status,
            "map_tier": map_tier,
            "svc_status": svc_pred["kategori_prediksi"],
            "svc_prob_pct": svc_pred["probabilitas_tidak_sehat_pct"],
            "delta_vs_csv": round(pm25 - p["avg_pm25"], 2),
            "observed_time": cur.get("time", ""),
        }

    payload = {
        "fetched_at": datetime.now().strftime("%d %b %Y, %H:%M WIB"),
        "source": "Open-Meteo Air Quality API (Real-Time) + LinearSVC",
        "realtime": results,
    }
    _realtime_cache["ts"] = now_ts
    _realtime_cache["data"] = payload
    return payload


_forecast3d_cache: Dict[str, Dict[str, Any]] = {}


@app.get("/api/provinces/{province_name}/forecast3d")
def api_province_forecast_3d(province_name: str) -> Dict[str, Any]:
    """Prakiraan kualitas udara 3 hari ke depan (H+1, H+2, H+3) dari Dataset CSV dan Open-Meteo API + LinearSVC."""
    prov_data = store.provinces.get(province_name)
    if not prov_data:
        raise HTTPException(status_code=404, detail=f"Provinsi '{province_name}' tidak ditemukan")

    now_ts = datetime.now().timestamp()
    cached = _forecast3d_cache.get(province_name)
    if cached and (now_ts - cached["ts"]) < 900:
        return cached["data"]

    api_3days: List[Dict[str, Any]] = []
    api_error: Optional[str] = None
    today_date_str = datetime.now().strftime("%Y-%m-%d")
    day_names = ["Besok (H+1)", "Lusa (H+2)", "H+3"]

    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": prov_data["lat"],
                "longitude": prov_data["lng"],
                "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,european_aqi",
                "forecast_days": 4,
                "timezone": "Asia/Jakarta",
            },
            timeout=10,
        )
        r.raise_for_status()
        hourly = r.json().get("hourly") or {}
        times = hourly.get("time") or []

        by_day: Dict[str, List[int]] = defaultdict(list)
        for idx, t_str in enumerate(times):
            day_key = t_str[:10]
            if day_key > today_date_str:
                by_day[day_key].append(idx)

        future_days = sorted(by_day.keys())[:3]
        for d_idx, day_key in enumerate(future_days):
            idxs = by_day[day_key]
            def _mean_col(col_name: str, fallback: float) -> float:
                vals = [hourly[col_name][i] for i in idxs if col_name in hourly and hourly[col_name][i] is not None]
                return round(float(np.mean(vals)), 1) if vals else fallback

            pm25_vals = [hourly["pm2_5"][i] for i in idxs if hourly.get("pm2_5") and hourly["pm2_5"][i] is not None]
            pm25_avg = round(float(np.mean(pm25_vals)), 1) if pm25_vals else prov_data["forecast"]["perkiraan_pm25"]
            pm25_max = round(float(np.max(pm25_vals)), 1) if pm25_vals else pm25_avg
            pm10_avg = _mean_col("pm10", prov_data["avg_pm10"])
            no2_avg = _mean_col("nitrogen_dioxide", prov_data["avg_no2"])
            so2_avg = _mean_col("sulphur_dioxide", prov_data["avg_so2"])
            co_avg = _mean_col("carbon_monoxide", prov_data["avg_co"])
            o3_avg = _mean_col("ozone", prov_data["avg_o3"])
            aqi_avg = int(round(_mean_col("european_aqi", 50.0)))

            svc_pred = store.model.predict([pm10_avg, so2_avg, no2_avg, co_avg, o3_avg])
            is_unhealthy = pm25_avg > 35.0
            dt_obj = datetime.strptime(day_key, "%Y-%m-%d")
            api_3days.append({
                "day_label": day_names[d_idx] if d_idx < len(day_names) else f"H+{d_idx + 1}",
                "date": dt_obj.strftime("%d %b"),
                "pm25": pm25_avg,
                "pm25_max": pm25_max,
                "aqi": aqi_avg,
                "status": "Tidak Sehat" if is_unhealthy else "Aman",
                "prob_pct": svc_pred["probabilitas_tidak_sehat_pct"],
                "svc_status": svc_pred["kategori_prediksi"],
            })
    except Exception as exc:
        api_error = str(exc)

    result = {
        "province": province_name,
        "dataset_3days": prov_data.get("forecast_3days_dataset", []),
        "api_3days": api_3days,
        "api_error": api_error,
        "fetched_at": datetime.now().strftime("%d %b %Y, %H:%M WIB"),
    }
    _forecast3d_cache[province_name] = {"ts": now_ts, "data": result}
    return result


@app.get("/api/provinces/{province_name}")
def api_province_detail(province_name: str) -> Dict[str, Any]:
    data = store.provinces.get(province_name)
    if not data:
        raise HTTPException(status_code=404, detail=f"Provinsi '{province_name}' tidak ditemukan")
    return data


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> Dict[str, Any]:
    res = store.model.predict([req.pm10, req.so2, req.no2, req.co, req.o3])
    return {
        "input": req.model_dump(),
        **res,
        "model": "LinearSVC (Spark MLlib Compatible)",
        "threshold_reference": "PM2.5 > 35.0 µg/m³ = Tidak Sehat",
    }


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Ekstrak JSON dari respons teks Gemini dengan toleransi markdown code block."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = match.group(1) if match else text.strip()
    if not candidate.startswith("{"):
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _fetch_open_meteo_today(lat: float, lng: float) -> Optional[Dict[str, Any]]:
    """Ambil data kualitas udara real-time hari ini dari Open-Meteo Air Quality API (gratis)."""
    try:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lng,
                "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,european_aqi",
            },
            timeout=8,
        )
        if r.status_code == 200:
            cur = r.json().get("current") or {}
            if cur.get("pm2_5") is not None:
                return cur
    except Exception:
        pass
    return None


@app.post("/api/gemini-compare")
def api_gemini_compare(req: GeminiCompareRequest) -> Dict[str, Any]:
    prov_data = store.provinces.get(req.province)
    if not prov_data:
        raise HTTPException(status_code=404, detail=f"Provinsi '{req.province}' tidak ditemukan.")

    api_key = (req.api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    today_str = datetime.now().strftime("%d %B %Y")

    # Ambil data real-time hari ini dari Open-Meteo (kecuali jika user isi manual)
    live_cur = None if req.today_pm25_manual is not None else _fetch_open_meteo_today(prov_data["lat"], prov_data["lng"])

    today_pm25_val: Optional[float] = req.today_pm25_manual
    today_aqi_val: Optional[int] = None
    live_svc_pred: Optional[Dict[str, Any]] = None

    if today_pm25_val is None and live_cur:
        today_pm25_val = round(float(live_cur["pm2_5"]), 1)
        if live_cur.get("european_aqi") is not None:
            today_aqi_val = int(round(float(live_cur["european_aqi"])))
        try:
            live_svc_pred = store.model.predict([
                float(live_cur.get("pm10") or prov_data["avg_pm10"]),
                float(live_cur.get("sulphur_dioxide") or prov_data["avg_so2"]),
                float(live_cur.get("nitrogen_dioxide") or prov_data["avg_no2"]),
                float(live_cur.get("carbon_monoxide") or prov_data["avg_co"]),
                float(live_cur.get("ozone") or prov_data["avg_o3"]),
            ])
        except Exception:
            live_svc_pred = None

    live_ctx = ""
    if req.today_pm25_manual is not None:
        live_ctx = (
            f"\nDATA HARI INI (INPUT MANUAL USER): PM2.5 = {req.today_pm25_manual} µg/m³"
            + (f", catatan: {req.today_note_manual}" if req.today_note_manual else "")
            + ".\n"
        )
    elif live_cur:
        live_ctx = (
            f"\nDATA PENGAMATAN REAL-TIME HARI INI (Open-Meteo, {live_cur.get('time', today_str)}):\n"
            f"- PM2.5 Hari Ini: {live_cur.get('pm2_5')} µg/m³\n"
            f"- PM10: {live_cur.get('pm10')} µg/m³, NO2: {live_cur.get('nitrogen_dioxide')} µg/m³, "
            f"SO2: {live_cur.get('sulphur_dioxide')} µg/m³, CO: {live_cur.get('carbon_monoxide')} µg/m³, O3: {live_cur.get('ozone')} µg/m³\n"
            f"- European AQI Hari Ini: {live_cur.get('european_aqi')}\n"
            + (
                f"- Klasifikasi LinearSVC Program User pada Data Hari Ini: {live_svc_pred['kategori_prediksi']} "
                f"(Risiko: {live_svc_pred['probabilitas_tidak_sehat_pct']}%)\n"
                if live_svc_pred
                else ""
            )
        )

    prompt = f"""Anda adalah asisten analitik kualitas udara untuk sistem Big Data Spark MLlib Indonesia.
Tugas Anda HANYA SATU: meringkas kondisi kualitas udara HARI INI ({today_str}) untuk wilayah {req.province} ({prov_data['hub']}), Indonesia, lalu membandingkannya secara langsung dengan data historis & model klasifikasi dari program user untuk menentukan apakah kualitas hari ini **Baik / Aman** atau **Tidak Sehat**.

DATA REFERENSI DARI PROGRAM & DATASET USER ({prov_data['province']}):
- Jumlah Observasi Dataset: {prov_data['total_observations']} rekaman stasiun
- Rata-rata Historis PM2.5: {prov_data['avg_pm25']} µg/m³ (Maksimum: {prov_data['max_pm25']} µg/m³)
- Perkiraan Model Spark MLlib LinearSVC: {prov_data['forecast']['perkiraan_status']} (Proyeksi PM2.5 ~{prov_data['forecast']['perkiraan_pm25']} µg/m³)
- ATURAN KLASIFIKASI PROGRAM USER: PM2.5 > 35.0 µg/m³ = "Tidak Sehat", PM2.5 <= 35.0 µg/m³ = "Baik / Aman"
{live_ctx}
INSTRUKSI WAJIB:
1. Tentukan apakah kualitas hari ini termasuk "Baik / Aman" atau "Tidak Sehat" berdasarkan aturan ambang batas program user (PM2.5 > 35.0 µg/m³ = Tidak Sehat).
2. Bandingkan kualitas hari ini secara langsung terhadap rata-rata historis dataset user ({prov_data['avg_pm25']} µg/m³) dan perkiraan Spark MLlib ({prov_data['forecast']['perkiraan_status']}).
3. Keluarkan hasil HANYA dalam format JSON valid berikut (dalam blok ```json ... ```):
```json
{{
  "wilayah": "{prov_data['province']}",
  "tanggal": "{today_str}",
  "pm25_hari_ini_ugm3": {today_pm25_val if today_pm25_val is not None else "null"},
  "aqi_hari_ini": {today_aqi_val if today_aqi_val is not None else "null"},
  "status_hari_ini": "Baik / Aman" atau "Tidak Sehat",
  "perbandingan_vs_dataset": "<1-2 kalimat membandingkan PM2.5/status hari ini dengan rata-rata historis dataset {prov_data['avg_pm25']} µg/m³ dan hasil model Spark MLlib>",
  "kesimpulan_singkat": "<1 kalimat tegas apakah hari ini lebih baik, sebanding, atau lebih buruk dibanding baseline program>",
  "sumber_data_hari_ini": "Open-Meteo Real-Time + Analisis Gemini"
}}
```"""

    parsed: Optional[Dict[str, Any]] = None
    raw_text = ""
    gemini_used_model = None
    gemini_error_note = None

    if api_key:
        candidate_models = ["gemini-3.8-flash-lite", "gemini-3.8-flash", "gemini-2.0-flash"]
        for model_name in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload_plain = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            }
            try:
                resp = requests.post(url, json=payload_plain, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        raw_text = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
                        parsed = _extract_json_block(raw_text)
                        gemini_used_model = model_name
                        break
                else:
                    try:
                        gemini_error_note = resp.json().get("error", {}).get("message", resp.text)
                    except Exception:
                        gemini_error_note = resp.text
                    # Jika quota/model error, coba model berikutnya
                    if resp.status_code in (400, 404, 429):
                        continue
                    break
            except Exception as exc:
                gemini_error_note = str(exc)
                continue
    else:
        gemini_error_note = "API Key Gemini belum diisi."

    # Jika Gemini gagal/kuota habis tapi Open-Meteo/manual tersedia, buat ringkasan perbandingan langsung agar tetap berjalan
    if parsed is None:
        if today_pm25_val is None:
            raise HTTPException(
                status_code=429 if "quota" in (gemini_error_note or "").lower() else 400,
                detail=f"Gagal memanggil Gemini API: {gemini_error_note}",
            )
        status_hari_ini = "Tidak Sehat" if today_pm25_val > 35.0 else "Baik / Aman"
        delta_val = round(today_pm25_val - prov_data["avg_pm25"], 2)
        arah = "lebih tinggi" if delta_val > 0 else "lebih rendah"
        parsed = {
            "wilayah": prov_data["province"],
            "tanggal": today_str,
            "pm25_hari_ini_ugm3": today_pm25_val,
            "aqi_hari_ini": today_aqi_val,
            "status_hari_ini": status_hari_ini,
            "perbandingan_vs_dataset": (
                f"Konsentrasi PM2.5 hari ini di {prov_data['province']} tercatat {today_pm25_val} µg/m³ "
                f"({status_hari_ini} menurut ambang batas 35.0 µg/m³), yaitu {abs(delta_val)} µg/m³ {arah} "
                f"dibanding rata-rata historis dataset Anda ({prov_data['avg_pm25']} µg/m³)."
            ),
            "kesimpulan_singkat": (
                f"Kualitas udara hari ini ({status_hari_ini}) "
                + ("lebih bersih dari rata-rata historis dataset Anda." if delta_val < 0 else "lebih buruk dari rata-rata historis dataset Anda.")
            ),
            "sumber_data_hari_ini": (
                f"Open-Meteo Real-Time + LinearSVC (Fallback karena kuota Gemini API habis)"
                if gemini_error_note
                else "Open-Meteo Real-Time + LinearSVC"
            ),
        }

    # Hitung selisih numerik langsung terhadap dataset user
    delta_pm25 = None
    status_program_hari_ini = None
    final_pm25 = parsed.get("pm25_hari_ini_ugm3") if parsed else today_pm25_val
    if final_pm25 is not None:
        try:
            today_pm25 = float(final_pm25)
            delta_pm25 = round(today_pm25 - prov_data["avg_pm25"], 2)
            status_program_hari_ini = "Tidak Sehat" if today_pm25 > 35.0 else "Baik / Aman"
        except (TypeError, ValueError):
            pass

    return {
        "province": prov_data["province"],
        "gemini_model": gemini_used_model,
        "gemini_fallback_note": gemini_error_note if not gemini_used_model else None,
        "dataset_baseline": {
            "avg_pm25": prov_data["avg_pm25"],
            "max_pm25": prov_data["max_pm25"],
            "forecast_status": prov_data["forecast"]["perkiraan_status"],
            "forecast_pm25": prov_data["forecast"]["perkiraan_pm25"],
            "threshold_pm25": 35.0,
        },
        "gemini_result": parsed,
        "program_rule_status_hari_ini": status_program_hari_ini,
        "delta_pm25_vs_dataset": delta_pm25,
        "web_sources": [],
        "raw_text": raw_text,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
