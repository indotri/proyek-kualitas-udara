/* Frontend Logic — Monad Parchment Style + Transparent Indonesia Choropleth Map */

const state = {
  provinces: [],
  provincesByName: {},
  realtimeByName: {},
  realtimeFetchedAt: null,
  dataSource: 'csv', // 'csv' | 'realtime'
  selectedProvince: null,
  activeTierFilter: 'ALL',
  map: null,
  geoJsonLayer: null,
  geoJsonData: null,
  provinceLayers: {},
  viewMode: 'map', // 'map' | 'histogram'
  geminiEnvConfigured: false,
};

const GEOJSON_TO_DATASET = {
  'DI. ACEH': 'Aceh',
  'SUMATERA UTARA': 'Sumatera Utara',
  'SUMATERA BARAT': 'Sumatera Barat',
  'RIAU': 'Riau',
  'JAMBI': 'Jambi',
  'SUMATERA SELATAN': 'Sumatera Selatan',
  'LAMPUNG': 'Lampung',
  'BANTEN': 'Banten',
  'DKI JAKARTA': 'DKI Jakarta',
  'JAWA BARAT': 'Jawa Barat',
  'JAWA TENGAH': 'Jawa Tengah',
  'DAERAH ISTIMEWA YOGYAKARTA': 'DI Yogyakarta',
  'JAWA TIMUR': 'Jawa Timur',
  'BALI': 'Bali',
  'NUSA TENGGARA TIMUR': 'Nusa Tenggara Timur',
  'KALIMANTAN BARAT': 'Kalimantan Barat',
  'KALIMANTAN TENGAH': 'Kalimantan Tengah',
  'KALIMANTAN SELATAN': 'Kalimantan Selatan',
  'KALIMANTAN TIMUR': 'Kalimantan Timur',
  'SULAWESI TENGAH': 'Sulawesi Tengah',
  'SULAWESI SELATAN': 'Sulawesi Selatan',
  'MALUKU': 'Maluku',
  'PAPUA': 'Papua',
};

function getActiveMetrics(provData) {
  if (!provData) return null;
  if (state.dataSource === 'realtime') {
    const rt = state.realtimeByName[provData.province];
    if (rt) {
      return {
        pm25: rt.pm25,
        status: rt.status,
        map_tier: rt.map_tier,
        sourceLabel: 'Real-Time Hari Ini',
        aqi: rt.aqi,
      };
    }
  }
  return {
    pm25: provData.forecast.perkiraan_pm25,
    status: provData.forecast.perkiraan_status,
    map_tier: provData.forecast.map_tier,
    sourceLabel: 'Perkiraan CSV / LinearSVC',
    aqi: null,
  };
}

function getChoroplethColor(provData) {
  const m = getActiveMetrics(provData);
  if (!m) return '#cbd5e1'; // Abu-abu: Tidak Ada Data
  const pm25 = m.pm25;
  if (pm25 <= 15.0) return '#10b981';   // Hijau Zamrud: Baik
  if (pm25 <= 25.0) return '#65a30d';   // Hijau: Aman
  if (pm25 <= 35.0) return '#eab308';   // Kuning: Sedang
  if (pm25 <= 45.0) return '#f97316';   // Oranye: Tidak Sehat
  return '#b91c1c';                     // Merah Tua: Buruk
}

function formatNumber(n, decimals = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '--';
  return Number(n).toLocaleString('id-ID', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function tierToBadgeClass(status) {
  if (!status) return 'warn';
  const s = status.toLowerCase();
  if (s.includes('tidak sehat') || s.includes('buruk')) return 'bad';
  if (s.includes('sedang')) return 'warn';
  return 'good';
}

function startClock() {
  const el = document.getElementById('live-clock');
  const tick = () => {
    const now = new Date();
    el.textContent =
      now.toLocaleTimeString('id-ID', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      }) + ' WIB';
  };
  tick();
  setInterval(tick, 1000);
}

function initMap() {
  state.map = L.map('indonesia-map', {
    center: [-2.2, 118.0],
    zoom: 5,
    minZoom: 4,
    maxZoom: 8,
    zoomControl: true,
    attributionControl: false,
    scrollWheelZoom: false,
  });
}

function styleFeature(feature) {
  const geoName = (feature.properties?.Propinsi || '').toUpperCase();
  const dsName = GEOJSON_TO_DATASET[geoName];
  const provData = dsName ? state.provincesByName[dsName] : null;
  const active = getActiveMetrics(provData);
  const isSelected = dsName && state.selectedProvince === dsName;

  let fillOpacity = 0.94;
  if (state.activeTierFilter !== 'ALL') {
    if (!active || active.map_tier !== state.activeTierFilter) {
      fillOpacity = 0.22;
    }
  }

  return {
    fillColor: getChoroplethColor(provData),
    weight: isSelected ? 2.5 : 1.0,
    opacity: 1,
    color: isSelected ? '#2b59d1' : '#f6f3f1',
    fillOpacity,
  };
}

function renderChoropleth() {
  if (!state.geoJsonData || !state.map) return;

  if (state.geoJsonLayer) {
    state.map.removeLayer(state.geoJsonLayer);
  }
  state.provinceLayers = {};

  state.geoJsonLayer = L.geoJSON(state.geoJsonData, {
    style: styleFeature,
    onEachFeature: (feature, layer) => {
      const geoName = (feature.properties?.Propinsi || '').toUpperCase();
      const dsName = GEOJSON_TO_DATASET[geoName];
      const provData = dsName ? state.provincesByName[dsName] : null;

      if (dsName) {
        state.provinceLayers[dsName] = layer;
      }

      const displayLabel = dsName || feature.properties?.Propinsi || 'Wilayah';
      const active = getActiveMetrics(provData);
      const tooltipHtml = provData && active
        ? `<div style="font-size: 12px; line-height: 1.45;">
             <div style="font-weight: 600; font-size: 13px; margin-bottom: 3px;">${provData.province}</div>
             <div>${active.sourceLabel}: <strong>${active.pm25} &mu;g/m&sup3;</strong></div>
             ${active.aqi !== null ? `<div>European AQI Hari Ini: <strong>${active.aqi}</strong></div>` : ''}
             <div>Status: <strong>${active.status}</strong></div>
             <div style="font-size: 10.5px; color: #a3a3a3; margin-top: 3px;">Klik untuk lihat detail &amp; cek Gemini</div>
           </div>`
        : `<div style="font-size: 12px;">
             <strong>${displayLabel}</strong><br/>
             <span style="color: #a3a3a3;">Tidak ada data stasiun</span>
           </div>`;

      layer.bindTooltip(tooltipHtml, {
        sticky: true,
        className: 'pihps-tooltip',
      });

      layer.on({
        mouseover: (e) => {
          const target = e.target;
          target.setStyle({
            weight: 2.5,
            color: '#2b59d1',
            fillOpacity: 1,
          });
          target.bringToFront();
        },
        mouseout: (e) => {
          state.geoJsonLayer.resetStyle(e.target);
          if (state.selectedProvince && state.provinceLayers[state.selectedProvince]) {
            state.provinceLayers[state.selectedProvince].bringToFront();
          }
        },
        click: () => {
          if (dsName) {
            selectProvince(dsName, false);
          }
        },
      });
    },
  }).addTo(state.map);

  if (!state._fittedOnce) {
    state.map.fitBounds(state.geoJsonLayer.getBounds(), { padding: [8, 8] });
    state._fittedOnce = true;
  }

  if (state.selectedProvince && state.provinceLayers[state.selectedProvince]) {
    state.provinceLayers[state.selectedProvince].bringToFront();
  }
}

function renderHistogram() {
  const container = document.getElementById('histogram-view');
  const sorted = [...state.provinces].sort(
    (a, b) => (getActiveMetrics(b)?.pm25 || 0) - (getActiveMetrics(a)?.pm25 || 0)
  );
  const maxPm25 = Math.max(80, ...sorted.map((p) => getActiveMetrics(p)?.pm25 || 0));
  const titleSrc = state.dataSource === 'realtime' ? 'REAL-TIME HARI INI (OPEN-METEO)' : 'PERKIRAAN CSV & MODEL LINEARSVC';

  container.innerHTML = `
    <div style="font-size: 12px; color: var(--color-graphite); margin-bottom: 12px; text-align: center;">
      PERBANDINGAN PM2.5 (${titleSrc}) &middot; AMBANG BATAS TIDAK SEHAT: 35.0 &mu;g/m&sup3;
    </div>
    <div class="hist-grid">
      ${sorted
        .map((p) => {
          const val = getActiveMetrics(p)?.pm25 || 0;
          const pct = Math.min(100, (val / maxPm25) * 100);
          const color = getChoroplethColor(p);
          const sel = p.province === state.selectedProvince ? 'selected' : '';
          return `<div class="hist-row ${sel}" data-province="${p.province}">
            <div class="hist-label">${p.province}</div>
            <div class="hist-bar-track">
              <div class="hist-bar-fill" style="width: ${pct.toFixed(1)}%; background: ${color};"></div>
            </div>
            <div class="hist-val">${formatNumber(val)}</div>
          </div>`;
        })
        .join('')}
    </div>
  `;

  container.querySelectorAll('.hist-row').forEach((row) => {
    row.addEventListener('click', () => {
      const prov = row.getAttribute('data-province');
      selectProvince(prov, false);
    });
  });
}

function setMapViewMode(mode) {
  state.viewMode = mode;
  const mapEl = document.getElementById('indonesia-map');
  const histEl = document.getElementById('histogram-view');
  const legendEl = document.getElementById('pihps-legend');
  const btnMap = document.getElementById('btn-view-map');
  const btnHist = document.getElementById('btn-view-histogram');

  if (mode === 'histogram') {
    mapEl.style.display = 'none';
    legendEl.style.display = 'none';
    histEl.style.display = 'block';
    btnHist.classList.add('active');
    btnMap.classList.remove('active');
    renderHistogram();
  } else {
    histEl.style.display = 'none';
    mapEl.style.display = 'block';
    legendEl.style.display = 'block';
    btnMap.classList.add('active');
    btnHist.classList.remove('active');
    if (state.map) {
      setTimeout(() => state.map.invalidateSize(), 50);
    }
  }
}

function renderSparkline(series) {
  const svg = document.getElementById('trend-sparkline');
  if (!series || series.length === 0) {
    svg.innerHTML = '';
    return;
  }

  const width = 360;
  const height = 72;
  const padX = 8;
  const padY = 10;
  const vals = series.map((d) => d.pm25);
  const maxVal = Math.max(45, ...vals) * 1.1;
  const minVal = 0;

  const xStep = (width - padX * 2) / Math.max(series.length - 1, 1);
  const yScale = (v) =>
    height - padY - ((v - minVal) / Math.max(maxVal - minVal, 1)) * (height - padY * 2);

  const threshY = yScale(35.0);
  const points = series.map((d, i) => `${(padX + i * xStep).toFixed(1)},${yScale(d.pm25).toFixed(1)}`);

  const areaPoints = [
    `${padX},${height - padY}`,
    ...points,
    `${(padX + (series.length - 1) * xStep).toFixed(1)},${height - padY}`,
  ].join(' ');

  const dots = series
    .map((d, i) => {
      const cx = (padX + i * xStep).toFixed(1);
      const cy = yScale(d.pm25).toFixed(1);
      const color = d.pm25 > 35.0 ? '#b91c1c' : '#10b981';
      return `<circle cx="${cx}" cy="${cy}" r="3" fill="${color}" stroke="#f6f3f1" stroke-width="1">
                <title>${d.datetime}: PM2.5 ${d.pm25} µg/m³ (${d.kategori})</title>
              </circle>`;
    })
    .join('');

  svg.innerHTML = `
    <line x1="${padX}" y1="${threshY.toFixed(1)}" x2="${width - padX}" y2="${threshY.toFixed(1)}"
          stroke="#b91c1c" stroke-width="1" stroke-dasharray="4 3" opacity="0.65" />
    <polygon points="${areaPoints}" fill="rgba(43, 89, 209, 0.12)" />
    <polyline points="${points.join(' ')}" fill="none" stroke="#2b59d1" stroke-width="2" stroke-linejoin="round" />
    ${dots}
  `;
}

function renderForecast3DGrid(elId, days, isApi) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!days || days.length === 0) {
    el.innerHTML = `<div class="forecast3d-cell" style="grid-column: 1 / -1; color: var(--color-smoke);">Data prakiraan belum tersedia.</div>`;
    return;
  }
  el.innerHTML = days
    .map((d) => {
      const cls = d.status === 'Tidak Sehat' ? 'bad' : 'good';
      const meta = isApi && d.aqi ? `AQI ${d.aqi} · Risiko ${d.prob_pct}%` : `Risiko ${d.prob_pct}%`;
      return `<div class="forecast3d-cell">
        <div class="forecast3d-day">${d.day_label} · ${d.date}</div>
        <div class="forecast3d-val">${formatNumber(d.pm25)} µg/m³</div>
        <div><span class="status-pill-tag ${cls}" style="padding: 2px 7px; font-size: 9.5px;">${d.status}</span></div>
        <div class="forecast3d-meta">${meta}</div>
      </div>`;
    })
    .join('');
}

async function renderForecast3D(provData) {
  // 1. Langsung tampilkan proyeksi 3 hari dari Dataset CSV + LinearSVC
  renderForecast3DGrid('forecast3d-dataset', provData.forecast_3days_dataset || [], false);

  // 2. Ambil prakiraan 3 hari live dari Open-Meteo API + LinearSVC
  const apiEl = document.getElementById('forecast3d-api');
  if (apiEl) {
    apiEl.innerHTML = `<div class="forecast3d-cell" style="grid-column: 1 / -1; color: var(--color-smoke);">Memuat prakiraan API live 3 hari…</div>`;
  }
  const targetProv = provData.province;
  try {
    const resp = await fetch(`/api/provinces/${encodeURIComponent(targetProv)}/forecast3d`);
    if (!resp.ok) throw new Error('Gagal memuat prakiraan API');
    const data = await resp.json();
    if (state.selectedProvince !== targetProv) return;
    if (data.api_3days && data.api_3days.length > 0) {
      renderForecast3DGrid('forecast3d-api', data.api_3days, true);
    } else {
      apiEl.innerHTML = `<div class="forecast3d-cell" style="grid-column: 1 / -1; color: var(--color-smoke);">Prakiraan API sedang tidak tersedia (${data.api_error || 'offline'}).</div>`;
    }
  } catch (err) {
    if (state.selectedProvince === targetProv && apiEl) {
      apiEl.innerHTML = `<div class="forecast3d-cell" style="grid-column: 1 / -1; color: var(--color-smoke);">Gagal memuat prakiraan API live.</div>`;
    }
  }
}

function selectProvince(provinceName, scrollToMap = false) {
  const p = state.provincesByName[provinceName];
  if (!p) return;

  state.selectedProvince = provinceName;
  document.getElementById('province-dropdown').value = provinceName;

  // Update Hero Pill Badge
  const heroPill = document.getElementById('hero-prov-pill');
  const heroDot = document.getElementById('hero-prov-dot');
  const active = getActiveMetrics(p);
  const cls = tierToBadgeClass(active.map_tier);
  heroPill.className = `metric-pill ${cls}`;
  heroDot.style.background = getChoroplethColor(p);
  document.getElementById('hero-prov-label').textContent =
    `${p.province}: ${formatNumber(active.pm25)} µg/m³ · ${active.status}`;

  renderChoropleth();
  if (state.viewMode === 'histogram') {
    renderHistogram();
  }
  renderProvinceTable(document.getElementById('table-search').value);

  if (scrollToMap) {
    document.getElementById('peta-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // Isi panel detail wilayah
  document.getElementById('detail-province').textContent = p.province;
  document.getElementById('detail-meta').textContent = `${p.hub} · ${p.pulau} · ${p.region_type}`;
  document.getElementById('inspector-count').textContent = `${p.total_observations.toLocaleString('id-ID')} OBS`;

  const badge = document.getElementById('detail-status-badge');
  badge.textContent = p.forecast.perkiraan_status;
  badge.className = `status-pill-tag ${cls}`;

  document.getElementById('forecast-pm25').textContent = `${formatNumber(p.forecast.perkiraan_pm25)} µg/m³`;
  document.getElementById('forecast-prob').textContent = `Risiko Tidak Sehat: ${p.forecast.probabilitas_tidak_sehat_pct}%`;
  document.getElementById('forecast-note').textContent = p.forecast.catatan_bmkg;

  document.getElementById('val-pm25').textContent = `${formatNumber(p.avg_pm25)} µg/m³`;
  document.getElementById('val-pm10').textContent = `${formatNumber(p.avg_pm10)} µg/m³`;
  document.getElementById('val-no2').textContent = `${formatNumber(p.avg_no2)} µg/m³`;
  document.getElementById('val-so2').textContent = `${formatNumber(p.avg_so2)} µg/m³`;
  document.getElementById('val-co').textContent = `${formatNumber(p.avg_co, 0)} µg/m³`;
  document.getElementById('val-o3').textContent = `${formatNumber(p.avg_o3)} µg/m³`;

  document.getElementById('val-temp').textContent = `${p.meteorology.temperature_c} °C`;
  document.getElementById('val-hum').textContent = `${p.meteorology.humidity_pct}%`;
  document.getElementById('val-wind').textContent = `${p.meteorology.wind_speed_kmh} km/j`;
  document.getElementById('val-stag').textContent = `${p.meteorology.stagnation_index}/10`;

  renderSparkline(p.recent_series);
  renderForecast3D(p);

  const stContainer = document.getElementById('stations-list');
  stContainer.innerHTML = p.stations
    .slice(0, 5)
    .map((s) => {
      const sCls = s.status === 'Tidak Sehat' ? 'bad' : 'good';
      return `<div class="station-row">
        <span>${s.station}</span>
        <span style="display: flex; align-items: center; gap: 8px;">
          <span>${s.avg_pm25} µg/m³</span>
          <span class="status-pill-tag ${sCls}" style="padding: 2px 8px; font-size: 10px;">${s.status}</span>
        </span>
      </div>`;
    })
    .join('');

  const gemRes = document.getElementById('gemini-result');
  const gemErr = document.getElementById('gemini-error');
  if (gemRes) gemRes.style.display = 'none';
  if (gemErr) gemErr.style.display = 'none';
}

function renderProvinceTable(query = '') {
  const q = query.trim().toLowerCase();
  const tbody = document.getElementById('province-table-body');

  const filtered = state.provinces.filter((p) => {
    if (!q) return true;
    return (
      p.province.toLowerCase().includes(q) ||
      p.hub.toLowerCase().includes(q) ||
      p.pulau.toLowerCase().includes(q)
    );
  });

  tbody.innerHTML = filtered
    .map((p) => {
      const selectedCls = p.province === state.selectedProvince ? 'selected-row' : '';
      const active = getActiveMetrics(p);
      const badgeCls = tierToBadgeClass(active.map_tier);
      return `<tr class="${selectedCls}" data-province="${p.province}">
        <td>
          <strong>${p.province}</strong>
          <div style="font-size: 11px; color: var(--color-smoke);">${p.hub}</div>
        </td>
        <td>${formatNumber(p.avg_pm25)}</td>
        <td>${formatNumber(p.max_pm25)}</td>
        <td><strong>${formatNumber(active.pm25)}</strong></td>
        <td><span class="status-pill-tag ${badgeCls}" style="padding: 3px 9px; font-size: 10.5px;">${active.status}</span></td>
      </tr>`;
    })
    .join('');

  tbody.querySelectorAll('tr').forEach((tr) => {
    tr.addEventListener('click', () => {
      const prov = tr.getAttribute('data-province');
      selectProvince(prov, true);
    });
  });
}

async function runGeminiCompare() {
  const prov = state.selectedProvince;
  if (!prov) return;

  const btn = document.getElementById('btn-gemini-compare');
  const btnText = document.getElementById('btn-gemini-text');
  const errBox = document.getElementById('gemini-error');
  const resBox = document.getElementById('gemini-result');

  const apiKey = document.getElementById('gemini-api-key').value.trim();
  if (apiKey) {
    localStorage.setItem('bmkg_gemini_api_key', apiKey);
  }

  const manualPm25Raw = document.getElementById('manual-pm25').value.trim();
  const manualPm25 = manualPm25Raw !== '' ? parseFloat(manualPm25Raw) : null;

  errBox.style.display = 'none';
  resBox.style.display = 'none';
  btn.disabled = true;
  btnText.textContent = `MEMBANDINGKAN ${prov.toUpperCase()}…`;

  try {
    const resp = await fetch('/api/gemini-compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        province: prov,
        api_key: apiKey || null,
        today_pm25_manual: manualPm25,
      }),
    });

    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || 'Gagal memanggil API Gemini');
    }

    const gem = data.gemini_result || {};
    const base = data.dataset_baseline || {};

    const todayPm25Str =
      gem.pm25_hari_ini_ugm3 !== null && gem.pm25_hari_ini_ugm3 !== undefined
        ? `${formatNumber(gem.pm25_hari_ini_ugm3)} µg/m³`
        : gem.aqi_hari_ini
        ? `AQI ${gem.aqi_hari_ini}`
        : 'Data Teks';

    document.getElementById('gem-today-pm25').textContent = todayPm25Str;

    const finalStatus = data.program_rule_status_hari_ini || gem.status_hari_ini || 'Tidak Diketahui';
    const badge = document.getElementById('gem-today-badge');
    badge.textContent = finalStatus;
    badge.className = `status-pill-tag ${tierToBadgeClass(finalStatus)}`;

    document.getElementById('gem-base-pm25').textContent = `${formatNumber(base.avg_pm25)} µg/m³`;

    if (data.delta_pm25_vs_dataset !== null && data.delta_pm25_vs_dataset !== undefined) {
      const sign = data.delta_pm25_vs_dataset > 0 ? '+' : '';
      const dir =
        data.delta_pm25_vs_dataset > 0 ? 'lebih tinggi dari rata-rata dataset' : 'lebih rendah dari rata-rata dataset';
      document.getElementById('gem-delta-text').textContent = `${sign}${data.delta_pm25_vs_dataset} µg/m³ (${dir})`;
    } else {
      document.getElementById('gem-delta-text').textContent = `Perkiraan MLlib: ${base.forecast_status}`;
    }

    document.getElementById('gem-compare-text').textContent = gem.perbandingan_vs_dataset || '';
    document.getElementById('gem-conclusion-text').textContent = gem.kesimpulan_singkat || '';

    const srcEl = document.getElementById('gem-sources');
    if (data.web_sources && data.web_sources.length > 0) {
      srcEl.style.display = 'block';
      srcEl.innerHTML =
        'Sumber Live Hari Ini: ' +
        data.web_sources
          .map((s) => `<a href="${s.uri}" target="_blank" rel="noopener noreferrer">${s.title}</a>`)
          .join('');
    } else if (gem.sumber_data_hari_ini) {
      srcEl.style.display = 'block';
      srcEl.textContent = `Sumber: ${gem.sumber_data_hari_ini}`;
    } else {
      srcEl.style.display = 'none';
    }

    resBox.style.display = 'block';
  } catch (err) {
    errBox.textContent = err.message;
    errBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    btnText.textContent = 'BANDINGKAN KUALITAS HARI INI';
  }
}

async function runSimulator() {
  const payload = {
    pm10: parseFloat(document.getElementById('sim-pm10').value) || 0,
    so2: parseFloat(document.getElementById('sim-so2').value) || 0,
    no2: parseFloat(document.getElementById('sim-no2').value) || 0,
    co: parseFloat(document.getElementById('sim-co').value) || 0,
    o3: parseFloat(document.getElementById('sim-o3').value) || 0,
  };

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    const resBox = document.getElementById('sim-result');
    const badge = document.getElementById('sim-badge');
    document.getElementById('sim-score-text').textContent =
      `Skor LinearSVC: ${data.raw_score} · Probabilitas Tidak Sehat: ${data.probabilitas_tidak_sehat_pct}%`;
    badge.textContent = data.kategori_prediksi;
    badge.className = `status-pill-tag ${data.kategori_prediksi === 'Tidak Sehat' ? 'bad' : 'good'}`;
    resBox.style.display = 'flex';
  } catch (err) {
    console.error('Simulator error:', err);
  }
}

function loadSelectedProvinceToSim() {
  const p = state.provincesByName[state.selectedProvince];
  if (!p) return;
  document.getElementById('sim-pm10').value = p.avg_pm10;
  document.getElementById('sim-so2').value = p.avg_so2;
  document.getElementById('sim-no2').value = p.avg_no2;
  document.getElementById('sim-co').value = p.avg_co;
  document.getElementById('sim-o3').value = p.avg_o3;
  runSimulator();
}

async function bootstrap() {
  startClock();
  initMap();

  const savedKey = localStorage.getItem('bmkg_gemini_api_key');
  const gemInput = document.getElementById('gemini-api-key');
  if (savedKey && gemInput) {
    gemInput.value = savedKey;
  }

  try {
    const [ovResp, provResp, geoResp] = await Promise.all([
      fetch('/api/overview'),
      fetch('/api/provinces'),
      fetch('/static/indonesia-prov.geojson'),
    ]);
    const overview = await ovResp.json();
    const provJson = await provResp.json();
    state.geoJsonData = await geoResp.json();

    state.provinces = provJson.provinces || [];
    state.provinces.forEach((p) => {
      state.provincesByName[p.province] = p;
    });

    document.getElementById('stat-provinces').textContent = overview.total_provinces;
    document.getElementById('stat-observations').textContent = Number(overview.total_observations).toLocaleString('id-ID');
    document.getElementById('stat-accuracy').textContent = `${overview.spark_mllib_metrics.accuracy}%`;

    state.geminiEnvConfigured = overview.gemini_env_configured;
    if (state.geminiEnvConfigured && !savedKey && gemInput) {
      gemInput.placeholder = 'API Key terdeteksi di server (.env) — siap digunakan';
    }

    const dot = document.getElementById('db-status-dot');
    const dbText = document.getElementById('db-status-text');
    const mongoSamples = document.getElementById('mongo-samples');

    if (overview.mongodb && overview.mongodb.connected) {
      dot.classList.remove('offline');
      dbText.textContent = `MONGODB AKTIF (${overview.mongodb.docs_hasil_prediksi} DOK)`;
      const samples = overview.mongodb.sample_predictions || [];
      mongoSamples.innerHTML = samples
        .map((s) => {
          const cls = s.kategori_prediksi === 'Tidak Sehat' ? 'bad' : 'good';
          return `<div class="station-row">
            <span><strong>${s.provinsi || '-'}</strong> · ${s.lokasi_stasiun || '-'}</span>
            <span class="status-pill-tag ${cls}" style="padding: 2px 8px; font-size: 10px;">${s.kategori_prediksi}</span>
          </div>`;
        })
        .join('');
    } else {
      dot.classList.add('offline');
      dbText.textContent = 'MODE DATASET CSV';
      mongoSamples.innerHTML =
        '<div class="station-row" style="color: var(--color-smoke);">MongoDB Docker sedang offline — aplikasi menggunakan dataset HDFS/CSV &amp; LinearSVC lokal.</div>';
    }

    const dropdown = document.getElementById('province-dropdown');
    state.provinces.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.province;
      opt.textContent = `${p.province} (${p.hub})`;
      dropdown.appendChild(opt);
    });

    const defaultProv = state.provincesByName['DKI Jakarta'] ? 'DKI Jakarta' : state.provinces[0]?.province;
    if (defaultProv) {
      selectProvince(defaultProv, false);
    }
    renderProvinceTable();
    runSimulator();
  } catch (err) {
    console.error('Gagal memuat data awal:', err);
  }

  document.getElementById('province-dropdown').addEventListener('change', (e) => {
    if (e.target.value) selectProvince(e.target.value, false);
  });

  document.getElementById('tier-select').addEventListener('change', (e) => {
    state.activeTierFilter = e.target.value;
    renderChoropleth();
  });

  document.getElementById('table-search').addEventListener('input', (e) => {
    renderProvinceTable(e.target.value);
  });

  document.getElementById('btn-src-csv').addEventListener('click', () => switchDataSource('csv'));
  document.getElementById('btn-src-realtime').addEventListener('click', () => switchDataSource('realtime'));
  document.getElementById('btn-view-map').addEventListener('click', () => setMapViewMode('map'));
  document.getElementById('btn-view-histogram').addEventListener('click', () => setMapViewMode('histogram'));
  const btnGem = document.getElementById('btn-gemini-compare');
  if (btnGem) btnGem.addEventListener('click', runGeminiCompare);
  document.getElementById('btn-run-sim').addEventListener('click', runSimulator);
  document.getElementById('btn-load-prov-sim').addEventListener('click', loadSelectedProvinceToSim);
}

async function switchDataSource(source) {
  const btnCsv = document.getElementById('btn-src-csv');
  const btnRt = document.getElementById('btn-src-realtime');
  const footerTitle = document.getElementById('map-footer-title');
  const footerSub = document.getElementById('map-footer-sub');

  if (source === 'realtime') {
    btnRt.disabled = true;
    btnRt.textContent = 'Memuat Real-Time…';
    try {
      if (Object.keys(state.realtimeByName).length === 0) {
        const resp = await fetch('/api/provinces/realtime');
        if (!resp.ok) throw new Error('Gagal memuat data real-time');
        const json = await resp.json();
        state.realtimeByName = json.realtime || {};
        state.realtimeFetchedAt = json.fetched_at || '';
      }
      state.dataSource = 'realtime';
      btnRt.classList.add('active');
      btnCsv.classList.remove('active');
      footerTitle.textContent = 'Kualitas Udara Real-Time Hari Ini (µg/m³)';
      footerSub.textContent = `Sumber: Open-Meteo Live (${state.realtimeFetchedAt}) · Klasifikasi Ambang 35.0 µg/m³`;
    } catch (err) {
      alert(err.message);
    } finally {
      btnRt.disabled = false;
      btnRt.textContent = 'Real-Time Hari Ini';
    }
  } else {
    state.dataSource = 'csv';
    btnCsv.classList.add('active');
    btnRt.classList.remove('active');
    footerTitle.textContent = 'Perkiraan PM2.5 Nasional (µg/m³)';
    footerSub.innerHTML =
      'Rentang Perkiraan Dataset: 11.4 – 70.1 &mu;g/m&sup3; &middot; <span id="stat-observations">10.000</span> Observasi di <span id="stat-provinces">23</span> Provinsi';
  }

  if (state.selectedProvince) {
    selectProvince(state.selectedProvince, false);
  } else {
    renderChoropleth();
    renderProvinceTable(document.getElementById('table-search').value);
  }
}

document.addEventListener('DOMContentLoaded', bootstrap);
