# Pendapatan per Kapita Chicago: GWR vs Decision Tree vs KNN — Rencana Implementasi

> **Untuk agen pelaksana:** SUB-SKILL WAJIB: gunakan superpowers:subagent-driven-development (disarankan) atau superpowers:executing-plans untuk menjalankan rencana ini task per task. Langkah memakai checkbox (`- [ ]`) untuk pelacakan.

**Tujuan:** Membangun `chicago-capita.ipynb` sehingga ia memuat shapefile community area Chicago, merekayasa tiga variabel sesuai spesifikasi ditambah centroid, melatih GWR, Decision Tree, dan KNN pada split yang sama, lalu ditutup dengan tabel perbandingan R², RMSE, dan MAE.

**Arsitektur:** Satu notebook, satu arah aliran data: shapefile → `gdf` → frame fitur `df` → split 80/20 → tiga model → `results` → tabel perbandingan. Setiap task menambah satu sel notebook dan diakhiri assertion yang gagal sebelum kode sel itu ada dan lolos sesudahnya. Kode model ditulis inline di notebook (deliverable tugas), bukan di dalam package.

**Tech Stack:** Python 3.13, geopandas 1.2.0 (pyogrio), mgwr 2.2.1 (libpysal/spglm), scikit-learn, pandas, jupyter nbconvert.

**Spec:** `chicago-capita.ipynb` (judul, tabel variabel, URL dataset)

## Batasan Global

- Dataset: `dataset/ComArea_ACS14_f.shp`, 77 baris, 86 kolom, CRS `GEOGCS["WGS84(DD)"]`.
- Y = `PerCInc14`.
- X₁ = `Unemp14 / LaborFrc`.
- X₂ = `NoHS14 / (NoHS14 + HSGrad14 + SmClg14 + ClgGrad14)`.
- Lokasi = centroid tiap community area, dihitung setelah reproyeksi ke EPSG:32616 (meter).
- Split 80/20 yang sama untuk ketiga model: `test_size=0.2, random_state=42`.
- Metrik: R², RMSE, MAE, dihitung pada 16 baris uji.
- Tiga sel markdown yang sudah ada (judul, tabel variabel, URL dataset) dibiarkan unchanged.
- Tidak ada baris yang dibuang: 77 baris lengkap untuk setiap kolom yang dipakai di sini.

## Fokus Review

- **Skala Y:** `PerCInc14` rata-rata ≈ 289.557, sekitar sepuluh kali pendapatan per kapita Chicago 2014, jadi kemungkinan disimpan dalam satuan persepuluh dolar. Laporkan metrik dalam satuan data aslinya dan sebutkan hal itu; jangan pernah rescale diam-diam.
- **Penyebut pendidikan:** spesifikasi menyebut lima kategori pendidikan, data hanya punya empat (`NoHS14`, `HSGrad14`, `SmClg14`, `ClgGrad14`). Pakai keempatnya; tambahkan satu baris markdown yang menyebutkan perbedaan ini agar pembaca tidak salah paham.
- **GWR out-of-sample:** prediksi dengan `gwr_res.predict(coords_test, X_test)` wajib dipakai untuk metrik. Menilai `predy` in-sample akan menggelembungkan R² GWR dibanding dua model lain yang dinilai pada baris uji.
- **Kebocoran scaling:** `StandardScaler` untuk KNN harus di-fit hanya pada `X_train`, lalu diterapkan ke `X_test`.
- **Sampel GWR kecil:** 61 titik latih untuk dua prediktor. Jika `Sel_BW.search()` gagal atau menghasilkan bandwidth yang tidak bisa dipakai, jatuh ke `fixed=True` dengan bandwidth terkecil yang diterima pencarian dan cetak nilai yang dipilih.

---

### Task 1: Environment dan kerangka notebook

**Files:**
- Modify: `chicago-capita.ipynb`
- Create: `.gitignore`

**Interfaces:**
- Menghasilkan: notebook yang bisa dieksekusi (`kernelspec` ada) dan repo git untuk commit per task.

- [ ] **Langkah 1: Install stack dan verifikasi import**

Jalankan:
```bash
python -m pip install geopandas mgwr scikit-learn pandas nbconvert
python -c "import geopandas, mgwr, sklearn, pandas, nbconvert; print('ok')"
```
Harapan: `ok`, plus output pip tanpa `Would install` yang berakhir dengan `Successfully installed`.

- [ ] **Langkah 2: Inisialisasi git dan abaikan checkpoint notebook**

Jalankan:
```bash
git init
Set-Content .gitignore ".ipynb_checkpoints/`n__pycache__/`n"
git add -A
git commit -m "chore: init chicago capita notebook project"
```

- [ ] **Langkah 3: Tambahkan kernelspec ke metadata notebook**

Di `chicago-capita.ipynb`, tambahkan ke `metadata`:
```json
"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
```

- [ ] **Langkah 4: Verifikasi notebook bisa dieksekusi**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit code 0, tanpa traceback `[NbConvertApp]`. Notebook masih hanya menampilkan tiga sel markdown yang sudah ada.

- [ ] **Langkah 5: Commit**

```bash
git add chicago-capita.ipynb .gitignore
git commit -m "chore: make notebook executable"
```

---

### Task 2: Memuat dataset

**Files:**
- Modify: `chicago-capita.ipynb` (tambahkan sel markdown `## 2. Load Dataset`, lalu sel kode)

**Interfaces:**
- Menghasilkan: `gdf: geopandas.GeoDataFrame` dengan 77 baris di EPSG:32616, kolom meliputi `ComAreaID`, `community`, `PerCInc14`, `Unemp14`, `LaborFrc`, `NoHS14`, `HSGrad14`, `SmClg14`, `ClgGrad14`, `geometry`.

- [ ] **Langkah 1: Tulis cek yang gagal**

Tambahkan sel load beserta assertion-nya lebih dulu:
```python
import geopandas as gpd

gdf = gpd.read_file("dataset/ComArea_ACS14_f.shp")
assert len(gdf) == 77, len(gdf)
assert {"ComAreaID", "community", "PerCInc14", "Unemp14", "LaborFrc",
        "NoHS14", "HSGrad14", "SmClg14", "ClgGrad14"} <= set(gdf.columns), sorted(gdf.columns)
gdf = gdf.to_crs(32616)
gdf[["ComAreaID", "community", "PerCInc14", "Unemp14", "LaborFrc"]].head()
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0. Jika shapefile tidak ada atau nama kolom berbeda huruf besar/kecil, pesan assertion akan mencetak kolom aktual; perbaiki path atau nama kolomnya, jangan melonggarkan assertion.

- [ ] **Langkah 3: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: load chicago community area shapefile"
```

---

### Task 3: Feature engineering

**Files:**
- Modify: `chicago-capita.ipynb` (markdown `## 3. Feature Engineering`, lalu sel kode)

**Interfaces:**
- Menghasilkan: DataFrame `df` berisi kolom asli ditambah `x1_unemp_rate`, `x2_no_hs`, `y`, `x_m`, `y_m`. Task berikutnya membaca `df[["x1_unemp_rate", "x2_no_hs"]]`, `df["y"]`, `df[["x_m", "y_m"]]`.

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
df = gdf.copy()
df["x1_unemp_rate"] = df["Unemp14"] / df["LaborFrc"]
df["x2_no_hs"] = df["NoHS14"] / df[["NoHS14", "HSGrad14", "SmClg14", "ClgGrad14"]].sum(axis=1)
df["y"] = df["PerCInc14"]
centroids = df.geometry.centroid
df["x_m"] = centroids.x
df["y_m"] = centroids.y

assert df[["x1_unemp_rate", "x2_no_hs", "y", "x_m", "y_m"]].isna().sum().sum() == 0
assert df["x1_unemp_rate"].between(0, 1).all()
assert df["x2_no_hs"].between(0, 1).all()
df[["community", "y", "x1_unemp_rate", "x2_no_hs", "x_m", "y_m"]].describe()
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0. `describe()` menunjukkan `x1_unemp_rate` dan `x2_no_hs` di antara 0 dan 1, serta `y` dalam rentang ratusan ribu.

- [ ] **Langkah 3: Tambahkan catatan perbedaan spesifikasi**

Sel markdown tepat di atas kode, satu baris: `Catatan: spesifikasi menyebut lima kategori pendidikan; dataset ACS ini hanya memuat empat (NoHS14, HSGrad14, SmClg14, ClgGrad14), jadi X2 memakai jumlah keempatnya.`

- [ ] **Langkah 4: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: engineer unemployment rate, no-HS share, and centroids"
```

---

### Task 4: Split dan helper evaluasi

**Files:**
- Modify: `chicago-capita.ipynb` (markdown `## 4. Train/Test Split`, lalu sel kode)

**Interfaces:**
- Mengonsumsi: `df` dari Task 3.
- Menghasilkan: `idx_train`, `idx_test` (pandas Index), `X_train`, `X_test` (array float n×2), `y_train`, `y_test` (array float n), `coords_train`, `coords_test` (array n×2 dalam meter), dan `evaluate(name: str, y_true, y_pred) -> dict` yang mengembalikan key `Model`, `R2`, `RMSE`, `MAE`.

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["x1_unemp_rate", "x2_no_hs"]
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)
X_train, X_test = df.loc[idx_train, FEATURES].to_numpy(), df.loc[idx_test, FEATURES].to_numpy()
y_train, y_test = df.loc[idx_train, "y"].to_numpy(), df.loc[idx_test, "y"].to_numpy()
coords_train = df.loc[idx_train, ["x_m", "y_m"]].to_numpy()
coords_test = df.loc[idx_test, ["x_m", "y_m"]].to_numpy()

def evaluate(name, y_true, y_pred):
    y_pred = np.asarray(y_pred).ravel()
    return {"Model": name,
            "R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred)}

results = {}
assert len(idx_train) == 61 and len(idx_test) == 16
assert X_train.shape == (61, 2) and X_test.shape == (16, 2)
print(idx_test.sort_values().tolist())
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0 dan 16 indeks data uji tercetak.

- [ ] **Langkah 3: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: add deterministic split and metric helper"
```

---

### Task 5: Model GWR

**Files:**
- Modify: `chicago-capita.ipynb` (markdown `## 5. Model 1: Geographically Weighted Regression`, lalu sel kode)

**Interfaces:**
- Mengonsumsi: `coords_train`, `coords_test`, `X_train`, `X_test`, `y_train`, `y_test`, `evaluate`, `results`.
- Menghasilkan: `results["GWR"]` berisi tiga metrik, plus `gwr_bw` (bandwidth terpilih).

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

y_train_col = y_train.reshape(-1, 1)
selector = Sel_BW(coords_train, y_train_col, X_train, fixed=False, kernel="bisquare")
try:
    gwr_bw = selector.search()
    gwr_fixed = False
except Exception as exc:
    print("adaptive bandwidth failed:", exc)
    gwr_bw, gwr_fixed = 30, True
    selector = Sel_BW(coords_train, y_train_col, X_train, fixed=gwr_fixed, kernel="bisquare")
    gwr_bw = selector.search(bw_min=2)

gwr_res = GWR(coords_train, y_train_col, X_train, gwr_bw,
              fixed=gwr_fixed, kernel="bisquare").fit()
gwr_pred = gwr_res.predict(coords_test, X_test).predictions.ravel()

results["GWR"] = evaluate("GWR", y_test, gwr_pred)
print(f"bandwidth={gwr_bw} fixed={gwr_fixed} in-sample R2={gwr_res.R2:.4f}")
results["GWR"]
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0, sebuah angka untuk `bandwidth`, dan dict berisi `R2`, `RMSE`, `MAE`. Jika `mgwr` melempar error pada 61 titik, pakai cabang fallback, jangan mengurangi ukuran data latih.

- [ ] **Langkah 3: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: fit geographically weighted regression"
```

---

### Task 6: Model Decision Tree

**Files:**
- Modify: `chicago-capita.ipynb` (markdown `## 6. Model 2: Decision Tree`, lalu sel kode)

**Interfaces:**
- Mengonsumsi: split yang sama seperti Task 4.
- Menghasilkan: `results["Decision Tree"]`.

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
results["Decision Tree"] = evaluate("Decision Tree", y_test, tree.predict(X_test))
results["Decision Tree"]
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0 dan dict metrik. Hyperparameter default memang disengaja; jangan tuning di sini agar perbandingan tetap reprodusibel.

- [ ] **Langkah 3: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: fit decision tree regressor"
```

---

### Task 7: Model KNN

**Files:**
- Modify: `chicago-capita.ipynb` (markdown `## 7. Model 3: K-Nearest Neighbors`, lalu sel kode)

**Interfaces:**
- Mengonsumsi: split yang sama seperti Task 4.
- Menghasilkan: `results["KNN"]`.

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
results["KNN"] = evaluate("KNN", y_test, knn.predict(X_test_scaled))
results["KNN"]
```

- [ ] **Langkah 2: Jalankan notebook dan lihat lolos**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0 dan dict metrik untuk KNN.

- [ ] **Langkah 3: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: fit scaled k-nearest neighbors"
```

---

### Task 8: Sel perbandingan dan eksekusi penuh

**Files:**
- Modify: `chicago-capita.ipynb` (sel terakhir: markdown `## 8. Hasil Perbandingan`, lalu sel kode)

**Interfaces:**
- Mengonsumsi: `results` yang diisi Task 5–7.
- Menghasilkan: DataFrame `comparison` terurut menurut `RMSE`, dan verdict tercetak yang menyebut model terbaik.

- [ ] **Langkah 1: Tulis cek yang gagal**

```python
comparison = pd.DataFrame(results.values()).sort_values("RMSE").reset_index(drop=True)
assert set(comparison["Model"]) == {"GWR", "Decision Tree", "KNN"}
assert comparison[["R2", "RMSE", "MAE"]].notna().all().all()
print(f"PerCInc14 mean = {df['y'].mean():.1f} (data units, not re-scaled)")
print(f"Best model by RMSE: {comparison.loc[0, 'Model']}")
comparison
```

- [ ] **Langkah 2: Tambahkan markdown verdict**

Di bawah sel tabel, sel markdown berisi tiga angka dan satu kalimat yang diisi dari hasil eksekusi nyata: model terbaik menurut RMSE, R²-nya, dan selisih RMSE dengan runner-up. Ditulis paling akhir, setelah tabel tercetak; jangan pernah ditebak sebelum eksekusi.

- [ ] **Langkah 3: Jalankan seluruh notebook dari awal sampai akhir**

Jalankan: `jupyter nbconvert --to notebook --execute --inplace chicago-capita.ipynb`
Harapan: exit 0, tabel tiga baris terurut naik menurut RMSE, satu baris `Best model by RMSE:`.

- [ ] **Langkah 4: Commit**

```bash
git add chicago-capita.ipynb
git commit -m "feat: add model comparison table and verdict"
```

---

## Self-Review

- **Cakupan spec:** judul/tabel variabel/URL dataset dipertahankan (Task 1); load (Task 2); X₁/X₂/Y/lokasi (Task 3); GWR, Decision Tree, KNN (Task 5–7); sel perbandingan akhir (Task 8). Tidak ada celah.
- **Pemeriksaan langkah:** setiap langkah menyebut satu sel, satu perintah, atau satu assertion. Satu-satunya pilihan terbuka bagi pelaksana adalah jalur fallback GWR, yang sudah dijelaskan pemicunya dan bukti cetaknya.
- **Konsistensi tipe:** `results` adalah dict berkunci nama model dan hanya dikonsumsi lewat `pd.DataFrame(results.values())`; `evaluate` didefinisikan sekali di Task 4 dan dipanggil dengan urutan argumen yang sama di Task 5–7; kolom `df` yang dibuat di Task 3 adalah satu-satunya yang dibaca sel berikutnya.
- **Fokus Review:** skala Y (Task 3 langkah 2, cetakan Task 8), penyebut pendidikan (Task 3 langkah 3), GWR out-of-sample (Task 5 memakai `predict(coords_test, X_test)`), kebocoran scaler (Task 7 fit pada `X_train`), sampel GWR kecil (fallback Task 5).
- **Proporsi:** rencana lebih panjang dari 3 sel notebook saat ini karena ia mendefinisikan interface setiap sel secara tepat; tidak ada body sel yang ditulis melebihi apa yang sudah ditentukan oleh signature API dan assertion.
