# TODO: Belajar ML & DL dari Nol

Dua tugas, masing-masing dikerjakan dua kali:

| Tugas | `ml/` (scikit-learn) | `dl/` (PyTorch) |
| --- | --- | --- |
| Kemiskinan Jawa Barat (regresi, data BPS) | Fase 1 | Fase 2 |
| Klasifikasi kode | Fase 4 | Fase 5 |

**Tujuan:** memahami apa yang dikerjakan scikit-learn "di balik layar" dengan menulis ulang sendiri di PyTorch. Kalau hasil PyTorch kamu mendekati scikit-learn, berarti kamu paham.

## Cara memakai file ini

- Kerjakan berurutan. Centang `[ ]` → `[x]` setiap selesai satu langkah.
- **Ketik sendiri kodenya, jangan copy-paste.** Tujuannya pemahaman, bukan sekadar kode yang jalan.
- Setiap langkah punya **Cek pemahaman**. Jawab dulu di kepala atau di catatan sebelum lanjut. Kalau ragu, tanyakan konsepnya ke Claude.
- Jalankan semua skrip **dari folder `project`** dengan `.venv` aktif:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python ml/kemiskinan/train.py
  ```
- Kode di tiap langkah **ditambahkan di bawah** kode langkah sebelumnya dalam file yang sama. Baris `import` baru selalu ditaruh di **bagian atas** file.
- Semua contoh kode di sini sudah diuji dengan library di `.venv` kamu (torch 2.15, scikit-learn 1.9, pandas 3.1). Angka di "Hasil yang diharapkan" adalah hasil uji coba tersebut. Hasilmu boleh sedikit berbeda di digit belakang.

## Struktur folder akhir

```text
project/
├── .venv/
├── requirements.txt
├── todo.md                        ← file ini
├── data/
│   ├── bps/                       ← file mentah BPS + SUMBER.md (jangan diedit)
│   ├── siapkan_kemiskinan.py      ← Fase 1 (A1)
│   ├── kemiskinan.csv             ← dihasilkan A1
│   ├── buat_dataset_kode.py       ← Fase 3
│   └── kode.csv                   ← dihasilkan Fase 3
├── ml/
│   ├── kemiskinan/train.py        ← Fase 1
│   └── klasifikasi_code/train.py  ← Fase 4
└── dl/
    ├── kemiskinan/train.py        ← Fase 2
    └── klasifikasi_code/train.py  ← Fase 5
```

---

## Fase 0: Persiapan ✅

- [x] Membuat venv: `python -m venv .venv`
- [x] Mengaktifkan venv: `.\.venv\Scripts\Activate.ps1`
- [x] Memasang library: numpy, pandas, scikit-learn, matplotlib, torch (CPU)
- [x] Mencatat library: `python -m pip freeze > requirements.txt`
- [x] Membuat folder `ml/` dan `dl/` beserta subfolder dan `train.py`

---

## Fase 1: Kemiskinan Jawa Barat dengan scikit-learn

**Pertanyaannya:** dari angka pengangguran (TPT) dan rata-rata lama sekolah (RLS), seberapa baik kita bisa menebak persentase penduduk miskin (P0) sebuah kab/kota di Jawa Barat?

**Data:** 3 file BPS Jabar di `data/bps/` (sumbernya dicatat di `data/bps/SUMBER.md`).

| File | Isi | Tahun |
| --- | --- | --- |
| `kemiskinan_jabar.csv` | **Target y:** persentase penduduk miskin (P0, %) | 2022–2026 |
| `tpt_jabar.csv` | Tingkat pengangguran terbuka (%) | 2020–2024 |
| `rls_jabar.csv` | Rata-rata lama sekolah penduduk usia 25+ (tahun) | 2022–2026 |

### [ ] A1. Menyiapkan data: dari 3 file mentah menjadi 1 tabel

File: `data/siapkan_kemiskinan.py`

```python
"""Menggabungkan data BPS Jabar (kemiskinan, TPT, RLS) menjadi satu tabel siap latih."""
import pandas as pd


def baca(path, nama_kolom):
    """Baca satu file BPS 'melebar', lalu ubah jadi 'memanjang'."""
    df = pd.read_csv(path)
    print(f"{path}: {df.shape} (bentuk asli)")

    # 1. Buang baris total provinsi: itu rata-rata, bukan kab/kota
    df = df[df["Wilayah"] != "Provinsi Jawa Barat"]

    # 2. Melebar -> memanjang: satu baris = satu wilayah di satu tahun
    df = df.melt(id_vars="Wilayah", var_name="tahun", value_name=nama_kolom)

    # 3. "Tahun2022" -> 2022 (angka)
    df["tahun"] = df["tahun"].str.replace("Tahun", "").astype(int)
    return df


p0 = baca("data/bps/kemiskinan_jabar.csv", "p0")
tpt = baca("data/bps/tpt_jabar.csv", "tpt")
rls = baca("data/bps/rls_jabar.csv", "rls")
print("\nSetelah melt:", p0.shape, tpt.shape, rls.shape)
print(p0.head())

# 4. Gabungkan berdasarkan wilayah DAN tahun.
#    how="inner": hanya pasangan (wilayah, tahun) yang ada di KETIGA file.
df = p0.merge(tpt, on=["Wilayah", "tahun"], how="inner")
df = df.merge(rls, on=["Wilayah", "tahun"], how="inner")

# 5. Fitur tambahan: 1 kalau kota, 0 kalau kabupaten
df["kota"] = df["Wilayah"].str.startswith("Kota").astype(int)

# 6. Rapikan nama dan urutan kolom
df = df.rename(columns={"Wilayah": "wilayah"})
df = df[["wilayah", "tahun", "kota", "tpt", "rls", "p0"]]
df = df.sort_values(["wilayah", "tahun"]).reset_index(drop=True)

# 7. Pemeriksaan: jangan percaya, cek!
assert df.isna().sum().sum() == 0, "Ada sel kosong!"
assert df["wilayah"].nunique() == 27, "Jabar seharusnya 27 kab/kota"
print("\nTahun yang dipakai:", sorted(df["tahun"].unique()))
print("Hasil akhir:", df.shape)
print(df.head())

df.to_csv("data/kemiskinan.csv", index=False)
print("\nDisimpan ke data/kemiskinan.csv")
```

Jalankan: `python data/siapkan_kemiskinan.py`

| Konsep | Penjelasan |
| --- | --- |
| **Melebar vs memanjang** | File BPS: 1 baris per wilayah, tahun menjadi kolom. Model butuh 1 baris per (wilayah, tahun), dengan setiap kolom berisi satu variabel. `melt` melakukan perubahan itu |
| **Baris total** | Baris "Provinsi Jawa Barat" adalah ringkasan dari baris lain. Kalau ikut dipakai, data dihitung dua kali |
| **`merge` inner** | Menyambung tabel berdasarkan kunci (wilayah + tahun). Pasangan yang tidak ada di salah satu file dibuang |
| **`assert`** | Pemeriksaan otomatis. Kalau syaratnya salah, program berhenti dengan pesan error. Lebih baik gagal di sini daripada model diam-diam belajar dari data rusak |

**Hasil yang diharapkan:**
```text
data/bps/kemiskinan_jabar.csv: (28, 6) (bentuk asli)
...
Setelah melt: (135, 3) (135, 3) (135, 3)
Tahun yang dipakai: [2022, 2023, 2024]
Hasil akhir: (81, 6)
```

**Cek pemahaman:**
1. Kenapa 28 baris × 5 kolom tahun menjadi 135 baris setelah `melt`? Hitung sendiri.
2. Kenapa hasil akhirnya hanya 81 baris (2022–2024)? Tahun mana yang hilang dan dari file mana?
3. Kenapa kunci penggabungannya `wilayah` **dan** `tahun`, bukan `wilayah` saja? Apa yang terjadi kalau hanya `wilayah`? (Coba saja, lalu lihat jumlah barisnya!)

### [ ] A2. Mengenal data

File: `ml/kemiskinan/train.py`

```python
import pandas as pd

df = pd.read_csv("data/kemiskinan.csv")
print(df.shape)
print(df.describe().round(2))
print(df[["p0", "tpt", "rls", "kota"]].corr()["p0"].round(3))
print(df.groupby("kota")[["p0", "tpt", "rls"]].mean().round(2))
```

**Hasil yang diharapkan:**
```text
p0      1.000
tpt    -0.247
rls    -0.733
kota   -0.513

        p0   tpt    rls
kota
0     9.25  6.79   7.99      ← kabupaten
1     6.34  7.93  10.59      ← kota
```

**Cek pemahaman:**
1. Bandingkan dengan tebakanmu sebelum melihat data: fitur mana yang paling kuat, dan arahnya positif atau negatif?
2. Korelasi TPT dengan kemiskinan **negatif**: makin tinggi pengangguran, makin *sedikit* penduduk miskin. Lihat tabel `groupby("kota")`. Bagaimana perbedaan kota dan kabupaten menjelaskan hal yang "aneh" ini?
3. Kenapa di kabupaten orang miskin justru jarang tercatat sebagai pengangguran? (Petunjuk: definisi BPS untuk "bekerja" adalah minimal 1 jam dalam seminggu terakhir.)

### [ ] A3. Regresi linear sederhana: rumus manual vs library

```python
import numpy as np                                    # ← atas
from sklearn.linear_model import LinearRegression     # ← atas

x = df["tpt"].to_numpy()
y = df["p0"].to_numpy()
Sxy = np.sum((x - x.mean()) * (y - y.mean()))
Sxx = np.sum((x - x.mean()) ** 2)
b = Sxy / Sxx
a = y.mean() - b * x.mean()
pred = a + b * x
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"Rumus manual : p0 = {a:.4f} + ({b:.4f}) * tpt   R2 = {r2:.4f}")

b_np, a_np = np.polyfit(x, y, 1)
print(f"np.polyfit   : p0 = {a_np:.4f} + ({b_np:.4f}) * tpt")

lr = LinearRegression().fit(df[["tpt"]], y)
print(f"scikit-learn : p0 = {lr.intercept_:.4f} + ({lr.coef_[0]:.4f}) * tpt")
```

Ini rumus yang sama dengan contoh di `panduan-asisten-ml-dan-code.md` (Sxy, Sxx, b, a), sekarang dipakai untuk data sungguhan.

- **b (kemiringan):** setiap pengangguran naik 1 poin persen, P0 berubah sebesar b poin persen.
- **a (intercept):** tebakan P0 kalau TPT = 0. Ini hanya titik awal garis, tidak selalu bermakna di dunia nyata.
- `df[["tpt"]]` memakai **dua** kurung siku, karena sklearn selalu meminta X berbentuk tabel (2 dimensi), walaupun fiturnya hanya satu.

**Hasil yang diharapkan:** ketiga cara memberi `p0 = 10.5840 + (-0.3211) * tpt`, dengan `R2 = 0.0612`.

✅ **Acuan berhasil:** rumus manualmu menghasilkan angka yang sama persis dengan `np.polyfit` dan sklearn.

**Cek pemahaman:** R² = 0,06. Dalam kalimat sehari-hari, apa artinya?

### [ ] A4. Menggambar grafik

```python
import matplotlib                      # ← atas
matplotlib.use("Agg")                  # ← atas: simpan grafik ke file, tanpa membuka jendela
import matplotlib.pyplot as plt        # ← atas

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, fitur, judul in [(axes[0], "tpt", "Pengangguran (TPT, %)"),
                         (axes[1], "rls", "Rata-rata lama sekolah (tahun)")]:
    for kota, warna, label in [(0, "tab:orange", "Kabupaten"), (1, "tab:blue", "Kota")]:
        bagian = df[df["kota"] == kota]
        ax.scatter(bagian[fitur], bagian["p0"], c=warna, label=label, alpha=0.7)
    kemiringan, potong = np.polyfit(df[fitur], df["p0"], 1)
    xs = np.linspace(df[fitur].min(), df[fitur].max(), 50)
    ax.plot(xs, potong + kemiringan * xs, "k--", label="garis regresi")
    ax.set_xlabel(judul)
    ax.set_ylabel("Penduduk miskin (%)")
    ax.legend()
fig.tight_layout()
fig.savefig("ml/kemiskinan/grafik.png", dpi=120)
print("Grafik disimpan ke ml/kemiskinan/grafik.png")
```

Buka `ml/kemiskinan/grafik.png` di VS Code.

**Yang akan kamu lihat:** di grafik kiri (TPT), titik-titiknya menyebar ke mana-mana dan garisnya hampir datar. Di grafik kanan (RLS), titik-titiknya rapi menurun mengikuti garis. Titik kota (biru) berkumpul di kanan bawah: sekolah lebih lama, lebih sedikit penduduk miskin.

**Cek pemahaman:**
1. Hanya dengan melihat grafik, tanpa angka: fitur mana yang lebih berguna untuk menebak kemiskinan?
2. Cari titik biru yang terpencil di kiri atas grafik RLS. Itu kota apa? (Petunjuk: lihat A9.)

### [ ] A5. Penilaian jujur: cross-validation per wilayah

```python
from sklearn.dummy import DummyRegressor                                # ← atas
from sklearn.metrics import mean_absolute_error, r2_score               # ← atas
from sklearn.model_selection import GroupKFold, cross_val_predict       # ← atas

cv = GroupKFold(n_splits=5)


def nilai(nama, model, fitur):
    pred = cross_val_predict(model, df[fitur], df["p0"], groups=df["wilayah"], cv=cv)
    mae = mean_absolute_error(df["p0"], pred)
    r2 = r2_score(df["p0"], pred)
    print(f"{nama:<28} MAE={mae:.3f}  R2={r2:.3f}")
    return pred


nilai("Baseline (rata-rata)", DummyRegressor(), ["tpt"])
```

R² di A3 (0,06) dihitung dari data yang **sama** dengan data latihnya. Itu seperti ujian dengan soal yang sudah dibocorkan. Sekarang kita menilai dengan jujur.

| Konsep | Penjelasan |
| --- | --- |
| **Kenapa tidak `train_test_split` biasa?** | Data hanya 81 baris. Kalau 20% disisihkan sebagai data uji, hanya sekitar 16 baris yang dipakai menilai, sehingga hasilnya sangat tergantung keberuntungan |
| **Cross-validation (CV)** | 27 wilayah dibagi 5 kelompok (*fold*). Bergiliran, 1 kelompok jadi data uji dan 4 sisanya jadi data latih. Akhirnya **setiap baris pernah diuji sekali**, oleh model yang tidak pernah melihatnya |
| **Kenapa per wilayah (`groups`)?** | Kab. Bogor muncul 3 kali (2022, 2023, 2024), dan angkanya mirip setiap tahun. Kalau Bogor 2022 ada di data latih dan Bogor 2023 di data uji, model tinggal "mengingat" Bogor. `GroupKFold` menjamin ketiga baris Bogor selalu berada di sisi yang sama |
| **`cross_val_predict`** | Mengembalikan tebakan untuk setiap baris, masing-masing dibuat oleh model yang tidak pernah melihat baris itu |

**Hasil yang diharapkan:** `Baseline (rata-rata) MAE=2.374  R2=-0.112`

**Cek pemahaman:** secara teori, R² model "selalu tebak rata-rata" = 0. Kenapa di sini **negatif**? (Petunjuk: rata-rata yang dipakai baseline dihitung dari data latih di setiap fold, bukan dari data uji.)

### [ ] A6. Menambah fitur satu per satu

```python
from sklearn.pipeline import make_pipeline           # ← atas
from sklearn.preprocessing import StandardScaler     # ← atas

for fitur in [["tpt"], ["rls"], ["tpt", "rls"], ["tpt", "rls", "kota"]]:
    nilai("Linear " + "+".join(fitur), make_pipeline(StandardScaler(), LinearRegression()), fitur)
```

**Hasil yang diharapkan:**
```text
Linear tpt                   MAE=2.365  R2=-0.127
Linear rls                   MAE=1.682  R2=0.429
Linear tpt+rls               MAE=1.745  R2=0.396
Linear tpt+rls+kota          MAE=1.735  R2=0.350
```

**Cek pemahaman:**
1. Model dengan TPT saja **lebih buruk dari baseline**. Apa artinya untuk rencana awalmu memakai pengangguran sebagai fitur?
2. Menambah fitur (TPT, lalu kota) justru **menurunkan** R² dibanding RLS saja. Bukankah lebih banyak informasi seharusnya lebih baik? (Kata kunci: overfitting dengan data sedikit, dan informasi `kota` sebagian besar sudah terkandung di RLS.)

### [ ] A7. Model non-linear: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor   # ← atas

nilai("Random Forest tpt+rls+kota",
      RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42),
      ["tpt", "rls", "kota"])
```

**Hasil yang diharapkan:** `MAE=1.981  R2=0.216`, kalah jauh dari regresi linear.

**Cek pemahaman:** Random Forest biasanya unggul di banyak kasus. Kenapa di sini kalah? (Petunjuk: hanya 63–66 baris latih per fold, dan hubungan RLS–kemiskinan di grafik A4 sudah hampir lurus.)

### [ ] A8. Model akhir dan membaca koefisiennya

```python
FITUR = ["tpt", "rls"]
akhir = LinearRegression().fit(df[FITUR], df["p0"])
print(f"p0 = {akhir.intercept_:.3f}"
      + "".join(f" + ({k:.3f} * {f})" for f, k in zip(FITUR, akhir.coef_)))
```

Setelah dinilai dengan CV, model akhir dilatih dengan **semua** data. Di sini tidak ada scaler, supaya koefisiennya bisa dibaca dalam satuan aslinya.

**Hasil yang diharapkan:** `p0 = 20.317 + (0.030 * tpt) + (-1.383 * rls)`

- **RLS −1,383:** setiap tambahan 1 tahun rata-rata lama sekolah, persentase penduduk miskin lebih rendah sekitar 1,4 poin persen, dengan TPT dianggap tetap.
- **TPT +0,030:** di A3 koefisiennya **−0,32**, sekarang **+0,03**. Tandanya berbalik!

**Cek pemahaman (penting!):**
1. Kenapa koefisien TPT berbalik tanda setelah RLS dimasukkan? (Petunjuk: di A2, kota punya TPT tinggi **dan** RLS tinggi. Di A3, TPT "meminjam" pengaruh RLS. Setelah RLS ada di model, pengaruh TPT yang tersisa hampir nol.) Fenomena ini disebut *confounding*.
2. Apakah koefisien −1,383 membuktikan bahwa menambah 1 tahun sekolah **menyebabkan** kemiskinan turun 1,4 poin? Kenapa belum tentu?

### [ ] A9. Daerah mana yang paling meleset?

```python
import joblib   # ← atas

df["tebakan"] = akhir.predict(df[FITUR])
df["selisih"] = df["p0"] - df["tebakan"]
print(df[df["tahun"] == 2024].sort_values("selisih")[["wilayah", "p0", "tebakan", "selisih"]]
      .round(2).to_string(index=False))

joblib.dump(akhir, "ml/kemiskinan/model_linear.joblib")
```

**Hasil yang diharapkan** (potongan, tahun 2024):
```text
                wilayah    p0  tebakan  selisih
     Kabupaten Sukabumi  6.87    10.38    -3.51     ← jauh lebih sedikit penduduk miskin dari tebakan
            Kota Banjar  5.85     8.27    -2.42
...
           Kota Cirebon  9.02     5.94     3.08
       Kota Tasikmalaya 11.10     7.19     3.91     ← jauh lebih banyak penduduk miskin dari tebakan
```

Selisih positif berarti kemiskinan sebenarnya **lebih tinggi** dari yang "seharusnya" menurut model.

**Cek pemahaman:**
1. Cari daerahmu sendiri di tabel. Apakah model menebak dengan tepat? Dari pengalamanmu, apa yang mungkin menjelaskan selisihnya?
2. Kota Tasikmalaya dan Kota Cirebon: RLS-nya tinggi, tapi kemiskinannya juga tinggi. Faktor apa yang tidak ada di model kita? Fitur apa yang ingin kamu tambahkan?

---

## Fase 2: Kemiskinan Jawa Barat dengan PyTorch

File: `dl/kemiskinan/train.py`

Di sini kamu menulis sendiri hal-hal yang di Fase 1 dikerjakan diam-diam oleh `.fit()` dan `cross_val_predict`.

### [ ] B1. Data yang sama, diubah ke tensor

```python
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

df = pd.read_csv("data/kemiskinan.csv")
FITUR = ["tpt", "rls"]
X = df[FITUR].to_numpy()
y = df["p0"].to_numpy()

scaler = StandardScaler()
Xt = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
print("Xt:", Xt.shape, "yt:", yt.shape)
```

- **Tensor** adalah array milik PyTorch, mirip array numpy, tapi bisa menghitung gradien otomatis.
- `float32`: neural network biasanya memakai presisi 32-bit.
- `.view(-1, 1)` mengubah bentuk `[81]` menjadi `[81, 1]` (satu kolom), supaya cocok dengan output model.
- `torch.manual_seed(42)`: bobot awal model selalu sama, sehingga hasilnya bisa diulang.

**Hasil yang diharapkan:** `Xt: torch.Size([81, 2]) yt: torch.Size([81, 1])`

**Cek pemahaman:** kenapa di sini data perlu di-scale? (Petunjuk: coba hapus scaler di B2 dan lihat apa yang terjadi pada loss.)

### [ ] B2. Training loop: inti dari semua deep learning

```python
def latih(model, X_latih, y_latih, epoch=500, lr=0.1, weight_decay=0.0, cetak=False):
    """Training loop: 5 langkah yang sama untuk semua model."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    for e in range(1, epoch + 1):
        model.train()
        loss = loss_fn(model(X_latih), y_latih)   # 1-2. tebak & hitung salahnya
        optimizer.zero_grad()                     # 3. hapus gradien lama
        loss.backward()                           # 4. hitung gradien baru
        optimizer.step()                          # 5. perbaiki bobot
        if cetak and e in (1, 5, 10, 20, 50, 100, 500):
            print(f"epoch {e:>3}  loss={loss.item():.4f}")
    return model


linear = latih(nn.Linear(len(FITUR), 1), Xt, yt, cetak=True)
```

Lima langkah di dalam loop ini adalah **inti dari semua deep learning**, dari model 3 bobot ini sampai LLM raksasa di file panduanmu.

| Istilah | Arti |
| --- | --- |
| `nn.Linear(2, 1)` | `p0 = b + w1·tpt + w2·rls`. Persamaan yang sama dengan `LinearRegression` |
| **Loss (MSE)** | Rata-rata kuadrat selisih tebakan dengan nilai asli |
| **Gradien** | Arah dan besarnya perubahan setiap bobot supaya loss turun |
| **lr (learning rate)** | Seberapa besar langkah perbaikan |
| **Epoch** | Satu kali putaran melihat seluruh data latih |
| **weight_decay** | Hukuman untuk bobot yang terlalu besar (regularisasi). Dipakai nanti untuk MLP |

Fungsi `latih()` sengaja dibuat terpisah, karena nanti dipakai lagi untuk 10 model (2 jenis × 5 fold).

**Hasil yang diharapkan:**
```text
epoch   1  loss=82.4324
epoch   5  loss=16.2572
epoch  10  loss=4.7449
epoch  20  loss=3.3411
epoch  50  loss=3.3183
epoch 100  loss=3.3183
epoch 500  loss=3.3183
```

**Cek pemahaman:**
1. Loss berhenti turun di 3,3183 sejak epoch 50. Kenapa tidak bisa turun ke 0?
2. Kenapa `zero_grad()` wajib? Hapus barisnya, lalu lihat apa yang terjadi.
3. Ubah `lr` menjadi `0.001`, lalu `1.5`. Apa yang terjadi pada loss?

### [ ] B3. Membuktikan bobotnya sama dengan sklearn

```python
sk = LinearRegression().fit(scaler.transform(X), y)
print(f"{'':<6} {'PyTorch':>9} {'sklearn':>9}")
print(f"{'bias':<6} {linear.bias.item():>9.4f} {sk.intercept_:>9.4f}")
for nama, w_pt, w_sk in zip(FITUR, linear.weight.detach().numpy().ravel(), sk.coef_):
    print(f"{nama:<6} {w_pt:>9.4f} {w_sk:>9.4f}")
```

**Hasil yang diharapkan:**
```text
         PyTorch   sklearn
bias      8.2817    8.2817
tpt       0.0612    0.0612
rls      -1.9873   -1.9873
```

✅ **Acuan berhasil:** semua bobot sama sampai 4 angka desimal. Kamu sudah membuktikan bahwa `LinearRegression().fit()` (sekali hitung dengan rumus) dan training loop-mu (500 langkah kecil) sampai di jawaban yang sama.

**Cek pemahaman:**
1. Angka ini berbeda dari A8 (`0.030` dan `-1.383`). Kenapa? (Petunjuk: di sini fiturnya di-scale.) Apa arti `-1.9873` dalam satuan simpangan baku?
2. Bias 8,2817 sama dengan angka apa di data? (Lihat `describe()` di A2.)

### [ ] B4. Cross-validation buatan sendiri: linear vs MLP

```python
cv = GroupKFold(n_splits=5)


def buat_mlp():
    return nn.Sequential(nn.Linear(len(FITUR), 8), nn.ReLU(), nn.Linear(8, 1))


hasil = {"Linear (PyTorch)": np.zeros(len(y)), "MLP (PyTorch)": np.zeros(len(y))}
for nomor, (idx_latih, idx_uji) in enumerate(cv.split(X, y, groups=df["wilayah"]), 1):
    sc = StandardScaler().fit(X[idx_latih])                  # fit di data latih fold ini saja
    X_latih = torch.tensor(sc.transform(X[idx_latih]), dtype=torch.float32)
    X_uji = torch.tensor(sc.transform(X[idx_uji]), dtype=torch.float32)
    y_latih = torch.tensor(y[idx_latih], dtype=torch.float32).view(-1, 1)

    for nama, buat, wd in [("Linear (PyTorch)", lambda: nn.Linear(len(FITUR), 1), 0.0),
                           ("MLP (PyTorch)", buat_mlp, 1e-3)]:
        model = latih(buat(), X_latih, y_latih, epoch=1000, lr=0.05, weight_decay=wd)
        model.eval()
        with torch.no_grad():
            hasil[nama][idx_uji] = model(X_uji).numpy().ravel()
    print(f"fold {nomor}: latih {len(idx_latih)} baris, uji {len(idx_uji)} baris "
          f"({df['wilayah'].iloc[idx_uji].nunique()} wilayah)")

for nama, pred in hasil.items():
    print(f"{nama:<20} MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.3f}")

torch.save(linear.state_dict(), "dl/kemiskinan/model_linear.pt")
```

Ini yang dikerjakan `cross_val_predict` di A5, ditulis tangan:
1. `cv.split(...)` memberi nomor baris latih dan uji untuk setiap fold.
2. Scaler di-`fit` **ulang di setiap fold**, hanya dari data latih fold itu, supaya tidak bocor.
3. Model baru dibuat di setiap fold, karena model dari fold sebelumnya sudah pernah melihat data uji fold ini.
4. Tebakan setiap fold disimpan di posisinya, lalu dinilai sekaligus di akhir.

| Konsep | Penjelasan |
| --- | --- |
| **MLP** | `Linear → ReLU → Linear`: lapisan tersembunyi berisi 8 "neuron". ReLU (`max(0, x)`) membuat model bisa belajar pola melengkung |
| **`model.eval()` + `torch.no_grad()`** | Mode menguji: tidak perlu menghitung gradien |

**Hasil yang diharapkan:**
```text
fold 1: latih 63 baris, uji 18 baris (6 wilayah)
...
Linear (PyTorch)     MAE=1.745  R2=0.396
MLP (PyTorch)        MAE=2.095  R2=0.123
```

✅ **Acuan berhasil:** R² linear PyTorch sama dengan `Linear tpt+rls` di A6 (0,396). Ini bukti bahwa CV buatanmu bekerja persis seperti `cross_val_predict`.

**Cek pemahaman:**
1. MLP kalah jauh dari linear. Dengan hanya sekitar 65 baris latih, kenapa model yang lebih "pintar" justru lebih buruk?
2. Coba ubah `weight_decay` MLP menjadi `1e-2` atau `1e-1`, atau ubah jumlah neuron dari 8 menjadi 2. Apakah MLP membaik? Kenapa?
3. Pelajaran besarnya: **kapan deep learning layak dipakai, dan kapan tidak?**

---

## Fase 3: Membuat dataset kode

File: `data/buat_dataset_kode.py` (buat folder `data` dulu)

**Idenya:** Python bawaan di laptopmu berisi ribuan fungsi asli buatan programmer profesional, misalnya modul `http`, `asyncio`, dan `json`. Kita ambil setiap fungsi, lalu beri label topik berdasarkan modul asalnya. Tugas model nanti: **membaca kode sebuah fungsi dan menebak topiknya.**

### [ ] C1. Skrip pembuat dataset

```python
"""Membuat dataset klasifikasi kode dari library bawaan Python."""
import ast
import sysconfig
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Modul bawaan Python dikelompokkan per topik
TOPIK = {
    "jaringan": ["http", "urllib", "socket", "socketserver", "ssl", "ftplib",
                 "smtplib", "imaplib", "poplib", "xmlrpc", "wsgiref", "ipaddress"],
    "konkurensi": ["asyncio", "concurrent", "threading", "multiprocessing",
                   "queue", "sched", "selectors", "subprocess"],
    "teks": ["re", "string", "textwrap", "difflib", "html", "xml", "json",
             "csv", "tomllib", "email", "configparser", "gettext"],
    "file": ["pathlib", "shutil", "glob", "fnmatch", "zipfile", "tarfile",
             "gzip", "bz2", "lzma", "tempfile", "filecmp", "fileinput",
             "ntpath", "posixpath", "genericpath"],
    "testing": ["unittest", "doctest", "pdb", "bdb", "trace", "traceback",
                "profile", "cProfile", "pstats", "timeit"],
    "matematika": ["statistics", "fractions", "_pydecimal", "random",
                   "numbers", "heapq", "bisect", "colorsys"],
}
MAKS_PER_TOPIK = 600   # supaya jumlah contoh tiap topik tidak terlalu timpang
MAKS_PER_FILE = 60     # supaya satu file raksasa tidak mendominasi satu topik
MIN_BARIS = 3          # fungsi yang terlalu pendek sulit ditebak

stdlib = Path(sysconfig.get_paths()["stdlib"])
print("Folder library bawaan:", stdlib)

baris_data = []
for topik, daftar_modul in TOPIK.items():
    for modul in daftar_modul:
        paket = stdlib / modul
        if paket.is_dir():
            # Lewati folder tes bawaan (misalnya unittest/test), karena isinya
            # kode tes, bukan kode milik modul itu sendiri
            files = [f for f in paket.rglob("*.py") if "test" not in f.parent.name]
        else:
            files = [stdlib / f"{modul}.py"]
        for file in files:
            if not file.exists():
                continue
            sumber = file.read_text(encoding="utf-8", errors="ignore")
            try:
                pohon = ast.parse(sumber)
            except SyntaxError:
                continue
            for node in ast.walk(pohon):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kode = ast.get_source_segment(sumber, node)
                    if kode and kode.count("\n") + 1 >= MIN_BARIS:
                        baris_data.append({
                            "kode": kode,
                            "label": topik,
                            "file": str(file.relative_to(stdlib)),
                        })

df = pd.DataFrame(baris_data).drop_duplicates(subset="kode")
print("Sebelum diseimbangkan:")
print(df["label"].value_counts(), "\n")

# Ambil maksimal MAKS_PER_TOPIK contoh per topik
# (acak dulu urutannya, lalu ambil N baris pertama dari tiap file dan tiap topik)
df = df.sample(frac=1, random_state=42)
df = df.groupby("file").head(MAKS_PER_FILE)
df = df.groupby("label").head(MAKS_PER_TOPIK).reset_index(drop=True)

# Bagi train/test BERDASARKAN FILE: semua fungsi dari satu file masuk ke sisi
# yang sama, supaya model tidak "menyontek" dari fungsi tetangganya.
# Stratified = proporsi tiap topik di train dan test diusahakan mirip.
# n_splits=5 berarti data dibagi 5 bagian; 1 bagian (20%) jadi data test.
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
idx_train, idx_test = next(sgkf.split(df, df["label"], groups=df["file"]))
df["split"] = "train"
df.loc[idx_test, "split"] = "test"

keluaran = Path(__file__).parent / "kode.csv"
df.to_csv(keluaran, index=False)
print("Sesudah diseimbangkan:")
print(pd.crosstab(df["label"], df["split"]))
print(f"\nDisimpan ke {keluaran} ({len(df)} baris)")
```

Jalankan: `python data/buat_dataset_kode.py`

| Konsep | Penjelasan |
| --- | --- |
| **`ast`** | Modul bawaan Python untuk membaca kode sebagai pohon struktur. Lebih andal daripada mencari kata `def` dengan regex |
| **Menyeimbangkan data** | Tanpa batas, `konkurensi` punya 1.583 contoh dan `testing` hanya 418. Model jadi condong menebak topik yang banyak |
| **Split per file** | Fungsi-fungsi dalam satu file biasanya mirip (nama variabel, gaya, import). Kalau satu file terbelah ke train dan test, model "menyontek" |
| **Kolom `split`** | Pembagian dilakukan sekali dan disimpan, sehingga Fase 4 dan Fase 5 dijamin memakai data uji yang sama |

**Hasil yang diharapkan:** sekitar **2.913 baris**. Tiap topik punya sekitar 20% data test (file 100, jaringan 121, konkurensi 120, matematika 50, teks 120, testing 71).

**Cek pemahaman:**
1. Buka `data/kode.csv` di VS Code. Ambil 3 contoh, lalu tebak sendiri topiknya sebelum melihat label. Mudah atau sulit?
2. Kenapa `random_state=42` penting di sini, padahal skrip ini hanya dijalankan sekali?

---

## Fase 4: Klasifikasi kode dengan scikit-learn

File: `ml/klasifikasi_code/train.py`

### [ ] D1. Membaca dataset

```python
import pandas as pd

df = pd.read_csv("data/kode.csv")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print("Train:", len(train), "Test:", len(test))

LABEL = sorted(df["label"].unique())
```

**Hasil yang diharapkan:** `Train: 2331 Test: 582`

### [ ] D2. Fungsi penilaian dan baseline

```python
from sklearn.dummy import DummyClassifier                    # ← atas
from sklearn.metrics import accuracy_score, f1_score         # ← atas


def nilai(nama, model):
    model.fit(train["kode"], train["label"])
    pred = model.predict(test["kode"])
    acc = accuracy_score(test["label"], pred)
    f1 = f1_score(test["label"], pred, average="macro")
    print(f"{nama:<22} akurasi={acc:.3f}  F1-macro={f1:.3f}")
    return model, pred


nilai("Baseline (terbanyak)", DummyClassifier(strategy="most_frequent"))
```

| Metrik | Arti |
| --- | --- |
| **Akurasi** | Persentase tebakan yang benar |
| **F1-macro** | Rata-rata F1 dari tiap topik, dengan bobot yang sama. Topik kecil (matematika) dianggap sama penting dengan topik besar |

**Hasil yang diharapkan:** `akurasi=0.206  F1-macro=0.057`

**Cek pemahaman:** akurasi baseline 20,6%, tapi F1-macro hanya 0,057. Kenapa selisihnya jauh sekali?

### [ ] D3. Mengubah kode menjadi angka: TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer   # ← atas

# token = nama variabel/fungsi/kata kunci: huruf atau _ diikuti huruf/angka/_
POLA_TOKEN = r"[A-Za-z_][A-Za-z0-9_]*"
tfidf = TfidfVectorizer(token_pattern=POLA_TOKEN, lowercase=True, min_df=2)
X_contoh = tfidf.fit_transform(train["kode"])
print("Ukuran matriks:", X_contoh.shape)
print("Contoh kosakata:", list(tfidf.vocabulary_)[:15])
```

Model hanya mengerti angka, jadi kode harus diubah dulu.

1. **Tokenisasi:** `def baca_file(path):` → `def`, `baca_file`, `path`.
2. **TF (term frequency):** seberapa sering sebuah token muncul di satu fungsi.
3. **IDF (inverse document frequency):** token yang muncul di hampir semua fungsi, seperti `self`, `def`, dan `return`, diberi bobot kecil karena tidak membedakan topik.
4. `min_df=2`: token yang hanya muncul di 1 fungsi dibuang.

**Hasil yang diharapkan:** `Ukuran matriks: (2331, 5562)`, artinya 2.331 fungsi × 5.562 token unik.

**Cek pemahaman:** sebagian besar isi matriks ini angka 0. Kenapa?

### [ ] D4. TF-IDF + Logistic Regression

```python
from sklearn.linear_model import LogisticRegression                       # ← atas
from sklearn.metrics import classification_report, confusion_matrix      # ← atas (gabung dengan import metrics tadi)
from sklearn.pipeline import make_pipeline                                # ← atas

model, pred = nilai("TF-IDF + LogReg", make_pipeline(
    TfidfVectorizer(token_pattern=POLA_TOKEN, lowercase=True, min_df=2,
                    sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=10),
))
print(classification_report(test["label"], pred, digits=3))
print(pd.DataFrame(confusion_matrix(test["label"], pred, labels=LABEL),
                   index=LABEL, columns=LABEL))
```

- **Logistic Regression:** walau namanya "regression", ini model klasifikasi. Setiap topik punya bobot untuk setiap token. Skor topik = jumlah (bobot × nilai TF-IDF), lalu skor itu diubah menjadi peluang.
- `sublinear_tf=True`: token yang muncul 20 kali tidak dianggap 20 kali lebih penting, melainkan pakai log.
- `C=10`: kebalikan dari kekuatan regularisasi. Makin besar C, makin bebas model mengikuti data train.
- **Confusion matrix:** baris = topik asli, kolom = tebakan model. Angka di diagonal adalah tebakan yang benar.

**Hasil yang diharapkan:** `akurasi=0.660  F1-macro=0.648`. Topik `testing` paling sulit (recall sekitar 0,32), dan sering tertukar dengan `teks` dan `jaringan`.

**Cek pemahaman:**
1. Akurasi 66% vs baseline 21%. Bagus atau jelek? (Bandingkan juga dengan tebakanmu sendiri di C1.)
2. Kenapa fungsi `testing` (misalnya dari `pdb` atau `traceback`) sering ditebak `teks`? Buka beberapa contohnya di CSV.

### [ ] D5. Mengintip "otak" model

```python
vec = model.named_steps["tfidfvectorizer"]
clf = model.named_steps["logisticregression"]
kata = vec.get_feature_names_out()
for i, topik in enumerate(clf.classes_):
    top = clf.coef_[i].argsort()[-8:][::-1]
    print(f"{topik:<11}", ", ".join(kata[top]))
```

**Hasil yang diharapkan** (kurang lebih):
```text
file        path, bytes, file, fileobj, tarinfo, mode, _buffer, os
jaringan    host, cookie, http, headers, response, url, network, request
konkurensi  loop, fd, process, thread, async, cancel, future, util
...
```

**Cek pemahaman:** apakah kata-kata ini masuk akal menurut pengalamanmu sebagai programmer? Ini kelebihan model linear: keputusannya bisa dijelaskan.

### [ ] D6. Mencoba kode buatanmu sendiri dan menyimpan model

```python
import joblib   # ← atas

contoh = '''
def ambil_halaman(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode("utf-8")
'''
print("Tebakan:", model.predict([contoh])[0])

joblib.dump(model, "ml/klasifikasi_code/model_logreg.joblib")
```

**Hasil yang diharapkan:** `Tebakan: jaringan`

Coba tulis 5 fungsimu sendiri, satu per topik, lalu lihat berapa yang ditebak benar.

---

## Fase 5: Klasifikasi kode dengan PyTorch

File: `dl/klasifikasi_code/train.py`

Di sini kamu membuat sendiri yang tadi dikerjakan `TfidfVectorizer`: tokenisasi dan kosakata.

### [ ] E1. Membaca dataset yang sama

```python
import re
import time
from collections import Counter

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

torch.manual_seed(42)

df = pd.read_csv("data/kode.csv")
train = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
```

### [ ] E2. Tokenisasi buatan sendiri

```python
POLA_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenisasi(kode):
    return [t.lower() for t in POLA_TOKEN.findall(kode)]


print(tokenisasi("def baca_file(path):\n    return open(path).read()"))
```

**Hasil yang diharapkan:** `['def', 'baca_file', 'path', 'return', 'open', 'path', 'read']`

Pola regex-nya sama dengan `POLA_TOKEN` di D3, supaya perbandingannya adil.

### [ ] E3. Kosakata (vocab) dari data train saja

```python
hitung = Counter(t for kode in train["kode"] for t in tokenisasi(kode))
vocab = {"<unk>": 0}                      # 0 = token yang tidak dikenal
for token, n in hitung.most_common():
    if n >= 2:
        vocab[token] = len(vocab)
print("Ukuran vocab:", len(vocab))

LABEL = sorted(df["label"].unique())
label_ke_id = {nama: i for i, nama in enumerate(LABEL)}


def ke_id(kode):
    return [vocab.get(t, 0) for t in tokenisasi(kode)] or [0]
```

- **Vocab:** kamus `token → nomor`. Neural network tidak membaca kata, hanya nomor.
- **`<unk>`:** token di data test yang tidak pernah muncul di train diberi nomor 0.
- `n >= 2`: sama seperti `min_df=2`, token yang terlalu jarang dibuang.

**Hasil yang diharapkan:** `Ukuran vocab: 7285`

**Cek pemahaman:** kenapa vocab hanya dibangun dari data train? Apa yang bocor kalau data test ikut dipakai?

### [ ] E4. Menyiapkan batch untuk EmbeddingBag

```python
def buat_batch(bagian):
    """Gabungkan semua token jadi satu deret panjang + 'offsets' penanda awal tiap kode."""
    daftar = [torch.tensor(ke_id(k)) for k in bagian["kode"]]
    offsets = torch.tensor([0] + [len(d) for d in daftar[:-1]]).cumsum(0)
    token = torch.cat(daftar)
    y = torch.tensor([label_ke_id[l] for l in bagian["label"]])
    return token, offsets, y
```

Panjang setiap fungsi berbeda-beda. Supaya tidak perlu diberi padding, semua token disambung jadi satu deret panjang, lalu `offsets` menandai di mana setiap fungsi dimulai.

Contoh: fungsi A punya 3 token dan fungsi B punya 5 token → `token` berisi 8 angka, `offsets = [0, 3]`.

### [ ] E5. Model

```python
class KlasifikasiKode(nn.Module):
    def __init__(self, n_vocab, n_dim, n_kelas):
        super().__init__()
        self.embedding = nn.EmbeddingBag(n_vocab, n_dim, mode="mean")
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(n_dim, n_kelas)

    def forward(self, token, offsets):
        vektor = self.embedding(token, offsets)   # 1 vektor per potongan kode
        return self.out(self.dropout(vektor))      # skor tiap kelas (logits)


model = KlasifikasiKode(len(vocab), 64, len(LABEL))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
```

| Konsep | Penjelasan |
| --- | --- |
| **Embedding** | Setiap token punya vektor 64 angka yang **dipelajari** selama training. Token yang sering muncul di konteks mirip akan punya vektor yang mirip. Bandingkan dengan TF-IDF, yang angkanya hanya dihitung, bukan dipelajari |
| **EmbeddingBag (mean)** | Mengambil vektor semua token di satu fungsi lalu merata-ratakannya, jadi satu vektor per fungsi |
| **Dropout** | Saat latihan, 30% angka dimatikan secara acak supaya model tidak terlalu bergantung pada token tertentu (mengurangi overfitting) |
| **Logits** | 6 skor mentah, satu per topik. Skor tertinggi = tebakan |
| **CrossEntropyLoss** | Loss untuk klasifikasi. Kecil kalau model yakin pada topik yang benar |
| **`nn.Module`** | Cara standar membuat model PyTorch: lapisan didefinisikan di `__init__`, alur datanya di `forward` |

### [ ] E6. Training loop dengan data validasi

```python
# Sisihkan ~15% data train sebagai validasi, dibagi PER FILE (sama seperti test).
# Data test TIDAK disentuh sampai akhir.
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
idx_latih, idx_val = next(gss.split(train, groups=train["file"]))
latih, val = train.iloc[idx_latih], train.iloc[idx_val]
X_val = buat_batch(val)

mulai = time.perf_counter()
for epoch in range(1, 31):
    model.train()
    acak = latih.sample(frac=1, random_state=epoch)
    total = 0.0
    for i in range(0, len(acak), 32):
        token, offsets, y = buat_batch(acak.iloc[i:i + 32])
        loss = loss_fn(model(token, offsets), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(X_val[0], X_val[1]).argmax(1)
        acc = (pred == X_val[2]).float().mean().item()
        print(f"epoch {epoch:>2}  loss={total / len(acak):.4f}  akurasi_val={acc:.3f}")
print(f"Lama latihan: {time.perf_counter() - mulai:.0f} detik")
```

Lima langkah training-nya **sama persis** dengan B2. Yang berbeda hanya data dan modelnya.

**Hasil yang diharapkan** (sekitar 15 detik):
```text
epoch  5  loss=0.3973  akurasi_val=0.466
epoch 10  loss=0.0960  akurasi_val=0.494
...
epoch 30  loss=0.0071  akurasi_val=0.489
```

Perhatikan: **loss train hampir 0, tapi akurasi validasi mentok sekitar 0,49 sejak epoch 10.** Model menghafal data train (overfitting).

### [ ] E7. Evaluasi akhir di data test dan simpan model

```python
X_test = buat_batch(test)
model.eval()
with torch.no_grad():
    pred = model(X_test[0], X_test[1]).argmax(1).numpy()
benar = X_test[2].numpy()
print(f"EmbeddingBag (PyTorch) akurasi={accuracy_score(benar, pred):.3f}  "
      f"F1-macro={f1_score(benar, pred, average='macro'):.3f}")
print(classification_report(benar, pred, target_names=LABEL, digits=3))

torch.save({"model": model.state_dict(), "vocab": vocab, "label": LABEL},
           "dl/klasifikasi_code/model_embbag.pt")
```

**Hasil yang diharapkan:** `akurasi=0.577  F1-macro=0.579`

✅ **Acuan berhasil:** akurasi > 0,50, jauh di atas baseline (0,206), dan selisihnya dengan sklearn (0,660) kurang dari 0,10.

Vocab ikut disimpan bersama model, karena tanpa vocab model tidak bisa menerjemahkan kode baru menjadi nomor.

**Cek pemahaman:**
1. Di sini model sklearn yang sederhana justru menang. Kenapa? (Petunjuk: dataset kecil, cuma sekitar 2.000 contoh. Embedding perlu banyak data untuk belajar.)
2. **Eksperimen kebocoran data (penting!):** ganti tiga baris `gss = ...` sampai `latih, val = ...` di E6 dengan:
   ```python
   val = train.sample(frac=0.1, random_state=42)
   latih = train.drop(val.index)
   ```
   Jalankan, lalu lihat `akurasi_val`. Kamu akan melihat sekitar **0,89**, padahal akurasi test tetap sekitar 0,6. Kenapa validasinya "berbohong"? Hubungkan dengan alasan split per file di Fase 3. Setelah itu, kembalikan ke versi `GroupShuffleSplit`.

---

## Fase 6: Bandingkan dan refleksi

### [ ] F1. Isi lembar hasil

Isi dengan angka dari laptopmu sendiri. Kolom "Acuan" adalah hasil uji coba instruktur.

**Kemiskinan Jabar (cross-validation 5 fold per wilayah, 81 baris):**

| Model | Fitur | Folder | R² kamu | R² acuan | MAE acuan |
| --- | --- | --- | --- | --- | --- |
| Baseline (rata-rata) | – | ml | | −0,112 | 2,374 |
| Linear | tpt | ml | | −0,127 | 2,365 |
| Linear | rls | ml | | 0,429 | 1,682 |
| Linear | tpt + rls | ml | | 0,396 | 1,745 |
| Linear | tpt + rls + kota | ml | | 0,350 | 1,735 |
| Random Forest | tpt + rls + kota | ml | | 0,216 | 1,981 |
| Linear buatan sendiri | tpt + rls | dl | | 0,396 | 1,745 |
| MLP | tpt + rls | dl | | 0,123 | 2,095 |

**Klasifikasi kode (data test, 582 fungsi):**

| Model | Folder | Akurasi kamu | Akurasi acuan | F1-macro acuan |
| --- | --- | --- | --- | --- |
| Baseline (terbanyak) | ml | | 0,206 | 0,057 |
| TF-IDF + LogReg | ml | | 0,660 | 0,648 |
| EmbeddingBag | dl | | 0,577 | 0,579 |

### [ ] F2. Daftar acuan keberhasilan

- [ ] A3: rumus manual Sxy/Sxx menghasilkan a dan b yang sama dengan `np.polyfit` dan sklearn
- [ ] B3: bobot linear PyTorch sama dengan sklearn (selisih < 0,001)
- [ ] B4: R² linear dari CV buatan sendiri sama dengan `cross_val_predict` (0,396)
- [ ] Bisa menjelaskan kenapa koefisien TPT berbalik tanda di A8, dan kenapa MLP kalah di B4
- [ ] D4: akurasi TF-IDF + LogReg > 0,60
- [ ] E7: akurasi EmbeddingBag > 0,50 dan selisihnya dengan sklearn < 0,10
- [ ] Bisa menjawab semua **Cek pemahaman** tanpa melihat jawaban

### [ ] F3. Tulis refleksi singkat (di `catatan.md`)

1. Apa yang dikerjakan `.fit()` di sklearn, kalau dijelaskan dengan 5 langkah training loop PyTorch?
2. Kapan kamu memilih sklearn, dan kapan PyTorch?
3. Apa itu kebocoran data? Sebutkan 3 tempat di proyek ini yang dijaga supaya tidak bocor.
4. Kenapa model yang paling rumit tidak selalu menang?

---

## Fase 7: Tantangan lanjutan (opsional)

Kerjakan setelah Fase 1–6 selesai. Setiap tantangan cukup diubah sedikit dari kode yang sudah ada.

- [ ] **Kemiskinan:** unduh indikator BPS Jabar lain (misalnya umur harapan hidup, PDRB per kapita, atau persentase rumah tangga dengan sanitasi layak), simpan di `data/bps/`, tambahkan ke `siapkan_kemiskinan.py`, lalu lihat apakah R² naik. Jangan lupa catat sumbernya di `SUMBER.md`.
- [ ] **Kemiskinan:** model RLS saja bisa memakai data 2022–2026 (135 baris), karena tidak bergantung pada TPT yang hanya sampai 2024. Apakah R² berubah dengan data lebih banyak?
- [ ] **Kemiskinan:** latih dengan data 2022–2023, lalu tebak 2024 (split berdasarkan waktu). Bandingkan dengan CV per wilayah. Mana yang lebih sulit, dan kenapa?
- [ ] **Kemiskinan:** buat grafik tebakan vs nilai asli (`plt.scatter(df["p0"], pred)` plus garis diagonal). Wilayah mana yang paling jauh dari diagonal?
- [ ] **Klasifikasi kode (dl):** lawan overfitting di E6. Tambahkan `weight_decay=1e-4` di Adam, early stopping (berhenti kalau akurasi validasi tidak naik beberapa epoch, lalu pakai bobot terbaik), atau perbesar dropout. Bisa mengejar 0,66?
- [ ] **Klasifikasi kode (ml):** ubah `C` (0,1 / 1 / 10 / 100) dan `min_df`. Catat pengaruhnya.
- [ ] **Klasifikasi kode (ml):** coba `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`, yaitu potongan huruf, bukan kata.
- [ ] **Git:** `git init`, lalu buat `.gitignore` berisi `.venv/`, `*.joblib`, `*.pt`, dan `data/kode.csv` (file model dan data bisa dibuat ulang, jadi tidak perlu disimpan di git).
- [ ] **Langkah besar berikutnya:** kembali ke `panduan-asisten-ml-dan-code.md`, yaitu fine-tuning LLM dengan LoRA. Sekarang kamu sudah paham dasarnya: tokenisasi, embedding, loss, optimizer, validasi, dan overfitting. Semuanya dipakai lagi di sana, hanya dalam skala yang jauh lebih besar.

---

## Lampiran: kalau ada masalah

| Masalah | Solusi |
| --- | --- |
| `No module named sklearn` / `torch` | `.venv` belum aktif. Jalankan `.\.venv\Scripts\Activate.ps1` |
| `FileNotFoundError: data/kode.csv` | Jalankan dari folder `project`, dan pastikan Fase 3 sudah dijalankan |
| `FileNotFoundError` saat menyimpan model | Jalankan dari folder `project`, bukan dari dalam subfolder |
| `FileNotFoundError: data/kemiskinan.csv` | Jalankan dulu `python data/siapkan_kemiskinan.py` (langkah A1) |
| `AssertionError: Jabar seharusnya 27 kab/kota` | Nama wilayah di salah satu file BPS berbeda tulisan, sehingga baris hilang saat `merge`. Bandingkan kolom `Wilayah` ketiga file |
| Grafik tidak muncul di layar | Memang disimpan ke file (`ml/kemiskinan/grafik.png`). Buka file itu di VS Code |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | Jumlah fitur tidak cocok dengan `nn.Linear(...)`. Cek `.shape` tensormu |
| Loss jadi `nan` | Learning rate terlalu besar, atau data belum di-scale |
| Angka sedikit beda dari acuan | Normal, terutama untuk MLP dan EmbeddingBag. Yang penting dekat dan lolos acuan |
