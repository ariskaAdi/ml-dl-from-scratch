# Panduan Membuat Asisten Machine Learning & Code Lokal

*Per 28 Oktober 2026*

## Ringkasan

Tujuannya adalah asisten lokal yang andal untuk dua hal:

1. **Machine learning**: menghitung regresi, mengolah matriks, membuat prediksi, membantu menentukan variabel y, dan bernalar dengan baik dalam bahasa Indonesia maupun Inggris.
2. **Coding**: menulis kode dengan penalaran yang baik dan mengikuti gaya Anda sendiri.

Model dasarnya **Qwen3.5 4B**, yang dijalankan di laptop (i3-1215U, RAM 16 GB, tanpa GPU NVIDIA) dan dilatih di GPU gratis Kaggle atau Colab.

Strategi intinya: **model menalar, Python menghitung.** Model 4B sering salah dalam aritmetika panjang, misalnya invers matriks atau koefisien regresi dari banyak data. Jadi asisten dilatih untuk menjelaskan konsep, menulis kode Python (NumPy, pandas, scikit-learn) untuk perhitungannya, lalu menafsirkan hasilnya.

Alur kerjanya:

1. Jalankan model di laptop dan beri system prompt yang tepat.
2. Uji dengan soal sungguhan dan catat kelemahannya.
3. Susun dataset latihan dari kelemahan itu.
4. Fine-tune dengan LoRA di GPU cloud.
5. Evaluasi, ekspor ke GGUF, lalu pakai lagi di laptop.

---

## Tahap 1: Menjalankan model di laptop

1. Unduh dan pasang **LM Studio** untuk Windows dari lmstudio.ai.
2. Di menu pencarian, ketik `Qwen3.5 4B` dan pilih versi **GGUF Q4_K_M** (sekitar 2,5–3 GB). Kalau terlalu lambat, coba versi 2B.
3. Tutup browser dan aplikasi berat, lalu muat model. Atur *context length* ke 4096 dulu.
4. Masukkan system prompt di bawah ini, lalu mulai bertanya.
5. Pasang **Python** beserta `numpy`, `pandas`, `scikit-learn`, dan `matplotlib` untuk menjalankan kode yang ditulis asisten.

System prompt asisten ML:

```text
Kamu adalah asisten machine learning yang teliti. Jawab dalam bahasa yang dipakai pengguna (Indonesia atau Inggris).

Cara kerja:
1. Pahami dulu masalahnya: apa tujuannya, data apa yang tersedia, dan variabel mana yang menjadi target (y).
2. Jelaskan alasan langkah demi langkah dengan singkat.
3. Untuk perhitungan dengan banyak angka (regresi, invers matriks, statistik), tulis kode Python dengan numpy/pandas/scikit-learn. Jangan menebak hasil hitungan.
4. Setelah hasil kode diberikan, tafsirkan artinya: koefisien, R², error, dan keterbatasannya.
5. Jika informasi kurang, ajukan satu pertanyaan klarifikasi.
6. Jika tidak yakin, katakan tidak yakin.
```

Perkiraan kecepatan di CPU i3: model 4B sekitar 5–10 token per detik, model 2B sekitar dua kali lebih cepat.

---

## Tahap 2: Uji awal dan catat kelemahan

Sebelum melatih apa pun, ukur dulu kemampuan model apa adanya. Siapkan sekitar 30 soal:

| Kategori | Contoh soal | Yang dinilai |
| --- | --- | --- |
| Regresi | "Hitung regresi linear dari x = [1,2,3,4,5], y = [2,4,5,4,5]" | Rumus benar, kode jalan, tafsiran koefisien dan R² |
| Matriks | "Cari invers dan determinan matriks 3×3 ini" | Langkah benar, hasil cocok dengan NumPy |
| Prediksi | "Data penjualan 12 bulan, prediksi bulan ke-13" | Pilihan model masuk akal, ada validasi |
| Variabel y | "Data pelanggan bank: kolom mana yang jadi target untuk memprediksi churn?" | Alasan pemilihan y dan fitur, waspada kebocoran data |
| Penalaran | Soal yang sama dalam bahasa Indonesia dan Inggris | Kualitas bahasa dan konsistensi jawaban |

Untuk setiap jawaban, catat: benar atau salah, jenis kesalahannya (hitungan, konsep, bahasa, format), dan jawaban yang seharusnya.

Simpan soal-soal ini sebagai **set uji tetap** dan jangan pernah memasukkannya ke data latihan.

---

## Tahap 3: Menyiapkan dataset latihan

Targetkan **1.000–3.000 contoh berkualitas** untuk putaran pertama.

| Jenis contoh | Porsi | Isi |
| --- | --- | --- |
| Soal hitungan dengan kode | 35% | Regresi, matriks, statistik: penalaran, kode Python, hasil, tafsiran |
| Pemilihan variabel y dan fitur | 20% | Skenario bisnis atau riset: target, fitur, risiko kebocoran data |
| Konsep ML | 20% | Bias-variance, overfitting, regularisasi, metrik evaluasi |
| Hitungan kecil manual | 15% | Soal 2×2 atau data sedikit, langkah demi langkah |
| Pertanyaan ambigu | 10% | Asisten bertanya balik atau mengakui tidak yakin |

Bagi kira-kira 60% bahasa Indonesia dan 40% bahasa Inggris.

### Format file

File JSONL: satu baris berisi satu percakapan. Contoh (ditampilkan dengan baris baru agar mudah dibaca; di file aslinya harus satu baris):

```json
{"messages": [
  {"role": "system", "content": "Kamu adalah asisten machine learning yang teliti..."},
  {"role": "user", "content": "Hitung regresi linear dari x = [1,2,3,4,5] dan y = [2,4,5,4,5]."},
  {"role": "assistant", "content": "Kita cari garis y = a + b·x dengan metode kuadrat terkecil.\n\n1. Rata-rata: x̄ = 3, ȳ = 4.\n2. Sxy = Σ(x−x̄)(y−ȳ) = 6, Sxx = Σ(x−x̄)² = 10.\n3. b = Sxy / Sxx = 0,6; a = ȳ − b·x̄ = 2,2.\n\nPersamaan: y = 2,2 + 0,6x. R² = 0,6, artinya 60% variasi y dijelaskan oleh x.\n\nVerifikasi:\n```python\nimport numpy as np\nb, a = np.polyfit([1,2,3,4,5], [2,4,5,4,5], 1)\nprint(a, b)  # 2.2 0.6\n```"}
]}
```

### Cara mengumpulkan data

1. **Tulis sendiri** 100–200 contoh terbaik sebagai standar gaya.
2. **Ambil dari materi kuliah atau buku**, lalu tulis ulang menjadi tanya-jawab.
3. **Buat data sintetis** dengan bantuan model yang lebih besar. Periksa syarat layanannya, karena sebagian penyedia melarang hasilnya dipakai untuk melatih model lain. Model open-weight berlisensi Apache 2.0 biasanya lebih aman.
4. **Gunakan dataset terbuka** di Hugging Face, dengan memperhatikan lisensinya.

### Kontrol kualitas

- [ ] Jalankan setiap blok kode Python secara otomatis; buang contoh yang error atau hasilnya tidak cocok.
- [ ] Cek ulang angka di teks agar sama dengan output kode.
- [ ] Hapus duplikat dan contoh yang terlalu mirip.
- [ ] Baca acak 50 contoh secara manual.
- [ ] Pastikan tidak ada soal dari set uji.

---

## Tahap 4: Fine-tuning dengan LoRA di GPU cloud

Gunakan **Kaggle Notebooks** (cadangan: Google Colab) dan **Unsloth**.

1. Buat akun Kaggle, verifikasi nomor HP agar GPU aktif, lalu buka notebook contoh Unsloth untuk keluarga Qwen terbaru (ada di GitHub Unsloth).
2. Unggah file JSONL sebagai Kaggle Dataset dan hubungkan ke notebook.
3. Muat Qwen3.5 4B dalam mode 4-bit.
4. Pasang adaptor LoRA dengan pengaturan di bawah.
5. Jalankan pelatihan dan pantau *loss*. Turun lalu mendatar itu normal; turun hampir ke nol berarti model menghafal.
6. Simpan adaptor LoRA ke output Kaggle atau akun Hugging Face Anda.

| Parameter | Nilai awal | Catatan |
| --- | --- | --- |
| LoRA rank (r) | 16 | Naikkan ke 32 jika model kurang menyerap pola |
| LoRA alpha | 16–32 | Biasanya sama dengan atau 2× rank |
| Learning rate | 2e-4 | Turunkan ke 1e-4 jika loss naik-turun tajam |
| Epoch | 2 | Lebih dari 3 berisiko overfitting |
| Panjang maksimum | 2048 token | Naikkan ke 4096 untuk data code |
| Batch efektif | 8–16 | Gabungan batch size dan gradient accumulation |
| Latih hanya jawaban | Ya | Loss dihitung hanya pada bagian asisten |

Untuk 2.000 contoh, satu putaran biasanya memakan sekitar 30–90 menit.

**Catatan konsep:** LoRA bukan jenis pembelajaran, melainkan cara hemat memperbarui bobot. Bobot asli dibekukan, dan yang dilatih hanya dua matriks kecil (ΔW ≈ B·A), biasanya kurang dari 1% parameter. Tahap ini adalah **SFT (supervised) dengan LoRA**.

**Opsional:** setelah putaran pertama berhasil, tambahkan **GRPO (reinforcement learning) dengan LoRA**, dengan hadiah otomatis jika angka jawaban cocok dengan NumPy atau kode lulus unit test.

---

## Tahap 5: Evaluasi, ekspor, dan kembali ke laptop

1. **Uji di notebook** dengan set uji, nilai dengan cara yang sama seperti Tahap 2.
2. **Bandingkan skor** per kategori, sebelum dan sesudah.
3. **Cek kemampuan umum** dengan 5–10 pertanyaan di luar topik. Kalau jadi kacau, kurangi epoch atau learning rate.
4. **Perbaiki data, bukan pengaturan.** Tambah 100–300 contoh untuk kategori yang lemah, lalu latih ulang.
5. **Ekspor ke GGUF.**
6. **Bawa ke laptop**, taruh di folder model LM Studio, muat dengan system prompt yang sama.

### Dari training ke file GGUF

| Langkah | Hasil | Ukuran (perkiraan) | Dipakai untuk |
| --- | --- | --- | --- |
| 1. Selesai training | `adapter_model.safetensors` + `adapter_config.json` | Puluhan–ratusan MB | Disimpan untuk melatih versi berikutnya |
| 2. Merge dengan model dasar | Model utuh format `safetensors` (16-bit) | Sekitar 8 GB | Bahan konversi |
| 3. Konversi + kuantisasi | File `.gguf` (q4_k_m) | Sekitar 2,5–3 GB | Dijalankan di LM Studio / Ollama |

Di Unsloth, langkah 2 dan 3 cukup satu perintah:

```python
model.save_pretrained_gguf("ml-asisten-v1", tokenizer, quantization_method="q4_k_m")
```

**Penting:** GGUF untuk menjalankan model, bukan untuk melatihnya lagi. Simpan adaptor LoRA dari langkah 1 untuk versi berikutnya.

Beri nama setiap versi dengan jelas, misalnya `ml-asisten-v1-q4.gguf`, beserta catatan skornya.

---

## Tambahan: Asisten Code dengan Gaya Anda

Alurnya sama persis (Tahap 1–5). Yang berbeda adalah isi data dan cara mengujinya.

### 1. Model dasar

Qwen3.5 4B tetap bisa dipakai. Jika ada varian khusus coding di ukuran kecil yang muat di laptop, bandingkan keduanya di set uji sebelum memilih.

### 2. Data untuk gaya Anda

- Ambil kode dari repo-repo Anda, lalu ubah menjadi pasangan instruksi dan jawaban. Contoh: "buat fungsi untuk membaca CSV dan membersihkan data kosong" dengan jawaban berupa fungsi asli tulisan Anda.
- Sertakan kebiasaan Anda: penamaan variabel, gaya komentar dan docstring, struktur folder, library favorit, cara menangani error.
- Tulis contoh *code review*: kode yang kurang bagus beserta perbaikan versi Anda. Ini mengajarkan selera, bukan hanya sintaks.
- **Hapus password, API key, dan data pribadi** dari kode sebelum dipakai.

### 3. Data untuk reasoning

Contoh jawaban sebaiknya memperlihatkan proses berpikir sebelum menulis kode:

1. Memahami masalah dan batasannya.
2. Memilih pendekatan dan alasannya.
3. Mempertimbangkan kasus khusus (input kosong, tipe data salah, data besar).
4. Menulis kode.
5. Menjelaskan kode dan cara mengujinya.

### 4. Evaluasi dengan unit test

- Setiap soal di set uji diberi beberapa tes otomatis (misalnya dengan `pytest`).
- Skor = persentase tes yang lulus, sebelum dan sesudah fine-tuning.
- Mekanisme yang sama bisa dipakai untuk GRPO: hadiah diberikan jika kode lulus tes. Ini cara paling efektif untuk menaikkan kemampuan reasoning.

### Contoh system prompt asisten code

```text
Kamu adalah asisten pemrograman yang teliti dan mengikuti gaya penulisan kode pengguna.

Cara kerja:
1. Pahami kebutuhan dan batasannya. Jika ambigu, ajukan satu pertanyaan klarifikasi.
2. Jelaskan pendekatan singkat sebelum menulis kode.
3. Pertimbangkan kasus khusus dan penanganan error.
4. Tulis kode yang rapi, dengan nama variabel jelas dan docstring.
5. Sertakan contoh cara menguji kode tersebut.
6. Jika tidak yakin suatu library atau fungsi ada, katakan tidak yakin.
```

### Mengelola dua asisten

| Pilihan | Kelebihan | Kekurangan |
| --- | --- | --- |
| Dua adaptor LoRA terpisah (ML dan code) | Masing-masing fokus, mudah diperbaiki sendiri-sendiri | Dua file GGUF, harus ganti model sesuai tugas |
| Satu model dari dataset gabungan | Praktis, ML dan coding saling melengkapi (asisten ML juga menulis Python) | Perbaikan satu bidang bisa memengaruhi bidang lain |

### Batasan

Model 4B cukup bagus untuk fungsi dan skrip pendek, tapi kesulitan memahami proyek besar dengan banyak file. Kombinasikan dengan **RAG atas codebase Anda** agar model bisa melihat file yang relevan saat menjawab.

---

## Batasan umum dan langkah lanjutan

Model 4B hasil fine-tuning bisa menjadi asisten yang sangat berguna, tapi tidak akan menyamai model frontier dalam penalaran panjang atau soal yang benar-benar baru. Fine-tuning mengasah gaya dan kebiasaan kerja; pengetahuan dasarnya tetap milik model awal.

- **Sambungkan ke Python otomatis** lewat aplikasi agen lokal yang bisa menjalankan kode buatan model. Ini peningkatan terbesar untuk akurasi hitungan.
- **Tambahkan RAG** untuk catatan kuliah, buku, dokumentasi, atau codebase Anda.
- **Naik ke model 9B** jika 4B terasa dangkal, dengan konsekuensi lebih lambat.
- **Upgrade hardware** ke RAM 32 GB atau GPU NVIDIA 8 GB+ agar bisa memakai model mixture-of-experts yang lebih pintar dan melatih sebagian di laptop sendiri.
