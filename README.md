# Implementasi Transformer dari Nol dengan NumPy
Proyek ini berisi implementasi forward pass dari arsitektur decoder-only Transformer (gaya GPT) yang dibangun sepenuhnya dari nol menggunakan NumPy. Tujuannya adalah untuk memahami mekanisme internal Transformer tanpa bergantung pada library deep learning seperti PyTorch atau TensorFlow.

## Fitur Utama
- Arsitektur Modular: Setiap komponen (Embedding, Attention, FFN, dll.) diimplementasikan dalam kelas atau fungsi terpisah untuk kejelasan dan kemudahan pengujian.

- Positional Encoding Modern: Menggunakan Rotary Positional Encoding (RoPE) untuk menyandikan informasi posisi secara relatif, sebuah pendekatan yang lebih canggih dari sinusoidal encoding.

- Self-Attention dengan Causal Mask: Implementasi Multi-Head Attention yang efisien dengan masking untuk memastikan model bersifat autoregresif (tidak bisa melihat token masa depan).

- Optimasi Parameter: Menerapkan Weight Tying antara lapisan embedding dan output untuk mengurangi jumlah parameter total dan meningkatkan performa model.

- Visualisasi: Dilengkapi dengan skrip visualisasi matriks attention menggunakan Matplotlib dan Seaborn untuk analisis dan debugging.

## Struktur Proyek
transformer_from_scratch.ipynb: Notebook Jupyter utama yang berisi seluruh implementasi kode, penjelasan langkah-demi-langkah, dan blok pengujian model.

## Cara Menjalankan
- Buka Notebook: Buka file transformer_from_scratch.ipynb menggunakan lingkungan google collab

- Jalankan Semua Sel: Untuk menjalankan keseluruhan alur, Anda bisa memilih "Run All" dari menu notebook. Alternatifnya, jalankan setiap sel secara berurutan dari atas ke bawah.

## Hasil yang Diharapkan
- Setelah menjalankan sel pengujian di bagian akhir notebook, Anda akan melihat output berikut di layar:

  - Verifikasi Dimensi: Pesan konfirmasi bahwa bentuk tensor logits yang dihasilkan sudah benar ([batch_size, seq_len, vocab_size]).

  - Validasi Softmax: Pemeriksaan yang menunjukkan bahwa total probabilitas dari fungsi softmax untuk setiap item di batch berjumlah 1.

  - Visualisasi Causal Mask: Sebuah heatmap yang menampilkan matriks bobot attention. Area segitiga kanan atas dari plot akan berwarna gelap (bernilai 0), yang secara visual membuktikan bahwa causal mask berhasil diterapkan.
