
# Daftar Pertanyaan Uji RAG Minci (STT Cipasung)

Disusun berdasarkan isi asli 4 dokumen: `KALENDER.docx`, `KRS.docx`, `PMB.docx`, `BIAYA.docx`.
Setiap fakta di dokumen punya minimal 1 pertanyaan. Beberapa ditulis dalam gaya
santai/singkatan untuk sekalian menguji routing & gate relevansi.

Cara pakai: jalankan tiap pertanyaan ke `ask_minci()`, cek apakah jawabannya
match dengan **kunci jawaban** di sebelahnya (angka/tanggal harus PERSIS sama).

---

## 1. KALENDER.docx

### Gelombang I

| # | Pertanyaan                                                 | Kunci Jawaban             |
| - | ---------------------------------------------------------- | ------------------------- |
| 1 | Kapan pendaftaran mahasiswa baru gelombang 1 dibuka?       | 02 Januari – 09 Mei 2026 |
| 2 | Seleksi penerimaan gelombang I tanggal berapa?             | 12 - 13 Mei 2026          |
| 3 | Pengumuman hasil seleksi gelombang 1 kapan?                | 16 Mei 2026               |
| 4 | Jadwal registrasi administrasi mahasiswa baru gelombang I? | 18 - 23 Mei 2026          |

### Gelombang II

| # | Pertanyaan                                           | Kunci Jawaban            |
| - | ---------------------------------------------------- | ------------------------ |
| 5 | Pendaftaran gelombang 2 mulai kapan sampai kapan?    | 11 Mei – 18 Juli 2026   |
| 6 | Tanggal seleksi PMB gelombang II?                    | 21 - 22 Juli 2026        |
| 7 | Pengumuman hasil seleksi gelombang 2 tanggal berapa? | 25 Juli 2026             |
| 8 | Registrasi administrasi maba gelombang II kapan?     | 27 Juli - 1 Agustus 2026 |

### Gelombang III

| #  | Pertanyaan                                       | Kunci Jawaban                                                           |
| -- | ------------------------------------------------ | ----------------------------------------------------------------------- |
| 9  | Gelombang 3 pendaftarannya kapan?                | 20 Juli - 15 Agustus 2026                                               |
| 10 | Seleksi penerimaan gelombang III tanggal berapa? | 19 - 20 Agustus 2026                                                    |
| 11 | Kapan pengumuman hasil seleksi gelombang 3?      | 24 Agustus 2026                                                         |
| 12 | Jadwal registrasi administrasi gelombang III?    | 26 - 31 Agustus 2026                                                    |
| 13 | min tau ga kapan pmb dibuka?                     | Harus menyebut ketiga gelombang (I, II, III) beserta rentang tanggalnya |

### Kegiatan Akademik Semester Gasal 2026/2027

| #  | Pertanyaan                                      | Kunci Jawaban                   |
| -- | ----------------------------------------------- | ------------------------------- |
| 14 | Kapan herregistrasi dan perwalian dilaksanakan? | 31 Agustus – 12 September 2026 |
| 15 | Jadwal Pra KTMB 2026?                           | 31 Agustus – 05 September 2026 |
| 16 | KTMB 2026 tanggal berapa?                       | 07 - 12 September 2026          |
| 17 | Kapan awal kuliah/praktikum mahasiswa?          | 14 September 2026               |
| 18 | Batas waktu KPRS atau cuti kuliah sampai kapan? | 30 September 2026               |
| 19 | Wisuda sarjana T.A. 2025/2026 kapan?            | 24 Oktober 2026                 |
| 20 | Jadwal UTS semester gasal ini?                  | 02 – 07 November 2026          |
| 21 | Libur natal tanggal berapa?                     | 25 Desember 2026                |
| 22 | Libur tahun baru kapan?                         | 01 Januari 2027                 |
| 23 | Kapan akhir kuliah/praktikum?                   | 02 Januari 2027                 |
| 24 | Jadwal UAS semester ini?                        | 04 – 09 Januari 2027           |
| 25 | Pengolahan nilai dilakukan tanggal berapa?      | 11 – 23 Januari 2027           |
| 26 | Rapat evaluasi akademik kapan?                  | 23 Januari 2027                 |

---

## 2. KRS.docx

### Cara login & pengisian KRS online

| #  | Pertanyaan                                               | Kunci Jawaban                                                     |
| -- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| 27 | Link buat isi KRS online di mana?                        | http://31.58.158.149:8060/                                        |
| 28 | Username buat login KRS online apa?                      | NIM (Nomor Induk Mahasiswa)                                       |
| 29 | Format password login KRS gimana?                        | Tanggal lahir format YYYYMMDD (contoh 20010512)                   |
| 30 | Kalau pembayaran belum terkonfirmasi, KRS bisa diisi ga? | Tidak bisa, wajib hubungi Bagian Keuangan di Tata Usaha dulu      |
| 31 | Setelah login, menu apa yang dipilih buat isi KRS?       | Menu Perkuliahan -> Kartu Rencana Studi Online                    |
| 32 | Gimana cara cetak KRS?                                   | Menu Laporan -> Kartu Rencana Studi -> Tampilkan Laporan -> Cetak |

### Definisi & istilah

| #  | Pertanyaan                         | Kunci Jawaban                                                                                                              |
| -- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 33 | Apa itu Herregistrasi?             | Proses pendaftaran ulang mahasiswa untuk mempertahankan/mengaktifkan status mahasiswa & lanjut ke semester berikutnya      |
| 34 | Perwalian itu apa sih?             | Proses bimbingan/konsultasi/pengawasan terstruktur antara dosen wali dan mahasiswa untuk merencanakan & mengevaluasi studi |
| 35 | KRS itu singkatan dan artinya apa? | Kartu Rencana Studi -- dokumen yang memuat rencana mata kuliah 1 semester, hasil konsultasi & persetujuan dosen wali       |

### Syarat & ketentuan

| #  | Pertanyaan                                                          | Kunci Jawaban                                              |
| -- | ------------------------------------------------------------------- | ---------------------------------------------------------- |
| 36 | Perwalian & KRS paling lambat diisi kapan sebelum kuliah mulai?     | Paling lambat 2 minggu sebelum awal perkuliahan            |
| 37 | Maksimal SKS per semester berapa buat mahasiswa IPK di atas 3?      | Maksimal 24 SKS (termasuk sks mengulang) untuk IPK >= 3.00 |
| 38 | Kalau IPK di bawah 3.00, ambil SKS-nya gimana?                      | Hanya diperkenankan mengambil mata kuliah paket semester   |
| 39 | Kalau mahasiswa tidak perwalian tapi tetap KRS-an gimana statusnya? | Dinyatakan tidak aktif pada semester berlangsung           |
| 40 | Kalau tidak aktif berturut-turut gimana?                            | Dinyatakan mengundurkan diri                               |
| 41 | Siapa yang bisa pantau laporan keaktifan mahasiswa?                 | Ka. Program Studi, melalui SEVIMA                          |

### Prosedur & pihak terkait

| #  | Pertanyaan                                 | Kunci Jawaban                      |
| -- | ------------------------------------------ | ---------------------------------- |
| 42 | Siapa yang menetapkan waktu pengisian KRS? | Wakil Ketua I                      |
| 43 | KRS diisi mahasiswa lewat sistem apa?      | SEVIMA                             |
| 44 | Siapa yang validasi KRS mahasiswa?         | Dosen wali                         |
| 45 | Referensi SOP KRS ini mengacu ke apa?      | Buku Pedoman Akademik STT Cipasung |

---

## 3. PMB.docx

### Syarat administrasi pendaftaran

| #  | Pertanyaan                                  | Kunci Jawaban                                                                                                                  |
| -- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 46 | Syarat daftar PMB apa aja?                  | Isi formulir online (s.id/pmbsttc), bayar biaya pendaftaran, scan ijazah/SKL, scan KTP, pas foto 4x6, scan KK, scan akta lahir |
| 47 | Link buat isi formulir pendaftaran di mana? | s.id/pmbsttc                                                                                                                   |

### Program studi

| #  | Pertanyaan                                       | Kunci Jawaban                                                                                                          |
| -- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| 48 | STT Cipasung punya prodi apa aja?                | S1 Teknik Industri dan S1 Informatika                                                                                  |
| 49 | Prodi Teknik Industri belajar apa?               | Perancangan sistem terintegrasi (manusia, mesin, material, energi, informasi) + ilmu manajemen & sosial                |
| 50 | Lulusan Teknik Industri kerja di bidang apa aja? | Produksi & mutu, sistem informasi, pemasaran, logistik, SDM, keuangan, konsultasi manajemen                            |
| 51 | Prodi Informatika belajar tentang apa?           | Ilmu komputer & analisis matematis untuk perancangan/pengujian/pengembangan software, sistem operasi, kinerja komputer |
| 52 | Prospek kerja lulusan Informatika apa aja?       | Software developer, konsultan IT, web engineer, network engineer, programmer, game developer, peneliti                 |

### Profil kampus

| #  | Pertanyaan                                | Kunci Jawaban                                                                 |
| -- | ----------------------------------------- | ----------------------------------------------------------------------------- |
| 53 | STT Cipasung didirikan tahun berapa?      | 1997, di bawah supervisi ITB                                                  |
| 54 | STT Cipasung itu berbasis lingkungan apa? | Lingkungan pesantren, memadukan ilmu pengetahuan dengan nilai-nilai keislaman |
| 55 | Apa motto STT Cipasung?                   | "Higher Education for All"                                                    |

### UKM / kemahasiswaan

| #  | Pertanyaan                            | Kunci Jawaban                                                                                    |
| -- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 56 | UKM apa aja yang ada di STT Cipasung? | Proclub, Kelapa, Sanggar Terasi, KDD, Dignity, UKM Kerohanian, UKM Olahraga, Rilis, Pencak Silat |
| 57 | Proclub itu UKM apa?                  | Programming Club                                                                                 |
| 58 | KDD singkatan dari apa?               | Keluarga Donor Darah                                                                             |

### Kontak & alamat

| #  | Pertanyaan                               | Kunci Jawaban                                        |
| -- | ---------------------------------------- | ---------------------------------------------------- |
| 59 | Alamat kampus STT Cipasung di mana?      | Jl. Raya Cisinga KM.1, Padakembang Tasikmalaya 46466 |
| 60 | Nomor telepon kampus berapa?             | (0265) 2550424                                       |
| 61 | Kontak pusat informasi PMB nomor berapa? | +6282117083998 / 0821-1708-3998                      |
| 62 | Email kampus apa?                        | info@sttcipasung.ac.id                               |
| 63 | Website resmi STT Cipasung apa?          | www.sttcipasung.ac.id                                |

### Beasiswa

| #  | Pertanyaan                                     | Kunci Jawaban                                                                                                |
| -- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 64 | Jenis beasiswa apa aja yang tersedia?          | KIP-K, UKT 100%, UKT 75%, UKT 50%                                                                            |
| 65 | Beasiswa KIP-K dapat fasilitas apa aja?        | Semua fasilitas beasiswa (bebas biaya awal, potongan biaya semester, bebas biaya sidang & wisuda, uang saku) |
| 66 | Beasiswa UKT 75% dapat potongan berapa persen? | Potongan biaya semester sebesar 75%                                                                          |
| 67 | min ada beasiswa ga buat yg kurang mampu?      | Ada, KIP-K -- dapat semua fasilitas beasiswa                                                                 |

---

## 4. BIAYA.docx

| #  | Pertanyaan                                              | Kunci Jawaban                                                                                   |
| -- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 68 | Biaya awal pendaftaran berapa dan termasuk apa aja?     | Rp 2.750.000 -- termasuk biaya pendaftaran, jas almamater, KTM, orientasi kampus, uang bangunan |
| 69 | UKT per semester berapa?                                | Rp 3.000.000, bisa dicicil per bulan                                                            |
| 70 | Total biaya semester 1 berapa?                          | Rp 5.750.000 (gabungan biaya awal + UKT)                                                        |
| 71 | Semester 2 bayar berapa?                                | Rp 3.000.000                                                                                    |
| 72 | Semester III biayanya berapa?                           | Rp 3.000.000                                                                                    |
| 73 | Biaya semester 4 berapa?                                | Rp 3.000.000                                                                                    |
| 74 | Semester V bayar berapa?                                | Rp 3.000.000                                                                                    |
| 75 | Semester 6 biayanya berapa?                             | Rp 3.000.000                                                                                    |
| 76 | Semester 7 kok lebih mahal, kenapa?                     | Rp 3.400.000 -- termasuk tambahan biaya Sidang KP Rp 400.000                                    |
| 77 | Semester 8 biaya berapa total?                          | Rp 6.000.000 -- termasuk tambahan Sidang TA & Wisuda Rp 3.000.000                               |
| 78 | Total biaya kuliah dari semester 1 sampai lulus berapa? | Rp 30.150.000                                                                                   |

---

## 5. Pertanyaan di luar konteks (harus fallback, BUKAN dijawab ngasal)

Gunakan ini untuk memastikan sistem **menolak menjawab** dan memberi fallback,
bukan malah mengarang atau salah ambil chunk dari dokumen yang tidak relevan.

| #  | Pertanyaan                             | Ekspektasi                                    |
| -- | -------------------------------------- | --------------------------------------------- |
| 79 | min tau kasus ini gak?                 | Fallback (tidak nyambung ke dokumen manapun)  |
| 80 | Siapa presiden Indonesia sekarang?     | Fallback (di luar cakupan dokumen kampus)     |
| 81 | Kalau saya sakit, obat apa yang cocok? | Fallback                                      |
| 82 | Rekomendasi laptop buat kuliah apa ya? | Fallback (bukan isi dokumen resmi kampus)     |
| 83 | (pesan kosong / cuma emoji "👍")       | Fallback atau chit-chat, bukan RAG dipaksakan |

---

## Cara menilai hasil

Untuk tiap baris, cek 3 hal:

1. **Ketepatan angka/tanggal** -- harus persis sama dengan kunci jawaban, tidak boleh dibulatkan/diubah.
2. **Routing benar** -- kalau `--debug` aktif, cek apakah chunk yang dipakai memang dari dokumen yang seharusnya (mis. pertanyaan biaya tidak boleh ambil chunk dari KRS.docx).
3. **Tidak mengarang** -- khusus bagian 5, pastikan sistem tidak "sok tahu" menjawab dari dokumen yang sebenarnya tidak relevan.
