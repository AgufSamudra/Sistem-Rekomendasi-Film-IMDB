# Laporan Proyek Machine Learning - Gufranaka Samudra
## _Sistem Rekomendasi Film_


## Project Overview

Sebuah perusahaan teknologi aplikasi Tube mempunyai layanan vidio film layar lebar dan memiliki banyak sekali Users dengan berbagai minat genre, perusahaan memiliki banyak sekali film berkelas yang ia layankan kepada Users. Namun Perusahaan ingin membuat sebuah sistem rekomendasi film berdasarkan genre kepada para pengguna setianya. Dengan rekomendasi, pengguna menjadi lebih mudah untuk memilih film kesukaan nya berdasarkan jenis genre yang serupa.

* Proyek ini sangat penting untuk menambah minat pengguna  dalam menggunakan aplikasi Tube, terutama dalam menonton sebuah film di dalam aplikasi tersebut.
* Dan juga dengan sistem rekomendasi akan memudahkan pengguna dalam memilih film kesukaan nya.
* Sumber Referensi
[Kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

## Business Understanding

Kita akan membangun sebuah sistem rekomendasi sebuah film kepada users, agar memudahkan users memilih film yang mungkin di sukai.

### Problem Statement
* Bagaimana cara membuat sistem rekomendasi film yang di peruntukkan untuk users berdasarkan genre kesukaan nya?
* Berapa film yang ingin di rekomendasikan kepada users berdasarkan genre yang ia tonton sebelumnya?


### Goals
* Membuat sistem rekomendasi berbasis Content Based Filtering.
* Berharap memberikan 5 rekomendasi film berdasarkan genre yang ia tonton sebelumnya.

### Solution statements
* Untuk membuat Content Based Filtering kita di bantu salah satu library Scikit-learn yaitu `TfidfVectorizer` dengan tujuan merubah teks menjadi representasi angka dan `cosine_similarity` untuk menghitung jumlah kata istilah yang muncul pada halaman-halaman yang diacu pada daftar indeks.
* Membuat fungsi bernama `film_recommendations` secara manual di bantu dengan hasil dari `TfidfVectorizer` dan `cosine_similarity` yang sudah kita lakukan sebelumnya.

## Data Understanding

Dataset movie_metadata adalah dataset IMDB movie yang berisi 5000 baris dengan 28 kolom, dataset di unduh dari website Kaggle. Tools yang akan kita gunakan adalah dengan Google Colab.
[Kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

Setelah kita melihat keseluruhan data dengan fungsi info(), kita bisa melihat semua variable pada dataset terdiri **13** tipe data `Float`, **3** tipe data `integer`, dan **12** tipe data `Object`.

Namun data yang akan kita gunakan hanya 3, yaitu `genres`, `movie_title` dan `num_user_for_reviews`

Setelah kita melihat data kita, ada beberapa problem pada data, yaitu,
* Missing Value.
* Perubahan data movie pada saat merubahnya ke dalam bentuk list.

### Missing Value
Ketika kita melihat data kita, ternyata mengandung sebuah Missing Value di dalamnya. Dengan fungsi `isnull().sum()` kita dapat melihatnya dengan jelas.

| Variable  | missing value |
| ----- | --- |
| genres   | 0 |
| movie_title   | 0 |
| num_user_for_reviews   | 21 |

Dari hasil di atas kita bisa mengambil kesimpulan bahwa hanyaa data `num_user_for_reviews` yang memiliki Missing Value sebanyak **21**.

### Perubahan Data Movie Saat Ubah Kedalam List
Setelah kita merubah setiap atribut ke dalam list dengan `tolist()` Data movie menjadi berubah, semisal pada film `Avatar` ketika di rumah menjadi `Avatar\xa0` penambahan `\xa0` akan mengganggu pada saat proses rekomendasi nanti.

### Variabel-variabel pada Sistem Rekomendasi dataset adalah sebagai berikut:
* color : Jenis film yang berwarna atau hitam putih.
* director_name : Berisi nama-nama director pembuat film.
* num_critic_for_reviews : Jumlah review kritik pada film.
* duration : Waktu lama tayang film berdasarkan (Menit).
* director_facebook_likes : Jumlah like director di facebook.
* actor_3_facebook_likes : jumlah like actor di facebook.
* actor_2_name : Nama-nama actor.
* actor_1_facebook_likes : jumlah like actor di facebook.
* genre : Nama genre dari masing-masing film.
* actor_1_name : Nama-nama actor.
* movie_title : Judul dari masing-masing film.
* num_voted_users : Jumlah vote dari users.
* cast_total_facebook_likes : Total like pemeran di facebook.
* actor_3_name : Nama-nama actor.
* facenumber_in_poster : Jumlah gambar wajah di dalam poster.
* plot_keywords : Kata kunci film.
* movie_imdb_link : Link film IMDB.
* num_user_for_reviews : Jumlah review dari Users.
* language : Bahasa masing-masing film.
* country : Negara yang membuat film tersebut.
* content_rating : Platform penayangan film.
* budget : Anggaran pembuatan film.
* title_year : Tahun rilis film.
* actor_2_facebook_likes : Jumlah like actor di facebook.
* imdb_score : Score dari IMDB.
* aspect_ratio : Ratio dari sebuah film.
* movie_facebook_likes : Jumlah like film di facebook.

## Data Preparation
Ada beberapa tahap yang akan kita lakukan terhadap dataset yang kita miliki, seperti sebagai berikut,

* Mengatasi Missing Value.
* Mensorting Data Berdasarkan Review Users
* Mengapus data bersifat duplikat agar memiliki satu nilai.
* Mengubah Variable Yang Dibutuhkan Kedalam List.
* Menghapus `\xa0` pada data movie setelah di ubah ke list.
* Menggabungkan semua atribute menjadi `Dictionary`


### Data Cleaning

#### Mengatasi Missing Value
Sebelumnya kita melihat adanya Missing Value pada data kita, tepatnya pada atribut `num_user_for_reviews` jadi kita akan menghapusnya dengan fungsi `dropna()`

#### Mensorting Data Berdasarkan Review Users
Di sini kami hanya ingin mengurutkan film yang memiliki review paling sedikit sampai yang terbanyak menggunakan `sort_values` dengan begitu kami memperoleh urutan data film dari yang terbesar hingga yang terbesar.

#### Mengapus data bersifat duplikat agar memiliki satu nilai
Kalo kita lihat pada data `num_user_for_reviews` memiliki nilai review yang sama, jadi kita akan menghapusnya agar menjadi setiap film memiliki nilai review yang berbeda. Cara yang kita gunakan adalah dengan `drop_duplicates()`

### Data Transforms
#### Mengubah Variable Yang Dibutuhkan Kedalam List
Pada tahap ini, kami mencoba untuk mengubah variable yang di butuhkan yaitu `genres` `movie_title` `num_user_for_reviews` ke dalam sebuah list. Dengan tujuan untuk menjadikannya data dictionary setelah itu.

#### Menghapus **\xa0** pada data movie setelah di ubah ke list

Pada saat setiap data atribut di ubah ke dalam bentuk list kita memiliki masalah, yaitu setiap nama film di akhiri dengan `\xa0`. Kita akan menghapus kalimat tersebut dengan `lambda` `Replace`

Contoh Code bisa di lihat di bawah,
```
movie = list(map(lambda st: str.replace(st, "\xa0", ""), movie))
```

Dengan menjalankan code di atas semua nama film sudah bersih dari kata yang di akhiri dengan `\xa0`.

#### Menggabungkan semua atribute menjadi Dictionary

Kita membuat variable baru dengan nama `movie_new` dengan isi Dictionary tiga atribute kita, di bantu dengan `pd.DataFrame()` dalam membuat Dictionary. Tujuan nya untuk menentukan key:value pada atrbiute dataset kita. 

## Modelling & Results

Tahap kali ini kita akan menggunakan Content Based Filltering yaitu, merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Di bantu dengan teknik `TF IDF` dan `Cosine Similarity` untuk mencari korelasi antara judul film satu dengan yang lainnya berdasarkan genre.

Dalam Cosine Similarity, objek data dalam kumpulan data diperlakukan sebagai vektor. Rumus untuk mencari persamaan kosinus antara dua vektor adalah,

![Cosine](data:image/png;base64,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)

di mana,
* A . B = hasil kali (titik) dari vektor 'A' dan 'B'.
* ||A|| dan ||B|| = panjang kedua vektor 'A' dan 'B'.
* ||A|| * ||B|| = perkalian silang dua vektor 'A' dan 'B'.

### Output
| movie | genre |
| --- | --- |
| Star Wars: Episode VI - Return of the Jedi | Action, Adventure, Fantasy, Sci-Fi |
| The Amazing Spider-Man 2 | Action, Adventure, Fantasy, Sci-Fi |
| Man of Steel | Action, Adventure, Fantasy, Sci-Fi |
| Star Wars: Episode III - Revenge of the Sith | Action, Adventure, Fantasy, Sci-Fi |
| Highlander: Endgame | Action, Adventure, Fantasy, Sci-Fi |

### kelebihan 
Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

### Kekurangan
Kekurangan utama yang pada metode ini yaitu metode ini tidak mampu merekomendasikan jenis item yang baru atau belum pernah dilihat kepada seorang pengguna. Hal ini dikarenakan metode ini dibuat berdasarkan item-item yang pernah dinilai oleh pengguna tersebut.

## Evaluasi 


Matrik yang kita gunakan untuk sistem rekomendasi Content Based Filtering adalah Matrik Precision, yaitu P = Hasil rekomendasi relevan/total rekomendasi. Sebagai gambaran kita bisa melihat illustrasi di bawah ini.

![TabelCM](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:819311f78d87da1e0fd8660171fa58e620211012160253.png)

Jika kita lihat hasil kita, kita memberi kepada users 5 rekomendasi film. Dari ke 5 rekomendasi ada 5 yang relevan, maka presisi adalah 100%. Bagaimana kita tau bahwa itu menghasilkan 100%?

Sebelumnya pengguna menonton film Avatar yang bergenre `Action`, `Adventure`, `Fantasy`, `Sci-Fi`. Kemudian program kita mencoba memberi 5 rekomendasi film yang serupa kepada pengguna berdasarkan genre.

Dan hasil yang di dapat adalah ke 5 rekomendasi sistem berhasil memberi 5 film dengan genre yang sama/relevan kepada user. Dengan begitu kita tau bahwa hasil presisi dari sistem rekomendasi untuk Avatar adalah 100%.