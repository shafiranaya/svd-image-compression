# svd-image-compression
Tugas Seleksi Ca-IRK 2019<br>
Spek: https://github.com/williammfu/svd-image-compression
## Penjelasan tentang algoritma SVD
Algoritma SVD atau Singular Value Decomposition adalah salah satu algoritma unsupervised learning pada machine learning yang sering digunakan. Dasar dari algoritma ini adalah konsep aljabar linear. Algoritma ini sangat berguna, karena dapat mendekomposisi matriks berisi data yang besar menjadi tiga submatriks. Penerapan algoritma SVD antara lain pada image compression, menyelesaikan sistem persamaan linear homogen, data mining, latent semantic analysis, dan lain-lain.
### Matriks U, S, V', dan Rank
Algoritma SVD mendekomposisi suatu matriks A menjadi tiga submatriks U, S, dan V'. Ketiga submatriks tersebut disusun sedemikian sehingga komponen-komponen yang penting (significant singular value) berada di bagian atas matriks U, S, dan V' tersebut, di mana matriks U berukuran m x n, matriks S berukuran n x n, dan matriks V' berukuran n x m. <br>
Pada algoritma SVD, komponen yang penting berada di atas matriks U, S, dan V'. Sehingga kita dapat membatasi rank matriks yang ingin kita ambil, misalnya kita ambil r sebuah integer positif yaitu 5 (r ini tidak boleh melebihi image size), maka yang akan dikomputasi untuk mendapatkan matriks A kembali adalah U berukuran m x r, S berukuran r x r, dan V' berukuran r x n. Jika ketiga matriks tersebut dikalikan, tetap akan menghasilkan matriks A berukuran m x n. Rank ini berpengaruh terhadap kualitas kompresi gambar. Semakin besar rank, maka kualitas gambar akan semakin baik atau semakin mendekati gambar aslinya. Begitu pula sebaliknya. Hal ini dikarenakan semakin besar nilai rank-nya, semakin banyak singular values yang digunakan dalam penghitungan matriks.

### Cara menghitung SVD
1. Hitung matriks A'A
2. Tentukan eigenvalue dari A'A lalu urutkan secara menurun
3. Bentuk matriks V'
<br>Cari eigenvector dari matriks A'A lalu lakukan normalisasi untuk setiap eigenvector. Kemudian, susun setiap eigenvector v<sub>i</sub> menjadi matriks V'. 
4. Bentuk matriks Sigma
<br>Dari eigenvalue yang telah terurut menurun, untuk setiap eigenvalue hitung singular value, yaitu akar dari eigenvalue. Susun singular value tersebut menjadi matriks diagonal Sigma. 
5. Bentuk matriks U
<br>Matriks U terdiri atas komponen-komponen u<sub>i</sub> yang dihitung dengan rumus u<sub>i</sub> = (1/s<sub>i</sub>)*A*v<sub>i</sub>. Lalu semua vektor kolom u<sub>i</sub> disusun menjadi matriks U.

## Cara penggunaan program
1. Install requirements terlebih dahulu dengan mengetikkan pada terminal:
```
pip install -r requirements.txt
```
2. Jalankan program dengan mengetikkan pada terminal:
```
python3 main.py
```
3. Masukkan nama file yang ingin dikompresi (harus sudah ada di folder in)
4. Pilih metode kompresi yang diinginkan
## Referensi
SVD
- https://www.lagrange.edu/academics/undergraduate/undergraduate-research/citations/18-Citations2020.Compton.pdf
- https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf
- https://www.d.umn.edu/~mhampton/m4326svd_example.pdf
- http://www.math.utah.edu/~goller/F15_M2270/BradyMathews_SVDImage.pdf
- https://www.youtube.com/watch?v=SU851ljMIZ8
- https://www.youtube.com/watch?v=2lXaCh-dnRk
Huffman Coding
- https://github.com/hemanth-nakshatri/Image-Compression
- https://github.com/TiongSun/DataCompression
Numpy Documentation - https://numpy.org/doc/
## Framework dan Library
- numpy, untuk melakukan operasi matriks, seperti matmul (matrix multiplication) serta melakukan operasi linalg (algoritma svd, mencari eigenvalue dan eigenvector), dan lain-lain
- math untuk menghitung akar (square root)
- opencv2, untuk mengubah gambar menjadi grayscale
- pillow (Image), untuk membaca dan menulis gambar
- datetime, untuk menghitung waktu eksekusi program
- os, untuk mendapatkan file size dari gambar sebelum kompresi dan sesudah kompresi
- heapq, untuk membentuk struktur data binary tree untuk pohon Huffman
- bitarray, untuk menyimpan data gambar dalam binary dari pohon Huffman