# svd-image-compression
Tugas Seleksi Ca-IRK 2019<br>
Spek: https://github.com/williammfu/svd-image-compression

## TODOLIST (nanti didelete)
- [ ] bikin format tingkat kompresi (-> kayak brp persen gitu nnt diconvert ke limit - gunain shape 
- [x] run time program
- [x] persentase ukuran memori gambar yang telah dikompresi thd gambar original -> ternyata masih anehh hasil kompresi nya lebih besar :< pdhl kualitas nya jelek
- [] readme
- [x] huffman coding
- [x] algoritma svd sendiri (belom perfect)
- [] rapihin kode2
## Cara penggunaan program


## Penjelasan tentang algoritma SVD
based on the idea that if the SVD is known, some of the singular values σ are significant while the others are small and not significant. Thus, if the significant values are kept and the small values are discarded then only the columns of U and V corresponding to the singular values are used. We will see in the following example, as more and more singular values are kept, the quality and representation compared to the original image improves.
### Matriks U, S, V', dan Rank
Algoritma SVD mereduksi suatu matriks A menjadi matriks U, S, dan V' di mana ketiga matriks tersebut diurutkan sehingga komponen-komponen yang penting (significant singular value) berada di bagian matriks U, S, dan V' tersebut. 
Matriks U adalah matriks berukuran ...
Matriks S adalah matriks berukuran ...
Matriks V' adalah matriks berukuran ...
Pada algoritma SVD, komponen yang penting berada di atas matriks U, S, dan V'. Sehingga kita dapat membatasi rank matriks yang ingin kita ambil, misalnya kita ambil r = 5, maka yang akan dikomputasi adalah
Walaupun direduksi, dapat dipastikan bahwa komponen yang penting tetap diambil.
### Cara menghitung SVD
1. Hitung matriks A'A
2. Tentukan eigenvalue dari A'A
3. Bentuk matriks V'
4. Bentuk matriks Sigma
5. Bentuk matriks U

## Referensi
https://www.youtube.com/watch?v=nbBvuuNVfco gatau deng ini guna gak

https://davetang.org/file/Singular_Value_Decomposition_Tutorial.pdf

https://www.d.umn.edu/~mhampton/m4326svd_example.pdf

http://www.math.utah.edu/~goller/F15_M2270/BradyMathews_SVDImage.pdf

https://www.lagrange.edu/academics/undergraduate/undergraduate-research/citations/18-Citations2020.Compton.pdf -> referensi ini adalah referensi utama. Penjelasannya sangat lengkap dan step by step sehingga mudah diimplementasikan dalam membuat algoritma SVD from scratch.

https://www.youtube.com/watch?v=SU851ljMIZ8 -> berguna :)
https://www.youtube.com/watch?v=2lXaCh-dnRk cukup memberi gambaran:)

## Framework dan Library
- Numpy, untuk melakukan operasi matriks, seperti matmul (matrix multiplication) serta melakukan operasi linalg (algoritma svd, mencari eigenvalue, eigenvector), dll
- Opencv2, untuk membaca gambar lalu mereturn gambar tersebut dalam bentuk matriks
- Datetime, untuk menghitung waktu eksekusi program