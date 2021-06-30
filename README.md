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