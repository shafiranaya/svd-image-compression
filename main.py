import numpy as np
import cv2
from svd import svd
import datetime
import os
from PIL import Image

def read_image(img_name):
    path = 'in/' + img_name
    img = np.asarray(Image.open(path))
    return img

def compress_channel(img, limit):
    n = len(img)
    U, S, Vt = np.linalg.svd(img)
    U_new = U[:,0:limit]
    S_new = np.diag(S)[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    for i, row in enumerate(img_new):
        for j, col in enumerate(row):
            if col < 0:
                img_new[i,j] = abs(col)
            if col > 255:
                img_new[i,j] = 255
    return img_new

# TODO mungkin benerin
def compress_scratch(img, limit):
    # n = len(img)
    U, S, Vt = svd(img)
    U_new = U[:,0:limit]
    S_new = S[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    for i, row in enumerate(img_new):
        for j, col in enumerate(row):
            if col < 0:
                img_new[i,j] = abs(col)
            if col > 255:
                img_new[i,j] = 255
    return img_new

# return matrix
def compress_grayscale(img, limit):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compressed_gray = compress_channel(gray,limit)
    compressed_gray = compressed_gray.astype(np.uint8)
    return compressed_gray

# return matrix
def compress_rgb(img, limit):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    red_new = compress_channel(red,limit)
    green_new = compress_channel(green,limit)
    blue_new = compress_channel(blue,limit)
    img_new = np.zeros(img.shape)
    img_new[:,:,0] = red_new
    img_new[:,:,1] = green_new
    img_new[:,:,2] = blue_new
    compressed_rgb = img_new.astype(np.uint8)
    return compressed_rgb

def compress_rgb_scratch(img, limit):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    red_new = compress_scratch(red,limit)
    green_new = compress_scratch(green,limit)
    blue_new = compress_scratch(blue,limit)
    img_new = np.zeros(img.shape)
    img_new[:,:,0] = red_new
    img_new[:,:,1] = green_new
    img_new[:,:,2] = blue_new
    compressed_rgb = img_new.astype(np.uint8)
    return compressed_rgb

# from image
def write_image(img, new_img_name, limit):
    path = 'out/' + new_img_name + '_' + str(limit) + '.jpeg'
    img = Image.fromarray(img)
    img.save(path)
    print("Berhasil save pada direktori: ", path)

def get_file_size(path):
    file_size = os.path.getsize(path)
    return file_size

# MAIN PROGRAM - TESTING
# print("Masukkan tingkat kompresi yang diinginkan, integer dari 0-")
lim = int(input("Masukkan tingkat kompresi yang diinginkan (k): "))
# lim gaboleh lebih besar dari shape

start = datetime.datetime.now()
file_name = 'momo.jpg'
momo = read_image(file_name)
file_size_awal = get_file_size('in/'+file_name)
print(momo.shape)
momo_compressed = compress_rgb_scratch(momo, lim)
compressed_file_name = 'momo_jpg_scratch_compressed'
write_image(momo_compressed,compressed_file_name, lim)
compressed_file_path = 'out/' + compressed_file_name + '_'+str(lim)+'.jpeg'
file_size_akhir = get_file_size(compressed_file_path)
end = datetime.datetime.now()
# print(momo_compressed.shape)
duration = (end-start).total_seconds()
print("Selesai dalam waktu", duration,"detik.")
print("File size awal:", file_size_awal, "bytes")
print("File size akhir:", file_size_akhir, "bytes")
persentase = round(file_size_akhir/file_size_awal * 100, 3)
print("Persentase: ", persentase, "%")

# # TESTING BENER APA ENGGA
# print("---TESTING---")
# A = cv2.cvtColor(momo, cv2.COLOR_BGR2GRAY)

# U, Sigma, Vt = svd(A)
# print("A: \n",A)
# A_test = np.matmul(np.matmul(U,Sigma), Vt)
# print("A_test: \n",A_test)
# print("U: \n",U)
# print("Sigma: \n",Sigma)
# print("Vt: \n",Vt)

# # BANDINGIN SAMA HASIL LIBRARY
# U_lib, S_lib, Vt_lib = np.linalg.svd(A)
# print("U_lib: \n",U_lib)
# print("S_lib: \n",S_lib)
# print("Vt_lib: \n",Vt_lib)