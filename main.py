import numpy as np
import cv2
from svd import svd
import datetime
import os

def read_image(img_name):
    path = 'in/' + img_name
    img = cv2.imread(path)
    file_size = os.path.getsize(path)
    return img, file_size

def compress_channel(img, limit):
    n = len(img)
    U, S, Vt = np.linalg.svd(img)
    # U, S, Vt = svd(img)
    U_new = U[:,0:limit]
    S_new = np.diag(S)[0:limit,0:limit]
    # S_new = S[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    imgs = []
    for i in range(n):
        imgs.append(S[i]*np.outer(U[:,i],Vt[i]))
    combined_imgs = []
    for i in range(n):
        img = sum(imgs[:i+1])
        combined_imgs.append(img)
    combined_imgs = combined_imgs[limit]
    return img_new, combined_imgs

def compress_grayscale(img, limit):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compressed_gray = compress_channel(gray,limit)
    return compressed_gray

def compress_rgb(img, limit):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    red_new = compress_channel(red,limit)
    green_new = compress_channel(green,limit)
    blue_new = compress_channel(blue,limit)
    compressed_rgb = cv2.merge((red_new, green_new, blue_new))
    return compressed_rgb

def write_image(img, img_name, limit):
    path = 'out/' + img_name + '_' + str(limit) + '.jpeg'
    cv2.imwrite(path, img)
    # print("berhasil write")
    file_size = os.path.getsize(path)
    return file_size

# MAIN PROGRAM - TESTING
lim = 111
# lim gaboleh lebih besar dari shape
start = datetime.datetime.now()
file_name = 'momo.jpg'
momo = read_image(file_name)[0]
file_size_awal = read_image(file_name)[1]
print(momo.shape)
# momo_compressed = compress_grayscale(momo, lim)[0]
momo_compressed = np.array(compress_grayscale(momo, lim)[1])
file_size_akhir = write_image(momo_compressed,'momo_jpg_grayscale_compressed', lim)

write_image(momo_compressed,'momo_jpg_grayscale_compressed', lim)
end = datetime.datetime.now()
print(momo_compressed.shape)
print("Selesai dalam waktu",(end-start),"seconds.")
print("File size awal:", file_size_awal, "bytes")
print("File size akhir:", file_size_akhir, "bytes")

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