import numpy as np
import cv2
from svd import svd
import datetime
import os
from PIL import Image

def read_image(img_name):
    path = 'in/' + img_name
    img = cv2.imread(path)
    return img

def read_image_pil(img_name):
    path = 'in/' + img_name
    img = np.asarray(Image.open(path))
    return img

def compress_channel(img, limit):
    n = len(img)
    U, S, Vt = np.linalg.svd(img)
    U_new = U[:,0:limit]
    
    S_new = np.diag(S)[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    # img_new = np.zeros((img.shape[0], img.shape[1]))
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    # img_new_inner = np.matmul(np.matmul(U_new,S_new),Vt_new)
    # print("---IMG NEW---\n",img_new)
    # img_new = img_new.astype('uint8')
    # imgs = []
    # for i in range(n):
    #     imgs.append(S[i]*np.outer(U[:,i],Vt[i]))
    # combined_imgs = []
    # for i in range(n):
    #     img = sum(imgs[:i+1])
    #     combined_imgs.append(img)
    # combined_imgs = combined_imgs[limit]
    for i, row in enumerate(img_new):
        for j, col in enumerate(row):
            if col < 0:
                img_new[i,j] = abs(col)
            if col > 255:
                img_new[i,j] = 255
    return img_new

# def compress_channel_dot(img, k):
#     u, s, vh = np.linalg.svd(img, full_matrices=False)
#     U_k = u[: , :k]
#     S_k = np.diag(s[:k])
#     Vh_k = vh[:k,:]
#     img_new = np.dot(U_k, np.dot(S_k, Vh_k))
#     # print("---IMG NEW---\n",img_new)
#     # img_new = img_new.astype('uint8')
#     # imgs = []
#     # for i in range(n):
#     #     imgs.append(S[i]*np.outer(U[:,i],Vt[i]))
#     # combined_imgs = []
#     # for i in range(n):
#     #     img = sum(imgs[:i+1])
#     #     combined_imgs.append(img)
#     # combined_imgs = combined_imgs[limit]
#     return img_new

# TODO mungkin benerin
def compress_scratch(img, limit):
    n = len(img)
    U, S, Vt = svd(img)
    U_new = U[:,0:limit]
    S_new = S[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    # img_new = img_new.astype('uint8')
    # imgs = []
    # for i in range(n):
    #     imgs.append(S[i]*np.outer(U[:,i],Vt[i]))
    # combined_imgs = []
    # for i in range(n):
    #     img = sum(imgs[:i+1])
    #     combined_imgs.append(img)
    # combined_imgs = combined_imgs[limit]

    return img_new

# return matrix
def compress_grayscale(img, limit):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compressed_gray = compress_channel(gray,limit)
    compressed_gray = compressed_gray.astype(np.uint8)
    return compressed_gray

# return image
def compress_grayscale_pil(img, limit):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    compressed_img_gray = Image.fromarray(compress_channel(gray,limit), mode=None)
    return compressed_img_gray

# return matrix
def compress_rgb(img, limit):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    red_new = compress_channel(red,limit)
    green_new = compress_channel(green,limit)
    blue_new = compress_channel(blue,limit)
    rimg = np.zeros(img.shape)
    rimg[:,:,0] = red_new
    rimg[:,:,1] = green_new
    rimg[:,:,2] = blue_new
    # compressed_rgb = cv2.merge((red_new, green_new, blue_new))
    # for ind1, row in enumerate(rimg):
    #     for ind2, col in enumerate(row):
    #         for ind3, value in enumerate(col):
    #             if value < 0:
    #                 rimg[ind1,ind2,ind3] = abs(value)
    #             if value > 255:
    #                 rimg[ind1,ind2,ind3] = 255
    compressed_rgb = rimg.astype(np.uint8)
    return compressed_rgb

# return image
def compress_rgb_pil(img, limit):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    red_new = Image.fromarray(compress_channel(red,limit), mode=None)
    green_new = Image.fromarray(compress_channel(green,limit), mode=None)
    blue_new = Image.fromarray(compress_channel(blue,limit), mode=None)
    compressed_img_rgb = Image.merge("RGB", (red_new, green_new, blue_new))
    return compressed_img_rgb

# from matrix
def write_image(img, img_name, limit):
    path = 'out/' + img_name + '_' + str(limit) + '.jpeg'
    cv2.imwrite(path, img)
    print("Berhasil save pada direktori: ", path)

# from image
def write_image_pil(img, new_img_name, limit):
    path = 'out/' + new_img_name + '_' + str(limit) + '.jpeg'
    img = Image.fromarray(img)
    img.save(path)
    print("Berhasil save pada direktori: ", path)

def get_file_size(path):
    file_size = os.path.getsize(path)
    return file_size

# MAIN PROGRAM - TESTING
lim = int(input("Masukkan tingkat kompresi yang diinginkan (k): "))
# lim gaboleh lebih besar dari shape

start = datetime.datetime.now()
file_name = 'momo.jpg'
momo = read_image_pil(file_name)
file_size_awal = get_file_size('in/'+file_name)
print(momo.shape)
momo_compressed = compress_grayscale(momo, lim)
# momo_compressed = np.array(compress_grayscale(momo, lim)[1])
write_image_pil(momo_compressed,'momo_jpg_gray_compressed', lim)
file_size_akhir = get_file_size('out/momo_jpg_gray_compressed_'+str(lim)+'.jpeg')
end = datetime.datetime.now()
# print(momo_compressed.shape)

print("Selesai dalam waktu",(end-start),"seconds.")
print("File size awal:", file_size_awal, "bytes")
print("File size akhir:", file_size_akhir, "bytes")
persentase = file_size_akhir/file_size_awal * 100
print("Persentase: ", persentase, "%")



test_gray = cv2.cvtColor(momo, cv2.COLOR_BGR2GRAY)
print(test_gray.shape)
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