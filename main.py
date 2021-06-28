import numpy as np
import cv2
from svd import svd

def read_image(img_name):
    path = 'in/' + img_name
    img = cv2.imread(path)
    return img

def compress_channel(img, limit):
    U, S, Vt = np.linalg.svd(img)
    U, S, Vt = svd(img)
    U_new = U[:,0:limit]
    S_new = np.diag(S)[0:limit,0:limit]
    # S_new = S[0:limit,0:limit]
    Vt_new = Vt[0:limit, : ]
    img_new = np.matmul(np.matmul(U_new,S_new),Vt_new)
    return img_new

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
    path = 'out/' + img_name + '_' + str(limit) + '.png'
    cv2.imwrite(path, img)

# MAIN PROGRAM - TESTING
n = 5
# n gaboleh lebih besar dari shape
momo = read_image('momo_kecil_gray.png')
print(momo.shape)
momo_compressed = compress_grayscale(momo, n)
write_image(momo_compressed,'momo_compressed_grayscale', n)
print("SELESAI")