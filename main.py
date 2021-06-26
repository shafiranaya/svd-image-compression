import numpy as np
from numpy import linalg
import cv2

path = 'in/momo.jpeg'
img = cv2.imread(path)
print(type(img))
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray)
print("gray shape",gray.shape)
U_gray, S_gray, Vt_gray = np.linalg.svd(gray)
print('A: \n',gray)
print('U: \n',U_gray)
print('S: \n',S_gray)
print('Vt: \n',Vt_gray)
[m,n] = gray.shape

limit = 150 # nanti ini bisa diganti
U_gray_new = U_gray[:,0:limit]
S_gray_new = np.diag(S_gray)[0:limit,0:limit]
Vt_gray_new = Vt_gray[0:limit, : ]
gray_new = np.matmul(np.matmul(U_gray_new,S_gray_new),Vt_gray_new)
print(gray_new)
print("gray new shape",gray_new.shape)
cv2.imwrite('momo_gray.png', gray)
new_img_name = 'restored_image_limit_'+str(limit)+'.png'
cv2.imwrite(new_img_name, gray_new)

# img_red = img[:,:,0]
# img_green = img[:,:,1]
# img_blue = img[:,:,2]

# U_red, S_red, Vt_red = np.linalg.svd(img_red)
# U_green, S_green, Vt_green = np.linalg.svd(img_green)
# U_blue, S_blue, Vt_blue = np.linalg.svd(img_blue)

# print(img_red)
# print(img)
# U_img, S_img, Vt_img = np.linalg.svd(img)
# cv2.imshow('image',img)