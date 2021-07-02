import numpy as np
from heapq import heapify, heappop, heappush
from bitarray import bitarray
import cv2

def frequency_table(array):
    freq_dict = {}
    for item in array:
        if item in freq_dict:
            freq_dict[item] += 1
        else:
            freq_dict[item] = 1
    return freq_dict

def create_heap(dictionary):
    heap = []
    # Nodenya adalah [frekuensi, [kode warna (0-255), representasi binary]]
    for key, val in dictionary.items():
        heap.append([val,[key,""]])
    return heap

def create_huffman_dict(heap):
    heapify(heap)
    while len(heap) > 1:
        right = heappop(heap)
        left = heappop(heap)
        for pair in right[1:]:
            pair[1] = '1' + pair[1]
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        # new node
        heappush(heap, [right[0]+left[0]] + right[1:] + left[1:])   
    huffman_list = right[1:] + left[1:]
    huffman_dict = {item[0]:bitarray(str(item[1])) for item in huffman_list}
    return huffman_dict

def huffman(img, path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgshape = img.shape
    img = np.reshape(img, (1, imgshape[0]*imgshape[1]))
    img = img.tolist()[0]
    frequency_dict = frequency_table(img)
    heap = create_heap(frequency_dict)
    huffman_dict = create_huffman_dict(heap)
    encoded_image = bitarray()
    encoded_image.encode(huffman_dict, img)
    decoded_out = encoded_image.decode(huffman_dict)
    with open('out/compressed_image.bin', 'wb') as w:
        encoded_image.tofile(w)
    decoded_out = bitarray()
    padding = 8 - (len(encoded_image) % 8)
    with open('out/compressed_image.bin', 'rb') as r:
        decoded_out.fromfile(r)
    decoded_out = decoded_out[:-padding] # remove padding
    decoded_out = decoded_out.decode(huffman_dict)
    decoded_out = np.array(decoded_out)
    img_new = np.reshape(decoded_out, (imgshape[0], imgshape[1]))
    img_new = img_new.astype(np.uint8)
    return img_new

# img = cv2.imread('in/momo.jpeg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgshape = img.shape
# print("Shape: ",imgshape)
# image = np.reshape(img, (1, imgshape[0]*imgshape[1]))
# print("Shape image reshape: ",image.shape)
# image = image.tolist()[0]

# frequency_dict = frequency_table(image)
# heap = create_heap(frequency_dict)
# heapify(heap)
# print("HEAPIFIED: \n",heap)

# print("----------")
# while len(heap) > 1:
#     right = heappop(heap)
#     # print("Right = ", right)
#     left = heappop(heap)
#     # print("Left = ", right)
#     for pair in right[1:]:
#         pair[1] = '1' + pair[1]
#     for pair in left[1:]:
#         pair[1] = '0' + pair[1]
#     heappush(heap, [right[0]+left[0]] + right[1:] + left[1:])

# huffman_list = right[1:] + left[1:]
# print("Len huffman list: ",len(huffman_list))
# huffman_dict = {item[0]:bitarray(str(item[1])) for item in huffman_list}
# #
# encoded_image = bitarray()
# encoded_image.encode(huffman_dict, image)
# # print(encoded_image)
# print("encoded shape: ",len(encoded_image))

# decoded_out = encoded_image.decode(huffman_dict)
# with open('compressed_file.txt', 'wb') as w:
#     encoded_image.tofile(w)
# decoded_out = bitarray()
# padding = 8 - (len(encoded_image) % 8)
# with open('compressed_file.txt', 'rb') as r:
#     decoded_out.fromfile(r)

# decoded_out = decoded_out[:-padding] # remove padding
# decoded_out = decoded_out.decode(huffman_dict)
# decoded_out = np.array(decoded_out)

# # print("decoded shape: ",decoded_out.shape)
# output = np.reshape(decoded_out, (imgshape[0], imgshape[1]))
# # print(output)
# path_out = 'out/' + 'hasil_huffman'+ '.jpeg'
# cv2.imwrite(path_out, output)
# out = len(encoded_image)
# im = len(image)*8
# compression = 1 - out/im
# print("Out: ",out)
# print("Image: ",im)
# print("Compression: ",compression)
# # print(image)