import numpy as np
from heapq import heapify, heappop, heappush
from bitarray import bitarray

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

def huffman(img):
    frequency_dict = frequency_table(img)
    heap = create_heap(frequency_dict)
    huffman_dict = create_huffman_dict(heap)
    encoded_image = bitarray()
    encoded_image.encode(huffman_dict, img)
    decoded_out = encoded_image.decode(huffman_dict)
    with open('out/compressed_file.bin', 'wb') as w:
        encoded_image.tofile(w)
    decoded_out = bitarray()
    padding = 8 - (len(encoded_image) % 8)
    with open('out/compressed_file.bin', 'rb') as r:
        decoded_out.fromfile(r)
    decoded_out = decoded_out[:-padding] # remove padding
    decoded_out = decoded_out.decode(huffman_dict)
    decoded_out = np.array(decoded_out)
    img_new = np.reshape(decoded_out, (img.shape[0], img.shape[1]))
    return img_new