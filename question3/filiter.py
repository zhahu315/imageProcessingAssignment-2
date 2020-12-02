import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def normalize(gray):
	temp = np.empty_like(gray)
	min = np.amin(gray)
	max = np.amax(gray)
	for i in range(gray.shape[0]):
		for j in range(gray.shape[1]):
			temp[i][j] = (gray[i][j] - min) / (max - min)
	gray = temp
	return gray


def dft2D(f):
	gray = np.array(f, dtype=complex)
	val = max(f.shape[0], f.shape[1])
	if val & (val - 1):
		d = pow(2, (int(math.log2(val)) + 1))
		temp = np.zeros((d, d), dtype=complex)
		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				temp[i][j] = f[i][j]
		gray = temp
	print('gray', gray.dtype, gray.shape, gray)
	for i in range(gray.shape[0]):
		gray[i] = np.fft.fft(gray[i])
	print(gray.dtype)
	print('fft1=', gray)
	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.fft(gray[:, j])
	print('fft2=', gray)
	# sys = np.fft.fft2(raw)
	# print(sys)
	# cv2.imwrite('2.tif', gray)
	# cv2.imshow('2', gray)
	# cv2.waitKey(0)
	return gray


def filter(f):
	row_mid = f.shape[0] // 2
	column_mid = f.shape[1] // 2
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if math.sqrt((i - row_mid) ^ 2 + (j - column_mid) ^ 2) < 100:
				f[i][j] = 0
			elif 400 < math.sqrt((i - row_mid) ^ 2 + (j - column_mid) ^ 2) < 512:
				f[i][j] = 0

	return f


def idft2D(f):
	gray = np.array(f, complex)

	for i in range(gray.shape[0]):
		gray[i] = np.fft.ifft(gray[i])

	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.ifft(gray[:, j])
	print('idft=', gray.dtype, gray)

	return gray



if __name__ == '__main__':
	f = cv2.imread('rose512.tif', cv2.IMREAD_GRAYSCALE)
	f = np.fft.fftshift(f)
	print(f)
	f = np.fft.ifftshift(filter(f)))
	cv2.imshow('filter', f)
	cv2.waitKey(0)
