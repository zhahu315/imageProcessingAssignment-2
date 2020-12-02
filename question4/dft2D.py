import numpy as np
import cv2
import matplotlib.pyplot as plt


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
	gray = np.array(f, complex)
	# raw = gray
	for i in range(gray.shape[0]):
		gray[i] = np.fft.fft(gray[i])
	print(gray.dtype)
	print('fft1=', gray)
	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.fft(gray[:, j])
	# gray = np.abs(gray)
	# print(gray.dtype)
	print('fft2=', gray)
	# sys = np.fft.fft2(raw)
	# print(sys)
	# cv2.imwrite('2.tif', gray)
	# cv2.imshow('2', gray)
	# cv2.waitKey(0)
	return gray


def center_fft(f):

	fft_raw = dft2D(f)
	abs_fft_raw = np.abs(fft_raw)

	F = normalize(abs_fft_raw)
	print('F=', F.dtype, F)

	center_F = normalize(center_shift(abs_fft_raw))
	print('center_F=', center_F.dtype, center_F)

	S = 1 + center_shift(abs_fft_raw)
	print(S.dtype, S)
	S = normalize(np.log(S))
	print('S=', S.dtype, S)

	cv2.imshow('F', F)
	cv2.imshow('center_F', center_F)
	cv2.imshow('S', S)
	cv2.waitKey(0)


def center_shift(f):
	temp = np.empty((f.shape[0], f.shape[1]), float)
	for i in range(f.shape[0] // 2):
		for j in range(f.shape[1] // 2):
			temp[i][j] = f[i + f.shape[0] // 2][j + f.shape[1] // 2]
	for i in range(f.shape[0] // 2, f.shape[0]):
		for j in range(f.shape[1] // 2, f.shape[1]):
			temp[i][j] = f[i - f.shape[0] // 2][j - f.shape[1] // 2]
	for i in range(f.shape[0] // 2, f.shape[0]):
		for j in range(f.shape[1] // 2):
			temp[i][j] = f[i - f.shape[0] // 2][j + f.shape[1] // 2]
	for i in range(f.shape[0] // 2):
		for j in range(f.shape[1] // 2, f.shape[1]):
			temp[i][j] = f[i + f.shape[0] // 2][j - f.shape[1] // 2]
	return temp


if __name__ == '__main__':
	# f = 'rose512.tif'
	# f = 'lena512.tif'
	# f = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	# g = idft2D(dft2D(normalize(f)))
	# g = np.abs(g)
	# diff(f, g)
	# cv2.imshow('test.tif', g)
	# cv2.imwrite('test.jpg', g)
	# cv2.waitKey(0)
	raw = np.zeros((512, 512), dtype='float32')
	for j in range(251, 260):
		for i in range(226, 285):
			raw[i][j] = 1
	cv2.imwrite('raw.tiff', raw)
	center_fft(raw)

