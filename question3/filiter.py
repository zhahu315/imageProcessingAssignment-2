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


def idft2D(f):
	gray = np.array(f, complex)

	for i in range(gray.shape[0]):
		gray[i] = np.fft.ifft(gray[i])

	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.ifft(gray[:, j])
	print('idft=', gray.dtype, gray)

	return gray


def diff(f, g):
	# print('f=', f)
	# print('g=', g)
	d = np.abs(f - g)
	print('diff= ', d.shape, d.dtype, d)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = range(d.shape[0])
	y = range(d.shape[1])
	Z = d[x][y]
	X, Y = np.meshgrid(x, y)
	ax.plot_surface(X, Y, Z, cmap='rainbow')
	plt.savefig('diff3D.png')
	plt.show()

	return d


if __name__ == '__main__':
	f = 'rose512.tif'
	f = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	f = np.array(f, dtype=float)
	f = normalize(f)
	f = f.astype(np.float32)
	cv2.imwrite('rose_normalize.tiff', f)
	g = np.abs(idft2D(dft2D(f)))
	# print('g=', g.dtype, g)
	d = diff(f, g)

	g = g.astype(np.float32)
	cv2.imshow('g', normalize(g))
	cv2.imwrite('rose_dft.tiff', g)

	d = d.astype(np.float32)
	cv2.imshow('d', normalize(d))
	cv2.imwrite('diff.tiff', d)

	cv2.waitKey(0)
