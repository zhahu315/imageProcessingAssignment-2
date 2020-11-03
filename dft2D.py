import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize(gray):
	for i in range(gray.shape[0]):
		for j in range(gray.shape[1]):
			gray[i][j] = (gray[i][j] - np.amin(gray)) / (np.amax(gray) - np.amin(gray))
	# print(gray.shape)
	# print(gray)
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


def idft2D(f):
	gray = f
	gray = np.array(gray, complex)

	for i in range(gray.shape[0]):
		gray[i] = np.fft.ifft(gray[i])
	print(gray.dtype)
	print('1=', gray)
	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.ifft(gray[:, j])
	# gray = np.abs(gray)
	# print(gray.dtype)
	print('2=', gray)

	return gray


def center_fft(f):

	fft_raw = dft2D(f)
	# print('fft_raw=', fft_raw)
	F = np.abs(fft_raw)
	print('F=', F.dtype, F)
	draw(F)
	center_F = center_shift(F)
	print('center_F=', center_F.dtype, center_F)
	draw(center_F)
	temp = np.ones((f.shape[0], f.shape[1]), dtype=float)
	S = temp + center_F
	draw(S)
	S = np.log(S)
	print('S=', S.dtype, S)

	# cv2.imshow('1', raw)
	cv2.imshow('F', normalize(F))
	cv2.imshow('center_F', normalize(center_F))
	cv2.imshow('S', normalize(S))
	# cv2.imshow('3', cv_fft_shift)
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


def draw(f):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = range(f.shape[0])
	y = range(f.shape[1])
	Z = f[x][y]
	X, Y = np.meshgrid(x, y)
	ax.plot_surface(X, Y, Z, cmap='rainbow')
	plt.show()


def diff(f, g):
	f = normalize(f)
	print('f=', f)
	print('g=', g)
	d = np.abs(f - g)
	# for i in range(d.shape[0]):
	# 	for j in range(d.shape[1]):
	# 		if d[i][j] > 10:
	# 			print(i, j)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = range(d.shape[0])
	y = range(d.shape[1])
	Z = d[x][y]
	X, Y = np.meshgrid(x, y)
	ax.plot_surface(X, Y, Z, cmap='rainbow')
	plt.show()


if __name__ == '__main__':
	# f = 'rose512.tif'
	f = 'lena512.tif'
	f = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	# g = idft2D(dft2D(normalize(f)))
	# g = np.abs(g)
	# diff(f, g)
	# cv2.imshow('test.tif', g)
	# cv2.imwrite('test.jpg', g)
	# cv2.waitKey(0)
	raw = np.zeros((512, 512), dtype=complex)
	for j in range(251, 260):
		for i in range(226, 285):
			raw[i][j] = 1

	center_fft(raw)

