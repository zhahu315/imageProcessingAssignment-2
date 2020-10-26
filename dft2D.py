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
	raw = gray
	gray[1] = np.fft.fft(gray[1])
	print(gray[1])
	for i in range(gray.shape[0]):
		gray[i] = np.fft.fft(gray[i])
	print(gray.dtype)
	print('1=', gray)
	for j in range(gray.shape[1]):
		gray[:, j] = np.fft.fft(gray[:, j])
	# gray = np.abs(gray)
	# print(gray.dtype)
	print('2=', gray)
	sys = np.fft.fft2(raw)
	print(sys)
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


def center_fft():
	raw = np.zeros([512, 512], float)
	for j in range(251, 261):
		for i in range(225, 285):
			raw[i][j] = 1
	fft_raw = dft2D(raw)
	fft_raw = np.abs(fft_raw)

	nor_fft_raw = normalize(fft_raw)

	F = center_shift(nor_fft_raw)
	print(F)
	temp = np.ones([F.shape[0], F.shape[1]])

	print(S)
	S = np.log10(S)
	print(S)

	cv2.imshow('1', raw)
	cv2.imshow('2', F)
	cv2.imshow('S', S)
	# cv2.imshow('3', cv_fft_shift)
	cv2.waitKey(0)


def center_shift(f):
	temp = np.zeros([f.shape[0], f.shape[1]], float)
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			temp[i][j] = f[int(abs(i - f.shape[0] / 2))][int(abs(j - f.shape[1] / 2))]
	return temp


if __name__ == '__main__':
	# f = 'rose512.tif'
	# f = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	# g = idft2D(dft2D(normalize(f)))
	# g = np.abs(g)
	# diff(f, g)
	# cv2.imshow('test.tif', g)
	# cv2.imwrite('test.jpg', g)
	# cv2.waitKey(0)
	center_fft()
