import numpy as np
import cv2
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
	val = max(gray.shape[0], gray.shape[1])
	if val & (val - 1):
		d = pow(2, (int(math.log2(val)) + 1))
		temp = np.zeros((d, d), dtype=complex)
		for i in range(gray.shape[0]):
			for j in range(gray.shape[1]):
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


def center_fft(f):
	fft_raw = dft2D(f)
	abs_fft_raw = np.abs(fft_raw)
	F = normalize(abs_fft_raw)
	print('F=', F.dtype, F.shape, F)
	# draw(F)
	center_F = center_shift(F)
	print('center_F=', center_F.dtype, F.shape, center_F)
	# draw(center_F)
	# temp = np.ones_like(center_F)
	S = 1 + center_F
	print(S.dtype, S)
	S = normalize(np.log(S))
	print('S=', S.dtype, F.shape, S)

	center_F = center_F.astype(np.float32)
	S = S.astype(np.float32)
	cv2.imwrite('center_F.tiff', center_F)
	cv2.imwrite('S.tiff', S)



if __name__ == '__main__':
	f = 'lunar_surface.tif'
	f = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	print(f)
	# g = idft2D(dft2D(normalize(f)))
	# g = np.abs(g)
	# diff(f, g)
	# cv2.imshow('test.tif', g)
	# cv2.imwrite('test.jpg', g)
	# cv2.waitKey(0)
	center_fft(f)
	exit(0)
