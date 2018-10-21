import numpy as np
import cv2
from keras.preprocessing import image

def nothing(x):
	pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	print(test_image.shape)
	test_image = np.expand_dims(test_image,axis=0)
	print(test_image.shape)
	result = classifier.predict(test_image)
	print(result.shape)

	if result[0][0] == 1:
		return 'A'
	elif result[0][1] ==1:
		return 'B'
	elif result[0][2] ==1:
		return 'C'
	elif result[0][3] ==1:
		return 'D'
	elif result[0][4] ==1:
		return 'E'
	elif result[0][5] ==1:
		return 'F'
	elif result[0][6] ==1:
		return 'G'
	elif result[0][7] ==1:
		return 'H'
	elif result[0][8] ==1:
		return 'I'
	elif result[0][9] ==1:
		return 'J'
	elif result[0][10] ==1:
		return 'K'
	elif result[0][11] ==1:
		return 'L'
	elif result[0][12] ==1:
		return 'M'
	elif result[0][13] ==1:
		return 'N'
	elif result[0][14] ==1:
		return 'O'
	elif result[0][15] ==1:
		return 'P'
	elif result[0][16] ==1:
		return 'Q'
	elif result[0][17] ==1:
		return 'R'
	elif result[0][18] ==1:
		return 'S'
	elif result[0][19] ==1:
		return 'T'
	elif result[0][20] ==1:
		return 'U'
	elif result[0][21] ==1:
		return 'V'
	elif result[0][22] ==1:
		return 'W'
	elif result[0][23] ==1:
		return 'X'
	elif result[0][24] ==1:
		return 'Y'
	elif result[0][15] ==1:
		return 'Z'
	

cam = cv2.VideoCapture(0)