import cv2

im = cv2.imread('face/images/train/angry/0.jpg')

print(type(im))
# <class 'numpy.ndarray'>

print(im.shape)
print(type(im.shape))
