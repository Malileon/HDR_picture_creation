import cv2 as cv
import numpy as np

img_fn = [r"Images\Tower2.png", r"Images\Tower2.png", r"Images\Tower3.png", 
r"Images\Tower4.png", r"Images\Tower5.png",r"Images\Tower6.png", r"Images\Tower7.png", 
r"Images\Tower8.png", r"Images\Tower9.png", r"Images\Tower10.png"]
img_list = [cv.imread(fn) for fn in img_fn]
for img in img_list:
    img = img.astype('float32') / 255.0
exposure_times = np.array([32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625], dtype=np.float32)

merge_robertson = cv.createMergeRobertson()
robertson = merge_robertson.process(img_list, exposure_times.copy())

calibrate_debevec = cv.createCalibrateDebevec()
response_debevec = calibrate_debevec.process(img_list, exposure_times)

merge_debevec = cv.createMergeDebevec()
debevec = merge_debevec.process(img_list, exposure_times, response_debevec)

cv.imwrite("debevec.hdr", debevec)
cv.imwrite('robertson.hdr', robertson)
