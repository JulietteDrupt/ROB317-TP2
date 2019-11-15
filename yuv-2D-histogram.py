import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize(hist) :
	valmax = np.amax(hist)
	valmin = np.amin(hist)
	hist = hist / (valmax - valmin) * 255
	return hist

cap = cv2.VideoCapture("./Vidéos/Extrait1-Cosmos_Laundromat1(340p).m4v")
ret, frame1 = cap.read()
yuv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
#hist0 = np.zeros_like(frame1)

u = np.array(yuv_frame[:,:,1]).flatten()
v = np.array(yuv_frame[:,:,2]).flatten()
hist, xbins, ybins = np.histogram2d(u, v, bins=(256,256), range=[[0,255],[0,255]])
#print(hist)

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)


while(ret):

	u = np.array(yuv_frame[:,:,1]).flatten()
	v = np.array(yuv_frame[:,:,2]).flatten()

	hist, xbins, ybins = np.histogram2d(u, v, bins=(256,256), range=[[0,255],[0,255]])

	hist = cv2.cvtColor(normalize(hist).astype('float32'), cv2.COLOR_GRAY2BGR)

	#hist0[0:hist.shape[0],0:hist.shape[0]] = hist

	#result = np.vstack((frame1,hist0))

	cv2.imshow('Histogramme', hist)
	cv2.imshow('Video', frame1)

	k = cv2.waitKey(15) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('Frame_%04d.png'%index,frame1)
	ret, frame1 = cap.read()
	if (ret):
		yuv_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)

cap.release()
cv2.destroyAllWindows()


