import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize(hist) :
	valmax = np.amax(hist)
	valmin = np.amin(hist)
	hist = (hist - valmin) / (valmax - valmin) * 255
	return hist

def histogram2d_Vx_Vy(flow) :
	Vx = flow[:,:,1].flatten()
	Vy = flow[:,:,0].flatten()
	hist, xbins, ybins = np.histogram2d(Vx, Vy, bins=(500,500), range=[[-50,50],[-50,50]])
	#hist = cv2.cvtColor(normalize(hist).astype('float32'), cv2.COLOR_GRAY2BGR)
	hist = normalize(hist).astype('float32')
	return hist

def generate_rltb_masks() :
	mask_r = np.zeros((500,500)).astype('uint8')
	mask_l = np.zeros((500,500)).astype('uint8')
	mask_t = np.zeros((500,500)).astype('uint8')
	mask_b = np.zeros((500,500)).astype('uint8')

	for i in range (500) :
		for j in range (500) :
			if i < j and i < 500 - j :
				mask_t[i,j] = 1
			elif i < j and i >= 500 - j :
				mask_l[i,j] = 1
			elif i >= j and i < 500 - j :
				mask_r[i,j] = 1
			else :
				mask_b[i,j] = 1
	return [mask_r, mask_l, mask_t, mask_b]


def find_direction(src) :
	sum_tot = np.sum(src)
	i = -1

	masks_rltb = generate_rltb_masks()

	rltb = [cv2.bitwise_and(src, src, mask = mask) for mask in masks_rltb]
	sums_rltb = [np.sum(im) for im in rltb]
	print(sums_rltb)
	maxi = max(sums_rltb)
	if maxi / sum_tot > 0.7 :
		i = sums_rltb.index(maxi)

	return i


def trav_h_or_pan(src) :
	sumy = np.sum(src,0)
	sumx = np.sum(src,1)
	longueur = min(max(sumx),250)
	l1 = np.mean(sumy[int(250 - longueur) : int(250 - longueur / 2)])
	l2 = np.mean(sumy[int(250 - longueur/2) : 250])
	print (l1, l2, longueur)



#Ouverture du flux video
cap = cv2.VideoCapture("./Vidéos/Pan.avi")

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image et Champ de vitesses (Farnebäck)', cv2.WINDOW_NORMAL)

ret, frame1 = cap.read() # Passe à l'image suivante
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # Passage en niveaux de gris
hsv = np.zeros_like(frame1) # Image nulle de même taille que frame1 (affichage OF)
hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

index = 1
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

mean_hist = np.zeros((500,500))

while(ret):
	index += 1
	flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 15, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 7, # Taille voisinage pour approximation polynomiale
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées 
                                        flags = 0)	
	mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
	hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
	hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	result = np.vstack((frame2,bgr))
	cv2.imshow('Image et Champ de vitesses (Farnebäck)',result)

	hist = histogram2d_Vx_Vy(flow)

	# Pour calculer l'histogramme 2D moyen
	mean_hist += hist;

	cv2.imshow('Histogramme', hist)

	k = cv2.waitKey(15) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		#cv2.imwrite('Frame_%04d.png'%index,frame2)
		#cv2.imwrite('OF_hsv_%04d.png'%index,bgr)
		cv2.imwrite('hist-dense.png',hist * 255)
	prvs = next
	ret, frame2 = cap.read()
	if (ret):
		next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY) 

cap.release()
cv2.destroyAllWindows()


# On divise la somme des histogrammes successifs par leur nombre et on ramène à des valeurs entre 0 et 255 en multipliant par 255.
mean_hist = mean_hist * 255 / index
# On met au maximum toutes les valeurs non nulles de mean_hist (puisque les pixels non-centraux ont toujours des valeurs faibles)
final = mean_hist > 50
final = final.astype('uint8')
plt.imshow(final, 'gray')
i = find_direction(final)
print(i)

trav_h_or_pan(final)

plt.show()


