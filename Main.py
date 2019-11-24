import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

import scipy.signal as signal
import scipy.spatial as sp

import skimage as sk

#Normalises the intensity of a gray scale picture within 0 and 255
def normalise(I):
  Imax = np.max(I)
  Imin = np.min(I)
  I = (255/(Imax-Imin))*I - (255/(Imax-Imin))*Imin
  return(I)

#Computes the absolute difference between two pictures of the same size
def difference(I,J):
    return(np.sum(np.abs(I-J))/(2*np.size(I)))

#Computes the euclidian distance between two pictures of the same size
def distance(I,J):
    return(np.sqrt(np.sum((I-J)**2)))

#n_match = Threshold on the number of matches used to evaluate simillarity between two frames
def cutDetection(histogramDifferences, grayFrames, n_match = 5,plot = False):

    #Compute first threshold for cut detection
    Tcut = np.average(histogramDifferences) + 2*np.std(histogramDifferences)
    print("Threshold for cut detection = "+str(Tcut))

    diffDifferences = []
    #Compute the first derivative of the differences
    for i in range(len(histogramDifferences)-1):
        diffDifferences.append(histogramDifferences[i+1] - histogramDifferences[i])
    diffDifferences.append(histogramDifferences[-1])

    if(plot):
        plt.figure()
        plt.plot(np.arange(0,len(histogramDifferences)),histogramDifferences,label="Histogram difference")
        plt.plot(np.arange(0,len(diffDifferences)),np.abs(diffDifferences),label="Absolute value of the derivative of the histogram difference")
        plt.plot(np.arange(0,len(histogramDifferences)),np.ones(len(histogramDifferences))*Tcut,label="Cut detection threshold")
        plt.title("Cut detection")
        plt.legend()
        plt.show()

    index = np.argwhere(np.abs(diffDifferences)>Tcut)
        
    print("First estimation of the cut indices : ")
    print(index)

    cutIndex = []

    #Filter good index using ORB feature detection
    for i in range(len(index)):
        print("Testing index : " + str(index[i][0]) + "...")

        #For the first index, we compare the corresponding frame with the one located 5 frames before - Arbitrary choice
        if(i==0):
            img1 = grayFrames[max(0,index[i][0]-5)]
        else:
            img1 = grayFrames[index[i-1][0]]
        img2 = grayFrames[index[i][0]]

        #Create ORB detector
        orb = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                        scaleFactor = 1.2,#Par défaut : 1.2
                        nlevels = 8)#Par défaut : 8

        #Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        try:
            #Compute matches
            matches = bf.knnMatch(des1,des2, k=2)
            good = []
            #Perform ration test
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append([m])

            #If there are less than n_match good matches, the frames aren't simillar -> Cut !
            if(len(good)<n_match):
                cutIndex.append(index[i][0])
                print("Good index !")
            else:
                print("Bad index...")

        except:
            #If no match can be found, the two frames must be very different
            print("Error in features detection -- Index supposed OK !")
            cutIndex.append(index[i][0])
            continue

    cutIndex.insert(0,0)
    cutIndex.append(len(grayFrames)-1)
    print("Second estimation of the cut indices :")
    print(cutIndex)

    if(plot):
        plt.figure()
        plt.plot(np.arange(0,len(diffDifferences)),diffDifferences,label="Aboslute value of the first derivative of the histogram difference")
        labelCounter = 0
        for index in cutIndex:
            if(labelCounter == 0):
                labelCounter+=1
                plt.plot(index,diffDifferences[index],'*',color='red',label="Cut")
            else:
                plt.plot(index,diffDifferences[index],'*',color='red')
        plt.title("Detected cuts")
        plt.legend()
        plt.show()

    return(cutIndex)

#n_match = Threshold on the number of matches used to evaluate simillarity between two frames
#len_dissolve = Minimum length of a dissolve sequence
def localDissolveDetection(histogramDifferences, averageIntensity, grayFrames, refIndex, n_match, len_dissolve, plot):
    dissolve = []

    diffDifferences = []
    #Compute the first derivative of the differences
    for i in range(refIndex[0],refIndex[-1]):
        diffDifferences.append(histogramDifferences[i+1] - histogramDifferences[i])
    diffDifferences.append(histogramDifferences[refIndex[-1]])

    #Compute second threshold for dissolve detection
    Tcut = np.average(differences[refIndex[0]:refIndex[-1]]) + np.std(differences[refIndex[0]:refIndex[-1]])
    print("Threshold for dissolve detection = "+str(Tcut))

    if(plot):
        plt.figure()
        plt.title("Threshold detection")
        plt.plot(np.arange(0,len(diffDifferences)),np.abs(diffDifferences),label="Absolute value of the derivative")
        plt.plot(np.arange(0,len(diffDifferences)),np.ones(len(diffDifferences))*Tcut,label="Dissolve detection threshold")
        plt.legend()
        plt.show()

    dissolveIndex = np.argwhere(np.abs(diffDifferences)>Tcut)

    print("First estimation of the dissolve indices : ")
    print(dissolveIndex)

    #Compute the filtered derivative of the gray scale intensity of the picture using a Stavisky-Golay filter 
    diffIntensity = signal.savgol_filter(avgIntensity[refIndex[0]:refIndex[-1]], window_length=5, polyorder=2, deriv=1) 
    
    #Set very small values to 0 to avoid errors
    for i in range(len(diffIntensity)):
        if(np.abs(diffIntensity[i])<10**(-4)):
            diffIntensity[i]=0

    #Detect windows with a continuous intensity increase -> Dissolves
    for i in range(len(dissolveIndex)):
        delta = diffIntensity[dissolveIndex[i][0]]
        print(len(diffIntensity))
        print(dissolveIndex)
        if(len(dissolve)!=0 and dissolve[-1][-1]>dissolveIndex[i][0]):
            continue
        else:
            counterNeg = 0
            counterPos = 0
            print(refIndex[-1])
            while(np.sign(diffIntensity[dissolveIndex[i][0]+counterPos]) == np.sign(delta)):
                if(dissolveIndex[i][0]+counterPos+1 >= len(diffIntensity)):
                    break
                else:
                    counterPos+=1
            while(np.sign(diffIntensity[dissolveIndex[i][0]+counterNeg]) == np.sign(delta)):
                if(dissolveIndex[i][0]+counterNeg-1 < 0):
                    break
                else:
                    counterNeg-=1

            print("Testing dissolve : " + str(refIndex[0]+dissolveIndex[i][0]+counterNeg) + " - " + str(refIndex[0]+dissolveIndex[i][0]+counterPos))

            if(counterPos+counterNeg < 10):
                print("Bad dissolve...")
            
            #Filter good index using ORB feature detection
            else:
                img1 = grayFrames[refIndex[0]:refIndex[-1]][dissolveIndex[i][0]+counterNeg]
                img2 = grayFrames[refIndex[0]:refIndex[-1]][dissolveIndex[i][0]+counterPos]

                #Create ORB detector
                orb = cv2.ORB_create(nfeatures = 500,#Par défaut : 500
                            scaleFactor = 1.2,#Par défaut : 1.2
                            nlevels = 8)#Par défaut : 8

                #Find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(img1,None)
                kp2, des2 = orb.detectAndCompute(img2,None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

                try:
                    #Compute matches
                    matches = bf.knnMatch(des1,des2, k=2)
                    good = []
                    #Perform ratio test
                    for m,n in matches:
                        if m.distance < 0.7*n.distance:
                            good.append([m])

                    #If the dissolve is long enough, and the first and last frame are really different -> Dissolve !
                    if(len(good)<len_dissolve):
                        dissolve.append([refIndex[0]+dissolveIndex[i][0]+counterNeg,refIndex[0]+dissolveIndex[i][0]+counterPos])
                        print("Good dissolve !")
                    else:
                        print("Bad dissolve...")

                except:
                    print("Error in feature detection -- Dissolvre supposed OK")
                    dissolve.append([refIndex[0]+dissolveIndex[i][0]+counterNeg,refIndex[0]+dissolveIndex[i][0]+counterPos])
                    continue
    print("Second estimation of the dissolve sections : ")
    print(dissolve)

    if(plot):        
        plt.figure()
        plt.plot(np.arange(0,len(avgIntensity[refIndex[0]:refIndex[-1]])),avgIntensity[refIndex[0]:refIndex[-1]],label="Average gray scale intensity")
        labelCounter = 0
        maxIntensity = np.max(avgIntensity[refIndex[0]:refIndex[-1]]) + 5
        for d in dissolve:
            if(labelCounter == 0):
                labelCounter += 1
                plt.plot(d,maxIntensity*np.ones(len(d)),'--',color='red',label="Dissolve")
            else:
                plt.plot(d,maxIntensity*np.ones(len(d)),'--',color='red')
        plt.title("Detected dissolves")
        plt.legend()
        plt.show()

    return(dissolve)

def globalDissolveDetection(cutIndex, histogramDifferences, averageIntensity, grayFrames, n_match = 5, len_dissolve = 5, plot = False):
    dissolveSequences = []

    for i in range(len(cutIndex)-1):
        refIndex=[cutIndex[i]+1,cutIndex[i+1]-1]
        dissolve = localDissolveDetection(histogramDifferences,averageIntensity,grayFrames,refIndex,n_match,len_dissolve,plot)
        for sequence in dissolve:
            dissolveSequences.append(sequence)

    return(dissolveSequences)

def extractShots(cutIndex,dissolveSequences):
    shots=[]

    if(len(dissolveSequences)==0):
        for i in range(len(cutIndex)-1):
            shots.append([cutIndex[i],cutIndex[i+1]])
    else:
        for i in range(len(cutIndex)-1):
            for j in range(len(dissolveSequences)):
                if(dissolveSequences[j][0]>cutIndex[i] and dissolveSequences[j][0]<cutIndex[i+1]):
                    if(len(shots) == 0 or shots[-1][-1]==cutIndex[i]):
                        shots.append([cutIndex[i],dissolveSequences[j][0]])
                        if(j<len(dissolveSequences)-1 and dissolveSequences[j+1][0]<cutIndex[i+1]):
                            shots.append([dissolveSequences[j][-1],dissolveSequences[j+1][0]])
                    elif(j<len(dissolveSequences)-1 and dissolveSequences[j+1][0]<cutIndex[i+1]):
                        shots.append([dissolveSequences[j][-1],dissolveSequences[j+1][0]])
                    else:
                        shots.append([dissolveSequences[j][-1],cutIndex[i+1]])
                else:
                    shots.append([cutIndex[i],cutIndex[i+1]])
                    break
    print("Identified shots : ")
    print(shots)
    return(shots)

def computeClosest(list):
    L = [l.flatten() for l in list]
    M = sp.distance_matrix(L,L)
    index = np.argmin(np.sum(M,axis=0))
    return(list[index],index)

def keyFrameExtraction(grayFrames,shots,method = "entropy", plot = False):
    keyFrames = []
    keyIndex = []
    for shot in shots:
        if(method=="entropy"):
            entropy = []
            for i in range(shot[0],shot[-1]+1):
                entropy.append(sk.measure.shannon_entropy(allFrames[i]))
            frame=allFrames[shot[0]+np.argmax(entropy)]
            keyIndex.append(shot[0]+np.argmax(entropy))
        else:
            frame,index=computeClosest(allFrames[shot[0]:shot[-1]])
            keyIndex.append(index)
        keyFrames.append(frame)
        if(plot):
            cv2.imshow("KeyFrame",frame)
            cv2.waitKey(0)
    print("Identified key frames : ")
    print(keyIndex)
    return(keyFrames)

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
			if j > 260 : # Comme pour les masques suivants, on masque les 10 premières lignes ou colones (selon le masque) à partir du centre. En effet, si les pixels sont concentrés au centre, peu importe leur répartition exacte, qui faussera la donne plus qu'autre chose.
				mask_r[i,j] = 1
			if j <= 240 :
				mask_l[i,j] = 1
			if i > 260 :
				mask_b[i,j] = 1
			if i <= 240 :
				mask_t[i,j] = 1

	"""
	for i in range (500) :
		for j in range (500) :
			if i < j and i < 500 - j :
				mask_t[i,j] = 1
			elif i < j and i >= 500 - j :
				mask_r[i,j] = 1
			elif i >= j and i < 500 - j :
				mask_l[i,j] = 1
			else :
				mask_b[i,j] = 1
	"""

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
	print(i)

	return i

def trav_h_or_pan(src,side) :
	"""
	side = 0 pour droite et side = 1 pour gauche
	"""
	kernel = np.ones((3,3),np.uint8)
	src = cv2.erode(src,kernel,iterations = 1)
	sumy = np.sum(src,0)
	sumx = np.sum(src,1)
	longueur = min(max(sumx),250)

	a,b,c,d = 0,0,0,0

	if side == 1 :
		a = int(250 - longueur)
		b = int(250 - longueur / 2)
		c = int(250 - longueur / 2)
		d = 250
	else :
		a = int(250 + longueur / 2)
		b = int(250 + longueur)
		c = 250
		d = int(250 + longueur / 2)

	l1 = np.mean(sumy[a : b])
	l2 = np.mean(sumy[c : d])
	#print(l1,l2,longueur)
	return (0.8 * l1 > l2)

def rotation_or_not(src) :
	# Si c'est une rotation alors on a une zone blanche beaucoup plus large que dans le cas d'un tilt, d'un plan fixe ou d'un zoom : on va ici considérer qu'à partir d'1/10 de l'image cette surface blache est suffisante pour reconnaître une rotation.
	sumtot = np.sum(src)
	return (sumtot > 0.1 * 500 ** 2)

def ta_or_not(src) :
	srcmed = src > np.median(src)
	srcmed = srcmed.astype('uint8')
	sumtot = np.sum(srcmed)
	return (sumtot > 0.2 * 500 ** 2)
	
def fixed_or_zoom(src) :
	sumtot = np.sum(src)
	#print(sumtot, sumtot / 500 ** 2)
	return (sumtot > 0.0003 * 500 **2)

def find_shot(src) :

    # On met au maximum toutes les valeurs de mean_hist supérieures à un seuil (puisque les pixels non-centraux ont toujours des valeurs faibles). Cela permet de voir les "cônes" des travellings horizontaux de manière satisfaisante. En revanche, src50 n'est pas exploitable pour les tilts puisqu'une partie importante des pixels significatifs pour reconnaître ce plan a des valeurs très faibles comparées aux autres.
    src50 = src > 50
    src50 = src50.astype('uint8')

    i = find_direction(src50)
    if i == -1 :
        print ("Plan fixe, travelling avant, rotation ou zoom")
        rot = rotation_or_not(src50)
        if rot :
            print("Rotation")
            string = "Rotation"
        else :
            ta = ta_or_not(src)
            if ta :
                print ("Travelling avant")
                string = "Travelling avant"
            else :
                zoom = fixed_or_zoom(src)
                if zoom :
                    print ("Zoom avant")
                    string = "Zoom avant"
                else :
                    print ("Plan fixe")
                    string = "Plan fixe"


    elif i == 0 :
        print ("Déplacement horizontal vers la gauche")
        ok = trav_h_or_pan(src50,i)
        if ok :
            print ("Pan")
            string = "Pan vers la droite"
        else :
            print ("Travelling")
            string = "Travelling vers la gauche"
    elif i == 1 :
        print ("Déplacement horizontal vers la droite")
        ok = trav_h_or_pan(src50,i)
        if ok :
            print ("Pan")
            string = "Pan vers la droite"
        else :
            print ("Travelling")
            string = "Travelling vers la droite"
    elif i == 2 :
        print ("Tilt vers le bas")
        string = "Tilt vers le bas"
    else :
        print ("Tilt vers le haut")
        string = "Tilt vers le haut"
    return src50, string

def shotsIdentification(shots,flow):
    Id = []

    for shot in shots:
        print("Identifying shot : ")
        print(shot)

        shotFlow = flow[shot[0]:shot[-1]]
        mean_hist = np.zeros((500,500))

        # On divise la somme des histogrammes successifs par leur nombre et on ramène à des valeurs entre 0 et 255 en multipliant par 255.
        for hist in shotFlow:
            mean_hist += hist
        mean_hist = mean_hist * 255 / len(shotFlow)

        src50,string = find_shot(mean_hist)
        plt.imshow(src50,'gray')
        plt.show()

        Id.append(string)

    print("Shot identification : ")
    print(Id)
    return(Id)

###MAIN###

cap = cv2.VideoCapture('Extrait3-Vertigo-Dream_Scene(320p).m4v')
ret, frame = cap.read() 

#Are the frames in gray scale or in color ?
n = len(frame.shape)

#Color
if(n == 3):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = True
#Gray scale
else:
    frame_gray = frame
    color = False

#Lists initialisation
allFrames = []
differences = []
avgIntensity = []

allFlow = []

if(color):
    #Compute 2D-histogram
    h, xedges, yedges = np.histogram2d(frame_yuv[:,:,1].flatten(),frame_yuv[:,:,2].flatten(),bins=(256,256),range=[[0, 255], [0, 255]])
    #Display histogram
    #Display in color - not recommanded
    #im=ax.imshow(normalise(h),cmap='gray')
    #Display in gray scale
    #cv2.imshow("Color histogram",h)
    #cv2.waitKey(0)

else:
    #Plot initialisation
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    #Compute 1D-histogram
    h, bins = np.histogram(frame_gray.flatten(),bins=256,range=[0,255])
    #Display histogram
    im=ax.bar(np.arange(0,256),h)

prevFrame = frame_gray

print("-- STARTING VIDEO --")
counter=0
while(ret and counter<500):
    counter+=1
    print(counter)
    
    ret,frame = cap.read()

    #Color
    if(color):
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Gray scale
    else:
        frame_gray = frame

    allFrames.append(frame_gray)

    if(color):
        #Compute 2D-histogram
        h1, xedges, yedges = np.histogram2d(frame_yuv[:,:,1].flatten(),frame_yuv[:,:,2].flatten(),bins=(256,256),range=[[0, 255], [0, 255]])
        #Update histogram
        #Update in color - not recommanded
        #im.set_data(normalise(h1))
        #Update in gray scale
        cv2.imshow('Color histogram',h1)

    else:
        #Compute 1D-histogram
        h1, bins = np.histogram(frame_gray.flatten(),bins=256,range=[0,255])
        #Update histogram
        ax.set_ylim(0,np.max(h1))
        for rectangle,height in zip(im.patches,h1):
            rectangle.set_height(height)

    plt.pause(0.05)
    
    #Compute difference between histograms
    diff = difference(h,h1)
    differences.append(diff)

    #Compute the average gray scale intensity of the frame
    avgIntensity.append(np.average(frame_gray))

    #Iteration
    h=h1

    #Compute optical flow
    hsv = np.zeros_like(frame) # Image nulle de même taille que frame1 (affichage OF)
    hsv[:,:,1] = 255 # Toutes les couleurs sont saturées au maximum

    flow = cv2.calcOpticalFlowFarneback(prevFrame,frame_gray,None, 
                                        pyr_scale = 0.5,# Taux de réduction pyramidal
                                        levels = 3, # Nombre de niveaux de la pyramide
                                        winsize = 9, # Taille de fenêtre de lissage (moyenne) des coefficients polynomiaux - w
                                        iterations = 3, # Nb d'itérations par niveau
                                        poly_n = 5, # Taille voisinage pour approximation polynomiale - Applicability
                                        poly_sigma = 1.5, # E-T Gaussienne pour calcul dérivées
                                        flags = 0)

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1]) # Conversion cartésien vers polaire
    hsv[:,:,0] = (ang*180)/(2*np.pi) # Teinte (codée sur [0..179] dans OpenCV) <--> Argument
    hsv[:,:,2] = (mag*255)/np.amax(mag) # Valeur <--> Norme 

    #Display optical flow
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('Optical flow',bgr)

    #Compute flow histogram
    hist = histogram2d_Vx_Vy(flow)
    allFlow.append(hist)

    #Display histogram
    cv2.imshow('Flow histogram', hist)

    #Iteration
    prevFrame = frame_gray

    #Display frame
    cv2.imshow("Film",frame)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break

print("-- ENDING VIDEO --")

cutIndex = cutDetection(differences,allFrames)
dissolveSequences = globalDissolveDetection(cutIndex,differences,avgIntensity,allFrames,plot=True)
shots = extractShots(cutIndex,dissolveSequences)
identification = shotsIdentification(shots,allFlow)
keyFrame = keyFrameExtraction(allFrames,shots) 


