import numpy as np
import cv2
import random
 
face_cascade = cv2.CascadeClassifier('face.xml')
mask1 = cv2.imread('mask1.png', -1)
mask2 = cv2.imread('mask2.png', -1)
mask3 = cv2.imread('mask3.png', -1)
mask4 = cv2.imread('mask4.png', -1)
ai_watermark = cv2.imread('alwaysai_logo.png',-1)
#ai_watermark = cv2.resize(ai,None,fx=1.1, fy=1.1, interpolation = cv2.INTER_CUBIC)
 
cap = cv2.VideoCapture(0) #webcam video
 
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
 
 
 
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = np.array(gray, dtype='uint8')
    faces=face_cascade.detectMultiScale(gray1, scaleFactor = 1.2, minNeighbors = 5)
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:

            glass_ymin = int(y)
            glass_ymax = int(y + h)
            sh_glass = glass_ymax - glass_ymin

            face_glass_roi_color = img[glass_ymin:glass_ymax, int(x):int(x+w)]
            #Generate a random number 1 - 4, this is the mask id
            specs_num = random.randint(1,5)
            if(specs_num == 1):
                specs = cv2.resize(mask1, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
                #specs = cv2.resize(specs,None,fx=1.337, fy=1.337, interpolation = cv2.INTER_CUBIC)
                transparentOverlay(face_glass_roi_color,specs)
            if(specs_num == 2):
                specs = cv2.resize(mask2, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
                #specs = cv2.resize(specs,None,fx=1.337, fy=1.337, interpolation = cv2.INTER_CUBIC)
                transparentOverlay(face_glass_roi_color,specs)
            if(specs_num == 3):
                specs = cv2.resize(mask3, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
                #specs = cv2.resize(specs,None,fx=1.337, fy=1.337, interpolation = cv2.INTER_CUBIC)
                transparentOverlay(face_glass_roi_color,specs)
            if(specs_num == 4):
                specs = cv2.resize(mask4, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
                #specs = cv2.resize(specs,None,fx=1.337, fy=1.337, interpolation = cv2.INTER_CUBIC)
                transparentOverlay(face_glass_roi_color,specs)

    transparentOverlay(img, ai_watermark, (-10, -10))
    cv2.namedWindow("alwaysAI Mask Demo", cv2.WND_PROP_FULLSCREEN)
   # cv2.setWindowProperty("HallMask", cv2.WND_PROP_FULLSCREEN, cv2.CV_WINDOW_FULLSCREEN)
    cv2.imshow("alwaysAI Mask Demo", img)
 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
cap.release()
 
cv2.destroyAllWindows()
