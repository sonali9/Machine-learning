
# coding: utf-8

# In[ ]:


import numpy as np
import cv2


# In[ ]:


cam = cv2.VideoCapture(0)


# In[ ]:


face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


# In[ ]:


data = []
i = 0

while True:
    ret, frame = cam.read()

    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        

        for (x, y, w, h) in faces:
            face_component = frame[y:y+h, x:x+w, :]
            fc = cv2.resize(face_component, (50, 50))
            if i%10 == 0 and len(data) < 20:
                data.append(fc)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        i += 1 
        cv2.imshow('frame', frame) 

        if cv2.waitKey(1) == 27 or len(data) >= 20:
            break
    else:
        print ("error")


# In[ ]:


cv2.destroyAllWindows()


# In[ ]:


data = np.asarray(data)
print(data.shape)
np.save('face_02', data)

