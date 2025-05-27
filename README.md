# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:
```python
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline
```
```python
model = cv2.imread('image_01.png',0)
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.jpeg',0)

plt.imshow(model,cmap='gray')
plt.show()
```

```python
plt.imshow(withglass,cmap='gray')
plt.show()
```

```python
plt.imshow(group,cmap='gray')
plt.show()
```

```python
face_cascade = cv2.CascadeClassifier('DIPT_PROJECT_FACEDETECTION/haarcascade_frontalface_default.xml')

import cv2

# Correct path (adjust if needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img) 
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

```

```python
result = detect_face(withglass)

plt.imshow(result,cmap='gray')
plt.show()result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```

```python
# Gets errors!
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 7) 
        
    return face_img


# Doesn't detect the side face.
result = adj_detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```

```python
eye_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_eye.xml')

import cv2

# Proper path to the eye detection Haar cascade
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return face_img

result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()

eyes = eye_cascade.detectMultiScale(withglass)

# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```

```python
cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()
```

## Images:

![Screenshot 2025-05-27 232910](https://github.com/user-attachments/assets/b8aa1b40-c0b1-4f1e-8679-049bbc1f014b)

![Screenshot 2025-05-27 232917](https://github.com/user-attachments/assets/4baf293e-3a76-47a7-8396-9d5950470295)

![Screenshot 2025-05-27 232922](https://github.com/user-attachments/assets/9cbd889f-9547-4c1c-954b-66c6781620eb)

![Screenshot 2025-05-27 232928](https://github.com/user-attachments/assets/61c0ce8d-947f-4251-8cc1-c3ec277cef42)

![Screenshot 2025-05-27 232946](https://github.com/user-attachments/assets/8f01a206-bb0f-40ad-8f39-a74a5f9a2788)

![Screenshot 2025-05-27 232952](https://github.com/user-attachments/assets/a9167a2f-7cd2-4f2e-9eda-17a848e0d44f)

![Screenshot 2025-05-27 232959](https://github.com/user-attachments/assets/81fbcb2e-dbe3-44ae-8ea2-eb131a83260e)


## Result:

Hence the program is executed successfully.




