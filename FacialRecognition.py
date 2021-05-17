from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import tkinter as tk
import os
from keras.models import load_model
import cv2
import numpy as np
import os
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from pathlib import Path

names = []
database = {}
global fln
cascade_path = 'haarcascade_frontalface_default.xml'

#load model
model=load_model('facenet_keras.h5',compile=False)

image_size=160
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_face(face, margin, img):
    #cascade = cv2.CascadeClassifier(cascade_path)
    #img = imread(Path(imagepath))
    aligned_image = []

    (x, y, w, h) = face
    cropped = img[y-margin//2:y+h+margin//2,
                  x-margin//2:x+w+margin//2, :]
    aligned = resize(img, (image_size, image_size), mode='reflect')
    aligned_image.append(aligned)
            
    return np.array(aligned_image)


def calc_emb(face,img, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_face(face, margin, img))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

#Eucledian distance from embeddings
def calc_dist(f0, f1):
    return distance.euclidean(f0, f1)

def addFace(name,imagepath):
	img = imread(Path(imagepath))
	cascade = cv2.CascadeClassifier(cascade_path)
	faces = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
	database[name]=calc_emb(faces[0],img)
	names.append("name")
addFace('Martin LK',Path("sample/MLK.jpg"))
def identify(face, database,img):
    '''Identify person
    image_path= path of target image
    database--precalculated embeddings
    model-- inception model'''

    enc=calc_emb(face,img)

    min_dist=100

    #find the closest finding from database

    for(name, db_enc) in database.items():

        #Compute eucledian distance between target and database
        dist=calc_dist(enc,db_enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.9:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return identity, min_dist











def addimage():
	global fln
	fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Select Person's Image", filetypes=(("JPG file",'*.jpg'),("PNG file",'*.png')))
	return None

def addperson():
	text=name.get()
	addFace(text,fln)
	print(text)
	return None

def testimage():
	print('singleimage')
	global fln
	cascade = cv2.CascadeClassifier(cascade_path);
	font = cv2.FONT_HERSHEY_SIMPLEX
	ImagePath=fln
	img=imread(Path(ImagePath))
	Oimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    
	faces = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
	for(x,y,w,h) in faces:
	    fa=0
	    cv2.rectangle(Oimg, (x,y), (x+w,y+h), (0,255,0), 2)
	    id, dist = identify(faces[fa],database,img)
	    fa+=1
	        #id='sam'
	        # If dist is less then 1  : perfect match 
	    if (dist > 0.9):
	        id = "unknown"
	            #confidence = "  {0}%".format(round(100 - confidence))
	        #else:
	        #    id = "unknown"
	            #confidence = "  {0}%".format(round(100 - confidence))
	        
	    cv2.putText(
	                Oimg, 
	                str(id), 
	                (x+5,y-5), 
	                font, 
	                1, 
	                (255,255,255), 
	                2
	               )
	    cv2.putText(
	                Oimg, 
	                str(dist), 
	                (x+5,y+h-5), 
	                font, 
	                1, 
	                (255,255,0), 
	                1
	               )  
	    
	 
	recognized=ImagePath[-6:]
	recoPath='recognized/'+recognized
	cv2.imshow('picture',Oimg)
	cv2.imwrite(recoPath, Oimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return None
def cam_reco():
	print('start camera and recognize')
	faceCascade = cv2.CascadeClassifier(cascade_path);
	font = cv2.FONT_HERSHEY_SIMPLEX
	cam = cv2.VideoCapture(0)
	cam.set(3, 640) # set video widht
	cam.set(4, 480) # set video height
	# Define min window size to be recognized as a face
	#minW = 0.1*cam.get(3)
	#minH = 0.1*cam.get(4)
	while True:
	    ret, img =cam.read()
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    
	    faces = faceCascade.detectMultiScale( 
	        gray,
	        scaleFactor = 1.3,
	        minNeighbors = 3,
	        minSize = (30, 30),
	       )
	    for(x,y,w,h) in faces:
	        fa=0
	        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	        id, dist = identify(faces[fa],database,img)
	        fa+=1
	        #id='sam'
	        # If dist is less then 1  : perfect match 
	        if (dist > 1):
	            id = "unknown"
	            #confidence = "  {0}%".format(round(100 - confidence))
	        #else:
	        #    id = "unknown"
	            #confidence = "  {0}%".format(round(100 - confidence))
	        
	        cv2.putText(
	                    img, 
	                    str(id), 
	                    (x+5,y-5), 
	                    font, 
	                    1, 
	                    (255,255,255), 
	                    2
	                   )
	        cv2.putText(
	                    img, 
	                    str(dist), 
	                    (x+5,y+h-5), 
	                    font, 
	                    1, 
	                    (255,255,0), 
	                    1
	                   )  
	    
	    cv2.imshow('camera',img) 
	    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
	    if k == 27:
	        break
	# Do a bit of cleanup
	print("\n [INFO] Exiting Program and cleanup stuff")
	cam.release()
	cv2.destroyAllWindows()
	return None




root=Tk()

img=ImageTk.PhotoImage(Image.open('logo.jpeg'))
pannel=Label(root,image=img)
pannel.pack()
frm=Frame(root)
frm.pack(side=TOP, padx=15, pady=15)

#lbl=Label(root)
#lbl.pack()
Label(frm,text="Person's Name::").pack(side=TOP)
name=Entry(frm,width=20)
name.pack(side=TOP)
btn=Button(frm,text='Browse Image',command=addimage)
btn.pack(side=tk.LEFT)

Pname=Button(frm,text='Add', command=addperson)
Pname.pack(side=tk.LEFT)


frm1=Frame(root)
frm1.pack(side=TOP, padx=30, pady=30)
T_img=Button(frm1,text='Recognize Image',command=testimage)
T_img.pack(side=tk.LEFT)

RT_recognition=Button(frm1,text='RT Recognition', command=cam_reco)
RT_recognition.pack(side=tk.LEFT)

btn2=Button(root,text='exit',command=lambda:exit())
btn2.pack(side=tk.LEFT, padx=10)



root.title("Face Recognition system")
root.geometry("300x450")
root.mainloop()