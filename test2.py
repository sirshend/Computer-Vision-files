#!/usr/bin/python3	
import os
import random
import cv2
import skimage
from skimage import data
import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_transformer = TfidfTransformer()
import matplotlib.pyplot as plt
from skimage import io
data_directory='/home/ankgaut/Desktop/Assignment2_Dataset/total'
file_names = [ os.path.join(data_directory, f) for f in
os.listdir(data_directory) if f.endswith(".jpg")]
#print(file_names)
image_dict={}
count=0		
#image_descriptor_sizes=[]
image_descriptor_sizes=[]
global_labels=[]
sift_kp=[]
sift_des=[]
category_map={}

images_numpy=[]
length=len(file_names)
print(length)
for f in file_names:
	l=f.split('/')
	g=l[6]
	final=g.split('.')
	final2=final[0].split('_')
	item=str(final2[0]+"_"+final2[1])
	#print(item)
	#print("\n")
	#print(f)
	#print("\n")
	global_labels.append(item)
	image=io.imread(f)
	images_numpy.append(image)
	sift = cv2.xfeatures2d.SIFT_create()
	kp,des = sift.detectAndCompute(io.imread(f),None)
	image_dict[count]=des
	y = np.atleast_2d(image_dict[count])
	image_descriptor_sizes.append(len(y))
	count=count+1
    #z=des
    #break

	#print(image_numpy)
	#break



	
