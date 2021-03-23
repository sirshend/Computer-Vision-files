#!/usr/bin/python3
import os
import random
import cv2
import skimage
from skimage import data
import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_transformer = TfidfTransformer()
import matplotlib.pyplot as plt
#import os
#filename = os.path.join(skimage.data_dir,

from skimage import io
#moon = io.imread(filename)
#print(moon)
#print(moon.shape)ectory
#path1 = "/home/sirshendu/Desktop/train/3m_high_tack_spray_adhesive"
#dirs = os.listdir( path )
#for file in dirs:
#   print (file)

global_labels=[]  ############################  what are the names for
##################these images
################################################################
images_np=[]   ################ matrices for the images
####################################################################
sift_kp=[]
sift_des=[]
category_map={}
#data_directory="/home/sirshendu/Desktop/train/combined"
data_directory='/home/ankgaut/Desktop/Assignment2_Dataset/total'
file_names = [ os.path.join(data_directory, f) for f in
os.listdir(data_directory) if f.endswith(".jpg")]

image_dict={}
count=0

image_descriptor_sizes=[]


#print (file_names)
#z=len(file_names)
#print(z)

length=len(file_names)
#print(length)
for f in file_names:
	l=f.split('/')
	# l=f.split('/')
	g=l[6]
	final=g.split('.')
	final2=final[0].split('_')
	item=str(final2[0]+"_"+final2[1])
	#print(item)
	#print("\n")
	#print(f)
	#print("\n")
	global_labels.append(item)	
	p=io.imread(f)
	#print(p)
	#global_labels.append(llabel)
	images_np.append(io.imread(f))
	sift = cv2.xfeatures2d.SIFT_create()
	kp, des = sift.detectAndCompute(io.imread(f),None)
	image_dict[count]=des
	y = np.atleast_2d(image_dict[count])
	image_descriptor_sizes.append(len(y))
	#print(des)
	count=count+1
	z=des
	break
	#sift_kp.append(kp)
	#sift_des.append(des)
	#print(sift_kp[0])
	#print(sift_kp[0].shape)
#print(image_dict[0].shape)
#y = np.atleast_2d(image_dict[0])
#l=len(y)
#print(l)
#np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
z=des
for f in file_names:
		if f != file_names[0]:
				l=f.split('/')
				#print (l)
				#gprint (l[6])
				l=f.split('/')
				g=l[6]
				final=g.split('.')
				final2=final[0].split('_')
				item=str(final2[0]+"_"+final2[1])
				#print (label_beta)
				#print(type(label_beta[0]))
				"""l=len(label_beta)
				for i in range(l-2):
				
				#print (llabel)
				#llabel=llabel-str('_')
				llabel=llabel[:-1]"""
				#print (llabel)
				#break
				p=io.imread(f)
				#print(p)
				global_labels.append(item)
				images_np.append(io.imread(f))
				sift = cv2.xfeatures2d.SIFT_create()
				kp, des = sift.detectAndCompute(io.imread(f),None)
				image_dict[count]=des
				y = np.atleast_2d(image_dict[count])
				image_descriptor_sizes.append(len(y))
				#print(des)
				print(count)
				count=count+1

				z=np.append(z,des,axis=0)
#print(global_labels)
#print(image_descriptor_sizes)
num_clusters=100
############### K-Means reimplemented here ################################################################################
kmeans=KMeans(n_clusters=num_clusters,init='k-means++').fit(z)

Visual_words=kmeans.cluster_centers_

final_Label=kmeans.predict(z)
#print(kmeans.cluster_centers_)
#print(len(kmeans.labels_))
sum=0
cumulative_sizes=np.zeros((length,1))
for i in range(length):
        sum=sum+image_descriptor_sizes[i]
        cumulative_sizes[i]=sum
#print(cumulative_sizes)
countere=1
new_dictionary={}
for i in range(length):
  if i==0:
    print(countere)
    print("\n")
    countere=countere+1
    classses=[str(int(a)) for a in final_Label[0:int(cumulative_sizes[i])]]
    new_dictionary[i]=' '.join(classses)



  else:
    
    classses=[str(int(a)) for a in final_Label[int(cumulative_sizes[i-1]):int(cumulative_sizes[i])]]
    new_dictionary[i]=' '.join(classses)


text=[new_dictionary[i] for i in range(length)]

vectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-zA-Z0-9_]+\\w*\\b')
L=vectorizer.fit(text)

TF_IDF_matrix=vectorizer.transform(text)
#print(TF_IDF_matrix.shape)
X=TF_IDF_matrix

######################################### Now SVM implementation ######################################################################

