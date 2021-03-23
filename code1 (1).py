
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
data_directory='/home/sirshendu/Desktop/small_data/3m_high_tack_spray_adhesive' 
file_names = [ os.path.join(data_directory, f) for f in
os.listdir(data_directory) if f.endswith(".jpg")]
data_directory='/home/sirshendu/Desktop/small_data/cheez_it_white_cheddar'
file_names2 = [ os.path.join(data_directory, f) for f in
os.listdir(data_directory) if f.endswith(".jpg")]

image_dict={}
count=0

image_descriptor_sizes=[]

for f in file_names2:
        file_names.append(f)
#print (file_names)
#z=len(file_names)
#print(z)

length=len(file_names)
#print(length)
for f in file_names:
        l=f.split('/')
        #print (l)
        #gprint (l[6])
        g=l[5]
        #p=l[6].split('.')
        #print (p[0])
        #print ('\n')
        #break
        #label_beta=p[0].split('_')
        llabel=g
        #print (label_beta)
        #print(type(label_beta[0]))
        #l=len(label_beta)
        """for i in range(l-2):
                llabel=llabel+label_beta[i]
                #if i != (l-2)
                llabel=llabel+'_'"""

        #print (llabel)
        #llabel=llabel-str('_')
        #llabel=llabel[:-1]
        #print (llabel)
        #break
        p=io.imread(f)
        #print(p)
        global_labels.append(llabel)
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
                g=l[5]
                """p=l[6].split('.')
                #print (p[0])
                #print ('\n')
                #break
                label_beta=p[0].split('_')"""
                llabel=g
                #print (label_beta)
                #print(type(label_beta[0]))
                """l=len(label_beta)
                for i in range(l-2):
                        llabel=llabel+label_beta[i]
                        #if i != (l-2)
                        llabel=llabel+'_'

                #print (llabel)
                #llabel=llabel-str('_')
                llabel=llabel[:-1]"""
                #print (llabel)
                #break
                p=io.imread(f)
                #print(p)
                global_labels.append(llabel)
                images_np.append(io.imread(f))
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(io.imread(f),None)
                image_dict[count]=des
                y = np.atleast_2d(image_dict[count])
                image_descriptor_sizes.append(len(y))
                #print(des)
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

new_dictionary={}
for i in range(length):
  if i==0:
    
    
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
Y=[]
for i in range(length):
        if global_labels[i]=='3m_high_tack_spray_adhesive':
                Y.append(int(0))        
        elif global_labels[i]=='cheez_it_white_cheddar':
                Y.append(int(1))
        elif global_labels[i]=='campbells_chicken_noodle_soup':
                Y.append(int(2))
        elif global_labels[i]=='aunt_jemima_original_syrup':
                Y.append(int(3))
        elif global_labels[i]=='cholula_chipotle_hot_sauce':
                Y.append(int(4))
        elif global_labels[i]=='clif_crunch_chocolate_chip':
                Y.append(int(5))
        elif global_labels[i]=='coca_cola_glass_bottle':
                Y.append(int(6))
        elif global_labels[i]=='detergent':
                Y.append(int(7))
        elif global_labels[i]=='expo_marker_red':
                Y.append(int(8))
        elif global_labels[i]=='listerine_green':
                Y.append(int(9))
        elif global_labels[i]=='nice_honey_roasted_almonds':
                Y.append(int(10))
        elif global_labels[i]=='nutrigrain_apple_cinnamon':
                Y.append(int(11))
        elif global_labels[i]=='palmolive_green':
                Y.append(int(12))
        elif global_labels[i]=='pringles_bbq':
                Y.append(int(13))
        elif global_labels[i]=='vo5_extra_body_volumizing_shampoo':
                Y.append(int(14))
        elif global_labels[i]=='vo5_split_ends_anti_breakage_shampoo':
                Y.append(int(15))
#print(Y)
#print(global_labels)













#########################################################################################################################################3
M=X.toarray()
#clf = svm.SVC(gamma='0.01', decision_function_shape='ovo',probability='True')
clf = svm.SVC(probability='True')
print(type(M[0]), Y[0])
#print (y[0])
label = np.asarray(Y)
print(label[0], type(label[0]))
clf.fit(M, label)

dec = clf.decision_function([[1]])
dec.shape[1]
clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1]






##############################  Assume opencv works from here
######################## t
##################################################################
#######################################################################################################################################################################
############################################################################################################################################################################################
#######################################################################################################################################################################
###########################################################################################################################################################################################
###################################################################################################################################################################
#print(shift_des.shape)
#counter1=0
#counter2=0
#length=0
#num_elem=[]
#num_cumul=[]
#fdesc=[]     ################# ALL-DESCRIPTORS are HERE
#################################################################################3
#for f in shift_des:
 #       length=length+1
  #      counter1=0
   #     for g in f:
    #            fdesc.append(g)
     #           counter1=counter1+1
      #          counter2=counter2+1
       # num_elem.append(counter1)
        #num_cumul.append(counter2)
################################ K-MEANS
########################K-MEANS##############################################################################################################

#print(fdesc.shape)
#kmeans = KMeans(n_clusters=2000)
#kmeans.fit(fdesc)
#clusterss=[]
#id_tags=[]
#for each f in kmeans.cluster_centers_:
 #       clusterss.append(f)
#for each f in kmeans.labels_:
#        id_tags.append(f)

################## image descriptors ############################
#i = 0
#j=0
#final_image_descriptors=[]
#for k in range(length):
 #       for i in range(num_cumul[j]):
  #              final_image_descriptors[j].append(id_tags[i])
   #     j=j+1
#new_wordlist=[]
#for i in range(length):
 #       new_wordlist[i]=''.join(str(e) for e in final_image_descriptors[i])
################################## we have got the image descriptors and
#parsed them as list of strings
####################################################
#vectorizer = TfidfVectorizer()
 #X = vectorizer.fit(new_wordlist)
# TFIDF_matrix=vectorizer.fit_transform(new_wordlist)
#integer_labels=[]
#lenn=len(filenames)gth

"""lin_clf = svm.LinearSVC()
#lin_clf = svm.LinearSVC()
lin_clf.fit(TFIDF_matrix  ,integer_labels)"""
