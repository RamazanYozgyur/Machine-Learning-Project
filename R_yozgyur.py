import keras
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
#from keras.applications.resnet50 import preprocess_input
from keras_preprocessing.image import load_img 
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import os
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


model =ResNet50(weights='imagenet', include_top=True, input_shape=(224, 224,3))

try:
    filename = sys.argv[1]
except IndexError:
    print("You must supply a file name.")
    sys.exit(2)

def ready_csv(data_list):
    resnet_feature_list = []
    a=open(data_list,"r")
    for i in a:
        if i[-1]=="\n":
           img = img_to_array(load_img(i[:-1],target_size=(224,224))) 
           img = preprocess_input(np.expand_dims(img,axis=0))
           resnet_feature = model.predict(img)
           resnet_feature_list.append(resnet_feature.flatten())
        else:
           img = img_to_array(load_img(i,target_size=(224,224)))
           img = preprocess_input(np.expand_dims(img,axis=0))
           resnet_feature = model.predict(img)
           resnet_feature_list.append(resnet_feature.flatten())
    return pd.DataFrame(resnet_feature_list)
          
csv=ready_csv(filename)

pca=PCA()
pca.fit(csv)
csv_new=PCA(n_components=2).fit_transform(csv)

kmeans=KMeans(n_clusters=3,random_state=22)
kmeans.fit(csv_new)
list_flnm=open(filename,"r")

groups = {}
for file, cluster in zip(list_flnm,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file.split("/")[-1])
    else:
        groups[cluster].append(file.split("/")[-1])

list_flnm.close()
output_file=open("output_file.txt","w")
for i in set(kmeans.labels_):
    for j in groups[i]:
        output_file.write(j.rstrip("\n")+" ")
    output_file.write("\n") 
output_file.close()

list_flnm=open(filename,"r")
new_path=list_flnm.readlines()[1][::-1].split("/",1)[1][::-1]
list_flnm.close()

output_html_file=open("output_html_file.html","w")
output_html_file.write("<!DOCTYPE html> \n")
output_html_file.write("<html> \n")
output_html_file.write("<body> \n")

for i in set(kmeans.labels_):
    output_html_file.write("<h2><p> cluster %s <p><h2> \n " %(i))
    for j in groups[i]:
        final_path= new_path+"/"+j
        output_html_file.write("<img src='%s' width='30' height='30'> \n" %(final_path))
    
    output_html_file.write("<HR>")
output_html_file.write("<body> \n")
output_html_file.write("<html> \n")

output_html_file.close()

print("\nYou can find output files in the folllowing path :", os.getcwd())
print("With filename 'output_file.txt' and output_html_file.html")








































