# Python-project-Scripts.

## This is a  Machine Learning  projects in  Python scripts.

### Description

In this project, I aim to cluster images which was given from the folder. Firstly, I need to prepare my image data which I want to get txt file as a list of names of images from the folder. After I get this txt I get it from the console as an input file. I need to prepare these images so that I can use them as data. For clustering, I used the K-means algorithm for images but I saw that the silhoutte score is not good. I decide to use the resnet50 model before applying Kmeans clustering. After I use the resnet50 model, I did dimension reduction on my data with PCA method. For different K in Kmeans clustering, I choose 3 after I measure the silhoutte score for different K. The program will be executed in around 8 - 10 minutes if you do not run any other stuff on your computer.
As an example you can find images file and sample.txt file in repo.
### HOW TO RUN PROGRAM
If you have folder of images, to get list of images' name in txt file apply the code below in your terminal:
find /home/master/test/data/obj/ -type f -name "*.jpg" > sample.txt


Now assume you have "sample.txt" file as an input, you must enter to terminal below code :

python3 R_yozgyur.py sample.txt
As an out put you will have path to one txt file and one html file.
