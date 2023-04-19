#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import urllib
import requests
import shutil
    
import certifi
import ssl
import gc
import ast
import os
import sys
import random
from os import listdir
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool,Lock
from datetime import datetime


#Only images related to these words are extracted
def getWords1():
    return ['kindergarten teacher','dental hygienist','speech-language pathologist','dental assistant','childcare worker',
            'medical records technician','secretary','medical assistant','hairdresser','dietitian','vocational nurse',
            'teacher assistant','paralegal','billing clerk','phlebotomist','receptionist','housekeeper','registered nurse',
            'bookkeeper','health aide','taper','steel worker','mobile equipment mechanic','bus mechanic',
            'service technician','heating mechanic','electrical installer','operating engineer','logging worker',
            'floor installer','roofer','mining machine operator','electrician','repairer',
            'conductor','plumber','carpenter','security system installer','mason','firefighter',
            'salesperson','director of religious activities','crossing guard','lifeguard',
            'lodging manager','healthcare practitioner','sales agent','mail clerk','electrical assembler',
            'insurance sales agent','insurance underwriter','medical scientist','statistician','training specialist',
            'judge','bartender','dispatcher','order clerk','mail sorter']


#Extracting the images from whole dataset
def downloadImageDataset(save_dir,datasetname,word):
    
    #for the downloading of images from the link
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent} 
    context = ssl._create_unverified_context()

    data = pd.read_parquet(datasetname, engine='pyarrow')
    dfn= data.to_numpy()
    rows = dfn.shape[0]
    #print("total rows: "+str(rows))
    cols = dfn.shape[1]
    val = -1 # inital value no word match
    
    wordPattern =  [ str(" "+word+" "), '-'.join((word.split(' ')))  ]
    save_dir2 = os.path.join(save_dir,'-'.join((word.split(' '))))
    for x in range(0, rows-1):
        attempts=0
        val =-1
        width = dfn[x][3]
        height = dfn[x][4]
        #print("current rows: "+str(x))
      
        for ws in wordPattern  :
            val=dfn[x][1].lower().find(ws)
            if val != -1 :  #val takes -1 if the word is not encountered
                break

        if val != -1:  # val has found the word
            url = dfn[x][0]
            text=dfn[x][1]
            sys.stdout.flush()
            
            try: 
                try:
                    response = requests.get(url,stream=True, timeout=1)
                    responseStatus=True
                except requests.exceptions.ReadTimeout as e:
                    pass
                if responseStatus == True:
                    #print("url: " + url)
                    #unique name for image
                    randomNum =str(str(random.randint(0,100000))+str(random.randint(0,100000)))
                    xxx=str('image'+ str(len(text))+str(width+height)+str(height)+str(len(url))+str(height))
                    name = str(xxx+randomNum+".jpg")

                    download_image=os.path.join(save_dir2,name) 
                    #print("Download_images: "+download_image)
                    with open(download_image, 'wb') as out_file:
                        shutil.copyfileobj(response.raw, out_file)
                    del response
                    responseStatus=False
            except Exception as e:
                continue  

    print("End of word: " + word + " for " + str(datasetname))

    
    
#creating muntiple worker for image extraction   
def getWorkers(save_dir,dataset_location,wordList):
    num_workers = 59  #number of processes
    with Pool(processes=num_workers) as pool:
        res= pool.map(partial(downloadImageDataset,save_dir,dataset_location),wordList)
    return 0

 


if __name__ == "__main__":
    dataset_dir= "/home/bhatta/dataset/laion-super-resolution"  #location of the dataset
    save_dir = "/home/bhatta/docs/fotos"  #folder to save the extracted images
    dataset_list = list(os.listdir(dataset_dir))
    #dataset_name = os.path.join(dataset_dir,dataset_list[0])
                 


    wordList=getWords1()
    
    #remove the spacing: child worker --> child-worker
    for word in wordList:
        os.chdir(save_dir)
        subfolder='-'.join((word.split(' ')))
        os.mkdir(subfolder)                 
                            
    #iterate through whole dataset                        
    for dataset_name in dataset_list:
        print("Dataset_name:" + dataset_name)
        dataset_location = os.path.join(dataset_dir,dataset_name)
        res = getWorkers(save_dir,dataset_location,wordList) # starting worker for image extraction
        print("End of dataset: "+ str(dataset_name))
    
        
   
                           
            
   


