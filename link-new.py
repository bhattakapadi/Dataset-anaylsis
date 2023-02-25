#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import urllib
import requests
    
import certifi
import ssl
import gc
import ast
import os
import sys
from os import listdir
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool,Lock
from datetime import datetime


def getWords1():
    return ['kindergarten teacher','dental hygienist','speech-language pathologist','dental assistant','childcare worker',
            'medical records technician','secretary','medical assistant','hairdresser','dietitian','vocational nurse',
            'teacher assistant','paralegal','billing clerk','phlebotomist','receptionist','housekeeper','registered nurse',
            'bookkeeper','health aide','taper','steel worker','mobile equipment mechanic','bus mechanic',
            'service technician','heating mechanic','electrical installer','operating engineer','logging worker',
            'floor installer','roofer','mining machine operator','electrician','repairer',
            'conductor','plumber','carpenter','security system installer','mason','firefighter',
            'salesperson','director of religious activities','crossing guard','photographer','lifeguard',
            'lodging manager','healthcare practitioner','sales agent','mail clerk','electrical assembler',
            'insurance sales agent','insurance underwriter','medical scientist','statistician','training specialist',
            'judge','bartender','dispatcher','order clerk','mail sorter']


def downloadImageDataset(sv_dir,datasetname,word):
    
    #for the downloading of images from the link
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,} 
    context = ssl._create_unverified_context()

    data = pd.read_parquet(datasetname, engine='pyarrow')
    dfn= data.to_numpy()
    rows = dfn.shape[0]
    print("total rows: "+str(rows))
    cols = dfn.shape[1]
    val = -1 # inital value no word match
    
    wordPattern =  [word, '-'.join((word.split(' ')))  ]

    for x in range(0, rows-1): #for the end of the dataset 
        print("current rows: "+str(x))
        for ws in wordPattern:
            val=dfn[x][1].lower().find(ws)
            if val != -1 :  #val takes -1 if the word is not encountered
                break

        if val != -1:  # val has found the word
            url = dfn[x][0]
            text=dfn[x][1]
            #print("+--------------------------------------+")
            sys.stdout.flush()
            try:
                #print("url:"+url)
                #print("Text: "+text)
                request=urllib.request.Request(url,None,headers) #The assembled request

                resource = urllib.request.urlopen(request,context=context)
                download_image_link= sv_dir + "/"+ str(text[4:11]) +".jpg"
                #print(text)
                output = open(download_image_link,"wb+")
                output.write(resource.read())
                output.close()
            except:
                continue  
    

        


if __name__ == "__main__":
    dataset_dir= "/home/ec2-user/dataset"
    save_dir = "/home/ec2-user/fotos"
    dataset_list = list(os.listdir(dataset_dir)) 
    dataset_name = os.path.join(dataset_dir,dataset_list[0])

    print("Dataset_name:" + dataset_name)
               

    wordList=getWords1()
    
    for word in wordList:
        os.chdir(save_dir)
        subfolder='-'.join((word.split(' ')))
        os.mkdir(subfolder)
        wordSavePath=os.path.join(save_dir,subfolder)
        print("wordSavePath: "+wordSavePath)
        downloadImageDataset(wordSavePath,dataset_name,word)
   
                           
            
   


