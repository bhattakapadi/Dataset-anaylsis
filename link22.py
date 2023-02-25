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
            'medical records technician','medical assistant','hairdresser','dietitian','vocational nurse',
            'teacher assistant','paralegal','billing clerk','phlebotomist','receptionist','housekeeper','registered nurse',
            'bookkeeper','health aide','taper','steel worker','mobile equipment mechanic','bus mechanic',
            'service technician','heating mechanic','electrical installer','operating engineer','logging worker',
            'floor installer','roofer','mining machine operator','electrician','repairer',
            'conductor','plumber','carpenter','security system installer','mason','firefighter',
            'salesperson','director of religious activities','crossing guard','photographer','lifeguard',
            'lodging manager','healthcare practitioner','sales agent','mail clerk','electrical assembler',
            'insurance sales agent','insurance underwriter','medical scientist','statistician','training specialist',
            'judge','bartender','dispatcher','order clerk','mail sorter']

def getWords2():
    return ['secretary']


def getString(word):
    y='-'.join((word.split(' ')))
    y= str('-'+y+'-')
    if ' ' in word :
        z=''.join((word.split(' ')))
        return [word,y,z]
    return [word,y]

def downloadImageDataset(sv_dir,dt_dir,ws,datasetname):
    
    #for the downloading of images from the link
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers={'User-Agent':user_agent,} 
    context = ssl._create_unverified_context()

    dtset = dt_dir + "/"+ datasetname
    data = pd.read_parquet(dtset, engine='pyarrow')
    dfn= data.to_numpy()
    rows = dfn.shape[0]
    cols = dfn.shape[1]
    val = -1 # inital value no word match
    
    ws =  [ws, '-'.join((ws.split(' ')))  ]
    #print(type(ws))
    #now = datetime.now()
    #timeNow = now.strftime('%d.%m %H:%M:%S')
    #print(timeNow + " Start of dataset : "+ datasetname + " for " + ws[1])
    for x in range(0, rows-1): #for the end of the dataset 
        for s in ws:
            val=dfn[x][1].lower().find(s)
            if val != -1 :  #val takes -1 if the word is not encountered
                break

        if val != -1:  # val has found the word
            url = dfn[x][0]
            text=dfn[x][1]
            #print("text: "+text)
            #print("+--------------------------------------+")
            sys.stdout.flush()

            try:
                #print("url:"+url)
                #print("Text: "+text)
                request=urllib.request.Request(url,None,headers) #The assembled request
                #resource = urllib.request.urlopen(request,context=ssl.create_default_context(cafile=certifi.where()))
                resource = urllib.request.urlopen(request,context=context)
                download_image_link= sv_dir + "/"+ str(text[4:11]) +".jpg"
                #print(text)
                output = open(download_image_link,"wb+")
                output.write(resource.read())
                output.close()
            except:
                continue  
                
    #now = datetime.now()
    timeNow = now.strftime('%d.%m %H:%M:%S')
    #print("Process Id: " + str(os.getpid()) + " end of dataset : "+ datasetname + " for " + ws[1])
    
def getWorkers(save_dir_job,dataset_dir,wordstring,dst_list):
    num_workers = mp.cpu_count()
    #num_workers=8
    with Pool(processes=num_workers) as pool:
        res= pool.map(partial(downloadImageDataset,save_dir_job,dataset_dir,wordstring),dst_list)
    return 0

def getWorkers2(save_dir_job,dataset_dir,wordstring,dataset_list):
    #num_workers = mp.cpu_count()
    res=[]
    for ds_name in dataset_list:
        res.append(downloadImageDataset(save_dir_job,dataset_dir,wordstring,ds_name))


def divide_chunks(l, n):     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]       
        


if __name__ == "__main__":
    
    #dataset_dir = "/home/bhatta/docs/dataset-tmp"
    dataset_dir= "/home/bhatta/dataset/laion-super-resolution"
    save_dir = "/home/bhatta/docs/fotos"
    #tmp_dir = "/home/manish/Documents" # directory to store temporary created image
    dataset_list = list(os.listdir(dataset_dir)) 
    n=64
    dataset_list2 = list(divide_chunks(dataset_list,n))
    
    #wordslist = [str([x, '-'.join((x.split(' ')))]) for x in getWords4()]
    os.chdir(save_dir)
    now = datetime.now()
    pathname = now.strftime('%d-%m--%H-%M-%S')
    os.mkdir(pathname)
    os.chdir(pathname)
    #num_workers = mp.cpu_count()
    www=getWords1()
    for wordstring in www:
        #paath = ast.literal_eval(wordstring)[1]
        #os.mkdir(paath)
        paath='-'.join((wordstring.split(' ')))
        os.mkdir(paath)
        #print(paath)
        save_dir_job=os.path.join(os.getcwd(),paath)
        index = 1
        #res = getWorkers(save_dir_job,dataset_dir,wordstring,dataset_list)
        for datast_name in dataset_list2:
            print("Dataset_name: " + str(datast_name))
            res = getWorkers(save_dir_job,dataset_dir,wordstring,datast_name)
            print(str(index )+" complete")
            index = index +1
            
   


