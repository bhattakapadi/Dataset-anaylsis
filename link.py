#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from os import listdir
from functools import partial
from multiprocessing import Pool


# In[ ]:


import pandas as pd
import urllib
import requests

import certifi
import ssl
import gc


# In[ ]:


#for the downloading of images from the link
user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,} 
context = ssl._create_unverified_context()


# In[ ]:


dataset_dir = "/home/bhatta/docs/dataset-tmp"
save_dir = "/home/bhatta/docs/fotos"
#tmp_dir = "/home/manish/Documents" # directory to store temporary created image


wordSearch=[" ceo "," CEO ","-ceo-","-CEO-","CHIEF EXECUTIVE OFFICER","chief executive officer"]

wordSearch2=[" police ","-police-","/police-"]

index = 0


dataset_list = list(os.listdir(dataset_dir))


# In[ ]:


def downloadImageDataset(sv_dir,dt_dir,ws,datasetname):
    print("save_dir: "+ sv_dir+"\n")
    print("dataset_dir: "+ dt_dir+"\n")
    print("wordSearch2:" + str(ws)+"\n")
    print("datasetname: "+datasetname+"\n")
    
    dtset = dt_dir + "/"+ datasetname
    data = pd.read_parquet(dtset, engine='pyarrow')
    dfn= data.to_numpy()
    rows = dfn.shape[0]
    cols = dfn.shape[1]
    val = -1 # inital value no word match
    
    for x in range(0, rows-1): #for the end of the dataset 
        for s in ws:
            val=dfn[x][1].lower().find(s)
            if val != -1 :  #val takes -1 if the word is not encountered
                break

        if val != -1:  # val has found the word
            url = dfn[x][0]
            #print("url: "+url)
            #print("+--------------------------------------+")
            text=dfn[x][1]
            try:
                request=urllib.request.Request(url,None,headers) #The assembled request
                #resource = urllib.request.urlopen(request,context=ssl.create_default_context(cafile=certifi.where()))
                resource = urllib.request.urlopen(request,context=context)
                download_image_link= sv_dir + "/"+ str(text[4:15]) +".jpg"
                output = open(download_image_link,"wb")
                output.write(resource.read())
                output.close()
            except:
                continue  
    print("end of one dataset")
    
    


# In[ ]:


wordSearch2='[" police ","-police-","/police-"]'

with Pool(processes=60) as pool:
    print("save_pool: "+ save_dir)
    result=pool.map(partial(downloadImageDataset,save_dir,dataset_dir,wordSearch2),dataset_list)
    #print(result)
    
    

    



# In[9]:


#print("save_dir: "+save_dir)
#for file in os.listdir(save_dir):
#    print(file)
#    os.remove(file)


# In[ ]:





# In[ ]:




