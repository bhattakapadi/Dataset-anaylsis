from deepface import DeepFace
import face_recognition
from PIL import Image
import os
import pickle
from os import listdir
from multiprocessing import Pool 
import torch
import torchvision.transforms as transforms
import numpy as np
from numpy import asarray
import torch.nn.functional as F
from emotion_detection.fer_data_utils import SkResize, HistEq, AddChannel, ToRGB
from vision_utils.custom_architectures import SepConvModel
#extra added
from vision_utils.custom_architectures import SepConvModel, initialize_model, PretrainedMT,SepConvModelMT
from multitask_rag.evaluate import predict_utk as test_image
#for  multi processing analysis of image
import time
import functools
from functools import partial





#resizing the data for democlassi
data_transforms = transforms.Compose([
    transforms.Resize((200,200),Image.Resampling.LANCZOS),
    transforms.ToTensor()
])



#hold image data 
class Stats:
    def __init__(self,framework,gender):
        self.totalImages=0
        self.framework = framework
        self.gender = gender
        self.probability = 0
        self.picAnalysed = 0
        self.age = 0
        self.time=0
    class Race:
        def __init__(self):
            self.white = 0
            self.black = 0
            self.asian = 0
            self.indian = 0
            self.unknown = 0
    class Emotion:
        def __init__(self):
            self.angry = 0
            self.disgust = 0
            self.fear = 0
            self.happy = 0
            self.sad = 0
            self.suprise = 0
            self.neutral = 0



#load the model from the address
def loadModel(PATHSEPCONV,PATHRESNETAGR):
    #emotion
    model = SepConvModel()
    model.load_state_dict(torch.load(PATHSEPCONV, map_location="cpu"))

    #age-race-gender
    resnet_model_agr = PretrainedMT(model_name='resnet')
    resnet_model_agr.load_state_dict(torch.load(PATHRESNETAGR, map_location="cpu"))

    #age-race-gender--Vgg model
    #vgg_model_agr = PretrainedMT(model_name='vgg')
    #vgg_model_agr.load_state_dict(torch.load(PATHRESNETAGR, map_location="cpu")) 
    #return model,resnet_model_agr
    

    #age-race-gender-- Seperable convulation
    #sep_conv_model_agr = SepConvModelMT()
    #sep_conv_model_agr.load_state_dict(torch.load(PATHRESNETAGR, map_location="cpu")) 
    #return model, sep_conv_model_agr

    return model,resnet_model_agr



#democlassi 

def preprocess_fer(image, transf_learn):
    if transf_learn:
        transf = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transf = transforms.Compose([
            HistEq(),
            AddChannel(),
            transforms.ToTensor()
        ])
        print(transf(image))  
    return transf(image).to(torch.float32).unsqueeze_(0)



#output the emotion for democlassi
def predict_fer(image, model, transf_learn=True):

    # process image
    image = preprocess_fer(image, transf_learn)
    
    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    image = image.to(device)

    # predict probabilities
    emotion = F.softmax(model(image), dim=1).detach().to('cpu').numpy()[0]
    target_names = ['Angry', 'Disgusted', 'Afraid', 'Happy', 'Sad', 'Surprised', 'Neutral']
    pred_label = target_names[np.argmax(emotion)]

    emotion_probs = dict(zip(target_names, emotion))
    return emotion_probs, pred_label




#get Evaluation from democlassi
def democlassiEvaluate(model,resnet_model,pil_image):  
    img_tensor= data_transforms(pil_image) 
    imge = pil_image.convert('L') 
    img = imge.resize((200,200),Image.Resampling.LANCZOS)
    numpydata = asarray(img)
    res1=predict_fer(numpydata,model,True) # predcit age,gender race
    res2 = test_image(img_tensor,resnet_model) #predict emotion
    return res1,res2 




# print output form deepface
def deepfaceOutput(obj1):
    print("Deepface comparision")
    print("----Deepface---")
    print("Age:---",{obj1['age']},"---")
    print("Gender:---",{obj1['gender']},"---")
    print("Race:---",{obj1['dominant_race']},"---")
    print("Emotion:---",{obj1['dominant_emotion']},"---")
    print('------------------------------------------- ')



#print output from democlassi
def printDemoClassiOutput(res1,res2):
    print("Democlassi comparision")
    print("---Democlassi---")
    print("Age:---",{round(res2[0], 0)},"---")
    print("Gender:---",{res2[2]},"---")
    print("Race:---",{res2[4]},"---")
    print("Emotion:---",{res1[1]},"---") #res1 for emotion



#update the output for of democlassi
def classiStats(stats,res1,res2,prob):
    stats.age = stats.age + round(res2[0],0)
    stats.probability = stats.probability + prob
    stats.picAnalysed = stats.picAnalysed + 1 
    
    if res2[4].lower() == "white":
        stats.race.white = stats.race.white+1
    elif res2[4].lower() == "black":
        stats.race.black = stats.race.black+1
    elif res2[4].lower() == "asian":
        stats.race.asian = stats.race.asian+1
    elif res2[4].lower() == "indian":
        stats.race.indian = stats.race.indian+1
    else:
        stats.race.unknown = stats.race.unknown+1
        
        
    if res1[1].lower() == "angry":
        stats.emotion.angry= stats.emotion.angry+1
    elif res1[1].lower() == "disgusted":
        stats.emotion.disgust = stats.emotion.disgust+1
    elif res1[1].lower() == "afraid":
        stats.emotion.fear = stats.emotion.fear+1
    elif res1[1].lower() == "happy":
        stats.emotion.happy = stats.emotion.happy+1
    elif res1[1].lower() == "sad":
        stats.emotion.sad = stats.emotion.sad+1
    elif res1[1].lower() == "surprised":
        stats.emotion.suprise = stats.emotion.suprise+1
    else:
        stats.emotion.neutral = stats.emotion.neutral+1


#update the output for of deepface
def deepfaceStats(stats,obj,prob):
    stats.age = stats.age + obj['age']
    stats.probability = stats.probability + prob
    stats.picAnalysed= stats.picAnalysed + 1 
    
    if obj['dominant_race'].lower() == "white":
        stats.race.white = stats.race.white+1
    elif obj['dominant_race'].lower()  == "black":
        stats.race.black = stats.race.black+1
    elif obj['dominant_race'].lower()  == "asian":
        stats.race.asian = stats.race.asian+1
    elif obj['dominant_race'].lower()  == "indian":
        stats.race.indian = stats.race.indian+1
    else:
        stats.race.unknown = stats.race.unknown+1
        
        
    if obj['dominant_emotion'].lower()== "angry":
        stats.emotion.angry= stats.emotion.angry+1
    elif obj ['dominant_emotion'].lower() == "disgust":
        stats.emotion.disgust = stats.emotion.disgust+1
    elif obj['dominant_emotion'].lower()== "fear":
        stats.emotion.fear = stats.emotion.fear+1
    elif obj['dominant_emotion'].lower()== "happy":
        stats.emotion.happy = stats.emotion.happy+1
    elif obj['dominant_emotion'].lower() == "sad":
        stats.emotion.sad = stats.emotion.sad+1
    elif obj['dominant_emotion'].lower() == "surprised":
        stats.emotion.suprise = stats.emotion.suprise+1
    else:
        stats.emotion.neutral = stats.emotion.neutral+1




#after a evaluation of the saved data is done
#save the values to a pickle format file
def save_data_to_file(save_data_loc,stats1,stats2,stats3,stats4,subFolder):
    os.chdir(save_data_loc)  #changing directory to place where the data will be stored
    #print(os.getcwd())
    file = subFolder+".pkl"
    #print("file: "+file)
    with open(file, 'wb+') as f:
        pickle.dump(stats1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(stats2, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(stats3, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(stats4, f, pickle.HIGHEST_PROTOCOL)
    


#evaluation of a folder containing images from democlassi
def evaluateDemoclassi(folder_to_eval,subFolder,sep_model,resnet_model,statsmen,statswomen):
    print("Democlassi evaluate for "+ subFolder)
    adr= folder_to_eval + "/"+subFolder
    statsmen.totalImages=len(os.listdir(adr))
    statswomen.totalImages=len(os.listdir(adr))
    for images in os.listdir(adr):
        if (images.endswith(".png") or images.endswith(".jpg")or images.endswith(".jpeg")):
            new=adr+"/"+images
            try: 
                image = face_recognition.load_image_file(new)
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="resnet")
                if len(face_locations) != 0 :
                    nofpeople=len(face_locations)
                    for face_location in face_locations:
                        top, right, bottom, left = face_location
                        face_image = image[top:bottom,left:right]
                        pil_image = Image.fromarray(face_image)
                        # things to consider if res1 and res2 don' rerurn value
                        res1=0
                        res2=0
                        res1,res2= democlassiEvaluate(sep_model,resnet_model,pil_image)
                        prob = 1/nofpeople  
                        if res2[2].lower() == "man" :
                            classiStats(statsmen,res1,res2,prob)
                        elif res2[2].lower() == "woman":
                            classiStats(statswomen,res1,res2,prob)     
                        #printDemoClassiOutput(res1,res2)
            except Exception as e:
                #print(str(e))
                continue
    



#evaluation of a folder containing images from demodeepface
def evaluateDeepface(folder_to_eval,subFolder,tmp_dir,deepstatsMen,deepstatsWomen):
    print("DeepFace evaluate for "+subFolder)
    adr= folder_to_eval +"/"+subFolder
    deepstatsMen.totalImages=len(os.listdir(adr))
    deepstatsWomen.totalImages=len(os.listdir(adr))
    for images in os.listdir(adr):
        if (images.endswith(".png") or images.endswith(".jpg")or images.endswith(".jpeg")):
            new=adr+"/"+images
            try: 
                image = face_recognition.load_image_file(new)
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="resnet")
                if len(face_locations) != 0 :
                    nofpeople=len(face_locations)
                    for face_location in face_locations:
                        
                        top, right, bottom, left = face_location
                        face_image = image[top:bottom,left:right]
                        pil_img = Image.fromarray(face_image)
                        pil_image = pil_img.resize((400,400),Image.Resampling.LANCZOS)
                        fileLocation=tmp_dir+'/'+'image: '+subFolder+'.jpg' #temporary saves the data
                        pil_image.save(fileLocation,"JPEG")
                        obj=0
                        obj= DeepFace.analyze(img_path=fileLocation, enforce_detection=False)
                        #probability of the person 
                        prob = 1/nofpeople
                        if obj['gender'].lower() == "man" :
                            deepfaceStats(deepstatsMen,obj,prob)
                        elif obj['gender'].lower() == "woman" :
                            deepfaceStats(deepstatsWomen,obj,prob)
                            #deepfaceOutput(obj)
            except Exception as e:
                #print(str(e))
                continue 


def printStats(stats):
    if stats.picAnalysed  != 0 :
        print("Framework:"+stats.framework+"stats for "+stats.gender)
        print("Total number of images: "+str(stats.totalImages))
        print("Number of image analyzed: " + str(stats.picAnalysed) +" in "+ str(stats.time)+" seconds")
        averageAge = stats.age / stats.picAnalysed 
        print("average age:" + str( round(averageAge,2)))
        print("probability:  " + str(round(stats.probability,2)))

        print("-----------------Race------------------")
        print("White percentage : "+ str( round((stats.race.white * 100)/stats.picAnalysed ,2) ))
        print("Black: " + str( round(( (stats.race.black) * 100)/stats.picAnalysed ,2)))
        print("Asian:" +str( round(((stats.race.asian )* 100)/stats.picAnalysed ,2)))
        print("Indian: " + str( round((stats.race.indian * 100)/stats.picAnalysed ,2)))
        print("Unknown: " + str( round((stats.race.unknown * 100)/stats.picAnalysed ,2)))

        print("---------Emotion------------------")
        print("Angry: " + str( round((stats.emotion.angry * 100)/stats.picAnalysed ,2)    ) )
        print("Disgust: " + str(round((stats.emotion.disgust * 100)/stats.picAnalysed ,2)  ) )
        print("fear: " + str(round((stats.emotion.fear * 100)/stats.picAnalysed ,2)    ) )
        print("Hppy: " + str( round((stats.emotion.happy * 100)/stats.picAnalysed ,2)   ) )
        print("sad: " + str(round((stats.emotion.sad * 100)/stats.picAnalysed ,2)   ) )
        print("Suprise:" + str( round((stats.emotion.suprise * 100)/stats.picAnalysed ,2)    ) )
        print("neutral: " + str( round((stats.emotion.neutral * 100)/stats.picAnalysed ,2)   ) )
    else: 
        print("no stats for detected"+stats.framework +" for "+ stats.gender)






def startDatasetAnalysis(folderLocation1,tmp_dir,save_output,subFolder):
    #democlassi stats evalation
    demostatsMen = Stats("democlassi","men")
    demostatsMen.race = Stats.Race()
    demostatsMen.emotion = Stats.Emotion()
    demostatsWomen= Stats("democlassi","women")
    demostatsWomen.race = Stats.Race()
    demostatsWomen.emotion = Stats.Emotion()

    ##deepface statstics evaluation
    deepstatsMen = Stats("deepface","men")
    deepstatsMen.race = Stats.Race()
    deepstatsMen.emotion = Stats.Emotion()
    deepstatsWomen= Stats("deepface","women")
    deepstatsWomen.race = Stats.Race()
    deepstatsWomen.emotion = Stats.Emotion()


    #democlassi path to pretrained models
    PATHSEPCONV='/home/bhatta/github/Image-Analysis/pretrained-models/emotion/sepconv_model_55_val_loss=1.175765.pth'
    PATHRESNETAGR='/home/bhatta/github/Image-Analysis/pretrained-models/age-race-gender/resnet_model_21_val_loss=4.275671.pth'
    PATHVGG='/home/bhatta/github/Image-Analysis/pretrained-models/age-race-gender/vgg_model_21_val_loss=4.139335.pth'
    PATHSEP='/home/bhatta/github/Image-Analysis/pretrained-models/age-race-gender/sep_conv_adam_model_33_val_loss=4.714899.pth'

    #load the pretrainedmodel for democlassi
    sep_model,res_model=loadModel(PATHSEPCONV,PATHRESNETAGR)
    
    st = time.time()
    evaluateDemoclassi(folderLocation1,subFolder,sep_model,res_model,demostatsMen,demostatsWomen)
    et = time.time()

    print("Time taken by democlassi  is : "+ str(round(et-st)) + " seconds"+" for " + subFolder)
    demostatsMen.time=round(et-st)
    demostatsWomen.time=round(et-st)
    
    st = time.time()
    evaluateDeepface(folderLocation1,subFolder,tmp_dir,deepstatsMen,deepstatsWomen)
    et = time.time()

    deepstatsMen.time=round(et-st)
    deepstatsWomen.time=round(et-st)
    print("Time taken by democlassi  is : "+ str(round(et-st)) + " seconds"+" for " + subFolder)
    
    save_data_to_file(save_output,demostatsMen,demostatsWomen,deepstatsMen,deepstatsWomen,subFolder)


if __name__ == "__main__":
    
    folderlocation1='/home/bhatta/docs/fotos' #folder to evalaute
    subFolderList = os.listdir(folderlocation1)
    save_output ='/home/bhatta/github/output' #location to save the evaluated output  
    tmp_dir ='/home/bhatta/tmpFolder' #location to save a temporaray output for deepface

    num_workers=15   #number of workers to start the evaluation
    #for wordStringbreakedList in wsbreakedList:
    with Pool(processes=num_workers) as pool:
        res= pool.map(partial(startDatasetAnalysis,folderlocation1,tmp_dir,save_output),subFolderList)







