# functions
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt

import scipy 

import torch
from torch.autograd import Variable
from torchmetrics.classification import BinaryAccuracy 
from torchmetrics import Accuracy, Specificity, Recall   

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
import joblib

import pandas as pd
import seaborn as sn

from tqdm import tqdm

from natsort import natsorted
import os

from imblearn.over_sampling import SMOTE
from models import *

def str2bool(v):
    """argparse does not handle booleans well, solution found at
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','y','1'):
        return True
    elif v.lower() in ('false','no','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

def load_data(path,to3D=True):
    """Loads data into numpy array
    returns either 2d array pixels by mass bins (sorted by pixelnumber)
    or 3d array with pixels by pixels by mass bins"""
    print("Opening "+path)
    data = np.load(path,allow_pickle=True)[()]

    if to3D:
        data = datato3D(data)
    else:
        data = datato2D(data)
    return data

def datato3D(data):
    """Loads IMS data into a 3D tensor, empty pixels are nan"""
    size = np.append(data['image_shape'][0][0:2],(np.shape(data['intensities'])[1]))
    print("Converting data to image of %d by %d pixels with %d mass bins." % (size[0],size[1],size[2]))
   
    recon = np.zeros(size[:2], dtype=np.float32).reshape(-1)
    recon = np.zeros([np.shape(recon)[0],np.shape(data['intensities'])[1]])

    recon[recon==0] = np.nan
    recon[data['pixel_order']] = data['intensities'][:,:]

    recon = recon.reshape(np.append(data['image_shape'][0][:2],np.shape(recon)[1]))
    return recon

def datato2D(data):
    "Returns data in 2D matrix spectra sorted by pixel number!"
    size = np.append(data['image_shape'][0][0:2],(np.shape(data['intensities'])[1]))
    print("Converting data to array of %d spectra with %d mass bins." % (size[0]*size[1],size[2]))

    recon = np.zeros(size[:2], dtype=np.float32).reshape(-1)
    recon = np.zeros([np.shape(recon)[0],np.shape(data['intensities'])[1]])

    recon[recon==0] = np.nan
    recon[data['pixel_order']] = data['intensities'][:,:]

    # remove nan columns
    recon = recon[~np.isnan(recon).any(axis=1),:]
    recon = recon[~np.isnan(recon).any(axis=1),:]

    # recon = recon[~np.isnan(recon).all(axis=1),:]
    print("After removing empty spectra, %d sectra remain." %np.shape(recon)[0])
    return recon

def plot_slice(data,massbin : list):
    """plot slice of 3d data, sliced at masbin"""
    if isinstance(massbin,int):
        plt.imshow(np.rot90(data[:,:,massbin]),origin='lower')
    else:
        #mutliplot implementation
        pass
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Intensity',rotation=270)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

def plot_spect(data,pixel,show=True,title=True):
    """Plot spectra, takes two arguments
    data: either 2D or 3D array
    pixel, the pixel of the spectra to plot
    
    if 3D array is given, pixel should be [x,y]
    if multiple pixels are given a subplot is created"""
    if isinstance(pixel,int):
        pixel = [pixel]
    if data.ndim<=2: #data is in 2D
        
        plt.subplots(len(pixel),sharex=True)
        plt.subplots_adjust(hspace=0)

        for i in range(len(pixel)):
            plt.subplot(len(pixel),1,i+1)
            if min(data[pixel[i]])<=-.8:
                bottom=-1
            else:
                bottom = 0
            bottom=0 #comment out if if only using other scaling than standard!
            plt.stem(data[pixel[i]],basefmt =' ',markerfmt=' ',bottom=bottom)
            plt.ylabel('Intensity')

            # if opt.pltlog == True:
            #     plt.yscale('log')   
            #plt.title("Pixel nr. %d" %pixel[i])
    else: #data is in 3D, every pixel needs coordinate
        # print(np.size(pixel))
        if np.size(pixel)>3:
            pass #subplots       
        else:
            spec = data[pixel[0],pixel[1],:]
            plt.stem(spec,markerfmt=' ')
            # plt.ylabel('Intensity')
    
    plt.xlabel('Weight index') #index of moleculair weight or nmf weight
    title_text = "Selected spectra:\n"+", ".join([str(i) for i in pixel])
    # print(title)
    if title:
        plt.suptitle(title_text)

    if show:
        plt.show()
    return plt
def get_nmf_labels(path):
    with np.load(path,allow_pickle=True) as nmfdata:
        w = nmfdata['w']
        h = nmfdata['h']
    classes = []
    for i in range(len(w)):
        classes.append(np.argmax(w[i]))
    # sort classes
    classes+=np.ones(np.shape(classes))
    data = np.load("data/2022-12-08-rat_kidney.npy",allow_pickle=True)[()]   
    clusters=np.zeros((np.max(data['pixel_order'])+1,1))
    clusters = np.squeeze(clusters)
    # print(np.shape(labels[:]))
    # print(data['pixel_order'][0])
    # print(clusters[data['pixel_order'][0]][0])
    # print(clusters[0][0])
    clusters[data['pixel_order'][0]]=classes
    clusters = clusters[clusters!=0]
    return clusters    


class dataloader(torch.utils.data.TensorDataset):
    """Custom dataloader to handle batchlearning"""
    def __init__(self,samples,labels,opt):
        self.data=[]
        for i in range(len(labels)):
            self.data.append([samples[:,i],labels[i]])
        self.device=opt.device
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample, classname = self.data[idx]
        if torch.cuda.is_available():
            sample = torch.from_numpy(sample).to(self.device)
        else:
            sample = torch.from_numpy(sample)
        return sample, classname
    
def train_GANs(opt,data,generator,discriminator,optimizers,adverloss,savedir=False,test_data=False):
    """
    This function trains the GANs.
    The generator, discriminator, loss and optimizers have to be initialized beforehand.
    If no savedir is specified, model will not be saved!
    """
    # print(torch.cuda.is_available())
    if opt.device=="cpu":
        Tensor = torch.FloatTensor
    else:
        if torch.cuda.is_available():
            Tensor = torch.cuda.FloatTensor
        else:
            Tensor = torch.FloatTensor
        # print('cuda')
    if savedir:
        folder = savedir
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("new directroy created called: %s" %folder)
    # needed for tracking progress
    traingenloss = []
    traindisloss = []
    trainfid = []
    testgenloss = []
    testdisloss = []
    testfid = []
    dissaccreal = []
    dissaccfake = []
    mean_gen_grad_norm = [] #added 8-12-24
    mean_dis_grad_norm = []

    optim_g = optimizers.optimizer_G
    optim_d = optimizers.optimizer_D
    print("Training ...")

    optim_g.zero_grad()
    z = Variable(torch.FloatTensor(np.random.normal(0,1,(opt.GANsbsize, opt.latent_dim))).to(opt.device))
    y = Variable(torch.randint(0,opt.numclasses,(opt.GANsbsize,))).to(opt.device)
    gen_samples = generator.forward(z,y) #added .forward during noGEN testing #static gen_samples

    if opt.WGAN_gp:
        pass
    else:
        print("Using original loss")
        adverloss_g=adverloss
        adverloss_d=adverloss
    stop = False
    for epoch in range(opt.GANs_n_epochs):
        if stop==True:
            break
        genlosslist = []
        dislosslist = []
        dissaccreallist = []
        dissaccfakelist = []
        fidlist = []
        with tqdm(data,unit="batch") as tepoch:
            tepoch.set_description("Epoch %d / %d" %(epoch+1,opt.GANs_n_epochs))
            for i, (sample, target) in enumerate(tepoch):
                # create real/fake labels
                if opt.WGAN_gp:
                    fake =  Variable(Tensor(sample.shape[0],1).fill_(-1.0),requires_grad=False)
                else:
                    fake =  Variable(Tensor(sample.shape[0],1).fill_(0.0),requires_grad=False)
                valid = Variable(Tensor(sample.shape[0],1).fill_(1.0),requires_grad=False)

                real_samples = Variable(sample.type(Tensor))
                real_targets = target.to(torch.int64).to(opt.device)
                # generator step
                update_gen=False
                if epoch>=0: #1 epoch ahead start if set to 1, #changed from > to >= on 19-03-24
                    if opt.dynamicdiscstep == True:
                        if len(dissaccfake)>0:
                            if dissaccfake[-1]>0.8 and dissaccreal[-1]>0.8:
                                update_gen=True
                                print("gen update")
                            else:
                                update_gen=False
                        else:
                            update_gen=False
                    else: 
                        if i % opt.discsteps == 0:
                            update_gen=True
                        else:
                            update_gen=False
                    if update_gen:
                        optim_g.zero_grad()
                        z = Variable(torch.FloatTensor(np.random.normal(0,1,(sample.shape[0], opt.latent_dim))).to(opt.device))
                        y = Variable(torch.randint(0,opt.numclasses,(sample.shape[0],))).to(opt.device)
                        gen_samples = generator.forward(z,y) #added .forward during noGEN testing
                        predic_fake = discriminator.forward(gen_samples,y)
                        if opt.WGAN_gp:
                            g_loss = -torch.mean(predic_fake)
                        else:
                            g_loss = adverloss_g(predic_fake,valid) #maximize wrong predictions on generated
                        if abs(g_loss.item())>=50 and epoch>=2:
                            count+=1
                            if count>=5:
                                print("_Generator loss equal to %d,too large, stopping training...",d_loss.item())
                                stop = True #break training loop but save finalgan
                        else:
                            count = 0
                        
                        g_loss.backward()
                        optim_g.step()
                if update_gen == False:
                    g_loss = torch.mean(torch.randn(1,1))
                #Discriminator step
                optim_d.zero_grad()
                predic_real = discriminator.forward(real_samples,real_targets) #.forward added during nogen testing
                # resample noise
                z = Variable(torch.FloatTensor(np.random.normal(0,1,(sample.shape[0], opt.latent_dim))).to(opt.device))
                # y = Variable(torch.randint(0,opt.numclasses,(sample.shape[0],))).to(opt.device) WGAN gp needs same targets!
                y = real_targets
                gen_samples = generator.forward(z,y) #added .forward during noGEN testing
                predic_fake = discriminator.forward(gen_samples,y)    

                if opt.WGAN_gp:
                    d_loss = -torch.mean(predic_real) + torch.mean(predic_fake) + opt.lambda_gp * WGAN_gradient_penalty(opt,discriminator,real_samples,gen_samples,real_targets,y)
                else:
                    real_loss = adverloss_d(predic_real,valid)
                    fake_loss = adverloss_d(predic_fake,fake)
                    d_loss = (real_loss + fake_loss)/2
                if abs(d_loss.item())>=50 and epoch>=2: #if loss explodes, stop the training 
                    count+=1
                    if count>=5:
                        print("_Discriminator loss equal to %d,too large, stopping training...",d_loss.item())
                        stop = True #break training loop but save finalgan
                else:
                    count = 0
                d_loss.backward()
                optim_d.step()

                # claculating norms in this way does not work for all models, so is not used by default
                # gen_grad_param_norms=[] #doesnt work for dcgan?
                # for param in generator.parameters():
                #     gen_grad_param_norms.append(param.grad.norm())
                # mean_gen_grad_norm.append(torch.tensor(gen_grad_param_norms).mean().item())
                # dis_grad_param_norms=[]
                # for param in discriminator.parameters():
                #     dis_grad_param_norms.append(param.grad.norm())
                # mean_dis_grad_norm.append(torch.tensor(dis_grad_param_norms).mean().item())

                tepoch.set_postfix(Gloss=g_loss.item(),Dloss=d_loss.item())
                genlosslist.append(g_loss.item())
                dislosslist.append(d_loss.item())
                if opt.WGAN_gp:
                    # dissaccreallist.append((predic_real.squeeze()>0).float().mean().cpu())
                    # dissaccfakelist.append((predic_fake.squeeze()<0).float().mean().cpu())
                    dissaccreallist=(predic_real.squeeze()>0).float().mean().cpu().item()
                    dissaccfakelist=(predic_fake.squeeze()<0).float().mean().cpu().item()
                else:
                    dissaccreallist=(predic_real.squeeze().round()==valid).float().mean().cpu().item()
                    dissaccfakelist=(predic_fake.squeeze().round()==fake).float().mean().cpu().item()

                # print(np.mean(real_samples.cpu().numpy(),axis=0))
                # print(len(genlosslist))
                # errormeas = error_val(real_samples.cpu().numpy(),gen_samples.detach().cpu().numpy())
                # fidlist.append(errormeas.FID())
            # add mean of epoch to list 

            # dissaccreal.append(np.mean(dissaccreallist))
            # dissaccfake.append(np.mean(dissaccfakelist))                    
            # just add the entire list.... (We are also intrested in info within a batch)

                # trainfid = trainfid+np.mean(fidlist)
                dissaccreal.append(dissaccreallist)
                dissaccfake.append(dissaccfakelist) 
                # traingenloss.append(genlosslist)
                # traindisloss.append(dislosslist)
            traingenloss.append(np.mean(genlosslist))
            traindisloss.append(np.mean(dislosslist))
            trainfid.append(np.mean(fidlist))
            if savedir:
                torch.save(generator,str(folder+"generator"+str(epoch)))
                torch.save(discriminator,str(folder+"discriminator"+str(epoch)))


            # tepoch.set_postfix(Gloss=traingenloss[-1],Dloss=traindisloss[-1])

            # run on testset
            if test_data: #run a test at every epoch, not implemented for WGAN
                if opt.WGAN_gp:
                    print("No online testing implemented for wgangp")
                else:
                    genloss, disloss, fid = test_GANs(opt,test_data,generator,discriminator,adverloss)
                    print("Testset generator loss: %.2f, discriminator loss: %.2f, FID: %.2f" %(genloss, disloss, fid))
                    testgenloss.append(genloss)
                    testdisloss.append(disloss)
                    testfid.append(fid)

    if savedir: #save all kinds of stuff about the training in the specified folder
        plot_progress(opt,traingenloss,traindisloss,trainfid,testgenloss,testdisloss,testfid,mean_gen_grad_norm,mean_dis_grad_norm,dissaccreal,dissaccfake)
        #save latest
        torch.save(generator,str(folder+"finalgen"))
        torch.save(discriminator,(str(folder+"finaldis")))
        #create gan summary
        with open(opt.savegans+'gan_summary.txt','w+') as f:
            f.write("Number of training epochs:%d \n" %opt.GANs_n_epochs)
            f.write("Learning rate: %f \n" %opt.GANslr)
            f.write("Discriminator steps/ generator step: %d \n" %opt.discsteps)
            f.write("Gradiant penalty: %f \n" %opt.lambda_gp)
            f.write("Momentum 1: %f \n" %opt.b1)
            f.write("Momentum 2: %f \n" %opt.b2)
            f.write("Batch size: %d \n" %opt.GANsbsize)
            f.write("Latent dimension %d\n" %opt.latent_dim)
            f.write(str("Optimizer: "+opt.optimizer+"\n"))
            f.write(str("Generator architecture: \n"+str(generator)+"\n"))
            f.write(str("Discriminator architecture: \n"+str(discriminator)+"\n"))
            f.write("Classes used for training:\n")
            labels=[]
            for img, lab in data:
                labels.extend(lab.data.cpu().numpy())
            for i in set(labels):
                f.write("Class %d: %d Samples\n" %(i,labels.count(i)))
        # plot progress and save plots
def plot_progress(opt,traingenloss=[],traindisloss=[],trainfid=[],testgenloss=[],testdisloss=[],testfid=[],gengradnorm=[],disgradnorm=[],dissaccfake=[],dissaccreal=[]):
    """This function plots the different arrays saved during training
    All figures are saved in the same folder as the GANs"""
    if len(dissaccfake)>1 and len(dissaccreal)>1:
        plt.figure()
        plt.plot(dissaccfake)
        plt.plot(dissaccreal)
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Discriminator accuracy')
        plt.legend(["Real accurayc","Fake accuracy"])
        plt.savefig(str(opt.savegans+'discriminator_accuracy.png'))
    
    if len(gengradnorm)>1:
        plt.figure()
        plt.plot(gengradnorm)
        plt.xlabel('Batch')
        plt.title('Generator mean gradient norm')
        plt.savefig(str(opt.savegans+'genparamgrad.png'))
    if len(disgradnorm)>1:
        plt.figure()
        plt.plot(disgradnorm)
        plt.xlabel('Batch')
        plt.title('Discriminator mean gradient norm')
        plt.savefig(str(opt.savegans+'disparamgrad.png'))

    plt.figure()
    plt.plot(traingenloss)
    plt.title("Mean Generator loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(str(opt.savegans+'loss_trainmeangen.png'))

    plt.figure()
    plt.plot(traindisloss)
    plt.title("Mean Discriminator loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(str(opt.savegans+'loss_trainmeandis.png'))

    plt.figure()
    plt.plot(trainfid)
    plt.title("Mean FD between generated and traing data")
    plt.xlabel("Epoch")
    plt.ylabel("FD")
    plt.savefig(str(opt.savegans+'trainmeanfd.png'))

    plt.figure()
    plt.plot(testgenloss)
    plt.title("Mean Generator loss during testing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(str(opt.savegans+'testmeangen.png'))

    plt.figure()
    plt.plot(testdisloss)
    plt.title("Mean Discriminator loss during testing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(str(opt.savegans+'testmeandis.png'))

    plt.figure()
    plt.plot(testfid)
    plt.title("Mean FD between generated and test data")
    plt.xlabel("Epoch")
    plt.ylabel("FD")
    plt.savefig(str(opt.savegans+'testmeanfd.png'))
def test_GANs(opt,test_data,generator,discriminator,adverloss):
    """This function is used to test the gans at a specific step
    Using a test set, This function is not used for wasserstein gans"""
    generator.eval()
    discriminator.eval()
    if opt.device=="cpu":
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor

    gen_loss_list = []
    dis_loss_list = []
    FID_list = []

    for sample,target in tqdm(test_data):
        valid = Variable(Tensor(sample.shape[0],1).fill_(1.0),requires_grad=False)
        fake =  Variable(Tensor(sample.shape[0],1).fill_(0.0),requires_grad=False)

        real_samples = Variable(sample.type(Tensor))
        real_targets = target.to(torch.int64).to(opt.device)

        z = Variable(torch.FloatTensor(np.random.normal(0,1,(sample.shape[0], opt.latent_dim))).to(opt.device))
        y = Variable(torch.randint(0,opt.numclasses,(sample.shape[0],))).to(opt.device)

        gen_samples = generator.forward(z,y)
        gen_loss_list.append(adverloss(discriminator.forward(gen_samples,y),valid).item())

        real_loss = adverloss(discriminator.forward(real_samples,real_targets),valid)
        fake_loss = adverloss(discriminator.forward(gen_samples.detach(),y), fake) 
        dis_loss_list.append(((real_loss + fake_loss)/2).item())

        y = Variable(torch.FloatTensor(target)).to(opt.device).long()        
        gen_samples = generator.forward(z,y)
        # errormeas = error_val(sample.detach().cpu().numpy(),gen_samples.detach().cpu().numpy())
        # FID_list.append(errormeas.FID())
    genloss = np.mean(gen_loss_list)
    disloss = np.mean(dis_loss_list)
    FID = np.mean(FID_list)
    generator.train()
    discriminator.train()
    return genloss, disloss, FID 

class evaluation_metrics():
    """This class defines some evaluation metrics used"""
    def __init__(self) -> None:
        pass

    def l1norm(fake,real):
        """Calculates avarage l1 norm"""
        # e = real[:len(fake)-1,:,:]-fake
        e = real-fake

        l1 = np.linalg.norm(e,ord=1,axis=0)
        av_l1=np.mean(l1)

        #eb = real[len(fake):2*len(fake),:,:]-fake
        #l1_base = np.linalg.norm(eb,ord=1,axis=0)
        av_l1_base = np.mean(l1)
        return av_l1 ,av_l1_base
    
    def l2norm(real,fake):
        """Calculates average l2 norm"""
        e = real-fake
        l1 = np.linalg.norm(e,ord=2,axis=0)
        av_l1=np.mean(l1)
        return av_l1

    def discriminator_accuracy(opt,real,fake,dis_model):
        # if torch.cuda.is_available():
        #     Tensor = torch.cuda.FloatTensor
        # else:
        Tensor = torch.FloatTensor

        target_real = torch.ones(len(real),1,1)
        target_fake = torch.zeros(len(fake),1,1)
        metric = BinaryAccuracy()
        if torch.cuda.is_available():
            real_pred = dis_model(real.to(opt.device))
            fake_pred = dis_model(fake.to(opt.device))
            metric = metric.to((opt.device))
            target_real = torch.ones(len(real),1,1).to(opt.device)
            target_fake = torch.zeros(len(fake),1,1).to(opt.device)                                                     
        else:
            real_pred = dis_model(real)
            fake_pred = dis_model(fake)
        acc_real = metric(real_pred,target_real)
        acc_fake = metric(fake_pred,target_fake)
        return acc_real, acc_fake

def get_full_ratkidney(opt,limitmajor=0,oversampler="No"):
    """change to only handle data augmentation"""
    data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)
    # data2D = np.expand_dims(data2D,axis=2)
    labels = get_nmf_labels('data/orthogonal_nmf.npz')
    labels -= np.ones(np.shape(labels))
    labels = list(labels)
    classes =  set(labels)

    X = data2D[[item in classes for item in labels]]
    y = np.array([item for item in labels if item in classes])

    if limitmajor==0: #limit the number of samples in the training set for all majority classes
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=opt.testsplit,stratify=y)  
    else:
        X_train, X_test, y_train, y_test=limit_samples(data2D,labels,trainsize=limitmajor,testsplit=opt.testsplit)

    if oversampler=="No":
        print("No oversampling used")
    elif oversampler=="SMOTE":
        print("Using smote to solve imbalance...")
        smote = SMOTE(random_state=42)
        X_train,y_train=smote.fit_resample(X_train,y_train)
    else:
        print("Technique for oversampling not implemented")

    X_train=np.transpose(np.expand_dims(X_train,axis=2))
    y_train=torch.Tensor(y_train)
    X_test=np.transpose(np.expand_dims(X_test,axis=2))
    y_test=torch.Tensor(y_test)
    data_train=dataloader(X_train,y_train)
    data_train=torch.utils.data.DataLoader(data_train,batch_size=opt.bsize,num_workers=0,shuffle=True,drop_last=True)
    data_test=dataloader(X_test,y_test)
    data_test=torch.utils.data.DataLoader(data_test,batch_size=opt.bsize,num_workers=0,shuffle=True,drop_last=True)
    opt.numclasses=int(max(classes)+1)
    return opt, data_test, data_train, set(classes)

def limit_samples(data,classes,trainsize=5000,testsplit=0.2):
    """Splits dataset in train- and testset
    with every class at most "trainsize" times represented in the set for training
    If more than trainsize+trainsize*testsplit samples are available of the same class, that class is split with trainsize samples for training
    and all leftover samples for testing
    If less than trainsize+trainsize*testplist samples are available of the same class, that class sis split according to the testsplit resulting
    in numver of trainingsamples 1-testplist*100% of samples and number of testsamples testsplit*100% of samples"""
    for i in set(classes):
        if classes.count(i)>=trainsize+trainsize*testsplit:
            x_tr, x_te, y_tr, y_te = train_test_split(data[[item in [i] for item in classes]],np.array([item for item in classes if item in [i]]),train_size=trainsize)
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(data[[item in [i] for item in classes]],np.array([item for item in classes if item in [i]]),test_size=testsplit)
        if i==list(set(classes))[0]:
            trainset = np.array(x_tr)
            trainlabel = np.array(y_tr)
            testset = np.array(x_te)
            testlabel = np.array(y_te)
        else:
            trainset = np.vstack((trainset,x_tr))
            trainlabel = np.hstack((trainlabel,y_tr))
            testset = np.vstack((testset,x_te))
            testlabel = np.hstack((testlabel,y_te))
    return trainset, testset, trainlabel, testlabel

# def train_class(opt,lossfun,optimizer,classifier,data_train,data_test):
        # """Alternative classifier training, not used in any of the experiments"""
#     # numclasses=len(set(data_test.))
#     accuracy=Accuracy(task='multiclass',num_classes=int(opt.numclasses)).to(opt.device)
#     specificity=Specificity(task='multiclass',num_classes=int(opt.numclasses)).to(opt.device)
#     loss_list=[]
#     acc_train=[]
#     acc_test=[]
#     spec_test=[]
#     testacc=0
#     testspec=0
#     for epoch in range(opt.CLASSn_epochs):
#         with tqdm(data_train,unit="batch") as tepoch:
#             tepoch.set_description("Epoch %d / %d"%(epoch+1,opt.CLASSn_epochs))
#             for sample, lab in tepoch:
#                 inputs, labels = sample.to(opt.device, dtype=torch.float), lab.type(torch.LongTensor).to(opt.device)
                
#                 optimizer.zero_grad()
#                 outputs=classifier(inputs).type(torch.FloatTensor).to(opt.device)
#                 loss = lossfun(outputs,labels.detach())
#                 loss.backward()
#                 optimizer.step()

#                 acc = accuracy(outputs,labels)
#                 tepoch.set_postfix(Loss=loss.item(),Train_Acc=acc.item(),Test_Acc=testacc,Test_spec=testspec)
#                 loss_list.append(loss.item())
#                 acc_train.append(acc.item())
#             runacc=[]
#             runspec=[]
#             for testinp, testlab in data_test:
#                 testinp = testinp.to(opt.device, dtype=torch.float)
#                 testlab = testlab.to(opt.device)#.unsqueeze(1)

#                 testout = torch.argmax(classifier(testinp),dim=1)
#                 runacc.append(accuracy(testout,testlab).item())
#                 runspec.append(specificity(testout,testlab).item())
                
#             testacc = np.mean(runacc)
#             testspec = np.mean(runspec)
#             acc_test.append(testacc)
#             spec_test.append(testspec)
#     return classifier, acc_test,loss_list,acc_train,spec_test

# class datahandling():
#     def __init__(self) -> None:
#         pass
def reducespec(opt,X):
    """Using NMF to reduce the spectral dimension, the feature matrix is saved as joblib file"""
    print("reducing spectum size with NMF to %d massbins" %opt.specsize)
    nmf = NMF(opt.specsize,init="nndsvd",max_iter=500,verbose=0) #note initialization method!
    X = nmf.fit_transform(X)
    joblib.dump(nmf,str(opt.datafile+'nmf.bin'),compress=True)
    return X

def create_labeled_dataf(opt):
    """Creates datafile with sperate training and testing data
    This function will try to load 'data/2022-12-08-rat_kidney.npy' and 'data/orthogonal_nmf.npz'
    make sure these files exist"""
    # load data
    data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)
    
    # load labels
    labels = get_nmf_labels('data/orthogonal_nmf.npz')
    labels -= np.ones(np.shape(labels))
    labels = list(labels)
    # combine
    if len(opt.limitclasses)>=1:
        classes = set(opt.limitclasses)
    else:
        classes =  set(labels)
    X = data2D[[item in classes for item in labels]]
    y = np.array([item for item in labels if item in classes])
    # add nmf to reduce spetra length
    if opt.specsize<np.shape(X)[1]:
        X = reducespec(opt,X)
    #
    if opt.scalemethod == "minmaxf":
        print("Using featurewise minmax scaling")
        minmaxed = 2*((X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)))-1
        X=minmaxed
    elif opt.scalemethod == "minmax":
        print("Using minmax scaling")
        minmaxed = 2*((X-np.min(X))/(np.max(X)-np.min(X)))-1
        X=minmaxed
    elif opt.scalemethod == "standard":
        print("Using standard scaling")
        scaler = StandardScaler()
        X=scaler.fit_transform(X)
        joblib.dump(scaler,str(opt.datafile+'scaler.bin'),compress=True)
    elif opt.scalemethod == "1/3standard":
        print("Using standard scaling * 1/3")
        scaler = StandardScaler()
        X=scaler.fit_transform(X)/3
    else:
        print("No scaling of the data!")
    for i in set(classes):
        if labels.count(i)>=opt.limitmajor*(1+opt.testsplit):
            x_tr, x_te, y_tr, y_te = train_test_split(X[[item in [i] for item in y]],np.array([item for item in labels if item in [i]]),train_size=opt.limitmajor)
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(X[[item in [i] for item in y]],np.array([item for item in labels if item in [i]]),test_size=opt.testsplit)
            # reassign!
        if i==list(set(classes))[0]:
            X_train = np.array(x_tr)
            y_train = np.array(y_tr)
            X_test = np.array(x_te)
            y_test = np.array(y_te)
        else:
            X_train = np.vstack((X_train,x_tr))
            y_train = np.hstack((y_train,y_tr))
            X_test = np.vstack((X_test,x_te))
            y_test = np.hstack((y_test,y_te))
    # save
    mapping ={}
    current_value=0
    for item in y_train:
        if item not in mapping:
            mapping[item]=current_value
            current_value+=1
    y_train = [mapping[item] for item in y_train]
    mapping ={}
    current_value=0
    for item in y_test:
        if item not in mapping:
            mapping[item]=current_value
            current_value+=1
    y_test = [mapping[item] for item in y_test]
    #save
    np.savez_compressed(opt.datafile,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
    with open(opt.datafile+'summary.txt','w+') as f:
        f.write("Number of training samples:%d \n" %len(X_train))
        f.write("Number of test samples: %d\n" %len(X_test))
        f.write("Train test split %f \n" %opt.testsplit)
        f.write("Classes included: %s\n"%set(classes))

def scaledback_comparing(opt):
    """This function generates data using the final generator and compares generated data to original data
    after scaling all data to its original scale"""
    # load gan
    genmodel = torch.load(str(opt.savegans+'finalgen'),map_location=opt.device)
    # load used data
    X_train,y_train,X_test,y_test = load_labeled_data(opt.datafile)
    # load original data
    data2D = load_data("data/2022-12-08-rat_kidney.npy",to3D=False)
    
    # load labels
    labels = get_nmf_labels('data/orthogonal_nmf.npz')
    labels -= np.ones(np.shape(labels))
    labels = list(labels)
    # combine
    if len(opt.limitclasses)>=1:
        classes = set(opt.limitclasses)
    else:
        classes =  set(labels)
    X_orig = data2D[[item in classes for item in labels]]
    y_orig = np.array([item for item in labels if item in classes])
    mapping ={}
    current_value=0
    for item in y_orig:
        if item not in mapping:
            mapping[item]=current_value
            current_value+=1
    y_orig = [mapping[item] for item in y_orig]
    # generate data
    z = torch.rand(len(y_test),opt.latent_dim).to(opt.device)
    yhat = torch.tensor(y_test).to(opt.device)
    xhat = genmodel(z,yhat).cpu().detach().numpy()
    
    # smote generated data
    _,counts = np.unique(y_train,return_counts=True)
    samplelength = max(counts)*2
    X_train_ex = np.vstack((X_train,np.zeros((samplelength,np.shape(X_train)[1]))))
    y_train_ex = np.hstack((y_train,np.ones(samplelength)*(max(y_train)+1)))
    sm = SMOTE()
    xsmote,ysmote = sm.fit_resample(X_train_ex,y_train_ex)

    X_train_smote = xsmote[np.shape(X_train_ex)[0]:,:]
    y_train_smote = ysmote[np.shape(y_train_ex)[0]:]

    _,counts = np.unique(y_test,return_counts=True)
    samplelength = max(counts)*2
    X_test_ex = np.vstack((X_test,np.zeros((samplelength,np.shape(X_test)[1]))))
    y_test_ex = np.hstack((y_test,np.ones(samplelength)*(max(y_test)+1)))
    sm = SMOTE()
    xsmote,ysmote = sm.fit_resample(X_test_ex,y_test_ex)

    X_test_smote = xsmote[np.shape(X_test_ex)[0]:,:]
    y_test_smote = ysmote[np.shape(y_test_ex)[0]:]
    # print(np.shape(xhat))
    # scale back all data
    if opt.scalemethod == "standard":
        try:
            scaler = joblib.load(str(opt.datafile+"scaler.bin"))
        except:
            print("scaler not found")
            return
        if not os.path.exists(str(opt.savegans+'/exampledata/')):
            os.makedirs(str(opt.savegans+'/exampledata/'))
        try:
            xtest_scaledback = scaler.inverse_transform(X_test)
            xtrain_scaledback = scaler.inverse_transform(X_train)
            xhat_scaledback = scaler.inverse_transform(xhat)
            xtest_smote_scaledback = scaler.inverse_transform(X_test_smote)
            xtrain_smote_scaledback = scaler.inverse_transform(X_train_smote)
            try:
                nmf = joblib.load(str(opt.datafile+"nmf.bin"))
                xtest_scaledback = nmf.inverse_transform(xtest_scaledback) 
                xtrain_scaledback = nmf.inverse_transform(xtrain_scaledback)
                xhat_scaledback = nmf.inverse_transform(xhat_scaledback)
                xtest_smote_scaledback = nmf.inverse_transform(xtest_smote_scaledback)
                xtrain_smote_scaledback = nmf.inverse_transform(xtrain_smote_scaledback)
            except:
                print("No reduction matrix saved")
        except ValueError:
            print("Can't scale data back, spectra size incompatalbe with selected scaler!")

    # compare testset to original
    if 'xtest_scaledback' in locals():
        with open(opt.savegans+'scorecomparison.txt','w+') as f:
            # Compare train and test in original
            err = error_val(xtest_scaledback,X_orig)
            f.write('\n')
            f.write('Error between test and original (total): %2f\n' %err.FID())
            for i in set(y_test):
                err = error_val(xtest_scaledback[y_test==i],X_orig[y_orig==i])
                f.write('Error between test and original (class %d): %.2f\n' %(i,err.FID()))
            # Compare scaled back to original
            err = error_val(xtest_scaledback,X_orig)
            f.write('\n')
            f.write('Error between test and original (total): %2f\n' %err.FID())
            for i in set(y_test):
                err = error_val(xtest_scaledback[y_test==i],X_orig[y_orig==i])
                f.write('Error between test and original (class %d): %.2f\n' %(i,err.FID()))
            # Compare train to test
            err = error_val(xtrain_scaledback,xtest_scaledback)
            f.write('\n')
            f.write('Error between train and test (total): %2f\n' %err.FID())
            for i in set(y_test):
                err = error_val(xtrain_scaledback[y_train==i],xtest_scaledback[y_test==i])
                f.write('Error between train and test (class %d): %.2f\n' %(i,err.FID()))
            # Compare generated to test
            err = error_val(xhat_scaledback,xtest_scaledback)
            f.write('\n')
            f.write('Error between GANs-generated and test (total): %2f\n' %err.FID())
            for i in set(y_test):
                err = error_val(xhat_scaledback[y_test==i],xtest_scaledback[y_test==i])
                f.write('Error between generated and test (class %d): %.2f\n' %(i,err.FID()))
            # compare generated to original
            err = error_val(xhat_scaledback,X_orig)
            f.write('\n')
            f.write('Error between GANs-generated and original (total): %2f\n' %err.FID())
            for i in set(y_test):
                err = error_val(xhat_scaledback[y_test==i],X_orig[y_orig==i])
                f.write('Error between GANs-generated and original (class %d): %.2f\n' %(i,err.FID()))
            # compare generated to original
            err = error_val(xtest_smote_scaledback,X_orig)
            f.write('\n')
            f.write('Error between SMOTE-generated and original (total): %2f\n' %err.FID())
            for i in set(y_test_smote):
                err = error_val(xtest_smote_scaledback[y_test_smote==i],X_orig[y_orig==i])
                f.write('Error between SMOTE-generated and original (class %d): %.2f\n' %(i,err.FID()))
            
def load_labeled_data(datafile):
    """Load datafile containing 
    training data:      X_train
    training labels:    y_train
    test data:          X_test
    test labels:        y_test"""
    try:
        data = np.load(datafile)
    except (FileNotFoundError):
        print("data split file not found, maybe set create_split to True!")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train,y_train,X_test,y_test

def confusionmatrix(path:str,y_pred,y_true,name=""):

    cf_matrix = confusion_matrix(y_true,y_pred)
    classes = set(y_true+1)
    avrecall = np.round(np.mean(np.diag(cf_matrix/np.sum(cf_matrix, axis=1)[:,None])),2)

    # classes = [chr(ord('@')+int(list(classes)[i]+1)) for i in range(len(classes))]
    support=np.sum(cf_matrix,axis=1)
    # print(support)
    df_cm = pd.DataFrame(np.round(cf_matrix/np.sum(cf_matrix, axis=1)[:,None],decimals=2),index=[i for i in classes],columns = [i for i in classes])

    # df_cm['Support'] = support
    # print(df_cm)

    plt.figure(figsize = (4,4))
    # sn.heatmap(df_cm,annot=True,vmin=0,vmax=1) #add support as col?
    sn.heatmap(df_cm,vmin=0,vmax=1,cmap='Blues',linewidths=0,linecolor='black')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(str("Recall matrix, average recall "+str(avrecall)))
    plt.savefig(path+'/cfm_output'+name+'.png')

def test_confusionmatrix():
    print('ff')
    path = 'test'
    y_pred = [0,1,4,5,3,4,4,1,1,1,2,2,2,2,4,3,4,3,4]
    y_true = [0,1,4,3,4,2,1,3,1,4,2,4,4,5,2,2,2,1,1]
    confusionmatrix(path,y_pred,y_true)

def plottraining(acc_test,loss_list,acc_train,test_spec,path):

    plt.figure()
    plt.plot(acc_test)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.title("Accuracy on test data")
    plt.savefig(path+"/testacc")

    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss on Training data")
    plt.savefig(path+"/loss")

    plt.figure()
    plt.plot(acc_train)
    plt.xlabel("Step")
    plt.ylabel("Accuracy [%]")
    plt.title("Accuracy on training data")
    plt.savefig(path+"/trainacc")

    plt.figure()
    plt.plot(test_spec)
    plt.xlabel("Epoch")
    plt.ylabel("Specificity [%]")
    plt.title("Specificity on test data")
    plt.savefig(path+"/trainspec")   

def create_summary(acc_test,opt,elapsed_time,path,y_pred,y_true,architecture=None,mode='classifier'):
    if mode == 'classifier':
        with open(path+'/summary.txt','w+') as f:
            f.write("Accuracy on test set: %.2f \n" %acc_test[-1])
            f.write("Accuracy on training set: Not included\n")
            f.write("Training epochs %d\n" %opt.CLASSn_epochs)
            f.write("Learning rate %f\n" %opt.CLASSlr)
            f.write("Training took %f minutes\n" %elapsed_time)
            f.write("Batch size %d\n" %opt.CLASSbsize)
            f.write(classification_report(y_true,y_pred))
            if architecture:
                f.write('\n Model architecture: \n')
                f.write(architecture)

def plot_spec_stats(testdata,synthetic_data=[],range=[0,100],compare_var=False):
    """plots mean and +- 3 standard divation range of 'testdata'
    also plots mean of given syntethic data (if given)
    range can be selected, standard range from 0 to 100"""
    testdata=testdata[:,np.min(range):np.max(range)]
    testmean = np.mean(testdata,axis=0)
    teststd = np.std(testdata,axis=0)
    plt.figure(figsize=(20,5))
    # plt.ylim([-1,np.min([1,np.max([testmean+teststd*3])])])
    # plt.ylim([np.min([-3,np.min([testmean-teststd*3])]),np.max([3,np.max([testmean+teststd*3])])])
    if len(synthetic_data)>=1:
        synmean=np.mean(synthetic_data[:,np.min(range):np.max(range)],axis=0)
        synstd=np.std(synthetic_data[:,np.min(range):np.max(range)],axis=0)
        # plt.ylim([-1,np.min([1,np.max([synstd*3+synmean,teststd*3+testmean])])])
        plt.ylim([np.min([testmean-teststd*3,synmean-synstd*3]),np.max([testmean+teststd*3,synmean+synstd*3])])
    # print(np.max(synstd+synmean))
    # print(np.min([1,np.max([np.max(synstd+synmean),np.max(teststd+testmean)])]))
    plt.fill_between(np.arange(np.min(range),np.max(range)),(testmean+3*teststd),(testmean-3*teststd),color='lightblue',alpha=0.75)
    # plt.plot(np.arange(0,np.shape(testdata)[1]),testdata[1])
    plt.plot(np.arange(np.min(range),np.max(range)),testmean,'b',alpha=0.75)
    plt.legend(('$\pm3 \sigma$ range','real mean'))
    # plt.ylim([-1,1])
    if compare_var:
        if len(synthetic_data)>=1:
            plt.plot(np.arange(np.min(range),np.max(range)),np.mean(synthetic_data[:,np.min(range):np.max(range)],axis=0),color='red',alpha=0.9)
            plt.fill_between(np.arange(np.min(range),np.max(range)),(synmean+3*synstd),(synmean-3*synstd),color='red',alpha=0.25)
            plt.legend(('$\pm3 \sigma$ range real','real mean','synthetic mean','$\pm3 \sigma$ range synthetic'))
    else:
        if len(synthetic_data)>=1:
            plt.plot(np.arange(np.min(range),np.max(range)),np.mean(synthetic_data[:,np.min(range):np.max(range)],axis=0),color='orange',alpha=0.9)
            #plot out of bounds lines
            plt.fill_between(np.arange(np.min(range),np.max(range)),min(testmean)-3*min(teststd),max(testmean)+3*max(teststd),where=np.mean(synthetic_data[:,np.min(range):np.max(range)],axis=0)>(testmean+3*teststd),color='red',alpha=0.25)
            plt.fill_between(np.arange(np.min(range),np.max(range)),min(testmean)-3*min(teststd),max(testmean)+3*max(teststd),where=np.mean(synthetic_data[:,np.min(range):np.max(range)],axis=0)<(testmean-3*teststd),color='red',alpha=0.25)
            plt.legend(('$\pm3 \sigma$ range','real mean','synthetic mean','out bounds'))
    # where to save?
    return plt

def test_plot_spec_stats():
    """Test function for plotting statistiscs of spectra"""
    return plot_spec_stats(0.5*(np.random.rand(10,200)*2-1),np.random.rand(10,200))

def scaled_back_examples(opt,xtest,xtrain,xhat,cluster=''):
    """If a standard scaler is avaible scales data back.
    If nmf was used to reduce the spectrumsize, this is reverted as well"""

    if opt.scalemethod == "standard":
        # scaler = joblib.load(str(opt.datafile+"scaler.bin"))
        try:
            scaler = joblib.load(str(opt.datafile+"scaler.bin"))
        except:
            print("scaler not found")
            return
        if not os.path.exists(str(opt.savegans+'/exampledata/')):
            os.makedirs(str(opt.savegans+'/exampledata/'))
        try:
            xtest_scaledback = scaler.inverse_transform(xtest)
            xtrain_scaledback = scaler.inverse_transform(xtrain)
            xhat_scaledback = scaler.inverse_transform(xhat)
            # fix nmf stuff
            try:
                nmf = joblib.load(str(opt.datafile+"nmf.bin"))
                xtest_scaledback = nmf.inverse_transform(xtest_scaledback) 
                xtrain_scaledback = nmf.inverse_transform(xtrain_scaledback)
                xhat_scaledback = nmf.inverse_transform(xhat_scaledback)
            except:
                print("No reduction matrix saved")
        
        # plot examples
            plot = plot_spect(xtest_scaledback,np.random.randint(0,np.shape(xtest_scaledback)[0],5),show=False,title=False)
            plot.savefig(str(opt.savegans+'/exampledata/example_test_scaledback_class'+str(cluster)))
            plot = plot_spect(xtrain_scaledback,np.random.randint(0,np.shape(xtrain_scaledback)[0],5),show=False,title=False)
            plot.savefig(str(opt.savegans+'/exampledata/example_train_scaledback_class'+str(cluster)))
            plot = plot_spect(xhat_scaledback,np.random.randint(0,np.shape(xhat_scaledback)[0],5),show=False,title=False)
            plot.savefig(str(opt.savegans+'/exampledata/example_fake_scaledback_class'+str(cluster)))

            plot = plot_spec_stats(xtest_scaledback,xhat_scaledback,range=[0,np.shape(xtest_scaledback)[1]],compare_var=True)
            plot.savefig(str(opt.savegans+'/exampledata/compare_test_fake_scaledback_class'+str(cluster)))
            plot = plot_spec_stats(xtrain_scaledback,xhat_scaledback,range=[0,np.shape(xtest_scaledback)[1]],compare_var=True)
            plot.savefig(str(opt.savegans+'/exampledata/compare_train_fake_scaledback_class'+str(cluster)))
        
            # load original data

            # compare test data and original (distribution wise)

            # compare xhat and original (distribution wise)
        except ValueError:
            print("Can't scale data back, spectra size incompatalbe with selected scaler!")
    else:
        print("Scaled back examples only implemented for standard scaling")
        pass

    
def eval_gan_training(opt,X_test,y_test,X_train=[],y_train=[]):
    """Evaluate the training of the gans,
    This function uses the different saved generators and discriminators"""

    classes = set(y_test)
    y_testgpu = torch.tensor(y_test).long().to(opt.device)
    X_testgpu = torch.tensor(X_test).long().to(opt.device)
    # print(sum(np.array(y_test==1)))
    adverloss=opt.loss
    dis_loss_hist = []
    gen_loss_hist = []
    FID_class_tot = []
    FID_class = []

    test_genloss = []
    test_disloss = []
    train_genloss = []
    train_disloss = []
    l2mean_test = []
    l2std_test = []
    l2mean_train = []
    l2std_train = []
    FID_test = []
    FID_train = []

    for file in tqdm(natsorted(os.listdir(opt.savegans))):
        # print(file)
        if 'generator' in file:
            # load generator
            fgen = os.path.join(opt.savegans, file)
            nr = file.replace('generator','')
            torch.cuda.empty_cache() #added 12-01-2024

            gen_model = torch.load(fgen,map_location=opt.device).eval() #added eval() 15-01-24
            # load discriminator
            fdis = os.path.join(opt.savegans, file.replace('generator','discriminator'))
            dis_model = torch.load(fdis,map_location=opt.device).eval()

            # generate fake per class
            dis_lossclass = []
            gen_lossclass = []
            FID_class = []
            for lab in classes:
                # create fake data
                z = Variable(torch.FloatTensor(np.random.normal(0,1, (sum(y_test==lab),opt.latent_dim)))).to(opt.device)
                y_hat = torch.tensor(np.ones((sum(np.array(y_test==lab))))*lab).to(opt.device).long()

                X_hat = gen_model(z,y_hat)
                #create reference fake/valid
                valid = Variable(torch.FloatTensor(len(y_test[y_test==lab]),1).fill_(1.0),requires_grad=False).to(opt.device)
                fake = Variable(torch.FloatTensor(len(y_test[y_test==lab]),1).fill_(0.0),requires_grad=False).to(opt.device)
                # calculate losses
                if opt.WGAN_gp:
                    pass
                else:
                    loss = adverloss(dis_model.forward(X_testgpu[y_test==lab],y_testgpu[y_test==lab]),valid)
                    dis_lossclass.append(loss.item())
                    loss = adverloss(dis_model.forward(X_hat,y_hat[y_hat==lab].long()),fake)
                    gen_lossclass.append(loss.item())

                #FID over time per class
                errormeas = error_val(X_test[y_test==lab],X_hat.cpu().detach().numpy())
                FID_class.append(errormeas.FID())

            dis_loss_hist.append(dis_lossclass)
            gen_loss_hist.append(gen_lossclass)
            FID_class_tot.append(FID_class)

            # total loss based on X_test and y_test
            valid = Variable(torch.FloatTensor(len(y_test),1).fill_(1.0),requires_grad=False).to(opt.device)
            fake = Variable(torch.FloatTensor(len(y_test),1).fill_(0.0),requires_grad=False).to(opt.device)
            z = Variable(torch.FloatTensor(np.random.normal(0,1,(len(y_test),opt.latent_dim)))).to(opt.device)
            y_hat = torch.tensor(y_test).to(opt.device).long()
            x_hat = gen_model(z,y_hat)
            if opt.WGAN_gp:
                pass
            else:
                test_genloss.append(adverloss(dis_model.forward(x_hat.long(),y_testgpu.long()),fake).item())
                test_disloss.append(adverloss(dis_model.forward(X_testgpu.long(),y_testgpu.long()),valid).item())

            errormeas = error_val(X_test,x_hat.cpu().detach().numpy())
            l2mean_test.append(errormeas.l2mean())
            l2std_test.append(errormeas.l2std())
            FID_test.append(errormeas.FID())

            # JS = scipy.spatial.distance.jensenshannon(X_test.cpu().detach().numpy(),x_hat.cpu().detach().numpy(),axis=0)
            # mean_js_test.append(np.mean(JS))

            if len(X_train)>0:
                y_traingpu = torch.tensor(y_train).to(opt.device)
                X_traingpu = torch.tensor(X_train).to(opt.device)
                valid = Variable(torch.FloatTensor(len(y_traingpu),1).fill_(1.0),requires_grad=False).to(opt.device)
                fake = Variable(torch.FloatTensor(len(y_traingpu),1).fill_(0.0),requires_grad=False).to(opt.device)
                z = Variable(torch.FloatTensor(np.random.normal(0,1,(len(y_traingpu),opt.latent_dim)))).to(opt.device)
                y_hat = torch.tensor(y_traingpu).to(opt.device).long()
                x_hat = gen_model(z,y_hat)

                if opt.WGAN_gp:
                    pass
                else:
                    train_genloss.append(adverloss(dis_model.forward(x_hat.long(),y_traingpu.long()),fake).item())
                    train_disloss.append(adverloss(dis_model.forward(X_traingpu.long(),y_traingpu.long()),valid).item())

                errormeas = error_val(X_train,x_hat.cpu().detach().numpy())

                l2mean_train.append(errormeas.l2mean())
                l2std_train.append(errormeas.l2std())
                FID_train.append(errormeas.FID())

            if opt.removeproggan:
                os.remove(fdis)
                os.remove(fgen)
    try:
        temp = X_hat
    except UnboundLocalError:
        print("No gan history to visualize")
        z = Variable(torch.FloatTensor(np.random.normal(0,1, (len(y_test),opt.latent_dim)))).to(opt.device)
        y_hat = torch.tensor(y_test).to(opt.device).long()
        fgen = os.path.join(opt.savegans, "finalgen")
        gen_model=torch.load(fgen,map_location=opt.device).eval()
        X_hat = gen_model(z,y_hat)

    y_hat = torch.tensor(y_test).to(opt.device).long()
    y_hat=y_hat.cpu().squeeze()
    X_hat = X_hat.cpu()

 

    plt.figure()
    plt.plot(np.add(test_genloss,test_disloss))
    if len(X_train)>0:
        plt.plot(np.add(train_disloss,train_disloss))
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("Combined loss over time")
    if len(X_train)>0:
        plt.legend(["Test set","Train set"])
    else:
        plt.legend("Test set")
    plt.savefig(str(opt.savegans+"combined_loss"))

    plt.figure()
    plt.plot(l2mean_test)
    plt.plot(l2mean_train)
    plt.xlabel("Epoch")
    plt.ylabel("l2 of mean difference")
    # plt.ylim([0,1.1])
    plt.title("mean difference between generated and real \n  (divided by l2 of real)")
    plt.legend(["Test set","Train set"])
    plt.savefig(str(opt.savegans+"mean_diff"))

    plt.figure()
    plt.plot(l2std_test)
    plt.plot(l2std_train)
    plt.xlabel("Epoch")
    plt.ylabel("l2 of std difference")
    plt.title("std difference between generated and real")
    plt.legend(["Test set","Train set"])
    plt.savefig(str(opt.savegans+"std_diff"))

    plt.figure()
    # plt.subplot(2,1,1)
    plt.plot(FID_test)
    # plt.title("FID test set")
    # plt.subplot(2,1,2)
    plt.plot(FID_train)
    plt.title("FD")
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    # plt.suptitle("FID")
    plt.legend(["Test set","Train set"])
    plt.savefig(str(opt.savegans+"FID"))

    plt.figure()
    plt.plot(test_genloss) #on test set
    plt.plot(test_disloss)
    if len(X_train)>0:
        plt.plot(train_genloss)
        plt.plot(train_disloss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over time")
    if len(X_train)>0:
        plt.legend(["Generator loss (test)","Discriminator loss (test)","Generator loss (train)","Discriminator loss (train)"])
    else:
        plt.legend(["Generator loss (test)","Discriminator loss (test)"])

    plt.savefig(str(opt.savegans+"total_loss"))

    plt.figure()
    plt.plot(FID_class_tot)
    plt.xlabel("Epoch")
    plt.ylabel("FD")
    plt.title("FD between fake and test set per class")
    plt.legend(["Class "+str(i) for i in set(np.array(y_test))])
    plt.savefig(str(opt.savegans+"FID_class"))

    plt.figure()
    plt.plot(dis_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator loss on test set")
    plt.legend(["Class "+str(i) for i in set(np.array(y_test))])
    plt.savefig(str(opt.savegans+"discrloss"))

    plt.figure()
    plt.plot(gen_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator loss on test set")
    plt.legend(["Class "+str(i) for i in set(np.array(y_test))])
    plt.savefig(str(opt.savegans+"genloss"))

    #generate data from last generator
   
    y_hat = torch.tensor(y_test).to(opt.device).long()#.unsqueeze(dim=-1)
    z = torch.rand((len(y_hat),opt.latent_dim)).to(opt.device)
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(y_hat), opt.latent_dim)))).to(opt.device)
    X_hat = gen_model.forward(z,y_hat).cpu().detach().numpy()
    y_hat = y_hat.cpu().detach().numpy()
    #create spectra visualization based on mean and variance per class
    for i in set(y_test):
        scaled_back_examples(opt,xtest=X_test[y_test==i],
                             xtrain=X_train[y_train==i],
                             xhat=X_hat[y_hat.squeeze()==i],
                             cluster=i)

        plot = plot_spec_stats(X_test[y_test.squeeze()==i],X_hat[y_hat.squeeze()==i],range=[0,np.shape(X_test)[1]],compare_var=True)
        plot.title(str("class "+str(i)))
        plot.savefig(str(opt.savegans+"spec_stat_class"+str(i)+".png"))

        plot = plot_spect(X_test[y_test.squeeze()==i],np.random.randint(0,len(y_test[y_test.squeeze()==i]),5),show=False,title=False)
        # plot.title(str("class "+str(i)))
        plot.savefig(str(opt.savegans+"spec_test_class"+str(i)+".png"))

        plot = plot_spect(X_hat[y_test.squeeze()==i],np.random.randint(0,len(y_test[y_test.squeeze()==i]),5),show=False,title=False)
        # plot.title(str("class "+str(i)))
        plot.savefig(str(opt.savegans+"spec_fake_class"+str(i)+".png"))
    # scaled back fd for intergan comparison
    try:
        scaledback_comparing(opt)
    except ValueError:
        print("cant perform smote operation: no scaled back data, put this inside function eventually!")
    except RuntimeError:
        print("Runtime error occured, scaling back of data only implemented for 1 specific dataset!")
    y_hat = torch.tensor(y_train).to(opt.device).long()#.unsqueeze(dim=-1)
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(y_hat), opt.latent_dim)))).to(opt.device)
    X_hat_train = gen_model.forward(z,y_hat).cpu().detach().numpy()
    replace_LDA(opt.savegans,X_hat_train,y_train,X_test,y_test)
    # this is generating a testset?

    basic_replacement(opt,X_train,y_train,X_test,y_test)
    smote_replace_training(opt,X_train,y_train,X_test,y_test)

def basic_replacement(opt,X_train,y_train,X_test,y_test):
    """Test replacement method using a simple estimator based on bias and standard diviation per mass bin
    saves classification report and confusionmatrix"""
    xhat=np.array([])
    #simple data augmentation
    for y in set (y_train):
        z = np.random.normal(0,1,size=(sum(y_train==y),np.shape(X_train)[1]))
        bias = np.mean(X_train[y_train==y],axis=0)
        std = np.std(X_train[y_train==y],axis=0)
        try: 
            xhat = np.vstack((xhat,z*std+bias))
        except ValueError:
            xhat = z*std+bias
    #linear estimator #works less good as data augmentation
    # for y in set(y_train):
    #     cov = np.cov(X_train[y_train==y][:,:opt.specsize],rowvar=False)
    #     U, s, _ = np.linalg.svd(cov)
    #     l = opt.latent_dim
        
    #     A = np.dot(U[:,:l],np.sqrt(np.diag(s[:l])))
    #     bias = np.mean(X_train[y_train==y][:,:opt.specsize],axis=0)
  
    #     z = np.random.normal(0,1,size=(sum(y_train==y),l))
    #     try: 
    #         xhat = np.vstack((xhat,np.transpose(A@np.transpose(z))+bias))
    #     except ValueError:
    #         xhat = np.transpose(A@np.transpose(z))+bias
    
    LDA=LinearDiscriminantAnalysis()
    LDA.fit(xhat,y_train)
    ypred = LDA.predict(X_test)
    confusionmatrix(opt.savegans,ypred,y_test,"simple_replace_training")
    
    with open(opt.savegans+'basic_replacement_report','w+') as f:
        f.write(classification_report(y_test,ypred))

def smote_replace_training(opt,X_train,y_train,X_test,y_test):
    _,counts = np.unique(y_train,return_counts=True)
    samplelength = max(counts)*2
    X_train_ex = np.vstack((X_train,np.zeros((samplelength,np.shape(X_train)[1]))))
    y_train_ex = np.hstack((y_train,np.ones(samplelength)*(max(y_train)+1)))
    sm = SMOTE()
    Xhat,yhat = sm.fit_resample(X_train_ex,y_train_ex)

    X_smote = Xhat[np.shape(X_train_ex)[0]:,:]
    y_smote = yhat[np.shape(y_train_ex)[0]:]

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_smote,y_smote)
    y_pred = lda.predict(X_test)

    confusionmatrix(opt.savegans,y_pred,y_test,name="_smote_replacement")
    with open(opt.savegans+'smote_replacement_report','w+') as f:
        f.write(classification_report(y_test,y_pred))

def PCA_class_visualisation(X_r,y_r,X_f,y_f,save_path="",showfake=True,dims=3):
    """performs pca on real data and plots reduced dimensions in 3 different 2d scatterplots
    Arguments X_r: real data, y_r: real labels"""
    #perform pca fit real, transform both
    pca = PCA(dims)
    X_r_pca = pca.fit_transform(X_r)
    if showfake:
        X_f_pca = pca.transform(X_f)
    leg = []
    for d in range(dims):
        plt.figure()
        plt.grid()
        for i in set(y_r):
            plt.scatter(X_r_pca[y_r==i][:,d],X_r_pca[y_r==i][:,(d+1) % dims],alpha=0.8,marker='x')
            leg.append("Real class %d" %i)
        if showfake:
            for i in set(y_f):
                plt.scatter(X_f_pca[y_f==i][:,d],X_f_pca[y_f==i][:,(d+1) % dims],alpha=0.8,marker='o')
                leg.append("Fake class %d" %i)
        plt.xlabel("Component %d"%(d+1))
        plt.ylabel("Component %d"%(((d+1) % dims)+1))
        plt.legend(leg)
        plt.savefig(str(save_path+"Clusters3D"+str(d)))
class augmented_oversampling():
    """Use for augmented oversampling"""
    def __init__(self) -> None:
        pass

    def data_augmentation(self,X_train,y_train):
        occur = []
        genlen = []
        for i in set(y_train):
            occur.append(sum(sum([y_train==i])))

        # X_aug = [[]]
        y_aug = []
        for i in range(len(occur)):
            genlen.append(max(occur)-occur[i])
            aug_spec = self.get_augmented_spec(X_train,y_train,i,genlen[i])

            # print(np.transpose(np.ones(genlen[i])*i))
            if genlen[i]>0:
                y_aug = np.hstack([y_aug,(np.transpose(np.ones(genlen[i])*i))])
                aug_spec = self.get_augmented_spec(X_train,y_train,i,genlen[i])
                try:
                    X_aug = np.vstack([X_aug,aug_spec])
                except:
                    X_aug = aug_spec
                # print(np.shape(X_aug))
        try:
            X_train = np.vstack([X_train,X_aug])
            y_train = np.hstack([y_train,y_aug])
        except:
            print("No augumented spectra added")
        return X_train, y_train

    def get_augmented_spec(self,X_train,y_train,label,genlen):
        mean = np.mean(X_train[y_train==label],axis=0)
        std = np.mean(np.std(X_train[y_train==label],axis=0))
        gen_spec = []
        for i in range(genlen):
            gen_spec.append(mean+std*np.random.uniform(-3,3,len(mean)))
        return gen_spec

    def test(self):
        X = np.array([[1,2,3,4,5],[2,4,3,5,8]])
        y=np.array([1,1])
        return self.get_augmented_spec(X,y,1,1)

class GAN_resampler():
    def __init__(self) -> None:
        pass
    def gen_samples(self,classes,model,opt,static_n=True):
        if static_n:
            # static = torch.rand(1,1,opt.latent_dim)
            static = Variable(torch.FloatTensor(np.random.normal(0,1,(opt.latent_dim))))
            noise = torch.tile((len(classes),static,1))  
        else:
            # noise = torch.rand((len(classes),1,opt.latent_dim))
            noise = Variable(torch.FloatTensor(np.random.normal(0,1,(len(classes),opt.latent_dim))))
        print(np.shape(torch.tensor(np.transpose([classes])).int().to(opt.device)))
        print(np.shape(noise.to(opt.device)))
        return model.forward(noise.to(opt.device),torch.tensor(np.transpose([classes])).squeeze().int().to(opt.device))

    def resample(self,opt,X_train,y_train):
        """Add gan-generated data untill dataset is balanced"""
        occur = []
        genlen = []
        for i in set(y_train):
            occur.append(sum(sum([y_train==i])))
        y_aug = []
        for i in range(len(occur)):
            genlen.append(max(occur)-occur[i])
            if genlen[i]>0:
                y_aug = np.hstack([y_aug,(np.transpose(np.ones(genlen[i])*i))])
        
        model = torch.load(str(opt.savegans+'finalgen'),map_location=opt.device).eval()
        X_aug = self.gen_samples(y_aug,model,opt,static_n=False)
        
        try:
            X_train = np.vstack([X_train,X_aug.detach().cpu()])
            y_train = np.hstack([y_train,y_aug])
        except:
            print("No augumented spectra added")
        return X_train, y_train
    
    def test(self,opt,X_train,y_train):
        self.resample(opt,X_train,y_train)

class error_val():
    """Quantative methods of comparing real and fake data"""
    def __init__(self,X_r,X_f):
        """The methods below all use the mean, covariance and standarddiviation"""
        self.mean_r = np.mean(X_r,axis=0)
        self.mean_f = np.mean(X_f,axis=0)

        self.Cov_r = np.cov(X_r,rowvar=False)
        self.Cov_f = np.cov(X_f,rowvar=False)

        self.std_r = np.std(X_r,axis=0)
        self.std_f = np.std(X_f,axis=0)

    def FID(self):
        """Gives Frechet distance of 2 datasets"""
        # FID = np.sqrt(np.linalg.norm((self.mean_r-self.mean_f)**2)+np.trace(self.Cov_r+self.Cov_f-2*scipy.linalg.sqrtm(self.Cov_r*self.Cov_f)))
        # FID = np.sqrt(np.linalg.norm((self.mean_r-self.mean_f)**2)+np.trace(self.Cov_r+self.Cov_f-2*scipy.linalg.sqrtm(self.Cov_r.dot(self.Cov_f))))

        mean_diff = self.mean_r-self.mean_f
        # covmean = scipy.linalg.sqrtm(self.Cov_r.dot(self.Cov_f))

        try:  
            covmean = scipy.linalg.sqrtm(self.Cov_r.dot(self.Cov_f))
        except:
            print("covmean trouble! in FD calculation")    
            covmean=np.zeros((2,2))

        trace = np.abs(np.trace(self.Cov_r)+np.trace(self.Cov_f)-2*np.trace(covmean))
        if not np.isnan(trace):
            FD = np.sqrt(mean_diff.dot(mean_diff)+trace)
        else:
            FD = np.sqrt(mean_diff.dot(mean_diff))
        return FD
    def KL(self):
        """Calculates kullback-leibler divergence based on assumption that distributions are mutlivariate normal"""
        S1inv = np.linalg.inv(self.Cov_r)
        k = len(self.Cov_r)

        Dkl = 0.5*np.trace(S1inv@self.Cov_f)-k+(self.mean_r-self.mean_f).T@S1inv@(self.mean_r-self.mean_f)+np.log(np.linalg.det(self.Cov_r)/np.linalg.det(self.Cov_f))
        return Dkl
    
    def l2mean(self,normalization=True):
        """Calculates l2 norm (euclidean distance) between real and fake mean
        if normalization, devide by faken mean to normalize"""
        # print("l2 real: ",np.linalg.norm(self.mean_r))
        # print("l2 fake: ",np.linalg.norm(self.mean_f))
        l2mean = np.linalg.norm(self.mean_r-self.mean_f)
        # print("l2 difference",l2mean)
        if normalization:
            l2mean = l2mean/np.linalg.norm(self.mean_f)
            # print("normalized l2: ",l2mean)
        return l2mean
    
    def l2std(self,normalization=True):
        """Calculates l2 norm (euclidean distance) between real and fake standard diviation"""
        l2std = np.linalg.norm(self.std_r-self.std_f)
        if normalization:
            l2std = l2std/np.linalg.norm(self.std_f)
        return l2std
    
    def test(self):
        for key in self.__dict__:
            print(key," has shape: ",np.shape(self.__dict__[key]))

def replace_LDA(path:str,x_hat,y_hat,x_test,y_test):
    """Function to evaluate quality
    Uses generated data to fit LDA and classifies test set
    output is confusionmatrix and report"""
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(x_hat,y_hat)
    y_pred = LDA.predict(x_test)
    print(classification_report(y_test,y_pred))
    with open(path+'replacement_report','w+') as f:
        f.write(classification_report(y_test,y_pred))
    confusionmatrix(path,y_pred,y_test,name="_replace_training")

class WGAN_gp_gen(nn.Module):
    def __init__(self):
        super(WGAN_gp_gen,self)

    def forward(self,gen_validity):
        return -torch.mean(gen_validity)


def WGAN_gradient_penalty(opt,discriminator,real_data,fake_data,labels_real,labels_fake):
    """Calculate gradient penalty of the wgan"""
    # random weight for interpolation
    alpha = torch.FloatTensor(np.random.random((real_data.size(0),1,))).to(opt.device) # vector Uniform (0,1)
    # get random interpolation between real and fake
    interp = (alpha * real_data + ((1-alpha) * fake_data)).requires_grad_(True)

    d_interp = discriminator(interp,labels_real.long())

    fake = Variable(torch.FloatTensor(real_data.shape[0],1).fill_(1.0), requires_grad=False).to(opt.device)
    # get gradients with respect to interpolations 
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs = interp,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        # allow_unused=True,
    )[0]
    gradients = gradients.view(gradients.size(0),-1)
    penalty = ((gradients.norm(2,dim=1) -1) ** 2).mean()

    return penalty

