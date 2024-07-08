"""
This file contains the different models, both GANs and classifiers
Not all these generators and discriminators are used in the main files

"""



import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) #mean 0, std 0.02
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) 
        torch.nn.init.constant_(m.bias.data, 0.0)
   
class cGANgenerator_4(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_4,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,self.specsize),
            # nn.BatchNorm1d(self.specsize,0.8),
            nn.Tanh()
        )

    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels.long()),noise), -1)
        # print(z.size())
        x = self.model(z)
        x = x.view(x.size(0),self.specsize)
        return x
class cGANgenerator_3(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_3,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,self.specsize),
            # nn.BatchNorm1d(self.specsize,0.8),
            nn.Tanh()
        )

    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels.long()),noise), -1)
        # print(z.size())
        x = self.model(z)
        x = x.view(x.size(0),self.specsize)
        return x
    
class simplest_generator(nn.Module):
    def __init__(self,opt) -> None:
        super(simplest_generator,self).__init__()
        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim+self.classes),self.specsize,bias=True),
            nn.BatchNorm1d(self.specsize),
            # nn.LeakyReLU(inplace=True)
        )
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels.long()),noise), -1)
        # print(noise.size())
        # z = noise
        x = self.model(z)
        x = x.view(x.size(0),self.specsize)
        return x

class cGANgenerator_fc_4(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_fc_4,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024,self.specsize),
            nn.BatchNorm1d(self.specsize)#,0.8)
        )
    
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels.long()),noise), -1)
        # print(z.size())
        x = self.model(z)
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x
class cGANgenerator_fc_3(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_fc_3,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512,self.specsize),
            nn.BatchNorm1d(self.specsize)#,0.8)
        )
    
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels.long()),noise), -1)
        # print(z.size())
        x = self.model(z)
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x

class cGANgenerator_fc_small(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_fc_small,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(64,self.specsize),
            nn.BatchNorm1d(self.specsize)#,0.8)
        )
    
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1)
        # print(z.size())
        x = self.model(z)
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x
class cGANgenerator_small(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_small,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(64,self.specsize),
            nn.BatchNorm1d(self.specsize),#,0.8)
            nn.Tanh()
        )
    
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1)
        # print(z.size())
        x = self.model(z)
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x


class cGANgenerator_fc_wide(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_fc_wide,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),256),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(2048,self.specsize),
            nn.BatchNorm1d(self.specsize)#,0.8)
        )
    
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1)
        # print(z.size())
        x = self.model(z)
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x

class cGANgenerator_fc_skipconnect(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANgenerator_fc_skipconnect,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.block1 = nn.Sequential(
            nn.Linear((self.latentdim + self.classes),128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256,512),
            nn.BatchNorm1d(512),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),#,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(1024,self.specsize),
            nn.BatchNorm1d(self.specsize)#,0.8)
        )
        self.upsample = nn.Linear((self.latentdim + self.classes),256)
        
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1)
        # print(z.size())
        x = self.block1(z)
        x = self.block2(x+self.upsample(z))
        # print(x.size())
        x = x.view(x.size(0),self.specsize)
        return x

class cGANdiscriminator_4(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANdiscriminator_4,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x
class cGANdiscriminator_3(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANdiscriminator_3,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels.long())),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x

class cGANdiscriminator_small(nn.Module):
    def __init__(self,opt) -> None:
        super(cGANdiscriminator_small,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,64),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(64,32),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(32,1),
            nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x

class simplestdisc(nn.Module):
    def __init__(self,opt) -> None:
        super(simplestdisc,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,1),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Linear(512,512),
            # nn.Dropout1d(0.4),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Linear(512,512),
            # nn.Dropout1d(0.4),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Linear(512,1),
            nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x
class cWGANdiscriminator_4(nn.Module):
    def __init__(self,opt) -> None:
        super(cWGANdiscriminator_4,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize 
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            # nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels.long())),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x
class cWGANdiscriminator_3(nn.Module):
    def __init__(self,opt) -> None:
        super(cWGANdiscriminator_3,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize # not yet included!!
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
            # nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels.long())),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x
    
class cWGANdiscriminator_small(nn.Module):
    def __init__(self,opt) -> None:
        super(cWGANdiscriminator_small,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize # not yet included!!
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,64),
            # nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(64,32),
            # nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(32,1),
            # nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x

class cWGANdiscriminator_wide(nn.Module):
    def __init__(self,opt) -> None:
        super(cWGANdiscriminator_wide,self).__init__()

        self.classes=opt.numclasses
        self.specsize = opt.specsize # not yet included!!
        # self.latentdim = opt.latent_dim

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.model = nn.Sequential(
            nn.Linear(self.classes+self.specsize,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Linear(512,512),
            # nn.Dropout1d(0.4),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,1024),
            nn.Dropout1d(0.4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,1),
            # nn.Sigmoid()
        )  
    
    def forward(self,x,labels):
        # self.label_embedding(labels)
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1)
            # torch.cat((self.label_embedding(labels),noise), -1)
        x = self.model(x)
        # return x[:,None]
        return x

class cDC_Discriminator(nn.Module):
    def __init__(self,opt):
        super(cDC_Discriminator,self).__init__()

        self.classes=opt.numclasses
        self.label_embedding = nn.Embedding(self.classes,self.classes)
        self.hidden_dim = 8

        self.conv = nn.Sequential(
            # state 1 x 573+labels
            nn.Conv1d(in_channels=1,out_channels=self.hidden_dim,kernel_size=5,stride=2,padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hidden_dim),
            # state 1 x (573+labels)/2
            nn.Conv1d(in_channels=self.hidden_dim,out_channels=self.hidden_dim*2,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*2),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*2,out_channels=self.hidden_dim*4,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*4),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*4,out_channels=self.hidden_dim*8,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*8),
            # state 512 x 36 (by hiddem_dim = 64)
            # # # find nice solution for kernel_size on this line
            # nn.Conv1d(in_channels=self.hiddem_dim*8,out_channels=1,kernel_size=8,stride=1,padding=0),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
            )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=int(312*self.hidden_dim),out_features=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
    def forward(self,x,labels):
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1).unsqueeze(1)
        # x = x.unsqueeze(1)
        # print(x.size())
        x = self.conv(x)
        # print("size before output",x.size())
        # print("size of view",x.view(x.size(0),-1).size())
        x = self.output(x.view(x.size(0),-1))
        # print("size after output",x.size())
        return x

class cDCWGANdiscriminator(nn.Module):
    def __init__(self,opt):
        super(cDCWGANdiscriminator,self).__init__()

        self.classes=opt.numclasses
        self.label_embedding = nn.Embedding(self.classes,self.classes)
        self.hidden_dim = 32 #base 32
        self.spec_size = opt.specsize

        self.input = nn.Sequential(
            nn.Linear(in_features=self.classes+self.spec_size,out_features=512),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.conv = nn.Sequential(
            # state 1 x 573+labels
            nn.Conv1d(in_channels=1,out_channels=self.hidden_dim,kernel_size=5,stride=2,padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.4),
            # nn.BatchNorm1d(self.hidden_dim),
            # state 1 x (573+labels)/2
            nn.Conv1d(in_channels=self.hidden_dim,out_channels=self.hidden_dim*2,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.4),
            # nn.BatchNorm1d(self.hidden_dim*2),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*2,out_channels=self.hidden_dim*4,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.4),
            # nn.BatchNorm1d(self.hidden_dim*4),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*4,out_channels=self.hidden_dim*8,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.4),
            # nn.BatchNorm1d(self.hidden_dim*8),
            # state 512 x 36 (by hiddem_dim = 64)
            # # # find nice solution for kernel_size on this line
            # nn.Conv1d(in_channels=self.hiddem_dim*8,out_channels=1,kernel_size=8,stride=1,padding=0),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
            )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=int(280*self.hidden_dim),out_features=1),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
        )
    def forward(self,x,labels):
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels.long())),-1).unsqueeze(1)
        x = self.input(x)
        # print(x.size())
        x = self.conv(x)
        # print("size before output",x.size())
        # print("size of view",x.view(x.size(0),-1).size())
        x = self.output(x.view(x.size(0),-1))
        # print("size after output",x.size())
        return x
class cDCWGANdiscriminator_addlayer(nn.Module):
    def __init__(self,opt):
        super(cDCWGANdiscriminator_addlayer,self).__init__()

        self.classes=opt.numclasses
        self.label_embedding = nn.Embedding(self.classes,self.classes)
        self.hidden_dim = 8 #base =32
        self.spec_size = opt.specsize

        self.input = nn.Sequential(
            nn.Linear(in_features=self.classes+self.spec_size,out_features=512),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.conv = nn.Sequential(
            # state 1 x 573+labels
            nn.Conv1d(in_channels=1,out_channels=self.hidden_dim,kernel_size=5,stride=2,padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim),
            # state 1 x (573+labels)/2
            nn.Conv1d(in_channels=self.hidden_dim,out_channels=self.hidden_dim*2,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*2),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*2,out_channels=self.hidden_dim*4,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*4),
            # state 1 x
            nn.Conv1d(in_channels=self.hidden_dim*4,out_channels=self.hidden_dim*8,kernel_size=4,stride=2,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.5),
            nn.BatchNorm1d(self.hidden_dim*8),
            # state 512 x 36 (by hiddem_dim = 64)
            # # # find nice solution for kernel_size on this line
            # nn.Conv1d(in_channels=self.hiddem_dim*8,out_channels=1,kernel_size=8,stride=1,padding=0),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
            )
        
        self.output = nn.Sequential(
            nn.Linear(in_features=int(280*self.hidden_dim),out_features=1),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
        )
    def forward(self,x,labels):
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1).unsqueeze(1)
        x = self.input(x)
        # print(x.size())
        x = self.conv(x)
        # print("size before output",x.size())
        # print("size of view",x.view(x.size(0),-1).size())
        x = self.output(x.view(x.size(0),-1))
        # print("size after output",x.size())
        return x

class cDC_Discriminator_aux(nn.Module):
    def __init__(self,opt):
        super(cDC_Discriminator_aux,self).__init__()

        self.classes=opt.numclasses
        self.label_embedding = nn.Embedding(self.classes,self.classes)
        self.hiddem_dim = 1

        self.conv = nn.Sequential(
            # state 1 x 573+labels
            nn.Conv1d(in_channels=1,out_channels=self.hiddem_dim,kernel_size=5,stride=1,padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hiddem_dim),
            # state 64 x 289 (by hiddem_dim = 64, stride = 2)
            nn.Conv1d(in_channels=self.hiddem_dim,out_channels=self.hiddem_dim*1,kernel_size=4,stride=1,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.25),
            nn.BatchNorm1d(self.hiddem_dim*1),
            # state 128 x 144 (by hiddem_dim = 64,stride = 2)
            nn.Conv1d(in_channels=self.hiddem_dim*1,out_channels=self.hiddem_dim*1,kernel_size=4,stride=1,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.25),
            nn.BatchNorm1d(self.hiddem_dim*1),
            # state 256 x 72 (by hiddem_dim = 64)
            nn.Conv1d(in_channels=self.hiddem_dim*1,out_channels=self.hiddem_dim*1,kernel_size=4,stride=1,padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.25),
            nn.BatchNorm1d(self.hiddem_dim*1),
            # state 512 x 36 (by hiddem_dim = 64)
            # # # find nice solution for kernel_size on this line
            # nn.Conv1d(in_channels=self.hiddem_dim*8,out_channels=1,kernel_size=8,stride=1,padding=0),
            # nn.BatchNorm1d(1),
            # nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=int(2332/4),out_features=5),
            # nn.BatchNorm1d(5),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )
    def forward(self,x,labels):
        x = torch.cat((x.view(x.size(0),-1), self.label_embedding(labels)),-1).unsqueeze(1)
        # print(x.size())
        x = self.conv(x)
        # print("size before output",x.size())
        # print("size of view",x.view(x.size(0),-1).size())
        x = self.output(x.view(x.size(0),-1))
        # print("size after output",x.size())
        _, pred = torch.max(x,1)
        return pred
   
class cDC_Generator_fc(nn.Module):    
    def __init__(self,opt):
        super(cDC_Generator_fc, self).__init__()

        self.latent_dim = opt.latent_dim
        self.classes=opt.numclasses
        self.specsize=opt.specsize

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.hidden_dim = 8

        self.input = nn.Sequential(
            nn.Linear(self.latent_dim+self.classes,128),
            # nn.Linear(100,128),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(1,self.hidden_dim//1,4,2,1),
            nn.BatchNorm1d(self.hidden_dim//1),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//1,self.hidden_dim//2,4,2,1),
            nn.BatchNorm1d(self.hidden_dim//2),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//2,self.hidden_dim//4,4,2,1),
            nn.BatchNorm1d(self.hidden_dim//4),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//4,self.hidden_dim//8,4,1,1),
            nn.BatchNorm1d(self.hidden_dim//8),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
            # # (64x1) x 32 = 2048
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim*1,1,4,2,1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
            # #573 x 1 x 64
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(1,1,4,2,1,output_padding=0),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
            )
            # nn.ConvTranspose1d(self.hidden_dim*8,573,4),
            # nn.BatchNorm1d(573) 
        self.output = nn.Sequential(
            nn.Linear(1025,self.specsize),
            nn.BatchNorm1d(self.specsize)
            # nn.Tanh()
            )
    def forward(self,noise,labels):
        # print(noise.unsqueeze(2).size())
        # print(labels.size())
        # print(self.label_embedding(labels))
        z = torch.cat((self.label_embedding(labels.long()),noise), -1).unsqueeze(1)
        # z = noise.unsqueeze(1) # test without label embedding (no possibility of generating different classes)
        # print(z.size())
        x = self.input(z)  
        # print(x.size())
        x = self.conv1(x)  
        # # print(x.size())
        x = self.conv2(x)  
        # print(x.size())
        x = self.conv3(x)  
        # # print(x.size())
        x = self.conv4(x)  
        # print(x.size())
        # x = self.conv6(x)
        # print(x.size())
        x = self.output(x.view(x.size(0),-1))
        # print(x.size())
        return x.squeeze(1)
    
    def layerinfo(self,noise,labels):
        # print(noise.unsqueeze(2).size())
        # print(labels.size())
        z = torch.cat((self.label_embedding(labels.long()),noise), -1).unsqueeze(1)
        print("Input size: ",z.size())
        x = self.input(z)  
        print("inputlayer output size: ",x.size())
        x = self.conv1(x)
        print("conv 1 output size: ",x.size())
        x = self.conv2(x)  
        print("conv 2 output size: ",x.size())
        x = self.conv3(x)  
        print("conv 3 output size: ",x.size())
        x = self.conv4(x)  
        print("conv 4 output size: ",x.size())
        # x = self.conv5(x)  
        # print("conv 5 output size: ",x.size())
        # x = self.conv6(x)
        # print("conv 6 output size: ",x.size())
        x = self.output(x.view(x.size(0),-1))
        print("output size: ",x.size())
         
class cDC_Generator_fc_new(nn.Module):    
    def __init__(self,opt):
        super(cDC_Generator_fc_new, self).__init__()

        self.latent_dim = opt.latent_dim
        self.classes=opt.numclasses

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.hidden_dim = 64

        self.input = nn.Sequential(
            # nn.Linear(self.latent_dim+self.classes,128),
            nn.Linear(100,128),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(100,self.hidden_dim//1,4,1,0),
            nn.BatchNorm1d(self.hidden_dim//1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(0.5)
            )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//1,self.hidden_dim//2,8,4,2),
            nn.BatchNorm1d(self.hidden_dim//2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(0.5)
            )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//2,self.hidden_dim//4,8,4,2),
            nn.BatchNorm1d(self.hidden_dim//4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(0.5)
            )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//4,self.hidden_dim//8,4,4,0),
            nn.BatchNorm1d(self.hidden_dim//8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout1d(0.5)
            )
            # # (64x1) x 32 = 2048
        self.conv5 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//8,1,4,2,1),
            nn.BatchNorm1d(1),
            # nn.LeakyReLU(inplace=True),
            # nn.Dropout1d(0.5)
            )
            # #573 x 1 x 64
        self.conv6 = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim//16,self.hidden_dim//32,4,2,1,output_padding=0),
            nn.BatchNorm1d(self.hidden_dim//32),
            # nn.LeakyReLU(inplace=True),
            )
        # self.conv6 = nn.Sequential(
        #     nn.ConvTranspose1d(self.hidden_dim//32,1,4,2,1,output_padding=0),
        #     nn.BatchNorm1d(self.hidden_dim//64),
        #     nn.LeakyReLU(inplace=True),
        #     )
            # nn.ConvTranspose1d(self.hidden_dim*8,573,4),
            # nn.BatchNorm1d(573) 
        self.output = nn.Sequential(
            # nn.Linear(1025,573),
            # nn.BatchNorm1d(573)
            nn.Tanh()
            )
    def forward(self,noise,labels):
        # print(self.label_embedding(labels))

        # z = torch.cat((self.label_embedding(labels),noise), -1).unsqueeze(1)
        # z = noise.unsqueeze(1) # test without label embedding (no possibility of generating different classes)
        # print(z.float().size())
        # x = self.input(z.float())
        # print(x.size())
        x = noise.unsqueeze(2)
        x = self.conv1(x)  
        # print(x.size())
        x = self.conv2(x)  
        # print(x.size())
        x = self.conv3(x)  
        # # print(x.size())
        x = self.conv4(x)  
        # print(x.size())
        x = self.conv5(x)  
        # print(x.size())
        # x = self.conv6(x)
        # print(x.size())
        # x = self.output(x.view(x.size(0),-1))
        # print(x.size())
        # x = self.output(x)
        return x.squeeze(1)
     

    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1).unsqueeze(2)
        x = self.conv(z).squeeze(2)  
        return x
    
class cDC_Generator(nn.Module):    #including tanh
    def __init__(self,opt):
        super(cDC_Generator, self).__init__()

        self.latent_dim = opt.latent_dim
        self.classes=opt.numclasses

        self.label_embedding = nn.Embedding(self.classes,self.classes)

        self.hidden_dim = 64

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(opt.latent_dim+opt.numclasses,self.hidden_dim*8,4,1,0),
            nn.BatchNorm1d(self.hidden_dim *8),
            nn.LeakyReLU(inplace=True),
            #(64x8) x 4 = 2048
            nn.ConvTranspose1d(self.hidden_dim *8,self.hidden_dim *4,4,2,1),
            nn.BatchNorm1d(self.hidden_dim*4),
            nn.LeakyReLU(inplace=True),
            #(64x4) x 8 = 2048
            nn.ConvTranspose1d(self.hidden_dim *4,self.hidden_dim *2,4,2,1),
            nn.BatchNorm1d(self.hidden_dim *2),
            nn.LeakyReLU(inplace=True),
            #(64x2) x 16 = 2048
            nn.ConvTranspose1d(self.hidden_dim*2,self.hidden_dim,4,2,1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            # (64x1) x 32 = 2048
            # nn.ConvTranspose1d(self.hidden_dim*1,573,4,2,1),
            # nn.BatchNorm1d(573),
            # nn.LeakyReLU(inplace=True),
            #573 x 1 x 64
            nn.Conv1d(self.hidden_dim,573,32),
            nn.BatchNorm1d(573),
            nn.Tanh() 
        )
    def forward(self,noise,labels):
        z = torch.cat((self.label_embedding(labels),noise), -1).unsqueeze(2)
        x = self.conv(z).squeeze(2)  
        return x

