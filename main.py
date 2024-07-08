from utils import * #all functions and other imports are in here!

def configure():
    """Argument parser, all variables are set to a default unless changed
    This function creates the opt class containing the complete experiment configuration"""
    parser = argparse.ArgumentParser(description="Set variables")
    # MAIN arguments
    parser.add_argument('--device',type=str, default='cuda:0',choices=['cpu','cuda:0','cuda:1','cuda:2'],help='Choose device for computing',required=False)
    #Data
    parser.add_argument('--create_split',type=bool,default=False,help='Load existing labeled data or create new train test split',required=False)
    parser.add_argument('--datafile',type=str,default='data/class01234.npz',help="Name of data file",required=False)
    parser.add_argument('--testsplit',type=float,default=0.2,help='Testsplit if a new traintestsplit is created!',required=False)
    parser.add_argument('--limitmajor',type=int,default=100000,help='maximum number of samples from 1 class used for training',required=False)
    parser.add_argument('--limitclasses',type=int,default=[],nargs='+',required=False,help="Select classes to be used for experiment")
    parser.add_argument('--numclasses',type=int,default=0,help="Number of distinct classes, gets set automatically",required=False)
    parser.add_argument("--scalemethod",type=str,default="standard",choices=["standard","minmaxf","minmax","1/3standard"],help="method for normalizing data",required=False)
    #activate steps #dont set bool type, since bool("False")=True
    parser.add_argument('--train_gans',type=str2bool,default=False,help="Train new GANs (Switch)")
    parser.add_argument('--evaluate_gans',type=str2bool,default=False,help="Evaluate pretrained GANs (Switch)")
    parser.add_argument('--trainclass_baseline',type=str2bool,default=False,help="Fit classifier on training data(Switch)(Switch)")
    parser.add_argument('--trainclass_SMOTE',type=str2bool,default=False,help="Fit classifier on SMOTE-oversampled data(Switch)")
    parser.add_argument('--trainclass_gans',type=str2bool,default=False,help="Fit classifier on GAN-oversampled data(Switch)") 
    parser.add_argument('--trainclass_augm',type=str2bool,default=False,help="Fit classifier on augmentation-oversampled data(Switch)")
    # Classifier arguments,
    # parser.add_argument('--smote',type=str2bool,default=True,help="")
    parser.add_argument('--CLASSlr',type=float,default=0.0001,help='Learning rate for the classifier',required=False)
    parser.add_argument('--CLASSbsize',type=int,default=32,help='Batch size for the classifier',required=False)
    parser.add_argument('--saveclass',default="models/ffclass/",help="Name of folder in which the Classifier will be saved",required=False)
    parser.add_argument('--CLASSn_epochs',type=int,default=10,help='Number of training iterations for classifier training',required=False)
    # GAN arguments
    parser.add_argument('--GANslr',type=float,default=0.0001,required=False,help="Learning rate for the GANs")
    parser.add_argument('--b1',type=float,default=0.5,required=False,help="First momentum term for Adam")
    parser.add_argument('--b2',type=float,default=0.9,required=False,help="Second momentum term for Adam")
    parser.add_argument('--GANsbsize',type=int,default=64,required=False,help="Batchsize for training GANs")
    parser.add_argument("--latent_dim",type=int,default=100,help="Dimension of latent vector",required=False)
    parser.add_argument("--GANs_n_epochs",type=int,default=100,help="Number of GANs training steps",required=False)
    parser.add_argument("--savegans",default="models/ffgans/",help="Name of folder in which the GANs progress will be saved",required=False)
    parser.add_argument("--removeproggan",type=str2bool,default=True,help="Removing all generators and discriminators except final ones",required=False)
    parser.add_argument("--optimizer",type=str,default="ADAM",help="Optimizer used for gan training",required=False)
    parser.add_argument("--loss",type=str,default="BCE",help="Lossfunction used for gan training",required=False)
    parser.add_argument("--ganmodel",type=str,default="cwgan_fc",help="Generator model selection",choices=["cwgan_fc","cGAN_fc","cwgan_fc_4","cGAN_fc_4"],required=False)
    parser.add_argument("--specsize",default=512,type=int,help="Length of spectra, will be set automatically",required=False)
    parser.add_argument("--WGAN_gp",default=True,help="If true uses WGAN-gp",required=False)
    parser.add_argument("--discsteps",type=int,default=100,help="#discriminator steps per generator step",required=False)
    parser.add_argument("--lambda_gp",default=10,type=int,help="gradient penalty modifier for wgan-gp",required=False)
    parser.add_argument("--dynamicdiscstep",default=False,required=False,help="decides if generator step iff discriminator accuarcy is high")
    opt=parser.parse_args()
    
    # check if cuda is available and set device correctly
    if opt.device == "cpu":
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device((opt.device) if torch.cuda.is_available() else "cpu")

    if opt.loss == "BCE":
        opt.loss = torch.nn.BCELoss()
    else:
        opt.loss = torch.nn.MSELoss()
    return opt


def train_gans(opt,X_train,y_train):
    #setup
    gan_train_loader = torch.utils.data.DataLoader(dataloader(np.transpose(X_train),torch.Tensor(y_train),opt),batch_size=opt.GANsbsize,num_workers=0,shuffle=True)
    # gan_test_loader =  torch.utils.data.DataLoader(dataloader(np.transpose(np.expand_dims(X_test,axis=2),opt),torch.Tensor(y_test)),batch_size=opt.GANsbsize,num_workers=0,shuffle=True)
    
    #select the generator and discriminator 
    if opt.ganmodel == "cGAN":
        generator = cGANgenerator_3(opt)
        discriminator = cGANdiscriminator_3(opt)
    elif opt.ganmodel == "cwgan_fc":
        generator = cGANgenerator_fc_3(opt)
        discriminator = cWGANdiscriminator_3(opt)
        # discriminator = cDCWGANdiscriminator_3(opt)
    elif opt.ganmodel == "cwgan_fc_4":
        generator = cGANgenerator_fc_4(opt)
        discriminator = cWGANdiscriminator_4(opt)
    elif opt.ganmodel == "cGAN_fc":
        generator = cGANgenerator_fc_3(opt)
        discriminator = cGANdiscriminator_3(opt)
    elif opt.ganmodel == "cGAN_fc_4":
        generator = cGANgenerator_fc_4(opt)
        discriminator = cGANdiscriminator_4(opt)
    elif opt.ganmodel == "cDC_GAN_fc":
        generator = cDC_Generator_fc_new(opt)
        discriminator = cDC_Discriminator(opt)
    elif opt.ganmodel == "cDC_GAN":
        generator = cDC_Generator(opt)
        discriminator = cDC_Discriminator(opt)
    # generator = cDCGANgenerator(opt)
    # discriminator = 

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    generator.to(opt.device)
    discriminator.to(opt.device)
    class optimizers():
        pass
    if opt.optimizer=="SGD":
        optimizers.optimizer_G = torch.optim.SGD(generator.parameters(),lr=opt.GANslr*int(opt.discsteps))#,betas=(opt.b1,opt.b2))
        optimizers.optimizer_D = torch.optim.SGD(discriminator.parameters(),lr=opt.GANslr)#,betas=(opt.b1,opt.b2))
    elif opt.optimizer=="ADAM":
        optimizers.optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.GANslr*int(opt.discsteps),betas=(opt.b1,opt.b2))
        optimizers.optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=opt.GANslr,betas=(opt.b1,opt.b2))    
    # adverloss = torch.nn.MSELoss()
    adverloss = opt.loss
    train_GANs(opt,gan_train_loader,generator,discriminator,optimizers,adverloss,savedir=opt.savegans)

def evaluate_gans(opt,X_test,y_test,X_train,y_train):
    # gan_test_loader =  torch.utils.data.DataLoader(dataloader(np.transpose(np.expand_dims(X_test,axis=2)),torch.Tensor(y_test)),batch_size=opt.GANsbsize,num_workers=0,shuffle=True)
    # X_test=np.transpose(np.expand_dims(X_test,axis=2),(0,2,1))

    eval_gan_training(opt,X_test,y_test,X_train,y_train)

def train_classifier(opt,X_train,y_train,X_test,y_test,method="LDA"):
    """Only LDA implemented, this function fits an LDA classifier and tests it on the test set
    A summary of the results is given in text file and a confusionmatrix"""    

    def fit_LDA(opt,X_train,y_train,X_test,y_test):
        print("creating LDA")
        LDA = LinearDiscriminantAnalysis()
        LDA.fit(X_train,y_train)

        y_pred=LDA.predict(X_test)

        if opt.saveclass:
            path = "models/"+opt.saveclass
            if not os.path.exists(path):
                os.makedirs(path)
        create_summary([0,0],opt,0,path,y_pred,y_test)

        confusionmatrix(path,y_pred,y_test)

    if method=="LDA":
        fit_LDA(opt,X_train,y_train,X_test,y_test)
    # elif method == "RESNET": #method removed in this version
    #     train_resnet(opt,X_train,y_train,X_test,y_test)
    else:
        print("unknown classifier selected!")

def train_aug_class(opt,X_train,y_train,X_test,y_test):
    """Create a classifier using augmented oversampling"""
    oversampler = augmented_oversampling()
    X_train,y_train = oversampler.data_augmentation(X_train,y_train)
    saveclass = opt.saveclass
    if opt.saveclass:
        opt.saveclass = str(opt.saveclass+"AUG/")
    train_classifier(opt,X_train,y_train,X_test,y_test)
    opt.saveclass = saveclass

def train_smote_classifier(opt,X_train,y_train,X_test,y_test):
    """Create a classifier using smote oversampling"""
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train,y_train)
    #train classifier
    saveclass = opt.saveclass
    if opt.saveclass:
        opt.saveclass = str(opt.saveclass+"SMOTE/")
    train_classifier(opt,X_train,y_train,X_test,y_test)
    opt.saveclass = saveclass

def train_gans_classifier(opt,X_train,y_train,X_test,y_test):
    """Create a classifier using gans oversampling"""

    oversampler = GAN_resampler()
    X_train,y_train = oversampler.resample(opt,X_train,y_train)

    saveclass = opt.saveclass
    if opt.saveclass:
        opt.saveclass = str(opt.saveclass+"GANs/")
    train_classifier(opt,X_train,y_train,X_test,y_test)
    opt.saveclass = saveclass

def main(opt):
    """"Main workflow"""
    if opt.create_split:
        create_labeled_dataf(opt)
    #no majority undersampling implemented
    X_train,y_train,X_test,y_test = load_labeled_data(opt.datafile)
    opt.specsize = np.shape(X_train)[1]
    opt.numclasses=len(set(y_train))

    # create dataloader objects, NOTE, this is before oversampling!
    # train GANS
    if opt.train_gans:
        print("Training GANs...")
        train_gans(opt,X_train,y_train)
    
    # evaluate GANs
    if opt.evaluate_gans:
        # opt.device='cpu'
        print("Evaluate GANs")
        evaluate_gans(opt,X_test,y_test,X_train,y_train)
        print("TODO?")

    # train baseline classifier with no data augmentation
    if opt.trainclass_baseline:
        print("Training baseline classifier...")
        train_classifier(opt,X_train,y_train,X_test,y_test)

    if opt.trainclass_augm:
        print("Training classifier with data augmentation...")
        train_aug_class(opt,X_train,y_train,X_test,y_test)
    # train SMOTE classifier
    if opt.trainclass_SMOTE:
        print("Training classifier with SMOTE...")
        train_smote_classifier(opt,X_train,y_train,X_test,y_test)

     # train GANs classifier
    if opt.trainclass_gans:
        print("Training classifier with GAN-generated data...")
        train_gans_classifier(opt,X_train,y_train,X_test,y_test)        

if __name__=='__main__':
    opt = configure()
    # print(type(opt.trainclass_baseline),type(opt.trainclass_gans),opt.trainclass_SMOTE)
    main(opt)