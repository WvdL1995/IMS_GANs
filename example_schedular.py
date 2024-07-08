import subprocess

# example run

# devided into 2 subprocesses
# the first process only generates the datafile
# the second process does all steps: training and evaluating gans, creating baseline and oversampled classifers
# using this method an additional subproces can be added that uses the same datafile

task = subprocess.run(["python","main.py",
                            "--create_split=True",
                            "--datafile=data/example_5_classes.npz",
                            "--limitclasses",' 0',' 1',' 2',' 6',' 7',
                            "--limitmajor=5000",
                            "--specsize=50",
                            "--scalemethod=standard"])
task1 = subprocess.run(["python","main.py",
                            "--datafile=data/example_5_classes.npz",
                                "--trainclass_baseline=True",
                                "--trainclass_SMOTE=True",
                                "--trainclass_augm=True",
                                "--saveclass=example_5_classes/",
                                "--train_gans=True",
                                "--trainclass_gans=True",
                                "--evaluate_gans=True",
                                "--savegans=models/example_5_classes/gans/",
                                "--GANs_n_epochs=1000",
                                "--GANslr=0.0001",
                                "--GANsbsiz=64", 
                                "--optimizer=ADAM",
                                "--latent_dim=100",
                                "--device=cuda:0",
                                "--ganmodel=cwgan_fc",
                                "--WGAN_gp=True",
                                "--discsteps=100"
                                ])

        # task2 = subprocess.run(["python","main.py",
        #                             "--datafile=data/xclass_5000_standard_nmf"+str(i)+"_"+str(j)+".npz",
        #                                 "--trainclass_baseline=True",
        #                                 "--trainclass_SMOTE=True",
        #                                 "--trainclass_augm=True",
        #                                 "--saveclass=exp_maj_cgan_nmf_c_"+str(i)+"_"+str(j)+"/",
        #                                 "--train_gans=True",
        #                                 "--trainclass_gans=True",
        #                                 "--evaluate_gans=True",
        #                                 "--savegans=models/exp_maj_cgan_nmf_c_"+str(i)+"_"+str(j)+"/gans/",
        #                                 "--GANs_n_epochs=500",
        #                                 "--GANslr=0.0002",
        #                                 "--GANsbsiz=64", 
        #                                 "--optimizer=ADAM",
        #                                 "--latent_dim=100",
        #                                 "--device=cuda:0",
        #                                 "--ganmodel=cGAN_fc",
        #                                 "--WGAN_gp=False",
        #                                 "--discsteps=1"
        #                                 ])
