# Here is the code for the recommender system.  This consists of a deep auto-encoder with a
#'fit' and 'predict' method like sklearn.
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle as pkl
import torch.autograd as autograd
import torch.nn as nn

#Below we have a helper function just so we don't have to initialize the create_recsys
#dimensions by hand.

def create_recsys(matrix,dropout=.1,latent_features=4,max_iter=100,lr=.001,epochs=5,temperature=1,batch_size=500):
    return recsys(matrix,matrix.shape[0],matrix.shape[1],latent_features,dropout,max_iter,epochs,temperature,lr,\
                 batch_size=batch_size)


class recsys(nn.Module):


    def __init__(self,ratings=None,users=100,sites=1000,latent_features=12,\
                 dropout=.1,max_iter=10,epochs=4,temperature=1,lr=.01,batch_size=500,\
                 losses=None):

        super(recsys,self).__init__()


        self.users=users
        self.sites=sites
        self.dropout=nn.Dropout(p=dropout)
        self.max_iter=max_iter
        self.lr=lr
        self.batch_size=batch_size
        self.temperature=temperature
        self.ratings=ratings
        # self.mask=torch.tensor(np.logical_not(np.isnan(ratings)).astype(int)).type(torch.ByteTensor)
        self.losses=None
        self.epochs=epochs
        self.linear1=nn.Linear(sites,latent_features)
        self.linear2=nn.Linear(latent_features,latent_features)
        self.linear3=nn.Linear(latent_features,sites)


    # The input x should have shape (number_of_users,sites)
    # Ratings needs to be a torch tensor of the same shape as x.


#     def get_mask(self,ratings=None):

#         try:
#             if ratings==None:
#                 pass
#         except:
#             self.ratings=ratings

#         mask=[]
#         for i in range(len(self.ratings)):
#             mask.append([0 if math.isnan(self.ratings[i,j]) else 1 for j in range(len(self.ratings[0]))])
#         return torch.tensor(mask)



    def imputer(self,x=None):
        #Need to make a function which takes in a ratings array and returns
        #an initial best guess.  For now I'll just mask the unkown variables
        #print(type(self.ratings))

        try:
            if x==None:
                ratings=self.ratings
        except:
            ratings=x


        ratings[np.isnan(ratings)] = 0
        return torch.tensor(ratings).float()



    def forward(self,x):
        # print("Hola", flush=True)
        x=self.imputer(x)
        # print("Ni hao", flush=True)
        x=self.linear1(x.float())
        # print("Bonjour", flush=True)
        x=torch.tanh(x)
        # print("Konnchi wa", flush=True)
        x=self.linear2(x.float())
        # print("Ohio gayzomouse", flush=True)
        # x=self.dropout(x.float())
        # print("Anyoung", flush=True)
        x=torch.tanh(x)
        # print("Ki ora", flush=True)
        x=self.linear3(x.float())
        # print("g'day", flush=True)
        return x


    def custom_loss(self,x,y):
        ct=0
        # mask = [not x for x in  np.isnan(ratings)]
        for i in range(len(x)):
            if (torch.norm(x[i])==0) or (torch.norm(y[i])==0):
                pass
            else:

                ct+=1-(x[i]@y[i])/(torch.norm(x[i])*torch.norm(y[i]))
                # ct+=1-(x[i][mask]@y[i][mask])/(torch.norm(x[i][mask])*torch.norm(y[i][mask]))
        return ct/len(x)


    def custom_loss2(self,x,y):
        return 1-(x@y)/(torch.norm(x)*torch.norm(y))



    def predict(self,x):
        # print("About to impute", flush = True)
        x=self.imputer(x)
        # print("Imputed", flush=True)
        return self.forward(x)

    def fit(self,ratings=None):

        try:
            if ratings==None:
                ratings=self.ratings
        except:
            pass

        ratings_clean=self.imputer(ratings)
        loss_function=nn.MSELoss()




        f= open('raw_data/losses','w+')

        losses=[]

        for i in range(1,self.epochs+1):

            optimizer = optim.Adam(self.parameters(),lr=self.lr/i)

            print(f'Epoch {i}')

            sample_indices=np.random.choice(range(len(ratings_clean)),self.batch_size,replace=False)
            sample=ratings_clean[sample_indices]
            # new_mask=self.mask[sample_indices]

            #print(sample_indices)

            for _ in range(self.max_iter):
                optimizer.zero_grad()
                out = self.forward(sample)
                # out_mask=torch.masked_select(out,new_mask)
                # sample_mask=torch.masked_select(sample,new_mask)
                # print("here is the type of out_mask",flush=True)
                # print(type(out_mask),flush=True)
                # print(sample_mask)
                # print(sample_mask.shape,flush=True)
                # print(out_mask.shape,flush=True)
                # print("that was it",flush=True)
                #out = self.forward(ratings_clean)
                #loss = loss_function(out,ratings_clean) #This one works!
                # loss = self.custom_loss2(out_mask,sample_mask)  #This one works
                loss=self.custom_loss(out,sample)
                #loss = self.custom_loss(out,ratings_clean)
                losses.append(float(loss.detach().numpy()))
                f.write(str(loss.detach().numpy())+',')
                loss.backward(retain_graph=True)

                optimizer.step()
            self.losses=losses
            self.lr=.7*self.lr
        f.close()
