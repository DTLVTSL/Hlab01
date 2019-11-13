# -*- coding: utf-8 -*-
#Linear Least Squares solution
#stochastic gradient algorithm using adam method
#conjugate gradient algorithm
#ridge regression (optimize λ)

#ADAM
#alpha. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
#beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
#beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
#epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).
#more infomation on the 
#https://hackernoon.com/implementing-different-variants-of-gradient-descent-optimization-algorithm-in-python-using-numpy-809e7ab3bab4
#https://github.com/Niranjankumar-c/GradientDescent_Implementation/blob/master/VectorisedGDAlgorithms.ipynb
#https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
#https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


#to do
#fix gradient
#put the error comparison curves
#



import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use('classic')

def normalz(dataset):#Normalizing entire dataframe but not few columns.
    dataNorm=(dataset-data_train.mean())/np.sqrt(data_train.std())
    dataNorm["subject#"]=dataset["subject#"]  #we are not nomalizing subject because we ae not using on the regression
    #dataNorm["age"]=dataset["age"] 
    #dataNorm["sex"]=dataset["sex"]
    dataNorm["test_time"]=dataset["test_time"] #we are not using test time on the regression
    #dataNorm["motor_UPDRS"]=dataset["motor_UPDRS"]
    #dataNorm["total_UPDRS"]=dataset["total_UPDRS"]
    return dataNorm

#Linear Least Squares solution with pseudoinvese
def SolveLLS(y,A,yT,AT,yV,AV):   # method SolveLLS (Y train, x train, y test, x test ,y validation x validation)  
    w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)  #pseudoinverse calculation
    yhat_train = np.dot(A,w)    #estimated y train UPDRS values based on calculated w
    yhat_test = np.dot(AT,w)    #estimated y test UPDS values based on calculated w
    yhat_val = np.dot(AV,w)     #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm= (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()  #transfom normalized in non normalized values
    yhat_test_nonnorm=  (np.array(yhat_test) * np.sqrt(y_tn.std()))+y_tn.mean()   #transfom normalized in non normalized values
    yhat_val_nonnorm=  (np.array(yhat_val) * np.sqrt(y_tn.std()))+y_tn.mean()     #transform normalized in non normalized values 
    y_train_nonnorm= (np.array(y) * np.sqrt(y_tn.std()))+y_tn.mean()              #transform normalized in non normalized values   
    e_train = (np.dot(A,w)-y) #error(estimated ytrain - original ytrain)
    e_test = (np.dot(AT,w)-yT)#error (estimated ytest - original ytest)
    e_val = (np.dot(AV,w)-yV) #error (estimated yvalidation - original yvalidation)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) #minimum square error on train 
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)#minimum square error on test
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  #minimum square eor on validation
    
    #error histogram for LLS
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for LLS, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    print ("LLS error TRAIN", mse_train)  
    print ("LLS error TEST", mse_test)
    print ("LLS error VALIDATION", mse_val)

    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="LLS", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="LLS", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="LLS", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="LLS", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    


    plt.figure(figsize=(13,6))
    plt.hist(y_train_nonnorm, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train_nonnorm, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for LLS")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(y_train_nonnorm, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test_nonnorm, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for LLS")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(y_train_nonnorm, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_val_nonnorm, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for LLS")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2) 
      
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for LLS")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)
  
    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction for LLS")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w
#gradient algorithm
def SolveGrad(y,A,yT,AT,yV,AV):#method to solve conjugated gradient (Y train, x train, y test, x test ,y validation x validation)  
    max_iterations = 100   #maximum and defined number of iterations 
    iterations = 0
    gamma=1.0e-8            #gamma or
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w=np.random.rand(Nf,1)# random initialization of w
    grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))
    GRADe_historyTRAIN = []
    GRADe_historyVAL = []
    GRADe_historyTEST = []
    e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
    e_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    e_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)
    
    for iterations in range(max_iterations):
        grad = 2 * np.dot(A.T,(np.dot(A,w)-y))
        w2 = w - gamma*grad
        if np.linalg.norm(w2-w) < 1e-4:
            w = w2
            break
        w=w2
        GRADe_historyTRAIN += [e_train] 
        GRADe_historyTEST += [e_test] 
        GRADe_historyVAL += [e_test]                                   
        e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
        e_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
        e_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)
    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(A,w)-y)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  
    
   
    #error histogram for gradient
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for GRAD, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    print ("GRAD error TRAIN", mse_train)  
    print ("GRAD error TEST", mse_test)
    print ("GRAD error VALIDATION", mse_val)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="GRAD", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="GRAD", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="GRAD", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="GRAD", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for GRAD")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for GRAD")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for GRAD")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction for GRAD")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w
 
def SteepDesc(y,A,yT,AT,yV,AV):
    max_iterations = 100 #maximum number of iterations defined
    Nf=A.shape[1] # number of columns 
    w=np.zeros((Nf,1),dtype=float) # column vector w 
    w=np.random.rand(Nf,1)# random initialization of w
    iterations = 0
    e_history = []
    e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    while iterations < max_iterations:
        iterations += 1
        e_history += [e_train]
        e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
        grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))
        hes= 4*np.dot(A.T,A)
        learncof=(np.linalg.norm(grad)**2)/(np.dot((np.dot(grad.T,hes)),grad))
        w=w-(learncof*grad)
        #err[it,1]=np.linalg.norm(np.dot(A,w)-y)
   
    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(A,w)-y)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  
  
    #error histogram for SteepDescend
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for SteepDesc, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    print ("SteepDesc error TRAIN", mse_train)  
    print ("SteepDesc error TEST", mse_test)
    print ("SteepDesc error VALIDATION", mse_val)  

    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="SteepDesc", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="SteepDesc", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="SteepDesc", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="SteepDesc", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for SteepDesc")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for SteepDesc")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction SteepDesc")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)
  
    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction SteepDesc")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w

def stochastic(A,yT,AT,yV,AV): #method to sollve by stochastic gradient
    max_iterations = 100  #defined maximum number of iterations
    learning_coefficient = 1.0e-05 #learning coeficient 
    Nf=A.shape[1]-1 # number of columns used to create a initial random w matrix
    w=np.random.rand(Nf,1) #random initialization of w
    y=A[F0].to_numpy().reshape(-1, 1) #convert from list to numpy vector
    y_train=A[F0].to_numpy().reshape(-1, 1) #convert from list to numpy vector
    X_train=A.drop(columns=[F0]).to_numpy() #remove F0 from train X, F0 vaiable to estimated
    e_train = (np.linalg.norm(np.dot(X_train,w) - y_train)**2)/len(y)
    #e_train = np.linalg.norm(X_train.dot(w) - y_train)**2
    iterations = 0  #start iterations from 0
    e_history = [e_train]  #create a list of historical values of train error
    batch_size = 10  # size of used batches
    shuffled = shuffle(A) #shuffle X train again
    while iterations < max_iterations: # do it for the total of defined iterations
        iterations += 1  #increase the counter of iterations
        shuffled = shuffle(A) # shuffle the X train again
        y_train = shuffled[F0].to_numpy().reshape(-1, 1)  #shuffle y train and convert to a numpy vector, this because i dont know how to shuffle using numpy
        X_train = shuffled.drop(columns=[F0]).to_numpy()  #shuffle X train and convert to a numpy vector, this because i dont know how to shuffle using numpy
        batch_prev = 0  #create the batch prev variable
        for batch in range(batch_size, len(X_train), batch_size): #do it for batches ,spliting the dataset in batches
            X_batch = X_train[batch_prev:batch]  # took batch size from X
            y_batch = y_train[batch_prev:batch] #took batch size from y
            #batch_gradient = 2*np.dot(X_batch.T,(np.dot(X_batch,w)-y_batch))
            batch_gradient=-2 * X_batch.T.dot(y_batch) + 2 * X_batch.T.dot(X_batch).dot(w) #calculate gradient 
            w = w - (learning_coefficient * batch_gradient) #w - (step size = gradient * learning rate)
            batch_prev = batch
        e_train =(np.linalg.norm(np.dot(X_train,w) - y_train)**2)/len(y) #calculate the error
        e_history += [e_train]

    y_train=A[F0].to_numpy().reshape(-1, 1)
    X_train=A.drop(columns=[F0]).to_numpy()
    yhat_train = np.dot(X_train,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(X_train,w)-y_train)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(X_train,w)-y_train)**2)/len(y_train)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  

    #error histogram for LLS
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for Stochastic, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    print ("Stochastic error TRAIN", mse_train)  
    print ("Stochastic error TEST", mse_test)
    print ("Stochastic error VALIDATION", mse_val)  

    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="stochastic", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    
        
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="stochastic", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for stochastic")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for stochastic")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:200], color="black", label="yhat_train")
    plt.plot(y_tn[:200], label="Y train")
    plt.title("Train prediction stochastic")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)   
    return w


def SteepDescADAM(y,A,yT,AT,yV,AV):
    alfa = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.00000001
    m = 0
    v = 0
    max_iterations = 1000
    Nf=A.shape[1] # number of columns
    w=np.zeros((Nf,1),dtype=float) # column vector w 
    w=np.random.rand(Nf,1)# random initialization of w
    iterations = 0
    e_history = []
    e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
   
    """
    for t in range(num_iterations):
    g = compute_gradient(x, y)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
    """
        
    while iterations < max_iterations:
        iterations += 1
        e_history += [e_train]
        e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
        grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))
        m= beta1 * m +(1-beta1)*grad
        v = beta2 * v +(1+beta2)*np.power(grad,2)
        m_hat = m / (1 - np.power(beta1, iterations))
        v_hat = v / (1 - np.power(beta2, iterations))
        w = w - alfa* m_hat / (np.sqrt(v_hat) + epsilon)

        #err[it,1]=np.linalg.norm(np.dot(A,w)-y)
   
    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(A,w)-y)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  
  
    #error histogram for LLS
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for SteepDescADAM, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    print ("SteepDescADAM error TRAIN", mse_train)  
    print ("SteepDescADAM error TEST", mse_test)
    print ("SteepDescADAM error VALIDATION", mse_val)  

    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="SteepDescADAM", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="SteepDescADAM", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="SteepDescADAM", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="SteepDescADAM", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for SteepDescADAM")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for SteepDescADAM")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction SteepDescADAM")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)
  
    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction SteepDescADAM")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w


def ridge(y,A,yT,AT,yV,AV): 
    max_iterations = 100
    gamma=1.0e-8
    lbda= 1.0e-5
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w=np.random.rand(Nf,1)# random initialization of w
    #e_train = (np.linalg.norm(A.dot(w) - y)**2)+(lbda*(np.linalg.norm(w)**2))
    e_train = (np.linalg.norm(np.dot(A,w) - y)**2)/len(y)
    grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w)) + 2*(lbda*w)
    iterations = 0
    e_history = []

    while np.linalg.norm(w-a_prev) > 1e-8 and iterations < max_iterations:
        iterations += 1
        #print iterations, np.linalg.norm(self.a-a_prev), self.e_train
        e_history += [e_train]                                           
        a_prev = w
        w = w - gamma * grad
        e_train = (np.linalg.norm(np.dot(A,w) - y)**2)/len(y)
        grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w)) + 2*(lbda*w)        

    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(A,w)-y)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  


    print ("ridge error TRAIN", mse_train)  
    print ("ridge error TEST", mse_test)
    print ("ridge error VALIDATION", mse_val)  
    #error histogram for LLS
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for Ridge, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="ridge", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="ridge", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="ridge", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="ridge", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for ridge")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for ridge")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for ridge")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction for ridge")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w



def ConjugateGRAD(y,A,yT,AT,yV,AV): 
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w=np.random.rand(Nf,1)# random initialization of w
    #e_train = (np.linalg.norm(A.dot(w) - y)**2)+(lbda*(np.linalg.norm(w)**2))
    e_train = (np.linalg.norm(np.dot(A,w) - y)**2)/len(y)
    iterations = 0
    e_history = []
    b=2*np.dot(A.T,y)
    d=b
    g=-b
    Q=2*np.dot(A.T,A)

    for it in range(Nf):  # Iterations on number of features
        alpha = -((np.dot(d.T,g))/(np.dot(np.dot(d.T,Q),d)))
        w = w + alpha*d
        #g = np.dot(Q,w) - b
        g = g + alpha*(np.dot(Q,d))
        beta = np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
        d = -g + beta*d
        # Errors on de-standardized vectors.


    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(y_tn.std()))+y_tn.mean() 
    e_train = (np.dot(A,w)-y)
    e_test = (np.dot(AT,w)-yT)
    e_val = (np.dot(AV,w)-yV)
    mse_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y)
    mse_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    mse_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)  


    print ("Conjugated GRAD error TRAIN", mse_train)  
    print ("Conjugated GRAD error TEST", mse_test)
    print ("Conjugated GRAD error VALIDATION", mse_val)  
    #error histogram for LLS
    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for Conjugated GRAD, Train, Test, Validation")
    plt.xlabel("error")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.scatter(y,yhat_train, label="Conjugated GRAD", marker="o", color="green", alpha=0.3)
    plt.plot(y,y,color="black", linewidth=0.4)
    plt.title("Train - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tn,yhat_train_nonnorm, label="Conjugated GRAD", marker="o", color="green", alpha=0.3)
    plt.plot(y_tn,y_tn,color="black", linewidth=0.4)
    plt.title("Train non normalized - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.scatter(yT,yhat_test, label="Conjugated GRAD", marker="o", color="orange", alpha=0.3)
    plt.plot(yT,yT,color="black", linewidth=0.4)
    plt.title("Test - y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)     
    
    plt.figure(figsize=(13,6))
    plt.scatter(y_tt,yhat_test_nonnorm, label="Conjugated GRAD", marker="o", color="orange", alpha=0.3)
    plt.plot(y_tt,y_tt,color="black", linewidth=0.4)
    plt.title("Test non normalized- y_true VS y_hat")
    plt.xlabel("y_true")
    plt.ylabel("y_hat")
    plt.legend(loc=2)    

    plt.figure(figsize=(13,6))
    plt.hist(y, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_train, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Train -histogram for Conjugated GRAD")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.hist(yT, bins=50, label="Y_train", alpha=0.4)
    plt.hist(yhat_test, bins=50, label="Yhat_train", alpha=0.4)
    plt.title("Test - histogram for Conjugated GRAD")
    plt.xlabel("F0")
    plt.ylabel("Occurrencies")
    plt.legend(loc=2)
    
    plt.figure(figsize=(13,6))
    plt.plot(yhat_test_nonnorm[:100], color="black", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for Conjugated GRAD")
    plt.xlabel("Sample index")
    plt.ylabel("Original value")
    plt.legend(loc=2)

    plt.figure(figsize=(13,6))
    plt.plot(yhat_train_nonnorm [:100], color="black", label="yhat_train")
    plt.plot(y_tn[:100], label="Y train")
    plt.title("Train prediction for Conjugated GRAD")
    plt.xlabel("Sample index")
    plt.ylabel("Original values")
    plt.legend(loc=2)    
    return w


if __name__ == "__main__":  

    plt.style.use('ggplot')
    np.random.seed (30)
    df = pd.read_csv("parkinsons_updrs.data")
    #df.test_time = df.test_time.apply(np.abs)
    #df["day"] = df.test_time.astype(np.int64)
    #df = df.groupby(["subject#", "day"]).mean()
    pd = shuffle(df)
    pd = shuffle(pd)
    pd = shuffle(pd)
    pd = shuffle(pd)
    total_rows = len(pd)
    wLLS=[]
    #subject,age,sex,test_time,motor_UPDRS,total_UPDRS,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE
    data_train = pd.iloc[:int(total_rows/2)]  #total 5875 rows - 2937 rows selected for training 
    data_val = pd.iloc[int(total_rows/2)+1:int((3*total_rows)/4)]
    data_test = pd.iloc[int((3*total_rows)/4)+1:total_rows+1]
    #print(len(data_train),len(data_val),len(data_test),len(data_train)+len(data_val)+len(data_test))
    data_train_norm = normalz(data_train)
    data_val_norm = normalz(data_val)
    data_test_norm = normalz(data_test)
   
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=0
    for i in range(2):
        for j in range(3):
            df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=7
    for i in range(2):
        for j in range(3):
            df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=13
    for i in range(2):
        for j in range(3):
            df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=19
    for i in range(1):
        for j in range(3):
            df.hist(column = df.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1

#After normatization
            
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=0
    for i in range(2):
        for j in range(3):
            data_train_norm.hist(column = data_train_norm.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=7
    for i in range(2):
        for j in range(3):
            data_train_norm.hist(column = data_train_norm.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=13
    for i in range(2):
        for j in range(3):
            data_train_norm.hist(column = data_train_norm.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    fig, ax = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.15)
    plt.margins(0.2)
    plt.xticks(rotation=90)
    fig.tight_layout()
    m=19
    for i in range(1):
        for j in range(3):
            data_train_norm.hist(column = data_train_norm.columns[m], bins = 12, ax=ax[i,j], figsize=(20, 18))
            m+=1
    
    #print(data_val)
    #print(data_val_norm)
    #print (data_test_norm.describe())
    #print (data_val_norm["Jitter(%)"].mean())
    #define F0 or implement a combination automatically TBD
    F0 = "total_UPDRS"
    tn_stoch = data_train_norm.drop(columns=["subject#","test_time"])
    y_train = data_train_norm[F0] #collum vector with the feature to be estimated
    X_train =data_train_norm.drop(columns=["subject#","test_time",F0])
    X_test = data_test_norm.drop(columns=["subject#","test_time",F0])       #data test norm by removing column F0
    X_val = data_val_norm.drop(columns=["subject#","test_time",F0])         #data val norm by removing column F0
    y_test=data_test_norm[F0]     #data test norm column F0
    y_val=data_val_norm[F0]       #data val norm column F0

    X_tn =data_train_norm.drop(columns=["subject#","test_time",F0])
    X_tt = data_test_norm.drop(columns=["subject#","test_time",F0])       #data test norm by removing column F0
    X_v = data_val_norm.drop(columns=["subject#","test_time",F0])         #data val norm by removing column F0
    y_tt=data_test_norm[F0]         #data test norm column F0
    y_vl=data_val_norm[F0]          #data val norm column F0
    y_tn = data_train_norm[F0]      #collum vector with the feature to be estimated   
    
    #print(X_train)
    y=y_train.to_numpy().reshape(len(y_train),1)
    A=X_train.to_numpy()
    yT=y_test.to_numpy().reshape(len(y_test),1)
    AT=X_test.to_numpy()
    yV=y_val.to_numpy().reshape(len(y_test),1)
    AV=X_val.to_numpy()
    y_tn = (data_train[F0]).to_numpy().reshape(len(data_train[F0]),1)
    y_tt = (data_test[F0]).to_numpy().reshape(len(data_test[F0]),1)
    
    wLLS=SolveLLS(y,A,yT,AT,yV,AV) # instantiate the object
    wGrad=SolveGrad(y,A,yT,AT,yV,AV) # instantiate the object
    wStD=SteepDesc(y,A,yT,AT,yV,AV)
    wStc=stochastic(tn_stoch,yT,AT,yV,AV)
    wStDA=SteepDescADAM(y,A,yT,AT,yV,AV)
    wRid=ridge(y,A,yT,AT,yV,AV)
    wConGrad=ConjugateGRAD(y,A,yT,AT,yV,AV)
    
    # multiple line plot
    print("PRINT WLLS", wLLS)
    plt.figure(figsize=(13,6))
    plt.title("W(n) matrix indices", loc='left', fontsize=12, fontweight=0, color='black')
    #plt.plot(wLLS, marker='o', markerfacecolor='black', markersize=5, color='blue', linewidth=1, label="LLS")
    plt.plot(wGrad, marker='o', markerfacecolor='black', markersize=5, color='olive', linewidth=1, label="GRAD")
    plt.plot(wStD, marker='o', markerfacecolor='black', markersize=5, color='red', linewidth=1, label="SteepDesc")
    plt.plot(wStDA, marker='o', markerfacecolor='black', markersize=5, color='blue', linewidth=1, label="SteepDescADAM")
    plt.plot(wStc, marker='o', markerfacecolor='black', markersize=5, color='green', linewidth=1, label="Stochastic")
    plt.plot(wRid, marker='o', markerfacecolor='black', markersize=5, color='orange', linewidth=2, label="RidgeR")
    plt.plot(wConGrad, marker='o', markerfacecolor='black', markersize=5, color='yellow', linewidth=2, label="ConGRAD")   
    plt.xlabel("n")
    plt.ylabel("w(n)")
    plt.legend()
    plt.grid()
    plt.show()
    
