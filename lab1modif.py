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


#https://hackernoon.com/implementing-different-variants-of-gradient-descent-optimization-algorithm-in-python-using-numpy-809e7ab3bab4


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.style.use('classic')

def normalz(dataset):#Normalizing entire dataframe but not few columns.
    dataNorm=(dataset-data_train.mean())/np.sqrt(data_train.std())
    dataNorm["subject#"]=dataset["subject#"]
    #dataNorm["age"]=dataset["age"]
    #dataNorm["sex"]=dataset["sex"]
    dataNorm["test_time"]=dataset["test_time"]
    #dataNorm["motor_UPDRS"]=dataset["motor_UPDRS"]
    #dataNorm["total_UPDRS"]=dataset["total_UPDRS"]
    return dataNorm

#LINEAR LEAST SQUARE -PSEUDOINVERSE
def SolveLLS(y,A,yT,AT,yV,AV):   # class SolveLLS belongs to class SolveMinProb  
    w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)  #pseudoinverse calculation
    yhat_train = np.dot(A,w)     
    yhat_test = np.dot(AT,w)
    yhat_val = np.dot(AV,w)
    yhat_train_nonnorm= (np.array(yhat_train) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_test_nonnorm=  (np.array(yhat_test) * np.sqrt(y_tn.std()))+y_tn.mean()
    yhat_val_nonnorm=  (np.array(yhat_val) * np.sqrt(y_tn.std()))+y_tn.mean() 
    y_train_nonnorm= (np.array(y) * np.sqrt(y_tn.std()))+y_tn.mean()
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

def SolveGrad(y,A,yT,AT,yV,AV): 
    max_iterations = 100
    gamma=1.0e-8
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w=np.random.rand(Nf,1)# random initialization of w
    grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))
    iterations = 0
    GRADe_historyTRAIN = []
    GRADe_historyVAL = []
    GRADe_historyTEST = []
    e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
    e_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
    e_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)
    while np.linalg.norm(w-a_prev) > 1e-8 and iterations < max_iterations:
        iterations += 1
        GRADe_historyTRAIN += [e_train] 
        GRADe_historyTEST += [e_test] 
        GRADe_historyVAL += [e_test]                                   
        a_prev = w
        w = w - gamma * grad
        e_train = (np.linalg.norm(np.dot(A,w)-y)**2)/len(y) 
        e_test = (np.linalg.norm(np.dot(AT,w)-yT)**2)/len(yT)
        e_val = (np.linalg.norm(np.dot(AV,w)-yV)**2)/len(yV)
        grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))        
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
    
    plt.figure(figsize=(13,6))
    plt.plot(GRADe_historyTRAIN, color="black", label="GRAD error TRAIN History")
    plt.plot(GRADe_historyTEST, label="GRAD error TEST History")
    plt.plot(GRADe_historyVAL, label="GRAD error VAL History")
    plt.title("GRAD ERROR")
    plt.xlabel("ITERATIONS")
    plt.ylabel("ERROR")
    plt.legend(loc=2) 
   
    #error histogram for LLS
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
    max_iterations = 100
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
  
    #error histogram for LLS
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

def stochastic(A,yT,AT,yV,AV):
    max_iterations = 100
    learning_coefficient = 1.0e-05
    Nf=A.shape[1]-1 # number of columns
    w=np.zeros((Nf,1),dtype=float) # column vector w 
    w=np.random.rand(Nf,1)# random initialization of w
    y=A[F0].to_numpy().reshape(-1, 1)
    y_train=A[F0].to_numpy().reshape(-1, 1)
    X_train=A.drop(columns=[F0]).to_numpy()
    e_train = (np.linalg.norm(np.dot(X_train,w) - y_train)**2)/len(y)
    #e_train = np.linalg.norm(X_train.dot(w) - y_train)**2
    iterations = 0
    e_history = [e_train]
    batch_size = 10
    shuffled = shuffle(A)
    while iterations < max_iterations:
        iterations += 1
        shuffled = shuffle(A)
        y_train = shuffled[F0].to_numpy().reshape(-1, 1)
        X_train = shuffled.drop(columns=[F0]).to_numpy()
        batch_prev = 0
        for batch in range(batch_size, len(X_train), batch_size):
            X_batch = X_train[batch_prev:batch]
            y_batch = y_train[batch_prev:batch]
            #batch_gradient = -2 * X_batch.T.dot(y_batch) + 2 * X_batch.T.dot(X_batch).dot(w)
            #batch_gradient = 2*np.dot(X_batch.T,(np.dot(X_batch,w)-y_batch))
            batch_gradient=-2 * X_batch.T.dot(y_batch) + 2 * X_batch.T.dot(X_batch).dot(w)
            w = w - (learning_coefficient * batch_gradient)
            batch_prev = batch
        e_train =(np.linalg.norm(np.dot(X_train,w) - y_train)**2)/len(y)
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
    max_iterations = 100
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
    #plt.plot(wCGrad, marker='o', markerfacecolor='black', markersize=5, color='red', linewidth=2, label="CGRAD")   
    plt.xlabel("n")
    plt.ylabel("w(n)")
    plt.legend()
    plt.grid()
    plt.show()
    
