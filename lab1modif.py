# -*- coding: utf-8 -*-
#Linear Least Squares solution
#stochastic gradient algorithm using adam method
#conjugate gradient algorithm
#ridge regression (optimize λ)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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
def SolveLLS(y,A,yT,AT,yV,AV,trainmean,trainstd):   # method SolveLLS (Y train, x train, y test, x test ,y validation x validation)  
    w   =   np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)  #pseudoinverse calculation
    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm  = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm)    #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm)    #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)     #error (estimated yvalidation - original yvalidation)
    mse_train = (np.linalg.norm(yhat_train_nonnorm-y_nonnorm)**2)/len(y)
    mse_test  = (np.linalg.norm(yhat_test_nonnorm-yT_nonnorm)**2)/len(yT)
    mse_val   = (np.linalg.norm(yhat_val_nonnorm-yV_nonnorm)**2)/len(yV)
    
    c = np.dot(AT,w)
    mean1 = yT.mean() 
    mean2 = c.mean()
    std1 = yT.std()
    std2 = c.std()
    corr = ((yT*c).mean()-mean1*mean2)/(std1*std2)
    print("LLS coefficient of determination R2",corr)
    print ("LLS error TRAIN", mse_train)  
    print ("LLS error TEST", mse_test)
    print ("LLS error VALIDATION", mse_val)                                                                                                                                                                                     

    #PLOTS for LLS
    plt.figure(figsize=(6,6))
    plt.scatter(y_nonnorm,yhat_train_nonnorm, label="LLS", marker="o",s=0.8, color="red", alpha=0.8)
    plt.plot(y_nonnorm,y_nonnorm,color="black", linewidth=0.5)
    plt.title("Train un-normalized - y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/lls_train_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yT_nonnorm,yhat_test_nonnorm, label="LLS", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yT_nonnorm,yT_nonnorm,color="black",linewidth=0.5)
    plt.title("Test un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/lls_test_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yV_nonnorm,yhat_val_nonnorm, label="LLS", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yV_nonnorm,yV_nonnorm,color="black",linewidth=0.5)
    plt.title("Validation un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/lls_validation_yvsyhat.png')
    plt.show() 
    plt.close() 
    
    plt.figure(figsize=(6,6))
    plt.plot(yhat_test_nonnorm[:100], color="red", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for LLS",fontsize=20)
    plt.xlabel("Sample index",fontsize=16)
    plt.ylabel("Original value",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/lls_test_frame_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for GRAD, Train, Test, Validation",fontsize=20)
    plt.xlabel("error",fontsize=16)
    plt.ylabel("Occurrencies",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/lls_histogram_error.png')
    plt.show()
    plt.close()
    return w
#Stochastic ADAM    
def StochasticADAM(y,A,yT,AT,yV,AV,trainmean,trainstd):
    alfa  = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.0000001
    m = 0
    v = 0
    max_iterations = 100000
    Nf = A.shape[1] # number of columns
    w  = np.zeros((Nf,1),dtype=float) # column vector w 
    w  = np.random.rand(Nf,1)# random initialization of w
    iterations = 0
    e_history  = []
    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm   = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm  = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm  = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
        
    while iterations < max_iterations:
        iterations += 1
        e_history += [e_train]
        e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
        grad = -2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w))
        m = beta1 * m +(1-beta1)*grad
        v = beta2 * v +(1+beta2)*np.power(grad,2)
        m_hat = m / (1 - np.power(beta1, iterations))
        v_hat = v / (1 - np.power(beta2, iterations))
        w = w - alfa * m_hat / (np.sqrt(v_hat) + epsilon)
   
    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm  = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
    mse_train = (np.linalg.norm(yhat_train_nonnorm-y_nonnorm)**2)/len(y)
    mse_test  = (np.linalg.norm(yhat_test_nonnorm-yT_nonnorm)**2)/len(yT)
    mse_val   = (np.linalg.norm(yhat_val_nonnorm-yV_nonnorm)**2)/len(yV)
  
    c=np.dot(AT,w)
    mean1 = yT.mean() 
    mean2 = c.mean()
    std1 = yT.std()
    std2 = c.std()
    corr = ((yT*c).mean()-mean1*mean2)/(std1*std2)
    print("Stochastic ADAM coefficient of determination R2",corr)
    print ("stochastic_ADAM error TRAIN", mse_train)  
    print ("stochastic_ADAM error TEST", mse_test)
    print ("stochastic_ADAM error VALIDATION", mse_val)  

    #PLOTS for stochastic ADAM
    plt.figure(figsize=(6,6))
    plt.scatter(y_nonnorm,yhat_train_nonnorm, label="stochastic_ADAM", marker="o",s=0.8, color="red", alpha=0.8)
    plt.plot(y_nonnorm,y_nonnorm,color="black", linewidth=0.5)
    plt.title("Train un-normalized - y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/stochastic_ADAM_train_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yT_nonnorm,yhat_test_nonnorm, label="stochastic_ADAM", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yT_nonnorm,yT_nonnorm,color="black",linewidth=0.5)
    plt.title("Test un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/stochastic_ADAM_test_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yV_nonnorm,yhat_val_nonnorm, label="stochastic_ADAM", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yV_nonnorm,yV_nonnorm,color="black",linewidth=0.5)
    plt.title("Validation un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/stochastic_ADAM_validation_yvsyhat.png')
    plt.show() 
    plt.close() 
    
    plt.figure(figsize=(6,6))
    plt.plot(yhat_test_nonnorm[:100], color="red", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for stochastic_ADAM",fontsize=20)
    plt.xlabel("Sample index",fontsize=16)
    plt.ylabel("Original value",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/stochastic_ADAM_test_frame_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for stochastic_ADAM, Train, Test, Validation",fontsize=20)
    plt.xlabel("error",fontsize=16)
    plt.ylabel("Occurrencies",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/stochastic_ADAM_histogram_error.png')
    plt.show()
    plt.close()
    return w

def ridge(y,A,yT,AT,yV,AV,trainmean,trainstd): 
    max_iterations = 100000
    gamma=1.0e-6
    lbda= 1.0e-5
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w=np.random.rand(Nf,1)# random initialization of w
    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm   = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm  = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm  = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
    grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w)) + 2*(lbda*w)
    iterations = 0
    e_history = []

    while np.linalg.norm(w-a_prev) > 1e-8 and iterations < max_iterations:
        iterations += 1
        e_history += [e_train]                                           
        a_prev = w
        w = w - gamma * grad
        e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
        grad=-2*(np. dot (A.T,y)) + 2*(np.dot(np.dot(A.T,A),w)) + 2*(lbda*w)        

    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm  = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
    mse_train = (np.linalg.norm(yhat_train_nonnorm-y_nonnorm)**2)/len(y)
    mse_test  = (np.linalg.norm(yhat_test_nonnorm-yT_nonnorm)**2)/len(yT)
    mse_val   = (np.linalg.norm(yhat_val_nonnorm-yV_nonnorm)**2)/len(yV)

    c=np.dot(AT,w)
    mean1 = yT.mean() 
    mean2 = c.mean()
    std1 = yT.std()
    std2 = c.std()
    corr = ((yT*c).mean()-mean1*mean2)/(std1*std2)
    print ("Ridge coefficient of determination R2",corr)
    print ("Ridge error TRAIN", mse_train)  
    print ("Ridge error TEST", mse_test)
    print ("Ridge error VALIDATION", mse_val)  

    #PLOTS for Ridge
    plt.figure(figsize=(6,6))
    plt.scatter(y_nonnorm,yhat_train_nonnorm, label="Ridge", marker="o",s=0.8, color="red", alpha=0.8)
    plt.plot(y_nonnorm,y_nonnorm,color="black", linewidth=0.5)
    plt.title("Train un-normalized - y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Ridge_train_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yT_nonnorm,yhat_test_nonnorm, label="Ridge", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yT_nonnorm,yT_nonnorm,color="black",linewidth=0.5)
    plt.title("Test un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/Ridge_test_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yV_nonnorm,yhat_val_nonnorm, label="Ridge", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yV_nonnorm,yV_nonnorm,color="black",linewidth=0.5)
    plt.title("Validation un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/Ridge_validation_yvsyhat.png')
    plt.show() 
    plt.close() 
    
    plt.figure(figsize=(6,6))
    plt.plot(yhat_test_nonnorm[:100], color="red", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for Ridge",fontsize=20)
    plt.xlabel("Sample index",fontsize=16)
    plt.ylabel("Original value",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Ridge_test_frame_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for Ridge, Train, Test, Validation",fontsize=20)
    plt.xlabel("error",fontsize=16)
    plt.ylabel("Occurrencies",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Ridge_histogram_error.png')
    plt.show()
    plt.close()
    return w

def ConjugateGRAD(y,A,yT,AT,yV,AV,trainmean,trainstd): 
    Nf=A.shape[1] # number of columns
    a_prev = np.ones(Nf)
    w  = np.zeros((Nf,1),dtype=float) # column vector w
    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm   = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm  = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm  = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
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

    yhat_train = np.dot(A,w)   #estimated y train UPDRS values based on calculated w
    yhat_test  = np.dot(AT,w)  #estimated y test UPDS values based on calculated w
    yhat_val   = np.dot(AV,w)  #estimated y validation UPDS values based on calculated w
    yhat_train_nonnorm = (np.array(yhat_train) * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_test_nonnorm  = (np.array(yhat_test)  * np.sqrt(trainstd)+trainmean)  #transfom normalized in non normalized values
    yhat_val_nonnorm   = (np.array(yhat_val)   * np.sqrt(trainstd)+trainmean)  #transform normalized in non normalized values 
    y_nonnorm  = (np.array(y)  * np.sqrt(trainstd)+trainmean)
    yT_nonnorm = (np.array(yT) * np.sqrt(trainstd)+trainmean)
    yV_nonnorm = (np.array(yV) * np.sqrt(trainstd)+trainmean)
    e_train = (yhat_train_nonnorm-y_nonnorm) #error(estimated ytrain - original ytrain)
    e_test  = (yhat_test_nonnorm-yT_nonnorm) #error (estimated ytest - original ytest)
    e_val   = (yhat_val_nonnorm-yV_nonnorm)  #error (estimated yvalidation - original yvalidation)
    mse_train = (np.linalg.norm(yhat_train_nonnorm-y_nonnorm)**2)/len(y)
    mse_test  = (np.linalg.norm(yhat_test_nonnorm-yT_nonnorm)**2)/len(yT)
    mse_val   = (np.linalg.norm(yhat_val_nonnorm-yV_nonnorm)**2)/len(yV)

    c=np.dot(AT,w)
    mean1 = yT.mean() 
    mean2 = c.mean()
    std1 = yT.std()
    std2 = c.std()
    corr = ((yT*c).mean()-mean1*mean2)/(std1*std2)
    print ("Conjugated GRAD coefficient of determination R2",corr)
    print ("Conjugated GRAD error TRAIN", mse_train)  
    print ("Conjugated GRAD error TEST", mse_test)
    print ("Conjugated GRAD error VALIDATION", mse_val)  

    #PLOTS for Conjugate_grad
    plt.figure(figsize=(6,6))
    plt.scatter(y_nonnorm,yhat_train_nonnorm, label="Conjugate_grad", marker="o",s=0.8, color="red", alpha=0.8)
    plt.plot(y_nonnorm,y_nonnorm,color="black", linewidth=0.5)
    plt.title("Train un-normalized - y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Conjugate_grad_train_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yT_nonnorm,yhat_test_nonnorm, label="Conjugate_grad", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yT_nonnorm,yT_nonnorm,color="black",linewidth=0.5)
    plt.title("Test un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/Conjugate_grad_test_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(yV_nonnorm,yhat_val_nonnorm, label="Conjugate_grad", marker="o",s=0.8, color="blue", alpha=0.8)
    plt.plot(yV_nonnorm,yV_nonnorm,color="black",linewidth=0.5)
    plt.title("Validation un-normalized- y_true VS y_hat",fontsize=20)
    plt.xlabel("y_true",fontsize=16)
    plt.ylabel("y_hat",fontsize=16)
    plt.legend(loc=2)  
    plt.savefig('imag/Conjugate_grad_validation_yvsyhat.png')
    plt.show() 
    plt.close() 
    
    plt.figure(figsize=(6,6))
    plt.plot(yhat_test_nonnorm[:100], color="red", label="yhat_test")
    plt.plot(y_tt[:100], label="Y test")
    plt.title("Test prediction for Conjugate_grad",fontsize=20)
    plt.xlabel("Sample index",fontsize=16)
    plt.ylabel("Original value",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Conjugate_grad_test_frame_yvsyhat.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(13,6))
    plt.hist(e_train, bins=50, label="error Yhat_train-Y_train", alpha=0.4)
    plt.hist(e_test, bins=50, label="error Yhat_test-Y_test", alpha=0.4)
    plt.hist(e_val, bins=50, label="error Yhat_validation-Y_validation", alpha=0.4)    
    plt.title("Error histogram for Conjugate_grad, Train, Test, Validation",fontsize=20)
    plt.xlabel("error",fontsize=16)
    plt.ylabel("Occurrencies",fontsize=16)
    plt.legend(loc=2)
    plt.savefig('imag/Conjugate_grad_histogram_error.png')
    plt.show()
    plt.close()
    return w

if __name__ == "__main__":  
    plt.style.use('seaborn-dark-palette')
    np.random.seed (30)
    df = pd.read_csv("parkinsons_updrs.data")
    #df.test_time = df.test_time.apply(np.abs)
    #df["day"] = df.test_time.astype(np.int64)
    #df = df.groupby(["subject#", "day"]).mean()
    pd = shuffle(df)
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
    F0 = "total_UPDRS"
    tn_stoch = data_train_norm.drop(columns=["subject#","test_time"])
    y_train = data_train_norm[F0] #collum vector with the feature to be estimated
    X_train =data_train_norm.drop(columns=["subject#","test_time",F0])
    X_test = data_test_norm.drop(columns=["subject#","test_time",F0])    #data test norm by removing column F0
    X_val = data_val_norm.drop(columns=["subject#","test_time",F0])      #data val norm by removing column F0
    y_test=data_test_norm[F0]     #data test norm column F0
    y_val=data_val_norm[F0]       #data val norm column F0

    X_tn =data_train_norm.drop(columns=["subject#","test_time",F0])
    X_tt = data_test_norm.drop(columns=["subject#","test_time",F0])       #data test norm by removing column F0
    X_v = data_val_norm.drop(columns=["subject#","test_time",F0])         #data val norm by removing column F0
    y_tt=data_test_norm[F0]         #data test norm column F0
    y_vl=data_val_norm[F0]          #data val norm column F0
    y_tn = data_train_norm[F0]      #collum vector with the feature to be estimated   
    
    #plot correlation matrix fo the data train 
    corrmat = X_train.corr() 
    f, ax = plt.subplots(figsize =(9, 8)) 
    sns.heatmap(corrmat, ax = ax,
                cmap ="BuPu",
                linewidths = 0.05,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 6})

    #print(X_train)
    y=y_train.to_numpy().reshape(len(y_train),1)
    A=X_train.to_numpy()
    yT=y_test.to_numpy().reshape(len(y_test),1)
    AT=X_test.to_numpy()
    yV=y_val.to_numpy().reshape(len(y_test),1)
    AV=X_val.to_numpy()
    y_tn = (data_train[F0]).to_numpy().reshape(len(data_train[F0]),1)
    y_tt = (data_test[F0]).to_numpy().reshape(len(data_test[F0]),1)
    trainmed=y_tn.mean()
    trainstd=y_tn.std()    
    wLLS=SolveLLS(y,A,yT,AT,yV,AV,trainmed,trainstd)        # call the linear least square method and pass the variables
    wStDA=StochasticADAM(y,A,yT,AT,yV,AV,trainmed,trainstd) # call the Stochastic method and pass the variables
    wRid=ridge(y,A,yT,AT,yV,AV,trainmed,trainstd)           # call the ridge method and pass the variables
    wConGrad=ConjugateGRAD(y,A,yT,AT,yV,AV,trainmed,trainstd) # call the Conjugated Gradient method and pass the variables

    # multiple line plot
    print("PRINT WLLS", wLLS)
    plt.figure(figsize=(13,6))
    plt.title("W(n) matrix indices", loc='left', fontsize=20, fontweight=0, color='black')
    #LLS not apppears due to to high indexes n values.
    #plt.plot(wLLS, marker='o', markerfacecolor='black', markersize=5, color='blue', linewidth=1, label="LLS")
    plt.plot(wStDA, marker='o', markerfacecolor='black', markersize=5, color='red', linewidth=1, label="Stochastic_ADAM")
    plt.plot(wConGrad, marker='o', markerfacecolor='black', markersize=5, color='green', linewidth=2, label="ConGRAD")
    plt.plot(wRid, marker='o', markerfacecolor='black', markersize=5, color='orange', linewidth=2, label="RidgeR")
    plt.xlabel("n", fontsize=16)
    plt.ylabel("w(n)", fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()
