# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:42:49 2019

@author: AhmedMaher
"""
import numpy as np 
import math

#      2
   #   2
    #  1
  #    1
 #     0.35 0.9 0.5
#
    
def initialize_weight( m , l  , n ):
    
    weights_hidden = np.random.rand( l , m )
    #bias_hidden = np.random.rand( l , 1 )
    weight_output = np.random.rand( n , l )
    #bias_output= np.random.rand( n , 1 )
    return weights_hidden , weight_output


def forwarding( intputts , hidden_weights , output_weights ):
    
    z_hidden = np.dot(   hidden_weights, inputt    )  #( l * m ) * (m * k  )  = (l * k ) \n",
    hidden_output = sigmoid( z_hidden )
    net_output = np.dot( output_weights, z_hidden   ) #( n * l ) * (l * k ) = ( n * k )\n",
    net_output=sigmoid(net_output)
    return hidden_output , net_output



def back_propagation(z_output,actual_output,inpt,output_weights,hidden_weights,hidden_output,learning_rate):
    
    #Calculating delta ll output neourouns 
    delta_output=z_output*(1-z_output)*(actual_output-z_output)
    #print(delta_output)
    
    #usin delta output calculate new output weights
    new_output_weights=output_weights+(learning_rate* (np.dot(hidden_output,delta_output)).T)
    #print(new_output_weights)
    
    
    #calculating delta ll hidden neourouns
    Tranposed_output_weights=output_weights.T
    
    hidden_nerouns_sums=(np.dot(Tranposed_output_weights,delta_output))  #de*wce+dx*wcx whkza
    delta_hidden=(hidden_output*(1-hidden_output)*hidden_nerouns_sums)
    #print(delta_hidden)
    
    #calculating new hidden neourouns wieghts 
    new_hidden_weights=hidden_weights+(learning_rate* np.dot(delta_hidden,inpt.T))
    
    return new_hidden_weights,new_output_weights

def sigmoid(Z):
    return 1 / (1 + np.exp( -1 * Z) ) 


def normalize(X,mn,mx):
    return (X-mn)/(mx-mn)

def de_normalize(X,mn,mx):
    return ((X*(mx-mn))+mn)
    
# 0.35 0.9 0.5    
    
    
def getboundries():
    file=open("train.txt","r")
    line=file.readline().split(' ')      
    m = int(line[0]) #input nodes    
    line=file.readline().split(' ') 
    k = int(line[0]) #trainig set size 
    bias=1
    inputt = []
    output = []
    for j in range (k):
        line = file.readline().split()
        line.append(line[-1])
        line[-2]=str(bias)
        print(line[-1],line[-2])
        inputt.append([float(i) for i in line[:m+1]])
        output.append([float(i) for i in line[m+1:]])
        
    inputt  = np.array(inputt)
    output = np.array(output)
    
    mxo=np.max(output)
    mno=np.min(output)
    
    mxi=np.max(inputt)
    mni=np.min(inputt)
    
    return mno,mxo,mni,mxi

def write_on_file(hidden_weights,output_weights):
        np.savetxt("hidden_weights.txt", hidden_weights)
        np.savetxt("output_weights.txt", output_weights)
    
if __name__=="__main__":

    file=open("train.txt","r")
    line=file.readline().split(' ')
    m = int(line[0])+1 #input nodes
    l = int(line[1])+1 #nodes in hidden
    n = int(line[2]) #output nodes
    line=file.readline().split(' ')
    k = int(line[0])
    
    learning_rate = 0.1    
    number_iterations = 500
    mno,mxo,mni,mxi=getboundries() 
    hidden_weights , output_weights = initialize_weight( m , l , n )
    MSE=0
    bias=1
    
    fhidden_weights=hidden_weights
    foutput_weights=output_weights
    for j in range (k):
        inputt = []
        output = []
        line = file.readline().split()
        line.append(line[-1])
        line[-2]=str(bias)                           #Bais appending
        inputt.append([float(i) for i in line[:m]])
        output.append([float(i) for i in line[m:]])
        inputt  = np.array(inputt)
        output = np.array(output)
        inputt =  inputt.T
        
        
        #hidden_weights=[[0.1,0.8],[0.4,0.6]]
        #output_weights=[[0.3,0.9]]
        output_weights=np.array(output_weights)

        inputt=normalize(inputt,mni,mxi)
        output=normalize(output,mno,mxo)
        #print()
        
        for it in range(number_iterations):
            hidden_output ,net_output=forwarding(inputt , hidden_weights , output_weights )
            hidden_weights,output_weights=back_propagation(net_output,output,inputt,output_weights,hidden_weights,hidden_output,learning_rate)
        fhidden_weights+=hidden_weights
        foutput_weights+=output_weights

        act_output=de_normalize(output,mno,mxo)
        print()
        print("Sample numper: ",j+1)
        print("Desired OutPut: ",act_output)
        net_output=de_normalize(net_output,mno,mxo)
        print("Netword OutPut: ",net_output)
        print("Error: ",np.round((net_output-act_output),6))
        print()
        MSE+=(net_output-act_output)*(net_output-act_output)
    print("MSE= ",math.sqrt(MSE)/k)
    foutput_weights/=k
    fhidden_weights/=k
    
    #write_on_file(hidden_weights,output_weights)
    write_on_file(fhidden_weights,foutput_weights)
        


               