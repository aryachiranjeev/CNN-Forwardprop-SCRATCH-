# CNN-Forwardprop-SCRATCH-
import numpy as np
def conv_forward(X,maxpooling=False):
    #####CONVOLUTION1####
    #(H_prev,W_prev)=X.shape
    filter1=[]
    num_f1=6
    np.random.seed(0)
    for f in range(6):
        filter1.append(np.random.randint(-1,1,25).reshape(5,5))

    f1=np.asarray(filter1) 
    print("f1=",f1)
    print("filter_shape=",f1.shape)
    #print("f1=",f1)
    #"""
    #for f in range(6):
    #    f1.append(np.random.randint(2,size=(2,2)))
        
    H1=(X.shape[0])-(f1.shape[1])+1
    W1=(X.shape[1])-(f1.shape[2])+1
    print("H1=",H1)  
    print("W1=",W1)
    C1=np.zeros(num_f1*H1*W1).reshape(num_f1,H1,W1)
    b1=np.ones(H1*W1).reshape(H1,W1)
    print("B1_shape=",b1.shape)
    for n_f1 in range(num_f1):
        for h1 in range(H1):
            for w1 in range(W1):
                x_slice=X[h1:h1+(f1.shape[1]),w1:w1+(f1.shape[2])]
                C1[n_f1,h1,w1]=np.sum(x_slice*f1[n_f1])
        final_C1=C1+b1
    print("\nfinal_C1=",final_C1)
    print("\nfinal_C1_SHAPE=",final_C1.shape)
    #final_C1=C1+b1
    #print("final_C1_shape",final_C1.shape)
    #print("final_C1",final_C1)
    
    ####MAXPOOL1####
    
    if max_pooling1:
        pool1=np.ones(12*12).reshape(12,12)
        print("pool1_shape=",pool1.shape)
        HP1=(final_C1.shape[1])-(pool1.shape[0])+1
        WP1=(final_C1.shape[2])-(pool1.shape[1])+1
        print("HP1=",HP1)  
        print("WP1=",WP1)
        CP1=np.zeros(num_f1*HP1*WP1).reshape(num_f1,HP1,WP1)
        for n_f1 in range(num_f1):
            for hp1 in range(HP1):
                for wp1 in range(WP1):
                    p_slice1=final_C1[n_f1][hp1:hp1+(pool1.shape[0]), wp1:wp1+(pool1.shape[1])]
                    CP1[n_f1,hp1,wp1]=np.max(p_slice1*pool1)
        
        #print("CP1=",CP1)
        print("shape_after_pooling=",CP1.shape)
    print("CP1=",CP1)   
    #return CP1
    
 X=np.random.randint(255,size=(28,28))
print("input_shape(X)=",X.shape)
conv_forward(X,max_pooling=True)
