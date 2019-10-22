##FUNCIONES PARA RN reconocimiento de digitos en 28x28 + 1 BIAS

def getCellError(target,estado):
    err=target-estado  
    return err

def updateCellWeights(weight_neuro, err, input,learnin_rate):
    for i in range(785):
        weight_neuro[i] = weight_neuro[i] + (learnin_rate * input[i] * err)
        #print ("*******",weight_neuro[i], input[i], err)  
        
def activation_function(output):
    #binario 
    if output >0:
        return 1
    else:   
        return 0

#normalizar input
def normalizaImg(vector):
    vector_t={}
    for i in range(785):
        if vector[i]>0:
            vector_t[i]=1
        else:
            vector_t[i]=0 
        #vector_t[i]=vector[i]/255 #255:rgb max. Normalización
    return vector_t

# #pasa un entero a su representación en vector 
def number2vector(numero):
    num_v={}
    for i in range(10):
        if i==numero:
            num_v[i]=1 
        else:
            num_v[i]=0
    return num_v

#entrena. 
def trainNeuron(input,weight_neuro, target,learnin_rate,modo):
    output=0
    for j in range (785):
        output= output+ (input[j] * weight_neuro[j])
        #normalizar output
    output= output/785

    estado=activation_function(output)
    if modo=='training':
        err=getCellError(target,estado)
        updateCellWeights(weight_neuro, err, input,learnin_rate)
    return estado
 