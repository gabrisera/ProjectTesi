from library import *

#funzione per normalizzare i dati 
def normalize(vettore):
    """
    Normalizza un vettore tra 0 e 1 utilizzando la normalizzazione min-max.
    
    Parameters:
    - vettore (numpy.ndarray): Il vettore da normalizzare.
    
    Returns:
    - numpy.ndarray: Il vettore normalizzato.
    """
    minimo = np.min(vettore)
    massimo = np.max(vettore)
    vettore_normalizzato = (vettore - minimo) / (massimo - minimo)
    return vettore_normalizzato

#funzione per decidere quali video finiscono per il testing e quali per il training 
def split_data(df,df1):
    Xt = np.empty((0, 4))  
    Yt = np.empty((0, 1))  

    Xl = np.empty((0, 4))
    Yl = np.empty((0, 1))

    for chiave, valore in df.items():
        if (chiave == "23-03-2016_t030" or chiave == "22-03-2016_t048" or chiave == "21-03-2016_t027" or chiave == "08-08-2019_t020"):
            if not Xt.size:  # Se Xt è vuoto, assegna direttamente la matrice
                Xt = np.array(valore.values)
                Yt = np.array(df1[chiave].values)
            else:  # Altrimenti, verifica le dimensioni prima di concatenare
                if Xt.shape[1] == valore.shape[1]:
                    Xt = np.vstack((Xt, valore.values))
                    Yt = np.vstack((Yt, df1[chiave].values))
                else:
                    print(f"Le dimensioni delle colonne di {chiave} non coincidono.")
        else:
            if not Xl.size:  # Se Xl è vuoto, assegna direttamente la matrice
                Xl = np.array(valore.values)
                Yl = np.array(df1[chiave].values)
            else:  # Altrimenti, verifica le dimensioni prima di concatenare
                if Xl.shape[1] == valore.shape[1]:
                    Xl = np.vstack((Xl, valore.values))
                    Yl = np.vstack((Yl, df1[chiave].values))
                else:
                    print(f"Le dimensioni delle colonne di {chiave} non coincidono.")
    return Xl,Yl,Xt,Yt

#funzione per aggiungere dei valori aggiuntivi alla label 
def add_values_c(Yl):
    for i in range(1, np.size(Yl)):
        if Yl[i] == 1 and Yl[i - 1] == 0:
            for j in range(9, 0, -1):
                if i + j + 1 < Yl.size:
                    if [i + abs(10-j)]==1 or [i - abs(10-j)]==1:
                        break
                    else:
                        Yl[i + abs(10-j)] = 1
                        Yl[i - abs(10-j)] = 1
                        #print(i + abs(5-j))
                else:
                    break

    return Yl
def add_values_r(Yl):
    for i in range(1, np.size(Yl)):
        if Yl[i] == 1:
            Yl[i]=100
            Yl[i-1]=100
            Yl[i-2]=100
            if all(Yl[i-1:i-9] == 0) and all(Yl[i+1:i+9] == 0) and i>10:
                for j in range(9, 1, -1):
                    if i + j < Yl.size and i - j >= 0:
                        Yl[i+10-j] = j*5
                        Yl[i-12+j] = j*5
                    else:
                        break
            else:
                continue
        else:
            continue
    return Yl
#funzione per stampare quanti e quali valori sono presenti in un vettore
def print_val_count(x):
    unique, count=np.unique(x,return_counts="true")
    x_val_count={ k:v for (k,v) in zip(unique,count) }
    print(f'conta valori: {x_val_count}')

#funzione per creare le sliding windows    
def sliding_windows(x,k):

    ns=x.size
    hk=math.floor(k/2) 
    win=np.zeros((6,ns-hk))
    #np.shape(win)
   
    print(hk)
    for i in range(hk,ns):
        stat=[]
        #print(hk,ns)
        if i+hk>=ns:
            break
        
        w=np.transpose(np.vstack([x[i-hk:i],x[i+1:i+hk]]))      
        media1=np.mean(w[:,:hk])
        media2=np.mean(w[:,hk:])
        media=np.abs(media2-media1)      
        varianza1=np.var(w[:,:hk])
        varianza2=np.var(w[:,hk:])
        varianza=np.abs(varianza2-varianza1)
        moda1=np.argmax(stats.mode(w[:,:hk]).count)
        moda2=np.argmax(stats.mode(w[:,hk:]).count)
        moda=np.abs(moda2-moda1)
        median1=np.median(w[:,:hk])
        median2=np.median(w[:,hk:])
        median=np.abs(median2-median1)
        std1=np.std(w[:,:hk])
        std2=np.std(w[:,hk:])
        std=np.abs(std2-std1)
        int1=np.std(w[:,:hk])
        int2=np.std(w[:,hk:])
        intervallo=np.abs(int2-int1)
        #corr_coeff=np.corrcoef(w[:,:hk],w[:,hk:])[0,1]
        '''if np.isnan(corr_coeff):
            corr_coeff=0
            '''
        stat.append([media,varianza,median,moda,std,intervallo])
        #print(media.shape,varianza.shape,median.shape,moda.shape,std.shape,intervallo.shape,corr_coeff.shape)
        #print(len(stat[0]),len(stat[1]),len(stat[2]),len(stat[3]),len(stat[4]),len(stat[5]),len(stat[6]))
        #print(st.shape)
        st=np.array(stat)
        #w=np.append(w,stat)
        #print(i)
        win[:,i]=st.reshape(-1)
        #print(win[:,i])
        del stat
        #w[:]
    
    return win

#funzione per split training e validation
def split_train_val(Q, Yl, train_ratio, val_ratio):
    
    
    print(f"Q {Q.shape} Yl {Yl.shape}")
    # Calcolo le dimensioni per ciascun set
    num_rows = Q.shape[0]
    train_size = int(num_rows * train_ratio)
    val_size = num_rows-train_size
    #test_size = num_rows - train_size - val_size

    # Genero gli indici per gli insiemi
    indices = np.arange(num_rows)
    np.random.shuffle(indices)

    # Divido gli indici nei vari set
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Creo i set utilizzando gli indici
    x_train = Q[train_indices, :]
    x_val = Q[val_indices, :]


    #divido le lablel in base agli indici
    Y_train=Yl[train_indices]
    Y_val=Yl[val_indices]
    return x_train, x_val, Y_train, Y_val

#funzione per downsampling
def downsampling(x_train,y_train,flag):
    df_train = pd.concat([pd.DataFrame(x_train), pd.Series(y_train.reshape(-1), name='target_column')], axis=1)

    # Separare le classi maggioritarie e minoritarie
    df_majority = df_train[df_train['target_column'] == 0]  # sostituisci 'target_column' con il tuo nome di colonna target
    if(not flag):
        df_minority = df_train[df_train['target_column'] == 1]
    else:
        df_minority = df_train[df_train['target_column'] == 100]
        df_minority1 = df_train[df_train['target_column'] == 45]
        df_minority2 = df_train[df_train['target_column'] == 40]
        df_minority3 = df_train[df_train['target_column'] == 35]
        df_minority4 = df_train[df_train['target_column'] == 30]
        df_minority5 = df_train[df_train['target_column'] == 25]
        df_minority6 = df_train[df_train['target_column'] == 20]
        df_minority7 = df_train[df_train['target_column'] == 15]
        df_minority8 = df_train[df_train['target_column'] == 10]
        
    # Downsampling delle classi maggioritarie
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=int(20*len(df_minority)), random_state=42)
    if not flag:
        df_downsampled=pd.concat([df_majority_downsampled,df_minority])
    else:
    # Combinare i campioni downsampled con la classe minoritaria
        df_downsampled = pd.concat([df_majority_downsampled, df_minority, df_minority1,
                                    df_minority2, df_minority3, df_minority4, df_minority5, 
                                    df_minority6, df_minority7, df_minority8,])

    # Agitare il DataFrame risultante
    df_downsampled = df_downsampled.sample(frac=1, random_state=42)

    # Separare nuovamente X_train e y_train
    X_train_downsampled = df_downsampled.drop('target_column', axis=1)
    y_train_downsampled = df_downsampled['target_column']
    return X_train_downsampled.values, y_train_downsampled.values