from library import *
from dataset import import_data
from functions import *

#impostazione output
np.set_printoptions(threshold=np.inf)

#prelievo dataset
df, df1= import_data()

#split dataset 1VL
Xl,Yl,Xt,Y_test=split_data(df,df1)
print(Xl.shape,Yl.shape,Xt.shape,Y_test.shape)
Yt=np.vstack([Yl,Y_test])

#Y1=Yl
#Y1_ts=Yt

#normalize data della colonna scelta (in questo caso la "3" che definisce la "direzione della testa" per la comp. "space")
x=np.vstack([Xl[:,3].reshape(-1,1),Xt[:,3].reshape(-1,1)])
print(x.shape)
x=normalize(x)
x1l=x[:Xl.shape[0]]
x1t=x[Xl.shape[0]:]

#aggiunta di valori alla label di riferimento 
#print_val_count(Yl)

#print_val_count(Yl)

#creazione sliding_windows
k=31
hk=int(np.floor(k/2))
x1l= x1l.reshape(-1, 1)
x1t= x1t.reshape(-1, 1)
x1=np.vstack([x1l,x1t])
print(f"x1.shape: { x1.shape}")
win=sliding_windows(x1,k)
print(win.shape)
 ## divisione sliding windows per learning e sliding windows per testing
winl=win[:,:x1l.shape[0]].T
wint=win[:,x1l.shape[0]:win.shape[1]].T

Yl=Yt[hk:winl.shape[0]+hk]
Y_test=Yt[winl.shape[0]+hk:]
Yl=add_values_r(Yl)
print(winl.shape,wint.shape)

#creazione set validation e preparazione dati per smv
x_train, x_val, Y_train, Y_val=split_train_val(winl,Yl,1.0,0.0)
x_test = wint
print(f"x_train shape {x_train.shape}")
Y_train=Y_train.reshape(-1)
Y_train=Y_train[:Y_train.shape[0]].T
#undersampling
print_val_count(Y_train)
x_train_o,Y_train_o=downsampling(x_train,Y_train,True)
#print(x_train_o.shape,Y_train_o.shape)
#print_val_count(Y_train_o)
#oversampling
#sm=SMOTE(sampling_strategy={100:10000,20:5000,30:5000,40:5000,50:5000,60:5000,70:5000,80:5000,90:5000}, random_state=12)
#x_train_o, Y_train_o=sm.fit_resample(x_train,Y_train)
print_val_count(Y_train_o)
#x_train_o=x_train
#Y_train_o=Y_train
#print_val_count(Y_train_o)
Y_train_o=Y_train_o.reshape(-1,1)
#print_val_count(Y_train)
#searching of best hyperparameters
param_grid = {
    'C': [ 10, 100],
    'gamma': [0.01, 0.1, 1],
}

# Crea un'istanza della Support Vector Regression (SVR) con kernel RBF
svr_model = SVR(kernel='rbf')

# Crea un'istanza di GridSearchCV
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_search = GridSearchCV(svr_model, param_grid, scoring=mse_scorer, cv=3)

# Esegui la ricerca della griglia
grid_search.fit(x_train_o, Y_train_o)
best_svr = grid_search.best_estimator_

# Ottenere i migliori iperparametri
best_params = grid_search.best_params_
# Valutazione sul test, calcolo dell'errore quadratico medio
test_predictions = best_svr.predict(x_test)
mse_test = mean_squared_error(Y_test[:test_predictions.size], test_predictions)

joblib.dump(best_svr,"best_svr_model.pkl")
joblib.dump(best_params,"best_svr_params.pkl")
joblib.dump(mse_test,"mse_test.pkl")
joblib.dump(test_predictions,"test_predictions.pkl")
joblib.dump(Y_test,"Y_test.pkl")









