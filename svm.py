from library import *
from dataset import import_data
from functions import *

#impostazione output
np.set_printoptions(threshold=np.inf)

#prelievo dataset
df, df1= import_data()

#split dataset 1VL
Xl,Yl,Xt,Y_test=split_data(df,df1)

#Y1=Yl
#Y1_ts=Yt

#normalize data della colonna scelta (in questo caso la "3" che definisce la "direzione della testa" per la comp. "space")
x=np.vstack([Xl[:,3].reshape(-1,1),Xt[:,3].reshape(-1,1)])
x=normalize(x)
x1l=x[:Xl.shape[0]]
x1t=x[Xl.shape[0]:x.shape[0]]

#aggiunta di valori alla label di riferimento 
#print_val_count(Yl)
Yl=add_values_c(Yl)
#print_val_count(Yl)

#creazione sliding_windows
k=30
x1l= x1l.reshape(-1, 1)
x1t= x1t.reshape(-1, 1)
x1=np.vstack([x1l,x1t])
win=sliding_windows(x1,k)
 ## divisione sliding windows per learning e sliding windows per testing
winl=win[:,1:x1l.shape[0]+1].T
wint=win[:,x1l.shape[0]+1:win.shape[1]].T
#print(winl.shape,wint.shape)

#creazione set validation e preparazione dati per smv
x_train, x_val, Y_train, Y_val=split_train_val(winl,Yl,1.0,0.0)
x_test = wint
#undersampling
x_train_o,Y_train_o=downsampling(x_train,Y_train,False)
print(x_train_o.shape,Y_train_o.shape)
print_val_count(Y_train_o)
#oversampling
sm=SMOTE(sampling_strategy=1, random_state=12)
x_train_o, Y_train_o=sm.fit_resample(x_train_o,Y_train_o)
print_val_count(Y_train_o)
#x_train_o=x_train
#Y_train_o=Y_train
#print_val_count(Y_train_o)
Y_train_o=Y_train_o.reshape(-1,1)
#print_val_count(Y_train)
#searching of best hyperparameters
param_grid = {
    'C': [ 1, 10,100],
    'gamma': [0.01, 0.1, 1],
}

# Creo un'istanza della Support Vector Machine (SVM) con kernel RBF
svm_model = SVC(kernel='rbf')

# Creo un'istanza di GridSearchCV
f1_scorer = make_scorer(f1_score)
grid_search = GridSearchCV(svm_model, param_grid, scoring=f1_scorer, cv=5)
#grid_search = GridSearchCV(svm_model, param_grid, scoring=f1_score,  cv=5)

# Unisco il training set e il validation set per la grid search
#x_train_val = np.concatenate((x_train_o, x_val), axis=0)
#Y_train_val = np.concatenate((Y_train_o, Y_val), axis=0)
#x_train_val_shuffled, Y_train_val_shuffled = shuffle(x_train_val, Y_train_val, random_state=42)
# Eseguo la grid search sul training set e validation set combinati
grid_search.fit(x_train_o, Y_train_o)
best_svm=grid_search.best_estimator_
#print(x_train_val)
# Ottenere i migliori iperparametri
best_params = grid_search.best_params_

#valutazione sul test, calcolo accuracy precision, recall, f1_score
test_predictions = best_svm.predict(x_test)
# Calcola accuracy
accuracy = accuracy_score(Y_test[:test_predictions.shape[0]], test_predictions)
print(f'Accuracy: {accuracy}')

# Calcola F1 score
f1 = f1_score(Y_test[:test_predictions.shape[0]], test_predictions)
print(f'F1 Score: {f1}')

# Calcola precision
precision = precision_score(Y_test[:test_predictions.shape[0]], test_predictions)
print(f'Precision: {precision}')

# Calcola recall
recall = recall_score(Y_test[:test_predictions.shape[0]], test_predictions)
print(f'Recall: {recall}')
print(f'Params:{best_params}')





