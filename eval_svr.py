from library import *
from functions import *

test_predictions=joblib.load("test_predictions.pkl")
Y_test=joblib.load("Y_test.pkl")
best_param=joblib.load("best_svr_params.pkl")
'''
joblib.dump(test_predictions,"test_pred_k=10_sm.pkl")
joblib.dump(Y_test,"Y_test_k=10_sm.pkl")
joblib.dump(test_predictions,"best_svr_k=10_sm.pkl")
'''
print(best_param)
print(test_predictions.shape)
print_val_count(Y_test)
soglia=0.4
test_p_c = np.where(test_predictions > soglia, 1, 0)
print_val_count(test_p_c)
#print_val_count(test_predictions)
# Calcolo scores
plt.plot(test_predictions)
plt.title('Grafico di un Array')
plt.xlabel('Indice')
plt.ylabel('Valore')
plt.show()
# Soglia per la classificazione
max=0
soglia1=5.5

'''''
for i in range(1000):
    soglia = i*0.1  # Puoi impostare la soglia in base alle tue esigenze
    test_p_c = np.where(test_predictions > soglia, 1, 0)
    f1 = f1_score(Y_test[:test_p_c.shape[0]], test_p_c,zero_division=1)
    if f1>max:
        max=f1
        soglia1=soglia
'''''
test_p_c = np.where(test_predictions > soglia1, 1, 0)
# Calcola le metriche di valutazione
accuracy = accuracy_score(Y_test[:test_p_c.shape[0]], test_p_c)
precision = precision_score(Y_test[:test_p_c.shape[0]], test_p_c)
recall = recall_score(Y_test[:test_p_c.shape[0]], test_p_c)
f1 = f1_score(Y_test[:test_p_c.shape[0]], test_p_c)

# Stampa le metriche
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Soglia 1:", soglia1)