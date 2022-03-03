from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def get_acc_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)