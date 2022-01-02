import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

breast_cancer = pd.read_csv("data.csv")
col_names = breast_cancer.columns


# Dropping the unknown data column
breast_cancer = breast_cancer.drop("Unnamed: 32",axis=1)

# Resetting the index column
breast_cancer.set_index("id", inplace=True)


# Mapping the diagnosis column or converting categorical to numeric
breast_cancer["diagnosis"] = breast_cancer["diagnosis"].map({"M":1,"B":0})

# Dividing the independent and dependent variable
x = breast_cancer.drop("diagnosis",axis=1)
y = breast_cancer["diagnosis"]


# Running Correlation check and deleting correlated columns
def corr_relation_check(data_tab):
    dat = data_tab                       # Referrencing 
    cortab = dat.corr() >0.75
    b=1
    c=2
    d = str
    for i in cortab:
        for j in cortab.iloc[b:c,b:]:
            if(str(cortab.loc[i,j])=="True"):
                if(i in dat and j in dat):
                    print(f"{i} and {j} are strongly corelated, Hence we are deleting any {i}")   # by default we are deleting 
                    try:
                        del dat[i]
                    except:
                        pass
        b+=1
        c+=1
    return dat

corr_relation_check(x)


# Training and Testing Data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)




# Hyper parameter tuning

# grid = [{"min_samples_split":[100,150,200],
#          "max_depth":[4,5,6],
#          "min_samples_leaf":[20,25,30,40],
#          "bootstrap":["True","False"]
# }]

# cv_rf = GridSearchCV(model,grid,cv=10,n_jobs=3)
# cv_rf.fit(x_train,y_train)
# print("The best parameters are: ",cv_rf.best_params_)

model = RandomForestClassifier(random_state=101)

# model.set_params(warm_start=True, oob_score=True)

# Calculating number of n_estimators to be used
# min_estimator = 100
# max_estimator = 1000

# error_rate = {}
# for i in range(min_estimator, max_estimator+1):
#     model.set_params(n_estimators=i)
#     model.fit(x_train, y_train)
#     oob_error = 1-model.oob_score_
#     error_rate[i] = oob_error

# Modelling
model.set_params(n_estimators=520,bootstrap= True, max_depth = 4, min_samples_leaf= 40, min_samples_split= 100)

model.fit(x_train,y_train)

accuracy = model.score(x_train,y_train)
print("accuracy of training set:",accuracy)
accuracy = model.score(x_test,y_test)
print("accuracy of training set:",accuracy)


model_dep = joblib.dump(model,"breast_cancer.pkl")
# pred = model.predict(x_test)

# from sklearn.metrics import precision_score, recall_score

# precision = precision_score(y_test, pred)

# precision

# recall = recall_score(y_test, pred)

# recall

# confusion_matrix(y_test, pred)

# y_test.shape

# (y_test==1).sum()

# x_train.columns

# x_train

# a = classification_report(y_test,pred,target_names=dx)

# dx = ["Benign", "Malignant"]

