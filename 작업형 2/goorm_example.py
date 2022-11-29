import pandas as pd
x_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
x_test = pd.read_csv('data/X_test.csv')

x_train.drop(columns = 'cust_id', inplace = True)
y_train.drop(columns = 'cust_id', inplace = True)
x_test_cust_id = x_test.pop('cust_id')

# print(x_train.isnull().sum())
x_train['환불금액'].fillna(0, inplace = True)
x_test['환불금액'].fillna(0, inplace = True)
# print(x_train.isnull().sum())
# print(x_test.isnull().sum())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x_train['주구매상품'] = encoder.fit_transform(x_train['주구매상품'])
x_train['주구매지점'] = encoder.fit_transform(x_train['주구매지점'])
x_test['주구매상품'] = encoder.fit_transform(x_test['주구매상품'])
x_test['주구매지점'] = encoder.fit_transform(x_test['주구매지점'])

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns = x_test.columns)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth = 5, n_estimators = 500, criterion = 'entropy', random_state = 10)

# 오류 메세지가 뜬다면 y_train에 .values.ravel() 붙여주자.
model.fit(x_train, y_train.values.ravel())
y_val_predict_proba = model.predict_proba(x_val)
y_val_predict_proba = pd.DataFrame(y_val_predict_proba)
y_val_predict_proba.columns = ['0', 'gender']
y_val_predict_proba.drop(columns = ['0'], inplace = True)
# print(y_val_predict_proba)
# print(y_val)

from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y_val, y_val_predict_proba))
# max_depth = 5, n_estimators = 100, criterion = 'entropy' -> score : 0.6733242706263725
# max_depth = 5, n_estimators = 100, criterion = 'gini' -> scoer : 0.6297063592145559
# max_depth = 7, n_estimators = 100, criterion = 'entropy' -> score : 0.6447587574355585
# max_depth = 5, n_estimators = 500, criterion = 'entropy' -> score : 0.6762119981639068

y_test_predict_proba = model.predict_proba(x_test)
y_test_predict_proba = pd.DataFrame(y_test_predict_proba)
y_test_predict_proba.columns = ['0', 'gender']
y_test_predict_proba.drop(columns = ['0'], inplace = True)
result = pd.concat([x_test_cust_id, y_test_predict_proba], axis = 1)
# print(result)

result.to_csv('12345.csv', index = False)
