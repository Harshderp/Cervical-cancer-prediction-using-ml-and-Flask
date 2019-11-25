from flask import Flask,render_template,request,url_for
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv(r'C:\Users\ADMIN\Desktop\Cervical_Cancer.csv', header=0, na_values='?')

dataset.drop(dataset.columns[[26, 27]], axis=1, inplace=True)

values = dataset.values
X = values[:, 0:33]
y = values[:, 33]

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

imputer = Imputer(strategy='median')
X = imputer.fit_transform(X)

iht = InstanceHardnessThreshold(random_state=12)
X= X.astype(int)
y = y.astype(int)
X, y = iht.fit_sample(X, y)
#print('Amount of each class after under-sampling: {0}'.format(Counter(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.predict(X_test)

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/send', methods = ['POST'])
def send(sum=sum):
    if request.method == 'POST':
        A = request.form['num1']
        B = request.form['num2']
        C = request.form['num3']
        D = request.form['num4']
        E = request.form['num5']
        F = request.form['num6']
        G = request.form['num7']
        H = request.form['num8']
        I = request.form['num9']
        J = request.form['num10']
        K = request.form['num11']
        L = request.form['num12']
        M = request.form['num13']
        N = request.form['num14']
        O = request.form['num15']
        P = request.form['num16']
        Q = request.form['num17']
        R = request.form['num18']
        S = request.form['num19']
        T = request.form['num20']
        U = request.form['num21']
        V = request.form['num22']
        W = request.form['num23']
        X = request.form['num24']
        Y = request.form['num25']
        Z = request.form['num26']
        A1 = request.form['num27']
        B1 = request.form['num28']
        C1 = request.form['num29']
        D1 = request.form['num30']
        F1 = request.form['num31']
        G1 = request.form['num32']
        H1 = request.form['num33']
        pred_args = [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,A1,B1,C1,D1,F1,G1,H1]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.astype(float)
        pred_args_arr = pred_args_arr.reshape(1,-1)
        sum = logreg.predict(pred_args_arr)
        return render_template('index.html',sum=sum)
    else:
        return render_template('index.html')
if __name__== '__main__' :
    app.run(debug=True)
