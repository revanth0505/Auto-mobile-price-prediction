import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets, linear_model
data = pd.read_csv("Automobile_data (1).csv")
data=data.replace('?',0)
data['price'] = data['price'].astype(float)
temp = data.copy()
table = temp.groupby(['make'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left',on='make')
bins = [0,10000,20000,40000]
cars_bin=['Budget','Medium','Highend']
data['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
data=data.replace('gas',1)
data=data.replace('diesel',2)
data=data.replace('std',1)
data=data.replace('turbo',2)
data=data.replace('front',1)
data=data.replace('rear',2)
data=data.replace('convertible',1)
data=data.replace('hatchback',2)
data=data.replace('sedan',3)
data=data.replace('wagon',4)
data=data.replace('hardtop',5)
data=data.replace('Budget',1)
data=data.replace('Medium',2)
data=data.replace('Highend',3)
data=data.drop(['make','num-of-doors','drive-wheels','fuel-system','engine-type','num-of-cylinders','symboling'],axis=1)
x=data.drop(['price','carsrange'],axis=1).astype(float)
y=data['carsrange'].astype(float)
sclar=MinMaxScaler()
x=sclar.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
X_train=pd.DataFrame(X_train)
tree=DecisionTreeClassifier()
model=DecisionTreeRegressor()
reg=linear_model.LinearRegression()
tree.fit(X_train,y_train)
x=data.drop(['price','carsrange'],axis=1).astype(float)
y=data['price'].astype(float)
sclar=MinMaxScaler()
x=sclar.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
X_train=pd.DataFrame(X_train)
reg.fit(X_train,y_train)
st.header("Automobile price prediction")
normalised=st.number_input("Enter normalised losses(0-300)")
fueltype = st.selectbox('Select fuel type',
                      ('1. Gas', '2. Diesel'))
aspiration = st.selectbox('Select aspiration',
                      ('1. Std', '2. turbo'))
bodystyle = st.selectbox('Select bodystyle',
                      ('1. convertible', '2. hatchback','3. sedan','4. wagon','5.  hardtop'))
enginelocation = st.selectbox('Select Engine location',
                      ('1. Front', '2. Rear'))
wheelbase = st.number_input("Enter you wheelbase(80-120)")
length=st.number_input("Enter length(150-200)")
width=st.number_input("Enter width(50-100)")
height=st.number_input("Enter height(40-60)")
curbweight=st.number_input("Enter curbweight(1000-3500)")
enginesize=st.number_input("Enter enginesize(70-200)")
bore=st.number_input("Enter bore(2-4)")
stroke=st.number_input("Enter stroke(2-5)")
compression=st.number_input("Enter compression(8-12)")
horsepower=st.number_input("Enter horsepower(50-200)")
peakrpm=st.number_input("Enter peak rpm(4000-6000)")
citympg=st.number_input("Enter citympg(20-40)")
highwaympg=st.number_input("Enter highwaympg(20-50)")
if fueltype=='1. Gas':
    fueltype=1
if fueltype=='2. Diesel':
    fueltype=2
if aspiration=='1. Std':
    aspiration=1
if aspiration=='2. turbo':
    aspiration=2
if enginelocation=='1. Front':
    enginelocation=1
if enginelocation=='2. Rear':
    enginelocation=2
if bodystyle=='1. convertible':
    bodystyle=1
if bodystyle=='2. hatchback':
    bodystyle=2
if bodystyle=='3. sedan':
    bodystyle=3
if bodystyle=='4. wagon':
    bodystyle=4
if bodystyle=='5.  hardtop':
    bodystyle=5
inputs={
    'normalised':[normalised],
    'fueltype':[fueltype],
    'aspiration':[aspiration],
    'bodystyle':[bodystyle],
    'enginelocation':[enginelocation],
    'wheelbase':[wheelbase],
    'length':[length],
    'width':[width],
    'height':[height],
    'curbweight':[curbweight],
    'enginesize':[enginesize],
    'bore':[bore],
    'stroke':[stroke],
    'compression':[compression],
    'horsepower':[horsepower],
    'peakrpm':[peakrpm],
    'citympg':[citympg],
    'highwaympg':[highwaympg]
}
df_in = pd.DataFrame.from_dict(inputs)
val = tree.predict(df_in)
val1 = reg.predict(df_in)
if val[0]==1:
    val='Budget Car'
if val[0]==2:
    val='Modrate price car'
if val[0]==3:
    val='High Budget car'
if st.button("Predict"):
    st.success(f"{val}, Predicted Value in rupees : {val1[0]}")