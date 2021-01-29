#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import seaborn as sb
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm


# In[4]:


cars = pd.read_csv('C:\\Users\\umutd\\Desktop\\duzenlenmis.csv')
print("Dimension of our data set is: ")
print(cars.shape)
cars.head(50)


# In[183]:


cars.info()


# DATA PREPROCESSING

# In[184]:


details = pd.DataFrame(cars, columns =['ilan No', 'Marka','Seri', 'Model','Yıl','KM','Yakit','Vites',
'Motor_Gucu_hp','Motor_Hacmi_cc','Çekiş','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz_km/h',
'Hizlanma_0_100','Garanti','Takas','Durumu','Fiyat_TL']) 


# In[185]:


details.isnull().sum()


# I need to clear NaN values.We are going to replace NaN values with average of column.But I can do it for only numeric values.

# In[186]:


mean_Agirlik=cars['Agirlik_kg'].mean()


# In[187]:


cars['Agirlik_kg'].fillna(value=cars['Agirlik_kg'].mean(),inplace=True)


# In[188]:


mean_Bagaj_Kap_lt=cars['Bagaj_Kap_lt'].mean()


# In[189]:


cars['Bagaj_Kap_lt'].fillna(value=cars['Bagaj_Kap_lt'].mean(),inplace=True)


# In[190]:


mean_Genislik_mm=cars['Genislik_mm'].mean()


# In[191]:


cars['Genislik_mm'].fillna(value=cars['Genislik_mm'].mean(),inplace=True)


# In[192]:


mean_Yukseklik_mm=cars['Yukseklik_mm'].mean()


# In[193]:


cars['Yukseklik_mm'].fillna(value=cars['Yukseklik_mm'].mean(),inplace=True)


# In[194]:


mean_Uzunluk_mm=cars['Uzunluk_mm'].mean()


# In[195]:


cars['Uzunluk_mm'].fillna(value=cars['Uzunluk_mm'].mean(),inplace=True)


# In[196]:


mean_Koltuk_Sayisi=cars['Koltuk_Sayisi'].mean()


# In[197]:


cars['Koltuk_Sayisi'].fillna(value=cars['Koltuk_Sayisi'].mean(),inplace=True)


# In[198]:


cars['Tork_nm'].fillna(value=cars['Tork_nm'].mean(),inplace=True)


# In[199]:


cars= cars.rename(columns={'Max_hiz_km/h': 'Max_hiz'})


# In[200]:


mean_Max_hiz=cars['Max_hiz'].mean()


# In[201]:


cars['Max_hiz'].fillna(value=cars['Max_hiz'].mean(),inplace=True)


# In[202]:


cars['Hizlanma_0_100'].fillna(value=cars['Hizlanma_0_100'].mean(),inplace=True)


# In[203]:


details = pd.DataFrame(cars, columns =['ilan No', 'Marka','Seri', 'Model','Yıl','KM','Yakıt','Vites',
'Motor_Gucu_hp','Motor_Hacmi_cc','Çekiş','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz',
'Hizlanma_0_100','Garanti','Takas','Durumu','Fiyat_TL']) 


# In[204]:


details.isnull().sum()


# I am filling NaN values in 'Vites' column according to a condition.

# In[205]:


cars


# In[206]:


conditions = [cars['Yıl'] < 2010, cars['Yıl'].between(2010,2015), cars['Yıl'] > 2015]
values = ['Yarı Otomatik', 'Otomatik', 'Manuel']


# In[207]:


cars['Vites']=np.where(cars['Vites'].isnull(),
                      np.select(conditions,values),
                      cars['Vites'])


# In[208]:


cars


# In[209]:


cars.isnull().sum()


# There is no null value anymore .Now I must check if there are text is written wrong in categorical variables

# In[210]:


cars.Marka.unique()


# In[211]:


cars.Seri.unique()


# There are same seri name but it is written differend like '5 serisi 530 xDrive','VW CC 1.4 TSI' . Now I try to give them same name.VW CC 1.4 TSI = CC   and   5 serisi 530 xDrive = '5 Serisi'    in fact  

# In[212]:


cars = cars.replace(to_replace ="5 Serisi 530i xDrive", value ="5 Serisi") 
cars = cars.replace(to_replace ="VW CC 2.0 TDI", value ="CC") 
cars = cars.replace(to_replace ="E Serisi E 200 d", value ="E Serisi")
cars = cars.replace(to_replace ="3", value ="3 Serisi")
cars = cars.replace(to_replace ="Beetle 1.4 TSI", value ="Bettle")
cars = cars.replace(to_replace ="S60 1.5 T3", value ="S60")
cars = cars.replace(to_replace ="Civic 1.6i DTEC", value ="Civic")
cars = cars.replace(to_replace ="V90 Cross Country 2.0 D D5", value ="V90")
cars = cars.replace(to_replace ="208 1.2 PureTech", value ="208")
cars = cars.replace(to_replace ="VW CC 1.4 TSI", value ="CC")
cars = cars.replace(to_replace ="Ibiza 1.0 EcoTSI", value ="Ibiza")
cars = cars.replace(to_replace ="V40 1.5 T3", value ="V40")
cars = cars.replace(to_replace ="206 +", value ="206")
cars = cars.replace(to_replace ="Toledo 1.4 TDI", value ="Toledo")
cars = cars.replace(to_replace ="C3 1.2 PureTech", value ="C3")


# In[213]:


cars.Seri.unique()


# I need to convert categoric values to numeric values to machine learning algorithms work with them

# I will keep my original 'cars' data set.Because I will change categorical values.If I experience any problem I will back to origanal 'cars' data set again.

# In[214]:


car=cars.copy()


# In[215]:


car['Marka']=car['Marka']+' '+car['Seri']


# In[216]:


del car['Seri']


# In[217]:


pd.set_option('display.max_columns', None)


# In[218]:


pip install --upgrade category_encoders


# In[219]:


car


# In[220]:


car['Garanti'] = car['Garanti'].map({'Evet': 1, 'Hayır': 0})

car['Takas']=car['Takas'].map({'Evet':1,'Hayır':0})

car['Durumu']=car['Durumu'].map({'İkinci El':1,'Sıfır':0})


# In[221]:


car


# In[222]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
car["marka_code"] = lb_make.fit_transform(car["Marka"])


# In[223]:


car.head(50)


# In[224]:


car.marka_code.unique()


# In[225]:


car.Marka.nunique()


# In[226]:


car.marka_code.nunique()


# In[227]:


pd.get_dummies(car,columns=["Yakıt"])


# In[228]:


pd.get_dummies(car,columns=["Çekiş"])


# In[229]:


car


# In[230]:


car=pd.get_dummies(car,columns=["Çekiş"])


# In[231]:


car=pd.get_dummies(car,columns=["Vites"])


# In[232]:


car=pd.get_dummies(car,columns=["Yakıt"])


# In[233]:


car


# In[234]:


del car['Marka']


# In[235]:


car1=car.copy()
car2=car.copy()
# I will work with car2 just numeric variables


# DATA VISUALIZATION

# Heatmaps is use to find relations between two variables in a dataset

# In[236]:


car2


# In[237]:


car2=car2.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,17]]


# In[238]:


car2


# In[239]:


sb.heatmap(car2.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# In[ ]:





# Let's look relationship car price with other features.Horse power(Motor_Gucu)=0.76 has much correlation with car price.Accerelation(Hizlanma_0_100) has negative correlation with car price.It means while car_price increase , accerelation decrease .

# In[240]:


car_n=car.copy()
#car_n  will be normalize dataframe of car


# In[241]:


from sklearn.model_selection import train_test_split


# Now , all columns are numeric value

# I will divide our car dara into 67% for learning, and 33% for testing.

# In[242]:


from sklearn.model_selection import train_test_split


# In[243]:


car_n_train, car_n_test= train_test_split(car_n, train_size=0.67, test_size=0.33, random_state = 0)


# Now I want to scaling my features except unit8 data dtype.

# In[244]:


from sklearn.preprocessing import StandardScaler,scale
from sklearn.compose import ColumnTransformer


# In[245]:


scaling_columns=['Yıl','KM','Motor_Gucu_hp','Motor_Hacmi_cc','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz','Hizlanma_0_100','Fiyat_TL']


# In[246]:


features_train=car_n_train[scaling_columns]


# In[247]:


scaler=StandardScaler().fit(features_train.values)


# In[248]:


features_train=scaler.transform(features_train.values)


# In[249]:


car_n_train[scaling_columns]=features_train


# In[250]:


features_test=car_n_test[scaling_columns]


# In[251]:


scaler_test=StandardScaler().fit(features_test.values)


# In[252]:


features_test=scaler_test.transform(features_test.values)


# In[253]:


car_n_test[scaling_columns]=features_test


# In[254]:


car_n_test


# In[255]:


car_n_train


# car dataset separation.car data set values normalized.

# In[256]:


Y_train = car_n_train.loc[:,car_n_train.columns == 'Fiyat_TL']

X_train = car_n_train.loc[:, car_n_train.columns != 'Fiyat_TL']

Y_test = car_n_test.loc[:,car_n_test.columns == 'Fiyat_TL']

X_test = car_n_test.loc[:,car_n_test.columns != 'Fiyat_TL']


# car1 dataset separation. car1 dataset values are not normalized

# In[257]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# Applying MULTIPLE LINEAR REGRESSION

# In[258]:


from sklearn.linear_model import LinearRegression


# In[259]:


x_train


# In[260]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[261]:


y_pred=regressor.predict(x_test)


# In[262]:


car1


# In[263]:


y_pred=pd.DataFrame(y_pred)


# In[264]:


y_pred


# In[265]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred]], axis=1)


# ACTUAL PRICE-PREDICTION PRICE


# In[266]:


from sklearn.metrics import r2_score 


# In[267]:


r2_score(y_test,y_pred)


# I am going to study which features effect car price more with BACKWARD ELIMINATION metod.

# In[268]:


import statsmodels.api as sm


# In[269]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28,29]].values,axis=1)


# In[270]:


X_list=car1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28,29]].values


# In[271]:


X_list=np.array(X_list,dtype=float)


# In[272]:


X_list=pd.DataFrame(X_list)


# In[273]:


X_list


# In[274]:


model=sm.OLS(car1.iloc[:,17:18],X_list).fit()


# In[275]:


model.summary()


# It means if P>|t| value close to zero car price is affected from this feature more.When P>|t| value close to 1 , car price is not affected this feature.0 th feature's P>|t| value equals to 0.795.This feature correspond to 'ilan_no' feuture. It means ilan_no does not affect car price much . Same thing is valid also 8th column='Yukseklik',11th column 'Tork'

# In[276]:


x_train


# In[277]:


X_list


# First I am going to remove column which has maximum P>|t|=0.915 value.It is feature 8='Yukseklik'

# In[278]:


car12=car1.copy()
del car12['Yukseklik_mm']


# In[279]:


car1


# In[280]:


car12


# In[281]:


import statsmodels.api as sm

X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,
                   20,21,22,23,24,25,26,27,28]].values,axis=1)
X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,
                   20,21,22,23,24,25,26,27,28]].values
X_list=np.array(X_list,dtype=float)


# In[282]:


model=sm.OLS(car12.iloc[:,16:17],X_list).fit()


# In[283]:


model.summary()


# In[284]:


car12


# I am applying prediction again without 'Yukseklik_mm' column=8

# In[285]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28]]


# In[286]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28]]


# In[287]:


regressor.fit(x_train,y_train)
y_pred2=regressor.predict(x_test)


# In[288]:


y_pred2=pd.DataFrame(y_pred2)


# In[289]:


y_pred2


# In[290]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2]], axis=1)

## ACTUAL PRICE-PREDICTION1-PREDICTION2


# Some values are predicted better but some values are predicted worse.
# 0. row worse,1. better ,2. worse,3.better,4. worse ....

# In[291]:


x_test


# In[292]:


del car12['ilan No']


# In[293]:


car12


# In[294]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,
                   20,21,22,23,24,25,26,27]].values,axis=1)


# In[295]:


X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,
                   20,21,22,23,24,25,26,27]].values


# In[296]:


X_list=pd.DaX_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)


# In[297]:


model=sm.OLS(car12.iloc[:,15:16],X_list).fit()


# In[298]:


model.summary()


# In[299]:


x_train


# In[300]:


x_train=x_train.iloc[:,1:]


# In[301]:


x_train


# In[302]:


x_test=x_test.iloc[:,1:]


# In[ ]:





# In[303]:


regressor.fit(x_train,y_train)
y_pred3=regressor.predict(x_test)
y_pred3=pd.DataFrame(y_pred)


# In[304]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3]], axis=1)


# I will delete another biggest p>|t| value column='Tork_nm'

# In[305]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,
                   20,21,22,23,24,25]]


# In[306]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,
                   20,21,22,23,24,25]]


# In[307]:


regressor.fit(x_train,y_train)
y_pred4=regressor.predict(x_test)
y_pred4=pd.DataFrame(y_pred4)


# In[308]:


y_pred4


# In[309]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4]], axis=1)


# In[310]:


del car12['Tork_nm']


# In[311]:


car12


# In[312]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,
                   20,21,22,23,24,25,26]].values,axis=1)


# In[313]:


X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,
                   20,21,22,23,24,25,26]].values


# In[314]:


X_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)


# In[315]:


model=sm.OLS(car12.iloc[:,14:15],X_list).fit()
model.summary()


# 12. column='Takas' looks high p>|t| value .It means it does not affect car price much. I must remove it

# In[316]:


x_train


# In[317]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,
                   20,21,22,23,24]]


# In[318]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,
                   20,21,22,23,24]]


# In[319]:


regressor.fit(x_train,y_train)
y_pred5=regressor.predict(x_test)
y_pred5=pd.DataFrame(y_pred5)


# In[320]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4,y_pred5]], axis=1)


# In[321]:


del car12['Takas']


# In[322]:


car12


# In[323]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,
                   20,21,22,23,24,25]].values,axis=1)
X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,
                   20,21,22,23,24,25]].values
X_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)
model=sm.OLS(car12.iloc[:,13:14],X_list).fit()
model.summary()


# I will remove 6th column 'genislik' which has 0.325 value P>|t|

# In[324]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22]]


# In[325]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22]]


# In[326]:


regressor.fit(x_train,y_train)
y_pred6=regressor.predict(x_test)
y_pred6=pd.DataFrame(y_pred6)
pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6]], axis=1)


# In[327]:


print("Performanse of backward elimination steps")
performance={'Multiple Linear Reg.':[r2_score(y_test,y_pred)],
            'BE 1th step':[r2_score(y_test,y_pred2)],
            'BE 2th step':[r2_score(y_test,y_pred3)],
            'BE 3th step':[r2_score(y_test,y_pred4)],
             'BE 4th step':[r2_score(y_test,y_pred5)],
             'BE 5th step':[r2_score(y_test,y_pred6)],
             
            }
performance=pd.DataFrame(performance)
performance


# I could not get efficiency from BACKWARD elimination method.Prediction didn t become better.It had same performance

# In[328]:


X_train


# In[329]:


car1


# In[330]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# ilan_No column is a ID column.It may be coused to memorizing the system.
# So I am removing 'ilan_No' column.I saw that P value is pointless in categoric variables.So I will remove categoric variables too.

# In[331]:


x_train=x_train.iloc[:,1:14]
x_test=x_test.iloc[:,1:14]


# In[332]:


x_test


# In[333]:


x_train


# I will try to find which features are more or less important when predict price.

# In[334]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


# In[335]:


import statsmodels.api as sm
model=sm.OLS(y_pred,x_test)


# In[336]:


model.fit().summary()


# 'Yukseklik_mm' column has huge P value. So I am going to remove that column

# In[337]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12]]


# In[338]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12]]


# In[339]:


linear=LinearRegression()
linear.fit(x_train,y_train)
y_pred_linear=linear.predict(x_test)
model=sm.OLS(y_pred_linear,x_test)
model.fit().summary()


# R square increased just a little bit but it is not satisfying

# In[340]:


x_train


# In[341]:


linear=LinearRegression()
linear.fit(x_train,y_train)
y_pred_linear=linear.predict(x_test)
y_pred_linear=pd.DataFrame(y_pred)
y_pred_linear
pd.concat([d.reset_index(drop=True) for d in [y_test,y_pred_linear]], axis=1)


# In[161]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# In[ ]:





# In[162]:


x_train=x_train.iloc[:,1:14]
x_test=x_test.iloc[:,1:14]


# In[163]:


ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)


# In[164]:


lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)


# In[165]:


en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
y_pred_en = en.predict(x_test)


# In[166]:


bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
y_pred_bay= bayesian.predict(x_test)


# In[167]:


y_pred_linear=pd.DataFrame(y_pred_linear)
y_pred_ridge=pd.DataFrame(y_pred_ridge)
y_pred_lasso=pd.DataFrame(y_pred_lasso)
y_pred_bay=pd.DataFrame(y_pred_bay)
y_pred_en=pd.DataFrame(y_pred_en)


# In[168]:


pd.concat([d.reset_index(drop=True) for d in [y_test,y_pred_linear, y_pred_ridge,y_pred_lasso,y_pred_bay,y_pred_en]], axis=1)
#             LINEAR        RIDGE       LASSO         BAYES       ELASTICNET


# In[169]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred_en]], axis=1)
#ACTUAL PRICE-ELASTIC NET REGRESSION


# In[170]:


from sklearn.metrics import explained_variance_score as evs


# In[ ]:





# In[171]:


print('EXPLAINED VARIANCE SCORE:')
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LINEAR REGRESSION model is {}'.format(evs(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of RIDGE REGRESSION model is {}'.format(evs(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LASSO REGRESSION model is {}'.format(evs(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of BAYESSIAN REGRESSION model is {}'.format(evs(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of ELASTIC NET REGRESSION  is {}'.format(evs(y_test,y_pred_en)))


# In[172]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of RIDGE REGRESSION model is {}'.format(r2_score(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LASSO REGRESSION model is {}'.format(r2_score(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of BAYESSIAN REGRESSION model is {}'.format(r2_score(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of ELASTIC NET REGRESSION  is {}'.format(r2_score(y_test,y_pred_en)))


# In[173]:


x_test


# In[174]:


mycar={'Yıl':[2015], 'KM':[83000],'Motor_Gucu_hp':[105], 'Motor_Hacmi_cc':[1197],'Agirlik_kg':[1134]
       ,'Bagaj_Kap_lt':[380],'Genislik_mm':[1816],'Yukseklik_mm':[1459],'Uzunluk_mm':[4263],
       'Koltuk_Sayisi':[5],'Tork_nm':[175],'Max_hiz':[191],'Hizlanma_0_100':[10] }


# In[175]:


mycar=pd.DataFrame(mycar)


# In[176]:


mycar


# In[177]:


mycarprice=lasso.predict(mycar)


# In[178]:


mycarprice=pd.DataFrame(mycarprice)


# In[179]:


mycarprice


# In[180]:


mycarprice_linear=ridge.predict(mycar)


# In[181]:


mycarprice_linear


# In[182]:


mycarprice_bay= bayesian.predict(mycar)


# In[183]:


mycarprice_bay


# In[184]:


car1


# In[185]:


del car1['ilan No']


# In[186]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# In[187]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred_linear=lin_reg.predict(x_test)
y_pred_linear=pd.DataFrame(y_pred_linear)
pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred_linear]], axis=1)


# In[188]:


r2_score(y_test,y_pred_linear)


# Great! I increased R2 value from 0.72 to 0.75 for Linear Regression

# In[189]:


ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)
y_pred_ridge=pd.DataFrame(y_pred_ridge)


# In[190]:


lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)
y_pred_lasso=pd.DataFrame(y_pred_lasso)


# In[191]:


bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
y_pred_bay= bayesian.predict(x_test)
y_pred_bay=pd.DataFrame(y_pred_bay)


# In[192]:


en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
y_pred_en = en.predict(x_test)
y_pred_en=pd.DataFrame(y_pred_en)


# In[193]:


print('EXPLAINED VARIANCE SCORE:')
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LINEAR REGRESSION model is {}'.format(evs(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of RIDGE REGRESSION model is {}'.format(evs(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LASSO REGRESSION model is {}'.format(evs(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of BAYESSIAN REGRESSION model is {}'.format(evs(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of ELASTIC NET REGRESSION  is {}'.format(evs(y_test,y_pred_en)))


# In[194]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of RIDGE REGRESSION model is {}'.format(r2_score(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LASSO REGRESSION model is {}'.format(r2_score(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of BAYESSIAN REGRESSION model is {}'.format(r2_score(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of ELASTIC NET REGRESSION  is {}'.format(r2_score(y_test,y_pred_en)))


# Ridge regression is looking more succesfull.So I will input my car values and predict my car's price

# In[195]:


mycar={'Yıl':[2015], 'KM':[83000],'Motor_Gucu_hp':[105], 'Motor_Hacmi_cc':[1197],'Agirlik_kg':[1134]
       ,'Bagaj_Kap_lt':[380],'Genislik_mm':[1816],'Yukseklik_mm':[1459],'Uzunluk_mm':[4263],
       'Koltuk_Sayisi':[5],'Tork_nm':[175],'Max_hiz':[191],'Hizlanma_0_100':[10],
      'Garanti':[0],'Takas':[0],'Durumu':[1],'marka_code':[89],'Çekiş_4WD (Sürekli)':[0],
      'Çekiş_Arkadan İtiş':[0],'Çekiş_Önden Çekiş':[1],'Vites_Manuel':[1],'Vites_Otomatik':[0],
      'Vites_Yarı Otomatik':[0],'Yakıt_Benzin':[1],'Yakıt_Benzin & LPG':[0],
      'Yakıt_Dizel':[0],'Yakıt_Elektrik':[0],'Yakıt_Hybrid':[0]}


# In[196]:


mycar=pd.DataFrame(mycar)


# In[197]:


y_pred_ridge = ridge.predict(mycar)


# In[198]:


y_pred_ridge


# In[199]:


y_pred_linear=lin_reg.predict(mycar)
y_pred_ridge = ridge.predict(mycar)
y_pred_lasso = lasso.predict(mycar)
y_pred_bay= bayesian.predict(mycar)
y_pred_en = en.predict(mycar)


# In[200]:


print('LINEAR REGRESSION prediction for my car price',y_pred_linear)
print('RIDGE REGRESSION prediction for my car price',y_pred_ridge)
print('LASSO REGRESSION prediction for my car price',y_pred_lasso)
print('BAYESIAN REGRESSION prediction for my car price',y_pred_bay)
print('ELASTIC NET REGRESSION for my car price',y_pred_en)


# I am going to use BACKWARD ELIMINATION method for improve my models success

# In[201]:


car_b=car1.copy()


# In[202]:


sb.heatmap(car_b.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# I will remove columns which has close to zero colleration with price

# In[203]:


del car_b['Takas']
del car_b['Yakıt_Dizel']
del car_b['Yakıt_Hybrid']
del car_b['Vites_Yarı Otomatik']


# In[204]:


car_b_train, car_b_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car_b_train.loc[:,car_b_train.columns == 'Fiyat_TL']

x_train = car_b_train.loc[:, car_b_train.columns != 'Fiyat_TL']

y_test = car_b_test.loc[:,car_b_test.columns == 'Fiyat_TL']

x_test = car_b_test.loc[:,car_b_test.columns != 'Fiyat_TL']


# In[205]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred_linear=lin_reg.predict(x_test)
y_pred_linear=pd.DataFrame(y_pred_linear)

ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)
y_pred_ridge=pd.DataFrame(y_pred_ridge)

lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)
y_pred_lasso=pd.DataFrame(y_pred_lasso)

bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
y_pred_bay= bayesian.predict(x_test)
y_pred_bay=pd.DataFrame(y_pred_bay)

en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
y_pred_en = en.predict(x_test)
y_pred_en=pd.DataFrame(y_pred_en)


# In[206]:


print('EXPLAINED VARIANCE SCORE:')
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LINEAR REGRESSION model is {}'.format(evs(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of RIDGE REGRESSION model is {}'.format(evs(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of LASSO REGRESSION model is {}'.format(evs(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of BAYESSIAN REGRESSION model is {}'.format(evs(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of ELASTIC NET REGRESSION  is {}'.format(evs(y_test,y_pred_en)))


# In[207]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of RIDGE REGRESSION model is {}'.format(r2_score(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LASSO REGRESSION model is {}'.format(r2_score(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of BAYESSIAN REGRESSION model is {}'.format(r2_score(y_test,y_pred_bay)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of ELASTIC NET REGRESSION  is {}'.format(r2_score(y_test,y_pred_en)))


# System success increase a little bit %75 to % 76 . But it is not enough

# In[208]:


sb.heatmap(car_b.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# I will remove columns have low correlation with price 

# I removed 'Yukseklik' and 'markacode' correlation.Imlemented same regression models.But I git less success.R2 and corr values decreased to %75

# In[209]:


car_b


# In[210]:


car_b_train, car_b_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)


y_train = car_b_train.loc[:,car_b_train.columns == 'Fiyat_TL']

x_train = car_b_train.loc[:, car_b_train.columns == 'Motor_Gucu_hp']

y_test = car_b_test.loc[:,car_b_test.columns == 'Fiyat_TL']

x_test = car_b_test.loc[:,car_b_test.columns == 'Motor_Gucu_hp']


# In[211]:


lin_reg = LinearRegression() 
  
lin_reg.fit(x_train, y_train) 


# In[212]:


from sklearn.preprocessing import PolynomialFeatures


# SECOND ORDER POLYNOMIAL REGRESSION

# In[213]:


poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x_train) 
  
poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 
y_pred_poly=lin2.predict(poly.fit_transform(x_test))


# In[214]:


plt.scatter(x_train, y_train, color = 'blue') 
  
plt.plot(x_train, lin_reg.predict(x_train), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('horse power') 
plt.ylabel('Price') 
  
plt.show() 


# In[215]:


plt.scatter(x_train, y_train, color = 'blue') 
  
plt.plot(x_train, lin2.predict(poly.fit_transform(x_train)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Horse Power') 
plt.ylabel('Pirice') 
  
plt.show() 


# In[216]:


mycar_hp={'Motor_Gucu_hp':[105]}


# In[217]:


mycar_hp=pd.DataFrame(mycar_hp)


# In[218]:


y_pred_lin=lin_reg.predict(x_test)


# In[219]:


y_pred_lin=pd.DataFrame(y_pred_lin)


# In[220]:


mycar_pred_price=lin_reg.predict(mycar_hp)


# In[221]:


mycar_pred_price


# In[222]:


lin2.predict(poly.fit_transform(mycar_hp)) 


# In[223]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_lin)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of 2nd ORDER POLYNOMIAL REGRESSION model is {}'.format(r2_score(y_test,y_pred_poly)))


# In[224]:


print('Explained Variance Score of LINEAR REGRESSION model is {}'.format(evs(y_test,y_pred_lin)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of POLYNOMIAL  is {}'.format(evs(y_test,y_pred_poly)))


# 3rd ORDER POLLYNOMIAL REGRESSION 

# In[225]:


poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(x_train) 
  
poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 
y_pred_poly=lin2.predict(poly.fit_transform(x_test))


# In[226]:


mycar_hp={'Motor_Gucu_hp':[105]}
mycar_hp=pd.DataFrame(mycar_hp)
y_pred_lin=lin_reg.predict(x_test)
y_pred_lin=pd.DataFrame(y_pred_lin)
mycar_pred_price=lin_reg.predict(mycar_hp)
lin2.predict(poly.fit_transform(mycar_hp)) 


# In[227]:


print('Explained R-squared of 2nd ORDER POLYNOMIAL REGRESSION model is {}'.format(r2_score(y_test,y_pred_poly)))


# When I increase degree of polynomial regression , R2 value decrease . So second order polynomial regression found more usefull solution

# In[228]:


car_b


# STEP FORWARD FEATURE SELECTION

# In[230]:


get_ipython().system('pip install mlxtend')


# In[232]:


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[240]:


car_train, car_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car_b_train.loc[:,car_train.columns == 'Fiyat_TL']

x_train = car_b_train.loc[:, car_train.columns != 'Fiyat_TL']

y_test = car_b_test.loc[:,car_test.columns == 'Fiyat_TL']

x_test = car_b_test.loc[:,car_test.columns != 'Fiyat_TL']

x_train=x_train.iloc[:,0:13]
x_test=x_test.iloc[:,0:13]


# In[247]:


sfs=SFS(RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=-1),
       k_features=7,
        forward=True,
        floating=False,
        verbose=2,
        scoring='accuracy',
        cv=4,
        n_jobs=-1
       ).fit(x_train,y_train)


# In[ ]:




