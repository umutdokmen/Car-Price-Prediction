#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


cars = pd.read_csv('C:\\Users\\umutd\\Desktop\\duzenlenmis.csv')
print("Dimension of our data set is: ")
print(cars.shape)
cars.head(50)


# In[3]:


cars.info()


# DATA PREPROCESSING

# In[4]:


details = pd.DataFrame(cars, columns =['ilan No', 'Marka','Seri', 'Model','Yıl','KM','Yakit','Vites',
'Motor_Gucu_hp','Motor_Hacmi_cc','Çekiş','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz_km/h',
'Hizlanma_0_100','Garanti','Takas','Durumu','Fiyat_TL']) 


# In[5]:


details.isnull().sum()


# I need to clear NaN values.We are going to replace NaN values with average of column.But I can do it for only numeric values.

# In[6]:


mean_Agirlik=cars['Agirlik_kg'].mean()


# In[7]:


cars['Agirlik_kg'].fillna(value=cars['Agirlik_kg'].mean(),inplace=True)


# In[8]:


mean_Bagaj_Kap_lt=cars['Bagaj_Kap_lt'].mean()


# In[9]:


cars['Bagaj_Kap_lt'].fillna(value=cars['Bagaj_Kap_lt'].mean(),inplace=True)


# In[10]:


mean_Genislik_mm=cars['Genislik_mm'].mean()


# In[11]:


cars['Genislik_mm'].fillna(value=cars['Genislik_mm'].mean(),inplace=True)


# In[12]:


mean_Yukseklik_mm=cars['Yukseklik_mm'].mean()


# In[13]:


cars['Yukseklik_mm'].fillna(value=cars['Yukseklik_mm'].mean(),inplace=True)


# In[14]:


mean_Uzunluk_mm=cars['Uzunluk_mm'].mean()


# In[15]:


cars['Uzunluk_mm'].fillna(value=cars['Uzunluk_mm'].mean(),inplace=True)


# In[16]:


mean_Koltuk_Sayisi=cars['Koltuk_Sayisi'].mean()


# In[17]:


cars['Koltuk_Sayisi'].fillna(value=cars['Koltuk_Sayisi'].mean(),inplace=True)


# In[18]:


cars['Tork_nm'].fillna(value=cars['Tork_nm'].mean(),inplace=True)


# In[19]:


cars= cars.rename(columns={'Max_hiz_km/h': 'Max_hiz'})


# In[20]:


mean_Max_hiz=cars['Max_hiz'].mean()


# In[21]:


cars['Max_hiz'].fillna(value=cars['Max_hiz'].mean(),inplace=True)


# In[22]:


cars['Hizlanma_0_100'].fillna(value=cars['Hizlanma_0_100'].mean(),inplace=True)


# In[23]:


details = pd.DataFrame(cars, columns =['ilan No', 'Marka','Seri', 'Model','Yıl','KM','Yakıt','Vites',
'Motor_Gucu_hp','Motor_Hacmi_cc','Çekiş','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz',
'Hizlanma_0_100','Garanti','Takas','Durumu','Fiyat_TL']) 


# In[24]:


details.isnull().sum()


# I am filling NaN values in 'Vites' column according to a condition.

# In[25]:


cars


# In[26]:


conditions = [cars['Yıl'] < 2010, cars['Yıl'].between(2010,2015), cars['Yıl'] > 2015]
values = ['Yarı Otomatik', 'Otomatik', 'Manuel']


# In[27]:


cars['Vites']=np.where(cars['Vites'].isnull(),
                      np.select(conditions,values),
                      cars['Vites'])


# In[28]:


cars


# In[29]:


cars.isnull().sum()


# There is no null value anymore .Now I must check if there are text is written wrong in categorical variables

# In[30]:


cars.Marka.unique()


# In[31]:


cars.Seri.unique()


# There are same seri name but it is written differend like '5 serisi 530 xDrive','VW CC 1.4 TSI' . Now I try to give them same name.VW CC 1.4 TSI = CC   and   5 serisi 530 xDrive = '5 Serisi'    in fact  

# In[32]:


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


# In[33]:


cars.Seri.unique()


# I need to convert categoric values to numeric values to machine learning algorithms work with them

# I will keep my original 'cars' data set.Because I will change categorical values.If I experience any problem I will back to origanal 'cars' data set again.

# In[34]:


car=cars.copy()


# In[35]:


car['Marka']=car['Marka']+' '+car['Seri']


# In[36]:


del car['Seri']


# In[37]:


pd.set_option('display.max_columns', None)


# In[38]:


pip install --upgrade category_encoders


# In[39]:


car


# In[40]:


car['Garanti'] = car['Garanti'].map({'Evet': 1, 'Hayır': 0})

car['Takas']=car['Takas'].map({'Evet':1,'Hayır':0})

car['Durumu']=car['Durumu'].map({'İkinci El':1,'Sıfır':0})


# In[41]:


car


# In[42]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
car["marka_code"] = lb_make.fit_transform(car["Marka"])


# In[43]:


car.head(50)


# In[44]:


car.marka_code.unique()


# In[45]:


car.Marka.nunique()


# In[46]:


car.marka_code.nunique()


# In[47]:


pd.get_dummies(car,columns=["Yakıt"])


# In[48]:


pd.get_dummies(car,columns=["Çekiş"])


# In[49]:


car


# In[50]:


car=pd.get_dummies(car,columns=["Çekiş"])


# In[51]:


car=pd.get_dummies(car,columns=["Vites"])


# In[52]:


car=pd.get_dummies(car,columns=["Yakıt"])


# In[53]:


car


# In[54]:


del car['Marka']


# In[55]:


car1=car.copy()
car2=car.copy()
# I will work with car2 just numeric variables


# DATA VISUALIZATION

# Heatmaps is use to find relations between two variables in a dataset

# In[56]:


car2


# In[57]:


car2=car2.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,17]]


# In[58]:


car2


# In[59]:


sb.heatmap(car2.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# In[ ]:





# Let's look relationship car price with other features.Horse power(Motor_Gucu)=0.76 has much correlation with car price.Accerelation(Hizlanma_0_100) has negative correlation with car price.It means while car_price increase , accerelation decrease .

# In[60]:


car_n=car.copy()
#car_n  will be normalize dataframe of car


# In[61]:


from sklearn.model_selection import train_test_split


# Now , all columns are numeric value

# I will divide our car dara into 67% for learning, and 33% for testing.

# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


car_n_train, car_n_test= train_test_split(car_n, train_size=0.67, test_size=0.33, random_state = 0)


# Now I want to scaling my features except unit8 data dtype.

# In[64]:


from sklearn.preprocessing import StandardScaler,scale
from sklearn.compose import ColumnTransformer


# In[65]:


scaling_columns=['Yıl','KM','Motor_Gucu_hp','Motor_Hacmi_cc','Agirlik_kg','Bagaj_Kap_lt','Genislik_mm'
,'Yukseklik_mm','Uzunluk_mm','Koltuk_Sayisi','Tork_nm','Max_hiz','Hizlanma_0_100','Fiyat_TL']


# In[66]:


features_train=car_n_train[scaling_columns]


# In[67]:


scaler=StandardScaler().fit(features_train.values)


# In[68]:


features_train=scaler.transform(features_train.values)


# In[69]:


car_n_train[scaling_columns]=features_train


# In[70]:


features_test=car_n_test[scaling_columns]


# In[71]:


scaler_test=StandardScaler().fit(features_test.values)


# In[72]:


features_test=scaler_test.transform(features_test.values)


# In[73]:


car_n_test[scaling_columns]=features_test


# In[74]:


car_n_test


# In[75]:


car_n_train


# car dataset separation.car data set values normalized.

# In[76]:


Y_train = car_n_train.loc[:,car_n_train.columns == 'Fiyat_TL']

X_train = car_n_train.loc[:, car_n_train.columns != 'Fiyat_TL']

Y_test = car_n_test.loc[:,car_n_test.columns == 'Fiyat_TL']

X_test = car_n_test.loc[:,car_n_test.columns != 'Fiyat_TL']


# car1 dataset separation. car1 dataset values are not normalized

# In[77]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# Applying MULTIPLE LINEAR REGRESSION

# In[78]:


from sklearn.linear_model import LinearRegression


# In[79]:


x_train


# In[80]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[81]:


y_pred=regressor.predict(x_test)


# In[82]:


car1


# In[83]:


y_pred=pd.DataFrame(y_pred)


# In[84]:


y_pred


# In[85]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred]], axis=1)


# ACTUAL PRICE-PREDICTION PRICE


# In[86]:


from sklearn.metrics import r2_score 


# In[87]:


r2_score(y_test,y_pred)


# I am going to study which features effect car price more with BACKWARD ELIMINATION metod.

# In[88]:


import statsmodels.api as sm


# In[89]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28,29]].values,axis=1)


# In[90]:


X_list=car1.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28,29]].values


# In[91]:


X_list=np.array(X_list,dtype=float)


# In[92]:


X_list=pd.DataFrame(X_list)


# In[93]:


X_list


# In[94]:


model=sm.OLS(car1.iloc[:,17:18],X_list).fit()


# In[95]:


model.summary()


# It means if P>|t| value close to zero car price is affected from this feature more.When P>|t| value close to 1 , car price is not affected this feature.0 th feature's P>|t| value equals to 0.795.This feature correspond to 'ilan_no' feuture. It means ilan_no does not affect car price much . Same thing is valid also 8th column='Yukseklik',11th column 'Tork'

# In[96]:


x_train


# In[97]:


X_list


# First I am going to remove column which has maximum P>|t|=0.915 value.It is feature 8='Yukseklik'

# In[98]:


car12=car1.copy()
del car12['Yukseklik_mm']


# In[99]:


car1


# In[100]:


car12


# In[101]:


import statsmodels.api as sm

X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,
                   20,21,22,23,24,25,26,27,28]].values,axis=1)
X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,
                   20,21,22,23,24,25,26,27,28]].values
X_list=np.array(X_list,dtype=float)


# In[102]:


model=sm.OLS(car12.iloc[:,16:17],X_list).fit()


# In[103]:


model.summary()


# In[104]:


car12


# I am applying prediction again without 'Yukseklik_mm' column=8

# In[105]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28]]


# In[106]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,
                   20,21,22,23,24,25,26,27,28]]


# In[107]:


regressor.fit(x_train,y_train)
y_pred2=regressor.predict(x_test)


# In[108]:


y_pred2=pd.DataFrame(y_pred2)


# In[109]:


y_pred2


# In[110]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2]], axis=1)

## ACTUAL PRICE-PREDICTION1-PREDICTION2


# Some values are predicted better but some values are predicted worse.
# 0. row worse,1. better ,2. worse,3.better,4. worse ....

# In[111]:


x_test


# In[112]:


del car12['ilan No']


# In[113]:


car12


# In[114]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,
                   20,21,22,23,24,25,26,27]].values,axis=1)


# In[115]:


X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,
                   20,21,22,23,24,25,26,27]].values


# In[116]:


X_list=pd.DaX_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)


# In[117]:


model=sm.OLS(car12.iloc[:,15:16],X_list).fit()


# In[118]:


model.summary()


# In[119]:


x_train


# In[120]:


x_train=x_train.iloc[:,1:]


# In[121]:


x_train


# In[122]:


x_test=x_test.iloc[:,1:]


# In[ ]:





# In[123]:


regressor.fit(x_train,y_train)
y_pred3=regressor.predict(x_test)
y_pred3=pd.DataFrame(y_pred)


# In[124]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3]], axis=1)


# I will delete another biggest p>|t| value column='Tork_nm'

# In[125]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,
                   20,21,22,23,24,25]]


# In[126]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,
                   20,21,22,23,24,25]]


# In[127]:


regressor.fit(x_train,y_train)
y_pred4=regressor.predict(x_test)
y_pred4=pd.DataFrame(y_pred4)


# In[128]:


y_pred4


# In[129]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4]], axis=1)


# In[130]:


del car12['Tork_nm']


# In[131]:


car12


# In[132]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,
                   20,21,22,23,24,25,26]].values,axis=1)


# In[133]:


X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,
                   20,21,22,23,24,25,26]].values


# In[134]:


X_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)


# In[135]:


model=sm.OLS(car12.iloc[:,14:15],X_list).fit()
model.summary()


# 12. column='Takas' looks high p>|t| value .It means it does not affect car price much. I must remove it

# In[136]:


x_train


# In[137]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,
                   20,21,22,23,24]]


# In[138]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,
                   20,21,22,23,24]]


# In[139]:


regressor.fit(x_train,y_train)
y_pred5=regressor.predict(x_test)
y_pred5=pd.DataFrame(y_pred5)


# In[140]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4,y_pred5]], axis=1)


# In[141]:


del car12['Takas']


# In[142]:


car12


# In[143]:


X=np.append(arr=np.ones((971,1)).astype(int),values=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,
                   20,21,22,23,24,25]].values,axis=1)
X_list=car12.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,
                   20,21,22,23,24,25]].values
X_list=np.array(X_list,dtype=float)
X_list=pd.DataFrame(X_list)
model=sm.OLS(car12.iloc[:,13:14],X_list).fit()
model.summary()


# I will remove 6th column 'genislik' which has 0.325 value P>|t|

# In[144]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22]]


# In[145]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,18,19,
                   20,21,22]]


# In[146]:


regressor.fit(x_train,y_train)
y_pred6=regressor.predict(x_test)
y_pred6=pd.DataFrame(y_pred6)
pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred,y_pred2,y_pred3,y_pred4,y_pred5,y_pred6]], axis=1)


# In[147]:


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

# In[148]:


X_train


# In[149]:


car1


# In[150]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# ilan_No column is a ID column.It may be coused to memorizing the system.
# So I am removing 'ilan_No' column.I saw that P value is pointless in categoric variables.So I will remove categoric variables too.

# In[151]:


x_train=x_train.iloc[:,1:14]
x_test=x_test.iloc[:,1:14]


# In[152]:


x_test


# In[153]:


x_train


# I will try to find which features are more or less important when predict price.

# In[154]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


# In[155]:


import statsmodels.api as sm
model=sm.OLS(y_pred,x_test)


# In[156]:


model.fit().summary()


# 'Yukseklik_mm' column has huge P value. So I am going to remove that column

# In[157]:


x_train=x_train.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12]]


# In[158]:


x_test=x_test.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12]]


# In[159]:


linear=LinearRegression()
linear.fit(x_train,y_train)
y_pred_linear=linear.predict(x_test)
model=sm.OLS(y_pred_linear,x_test)
model.fit().summary()


# R square increased just a little bit but it is not satisfying

# In[160]:


x_train


# In[161]:


linear=LinearRegression()
linear.fit(x_train,y_train)
y_pred_linear=linear.predict(x_test)
y_pred_linear=pd.DataFrame(y_pred)
y_pred_linear
pd.concat([d.reset_index(drop=True) for d in [y_test,y_pred_linear]], axis=1)


# In[162]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# In[ ]:





# In[163]:


x_train=x_train.iloc[:,1:14]
x_test=x_test.iloc[:,1:14]


# In[164]:


ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)


# In[165]:


lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)


# In[166]:


en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
y_pred_en = en.predict(x_test)


# In[167]:


bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
y_pred_bay= bayesian.predict(x_test)


# In[168]:


y_pred_linear=pd.DataFrame(y_pred_linear)
y_pred_ridge=pd.DataFrame(y_pred_ridge)
y_pred_lasso=pd.DataFrame(y_pred_lasso)
y_pred_bay=pd.DataFrame(y_pred_bay)
y_pred_en=pd.DataFrame(y_pred_en)


# In[169]:


pd.concat([d.reset_index(drop=True) for d in [y_test,y_pred_linear, y_pred_ridge,y_pred_lasso,y_pred_bay,y_pred_en]], axis=1)
#             LINEAR        RIDGE       LASSO         BAYES       ELASTICNET


# In[170]:


pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred_en]], axis=1)
#ACTUAL PRICE-ELASTIC NET REGRESSION


# In[171]:


from sklearn.metrics import explained_variance_score as evs


# In[ ]:





# In[172]:


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


# In[173]:


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


# In[174]:


x_test


# In[175]:


mycar={'Yıl':[2015], 'KM':[83000],'Motor_Gucu_hp':[105], 'Motor_Hacmi_cc':[1197],'Agirlik_kg':[1134]
       ,'Bagaj_Kap_lt':[380],'Genislik_mm':[1816],'Yukseklik_mm':[1459],'Uzunluk_mm':[4263],
       'Koltuk_Sayisi':[5],'Tork_nm':[175],'Max_hiz':[191],'Hizlanma_0_100':[10] }


# In[176]:


mycar=pd.DataFrame(mycar)


# In[177]:


mycar


# In[178]:


mycarprice=lasso.predict(mycar)


# In[179]:


mycarprice=pd.DataFrame(mycarprice)


# In[180]:


mycarprice


# In[181]:


mycarprice_linear=ridge.predict(mycar)


# In[182]:


mycarprice_linear


# In[183]:


mycarprice_bay= bayesian.predict(mycar)


# In[184]:


mycarprice_bay


# In[185]:


car1


# In[186]:


del car1['ilan No']


# In[187]:


car1_train, car1_test= train_test_split(car1, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# In[188]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)
y_pred_linear=lin_reg.predict(x_test)
y_pred_linear=pd.DataFrame(y_pred_linear)
pd.concat([d.reset_index(drop=True) for d in [y_test, y_pred_linear]], axis=1)


# In[189]:


r2_score(y_test,y_pred_linear)


# Great! I increased R2 value from 0.72 to 0.75 for Linear Regression

# In[190]:


ridge = Ridge(alpha = 0.5)
ridge.fit(x_train, y_train)
y_pred_ridge = ridge.predict(x_test)
y_pred_ridge=pd.DataFrame(y_pred_ridge)


# In[191]:


lasso = Lasso(alpha = 0.01)
lasso.fit(x_train, y_train)
y_pred_lasso = lasso.predict(x_test)
y_pred_lasso=pd.DataFrame(y_pred_lasso)


# In[192]:


bayesian = BayesianRidge()
bayesian.fit(x_train, y_train)
y_pred_bay= bayesian.predict(x_test)
y_pred_bay=pd.DataFrame(y_pred_bay)


# In[193]:


en = ElasticNet(alpha = 0.01)
en.fit(x_train, y_train)
y_pred_en = en.predict(x_test)
y_pred_en=pd.DataFrame(y_pred_en)


# In[194]:


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


# In[195]:



print('-------------------------------------------------------------------------------')
print('Explained R-squared of BAYESIAN REGRESSION model is {}'.format(r2_score(y_test,y_pred_ridge)))


# Bayesian regression is looking more succesfull.So I will input my car values and predict my car's price

# In[196]:


mycar={'Yıl':[2015], 'KM':[83000],'Motor_Gucu_hp':[105], 'Motor_Hacmi_cc':[1197],'Agirlik_kg':[1134]
       ,'Bagaj_Kap_lt':[380],'Genislik_mm':[1816],'Yukseklik_mm':[1459],'Uzunluk_mm':[4263],
       'Koltuk_Sayisi':[5],'Tork_nm':[175],'Max_hiz':[191],'Hizlanma_0_100':[10],
      'Garanti':[0],'Takas':[0],'Durumu':[1],'marka_code':[89],'Çekiş_4WD (Sürekli)':[0],
      'Çekiş_Arkadan İtiş':[0],'Çekiş_Önden Çekiş':[1],'Vites_Manuel':[1],'Vites_Otomatik':[0],
      'Vites_Yarı Otomatik':[0],'Yakıt_Benzin':[1],'Yakıt_Benzin & LPG':[0],
      'Yakıt_Dizel':[0],'Yakıt_Elektrik':[0],'Yakıt_Hybrid':[0]}


# In[197]:


mycar=pd.DataFrame(mycar)


# In[198]:



y_pred_bay= bayesian.predict(mycar)


# In[199]:



print('BAYESIAN REGRESSION prediction for my car price',y_pred_bay)


# I am going to use BACKWARD ELIMINATION method for improve my models success

# In[200]:


car_b=car1.copy()


# In[201]:


sb.heatmap(car_b.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# I will remove columns which has close to zero colleration with price

# In[202]:


del car_b['Takas']
del car_b['Yakıt_Dizel']
del car_b['Yakıt_Hybrid']
del car_b['Vites_Yarı Otomatik']


# In[203]:


car_b_train, car_b_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car_b_train.loc[:,car_b_train.columns == 'Fiyat_TL']

x_train = car_b_train.loc[:, car_b_train.columns != 'Fiyat_TL']

y_test = car_b_test.loc[:,car_b_test.columns == 'Fiyat_TL']

x_test = car_b_test.loc[:,car_b_test.columns != 'Fiyat_TL']


# In[204]:


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


# In[205]:


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


# In[206]:


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


# System success increase a little bit %75 to % 76 .

# In[207]:


car_b


# sample: ford focus 1.5 TDCi ST Line

# In[208]:


samplecar={'Yıl':[2018], 'KM':[33.500],'Motor_Gucu_hp':[120], 'Motor_Hacmi_cc':[1500],'Agirlik_kg':[1398]
       ,'Bagaj_Kap_lt':[511],'Genislik_mm':[1848],'Yukseklik_mm':[1454],'Uzunluk_mm':[4647],
       'Koltuk_Sayisi':[5],'Tork_nm':[300],'Max_hiz':[196],'Hizlanma_0_100':[10],
      'Garanti':[1],'Durumu':[1],'marka_code':[34],'Çekiş_4WD (Sürekli)':[0],
      'Çekiş_Arkadan İtiş':[0],'Çekiş_Önden Çekiş':[1],'Vites_Manuel':[0],'Vites_Otomatik':[1],
      'Yakıt_Benzin':[0],'Yakıt_Benzin & LPG':[0],
      'Yakıt_Elektrik':[0]}


# In[209]:


samplecar=pd.DataFrame(samplecar)


# In[210]:


y_pred_linear=lin_reg.predict(samplecar)
y_pred_ridge = ridge.predict(samplecar)
y_pred_lasso = lasso.predict(samplecar)
y_pred_bay= bayesian.predict(samplecar)
y_pred_en = en.predict(samplecar)
print('LINEAR REGRESSION prediction for my car price',y_pred_linear)
print('RIDGE REGRESSION prediction for my car price',y_pred_ridge)
print('LASSO REGRESSION prediction for my car price',y_pred_lasso)
print('BAYESIAN REGRESSION prediction for my car price',y_pred_bay)
print('ELASTIC NET REGRESSION for my car price',y_pred_en)


# In[211]:


sb.heatmap(car_b.corr(), annot = True, cmap = 'magma')

plt.savefig('heatmap.png')
plt.show()


# I will remove columns have low correlation with price 

# I removed 'Yukseklik' and 'markacode' correlation.Imlemented same regression models.But I git less success.R2 and corr values decreased to %75

# In[212]:


car_b


# In[213]:


car_b_train, car_b_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)


y_train = car_b_train.loc[:,car_b_train.columns == 'Fiyat_TL']

x_train = car_b_train.loc[:, car_b_train.columns == 'Motor_Gucu_hp']

y_test = car_b_test.loc[:,car_b_test.columns == 'Fiyat_TL']

x_test = car_b_test.loc[:,car_b_test.columns == 'Motor_Gucu_hp']


# In[214]:


lin_reg = LinearRegression() 
  
lin_reg.fit(x_train, y_train) 


# In[215]:


from sklearn.preprocessing import PolynomialFeatures


# SECOND ORDER POLYNOMIAL REGRESSION

# In[216]:


poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x_train) 
  
poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 
y_pred_poly=lin2.predict(poly.fit_transform(x_test))


# In[217]:


plt.scatter(x_train, y_train, color = 'blue') 
  
plt.plot(x_train, lin_reg.predict(x_train), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('horse power') 
plt.ylabel('Price') 
  
plt.show() 


# In[218]:


plt.scatter(x_train, y_train, color = 'blue') 
  
plt.plot(x_train, lin2.predict(poly.fit_transform(x_train)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Horse Power') 
plt.ylabel('Pirice') 
  
plt.show() 


# In[219]:


mycar_hp={'Motor_Gucu_hp':[105]}


# In[220]:


mycar_hp=pd.DataFrame(mycar_hp)


# In[221]:


y_pred_lin=lin_reg.predict(x_test)


# In[222]:


y_pred_lin=pd.DataFrame(y_pred_lin)


# In[223]:


mycar_pred_price=lin_reg.predict(mycar_hp)


# In[224]:


mycar_pred_price


# In[225]:


lin2.predict(poly.fit_transform(mycar_hp)) 


# In[226]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_lin)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of 2nd ORDER POLYNOMIAL REGRESSION model is {}'.format(r2_score(y_test,y_pred_poly)))


# In[227]:


print('Explained Variance Score of LINEAR REGRESSION model is {}'.format(evs(y_test,y_pred_lin)))
print('-------------------------------------------------------------------------------')
print('Explained Variance Score of POLYNOMIAL  is {}'.format(evs(y_test,y_pred_poly)))


# 3rd ORDER POLLYNOMIAL REGRESSION 

# In[228]:


poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(x_train) 
  
poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 
y_pred_poly=lin2.predict(poly.fit_transform(x_test))


# In[229]:


mycar_hp={'Motor_Gucu_hp':[105]}
mycar_hp=pd.DataFrame(mycar_hp)
y_pred_lin=lin_reg.predict(x_test)
y_pred_lin=pd.DataFrame(y_pred_lin)
mycar_pred_price=lin_reg.predict(mycar_hp)
lin2.predict(poly.fit_transform(mycar_hp)) 


# In[230]:


print('Explained R-squared of 2nd ORDER POLYNOMIAL REGRESSION model is {}'.format(r2_score(y_test,y_pred_poly)))


# When I increase degree of polynomial regression , R2 value decrease . So second order polynomial regression found more usefull solution

# In[231]:


car_b


# STEP FORWARD FEATURE SELECTION

# In[232]:


get_ipython().system('pip install mlxtend')


# In[233]:




from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[234]:


car_b


# In[235]:


X=car_b.iloc[:,:13]


# Forward Selection Method — SFS() from mlxtend

# In[236]:


X


# In[237]:


y=car_b['Fiyat_TL']


# In[238]:


sfs=SFS(LinearRegression(),k_features=5,forward=True,floating=False,scoring='r2',cv=0)


# In[239]:


sfs.fit(X,y)


# In[240]:


SFS_results=pd.DataFrame(sfs.subsets_).transpose()


# In[241]:


SFS_results


# 5 most important features are iteratively added to the subset in a step-wise manner based on R-squared scoring

# In[242]:


mycar={'Yıl':[2015],'KM':[83000],'Motor_Gucu_hp':[105],'Agirlik_kg':[1134],'Hizlanma_0_100':[10]}


# In[243]:


car1_train, car1_test= train_test_split(car_b, train_size=0.67, test_size=0.33, random_state = 0)

y_train = car1_train.loc[:,car1_train.columns == 'Fiyat_TL']

x_train = car1_train.loc[:, car1_train.columns != 'Fiyat_TL']

y_test = car1_test.loc[:,car1_test.columns == 'Fiyat_TL']

x_test = car1_test.loc[:,car1_test.columns != 'Fiyat_TL']


# In[244]:


x_train=x_train.iloc[:,[0,1,2,4,12]]


# In[245]:


x_test=x_test.iloc[:,[0,1,2,4,12]]


# In[246]:


lin_reg=LinearRegression()
lin_reg.fit(x_train,y_train)


# In[247]:


y_pred_linear=lin_reg.predict(x_test)


# In[248]:


y_pred=pd.DataFrame(y_pred)
y_pred


# In[249]:


mycar=pd.DataFrame(mycar)


# In[250]:


mycar


# In[251]:


y_pred_linear=lin_reg.predict(mycar)


# In[252]:


mycar_hp=pd.DataFrame(mycar_hp)


# In[253]:


mycar_pred_price=lin_reg.predict(mycar)


# In[254]:


mycar_pred_price


# In[255]:


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


# In[256]:


print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LINEAR REGRESSION model is {}'.format(r2_score(y_test,y_pred_linear)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of RIDGE REGRESSION model is {}'.format(r2_score(y_test,y_pred_ridge)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of LASSO REGRESSION model is {}'.format(r2_score(y_test,y_pred_lasso)))
print('-------------------------------------------------------------------------------')
print('Explained R-squared of ELASTIC NET REGRESSION  is {}'.format(r2_score(y_test,y_pred_en)))


# In[257]:


y_pred_linear=lin_reg.predict(mycar)
y_pred_ridge = ridge.predict(mycar)
y_pred_lasso = lasso.predict(mycar)
y_pred_bay= bayesian.predict(mycar)
y_pred_en = en.predict(mycar)


# In[258]:


print('LINEAR REGRESSION prediction for my car price',y_pred_linear)
print('RIDGE REGRESSION prediction for my car price',y_pred_ridge)
print('LASSO REGRESSION prediction for my car price',y_pred_lasso)
print('ELASTIC NET REGRESSION for my car price',y_pred_en)


# In[ ]:




