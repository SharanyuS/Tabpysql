import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator,TransformerMixin): 
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns=columns
        self.mean_=None
        self.std_=None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
        
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        with open('Abs_model','rb') as model_file, open('Abs_scaler','rb') as scaler_file:
            self.LR = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    def load_and_clean_data(self, data_file):
        df=pd.read_csv(data_file,delimiter=',')

        df1 = df.drop(['ID'],axis=1)
        
        df1['Absenteeism Time in Hours'] = 'NaN'

        df2=df1

        reason_columns = pd.get_dummies(df['Reason for Absence'])

        reason_columns['check'] = reason_columns.sum(axis=1)

        pd.set_option('display.max_rows', None)

        reason_columns = reason_columns.drop(['check',0],axis=1)

        df2 = df2.drop(['Reason for Absence'],axis=1)

        rg1 = reason_columns.iloc[:,0:14]
        rg2 = reason_columns.iloc[:,14:17]
        rg3 = reason_columns.iloc[:,17:20]
        rg4 = reason_columns.iloc[:,20:]

        rg1=rg1.max(axis=1)
        rg2=rg2.max(axis=1)
        rg3=rg3.max(axis=1)
        rg4=rg4.max(axis=1)

        df2 = pd.concat([rg1,rg2,rg3,rg4,df2],axis=1)
        df2 = df2.rename(columns = {0:'rg1',1:'rg2',2:'rg3',3:'rg4'})

        df_checkpoint = df2.copy()

        df2['Date'] = pd.to_datetime(df2['Date'],format='%d/%m/%Y')

        df2['month_value'] = df2['Date'].dt.month

        df2['Day'] = df2['Date'].dt.weekday


        df2 = df2.drop(['Date'],axis=1)

        df2.columns

        rename = ['rg1', 'rg2', 'rg3', 'rg4','month_value', 'Day', 'Transportation Expense',
               'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index',
               'Education', 'Children', 'Pets', 'Absenteeism Time in Hours'
               ]
        df2=df2[rename]
       
        df2['Education'] = df2['Education'].map({1:0,2:1,3:1,4:1})
        
        df2 = df2.fillna(value=0)
       
        df2 = df2.drop(['Absenteeism Time in Hours','Daily Work Load Average','Distance to Work','Day'],axis=1)
        
        self.preprocessed_data = df2.copy()
            
        self.data = self.scaler.transform(df2)
        
         
        
    def predicted_probability(self):
        if (self.data is not None):  
                pred = self.LR.predict_proba(self.data)[:,1]
                return pred
    
    def predicted_output_category(self):
        if (self.data is not None):
                pred_outputs = self.LR.predict(self.data)
                return pred_outputs
        
    def predicted_outputs(self):
        if (self.data is not None):
                self.preprocessed_data['Probability'] = self.LR.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.LR.predict(self.data)
                return self.preprocessed_data





