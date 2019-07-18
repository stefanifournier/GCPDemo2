
import datetime
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import subprocess
from google.cloud import storage
import pandas as pd
import numpy as np

from sklearn import preprocessing
from math import sqrt
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

# Fill in your Cloud Storage bucket name
BUCKET_ID = 'mlspec_demo2'



#Read the data from the bucket
train = pd.read_csv('gs://mlspec_demo2/Data/train.csv')
test = pd.read_csv('gs://mlspec_demo2/Data/test.csv')


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes


# categorical columns
categorical_columns = ["Product_ID", "Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years",
                       "Marital_Status", "Product_Category_1", "Product_Category_2", "Product_Category_3"]


# Join Train and Test Dataset so it can be cleaned all at once
train['source']='train'
test['source']='test'

data = pd.concat([train,test], ignore_index = True, sort = False)

#imput the value -2 for NaN since NaN most likely means that the person did not buy products from these categories
data["Product_Category_2"]= \
data["Product_Category_2"].fillna(-2.0).astype("float")

data["Product_Category_3"]= \
data["Product_Category_3"].fillna(-2.0).astype("float")

#Get index of all columns with product_category_1 equal 19 or 20 from train and remove since not populated
condition = data.index[(data.Product_Category_1.isin([19,20])) & (data.source == "train")]
data = data.drop(condition)


# convert categorical data to to numerical values.
# convert data in categorical columns to numerical values
encoders = {col:LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = encoders[col].fit_transform(data[col])

#create count features of each user
def getCountVar(compute_df, count_df, var_name):
    grouped_df = count_df.groupby(var_name)
    count_dict = {}
    for name, group in grouped_df:
        count_dict[name] = group.shape[0]

    count_list = []
    for index, row in compute_df.iterrows():
        name = row[var_name]
        count_list.append(count_dict.get(name, 0))
    return count_list

data["Age_Count"]  =getCountVar(data, data, "Age")
data["Occupation_Count"]  =getCountVar(data, data, "Occupation")
data["Product_Category_1_Count"]  =getCountVar(data, data,"Product_Category_1")
data["Product_Category_2_Count"]  =getCountVar(data, data, "Product_Category_2")
data["Product_Category_3_Count"]  =getCountVar(data, data,"Product_Category_3")
data["Product_ID_Count"]  =getCountVar(data, data, "Product_ID")

#Divide into test and train
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

# remove column we are trying to predict ('Purchase') from features list
train_features = train.drop('Purchase', axis=1)
# create training labels list
train_labels = (train['Purchase'])

# load data into DMatrix object
dtrain = xgb.DMatrix(train_features, train_labels)
# train model
bst = xgb.train({}, dtrain, 20)


# Export the model to a file
model = 'model.bst'
bst.save_model(model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('blackfriday_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)


















