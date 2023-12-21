import pandas as pd
import numpy as np
#from keras.models import Model
# from keras.layers import Input,Dense
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from keras.layers import BatchNormalization
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import mlflow
# import mlflow.s3
# import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request,jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV

# def ok():
#     mlflow.set_tracking_uri("http://airflow-docker-mlflow-server-1:5000/")
#     mlflow_experiment_id = "0"
#     df = pd.read_csv("dags/ML/CC GENERAL.csv")
#     imputer= SimpleImputer(strategy='mean')
#     df = df.iloc[:,lambda df:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
#     X_tran_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(X_tran_imputed)
#     input_layer = Input(shape=17)
#     encoded = Dense(17,activation='relu')(input_layer)
#     decoded = Dense(17, activation='sigmoid')(encoded)
#     autoencoder = Model(input_layer,decoded)
#     autoencoder.compile(optimizer='adam',loss='mse')
#     autoencoder.fit(scaled_data,scaled_data,epochs=10,batch_size=5, shuffle=True)
#     # mlflow.keras.log_model(autoencoder, "model",'${AIRFLOW_PROJ_DIR:-.}/mlflow-data:/opt/airflow/mlflow-data')
    
#     with mlflow.start_run(run_name="PARENT_RUN",experiment_id=mlflow_experiment_id):
#         mlflow.keras.log_model(autoencoder,'model',registered_model_name='MLTest')


def ok():
    # set the experiment id
    mlflow.set_tracking_uri("http://airflow-docker-mlflow-server-1:5000/")
    mlflow.set_experiment(experiment_id="0")
    # mlflow.create_experiment('s3',artifact_location="s3://ss-datalake-dev/mlflow/") 

    mlflow.sklearn.autolog(registered_model_name='MLTest2')
    db = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
    # model = mlflow.sklearn.load_model()
    # return model
    
    # with mlflow.start_run():
    #     mlflow.log_param('n_estimators',100)
    #     mlflow.log_param('max_depth',6)
    #     mlflow.log_param('max_feature',3)
        
    #     mlflow.sklearn.log_model(rf,"random_forest_model")
    return 'deployML'

def ML(**variable):
    mlflow.set_tracking_uri("http://airflow-docker-mlflow-server-1:5000/")
    mlflow.set_experiment(experiment_id="2")
    # mlflow.create_experiment(name='s3',artifact_location="s3://ss-datalake-dev/mlflow/") 
    mlflow.sklearn.autolog(registered_model_name='MLTest3')
    df = pd.read_excel("dags/ML/Sample Data With OneHot (1).xlsx")
    # ti = variable['ti']
    # df = ti.xcom_pull(task_ids=[
    #     'training_model_A',
    #     'training_model_B',
    #     'training_model_C'
    # ])
    y= df['เป็นหนี้']
    X= df[['ภาคกลาง','ภาคตะวันออกเฉียงเหนือ','ภาคเหนือ','ภาคใต้','รายได้ครอบครัวต่อเดือน','ที่พักอาศัย','อายุ','เพศ','คณะเภสัชศาสตร์','คณะเทคโนโลยีสารสนเทศฯ',
       'คณะพยาบาลศาสตร์','คณะครุศาสตร์','คณะวิศวกรรมศาสตร์','คณะนิเทศศาสตร์','คณะเศรษฐศาสตร์','คณะมนุษยศาสตร์','คณะสถาปัตยกรรมศาสตร์','คณะสหวิทยาการ เทคโนโลยีและนวัตกรรม',
       'คณะบริหารธุรกิจ','คณะวิทยาศาสตร์','คณะรัฐศาสตร์','คณะสหเวชศาสตร์','คณะวิทยาศาสตร์การกีฬา','คณะทันตแพทยศาสตร์','คณะบัญชี','คณะดิจิทัลมีเดีย','คณะศิลปกรรมศาสตร์',
       'คณะจิตวิทยา','ชั้นปีที่ 1','ชั้นปีที่ 2','ชั้นปีที่ 3','ชั้นปีที่ 4','ชั้นปีที่ 5','ชั้นปีที่ 6']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    transformer = make_column_transformer(
    (StandardScaler(),['รายได้ครอบครัวต่อเดือน','อายุ'])
    )
    X_train_transformed = X_train
    X_test_transformed = X_test
    X_train_transformed[['รายได้ครอบครัวต่อเดือน','อายุ']] = transformer.fit_transform(X_train)
    X_test_transformed[['รายได้ครอบครัวต่อเดือน','อายุ']] = transformer.fit_transform(X_test)
    # dtree= DecisionTreeClassifier()
    # dtree.fit(X_train_transformed,y_train)
    # dtree.score(X_train_transformed, y_train)
    # dtree.score(X_test_transformed, y_test)
    # cross_validate(dtree,X_train_transformed, y_train, cv=5, return_train_score=True)
    # dtree.fit(X_train_transformed, y_train)
    param_grid = {'max_depth':[5,9],
              'max_leaf_nodes':[10,20,30]}
    grid_search = GridSearchCV(DecisionTreeClassifier(),param_grid)
    grid_search.fit(X_train_transformed, y_train)
    grid_search.score(X_train_transformed, y_train)
    grid_search.score(X_test_transformed, y_test)
    y_predicted = grid_search.predict(X_test_transformed)
    # mlflow.s3.log_artifacts(artifact_path="s3://ss-datalake-dev/mlflow/",bucket="ss-datalake-dev")
    return 'deployML'



    
    