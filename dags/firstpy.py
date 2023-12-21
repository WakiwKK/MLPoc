from airflow import DAG
from airflow.operators.python import PythonOperator,BranchPythonOperator
from datetime import datetime
from random import randint
from airflow.operators.bash import BashOperator
from airflow.operators.postgres_operator import PostgresOperator
from ML.MachineLearning import ML, ok
import subprocess

def _choose_best_model(**kwargs):
    ti= kwargs['ti']
    accuracies = ti.xcom_pull(task_ids=[
        'training_model_A',
        'training_model_B',
        'training_model_C'
    ])
    best_accuracy = max(accuracies)
    if(best_accuracy > 8):
        return 'ml'
    return 'ml'
    
    
def _training_model():
    return randint(1,10)

def deployML():
    subprocess.run(["python","deploy/deploy.py"])

    


with DAG("my_dag",start_date=datetime(2023,11,21),
    schedule_interval="*/1 * * * *",catchup=False) as dag:
    
        training_model_A= PythonOperator(
            task_id="training_model_A",
            python_callable = _training_model
        )
        
        training_model_B= PythonOperator(
            task_id="training_model_B",
            python_callable = _training_model
        )
        
        training_model_C= PythonOperator(
            task_id="training_model_C",
            python_callable = _training_model
        )
        
        choose_best_model= BranchPythonOperator(
            task_id="choose_best_model",
            python_callable = _choose_best_model
        )
        
        accurate = BashOperator(
            task_id="accurate",
            bash_command="echo 'accurate'",
        )
        
        inaccurate = BashOperator(
            task_id="inaccurate",
            bash_command="echo 'inaccurate'",
        )
        
        ml = PythonOperator(
            task_id='ml',
            python_callable = ML
        )
        
        deploy = PythonOperator(
            task_id='deployML',
            python_callable = deployML
        )
        
        # run_this = PapermillOperator(
        #     task_id='run_example_notebook',
        #     input_nb='/dags/ML/ml.ipynb',
        #     output_nb="/dags/ML/out-1{{executeion_date}}.ipynb",
        #     parameters={"msgs":"Ran from Airflow at {{execution_date}}!!"}
        # )
        
        
        [training_model_A, training_model_B, training_model_C] >> choose_best_model >> ml >> deploy