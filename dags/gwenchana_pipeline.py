import sys
if "/opt/airflow" not in sys.path:
    sys.path.insert(0, "/opt/airflow")
    
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.main import prepare_data , train , evaluate

with DAG(
    dag_id = "gwenchana_pipeline",
    description = "Student Mental Health Risk Prediction Pipeline",
    start_date = datetime(2026 , 11 , 1),
    schedule = "@weekly",
    catchup = False
) as dag:
    
    prepare_data_task = PythonOperator(
        task_id = "prepare_data",
        python_callable = prepare_data
    )

    train_models_task = PythonOperator(
        task_id = "train",
        python_callable = train
    )

    evaluate_models_task = PythonOperator(
        task_id = "evaluate",
        python_callable = evaluate
    )

    prepare_data_task >> train_models_task >> evaluate_models_task