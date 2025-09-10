from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(dag_id='fraud_detection_pipeline', start_date=datetime(2025,1,1), schedule_interval='@daily', catchup=False) as dag:
    def placeholder_task():
        print('This is a placeholder fraud detection DAG')

    run = PythonOperator(
        task_id='run_pipeline',
        python_callable=placeholder_task
    )

    run
