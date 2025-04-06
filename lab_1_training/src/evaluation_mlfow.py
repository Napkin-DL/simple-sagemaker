import json
import pathlib
import pickle
import tarfile
import mlflow
import joblib
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import subprocess, sys
# 필요한 패키지 설치
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', 'matplotlib'])
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
except:
    print("Could not install seaborn, will use matplotlib only")

import xgboost
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import logging
import logging.handlers

def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  

logger = _get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default="/opt/ml/processing")    
    parser.add_argument('--model_path', type=str, default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument('--test_path', type=str, default="/opt/ml/processing/test/test.csv")
    parser.add_argument('--output_evaluation_dir', type=str, default="/opt/ml/processing/output")
    
    # MLflow 관련 인자 추가
    parser.add_argument('--tracking-uri', type=str, required=True, help="MLflow 추적 서버 URI")
    parser.add_argument('--experiment-name', type=str, required=True, help="MLflow 실험 이름")
    parser.add_argument('--run-id', type=str, required=True, help="부모 MLflow 실행 ID")
    
    # parse arguments
    args = parser.parse_args()     
    
    logger.info("#############################################")
    logger.info(f"args.model_path: {args.model_path}")
    logger.info(f"args.test_path: {args.test_path}")    
    logger.info(f"args.output_evaluation_dir: {args.output_evaluation_dir}")
    logger.info(f"args.tracking-uri: {args.tracking_uri}")
    logger.info(f"args.experiment-name: {args.experiment_name}")
    logger.info(f"args.run-id: {args.run_id}")

    model_path = args.model_path
    test_path = args.test_path    
    output_evaluation_dir = args.output_evaluation_dir
    base_dir = args.base_dir
    
    # MLflow 설정
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Traverse all files
    logger.info(f"****** All folder and files under {base_dir} ****** ")
    for file in os.walk(base_dir):
        logger.info(f"{file}")
    logger.info(f"************************************************* ")        

    # 부모 실행 내에서 중첩 실행 시작
    with mlflow.start_run(run_id=args.run_id) as parent_run:
        with mlflow.start_run(run_name="ModelEvaluation", nested=True) as eval_run:
            # 모델 로드
            with tarfile.open(model_path) as tar:
                tar.extractall(path=".")
            
            model = xgboost.XGBRegressor()
            model.load_model("xgboost-model")
            logger.info(f"model is loaded")    
            
            # 테스트 데이터 로드
            df = pd.read_csv(test_path)
            logger.info(f"test df sample \n: {df.head(2)}")
            
            # MLflow에 입력 데이터 기록 - 간소화된 방식으로 변경
            # 임시 파일로 저장
            temp_csv_path = "/tmp/test_data.csv"
            df.to_csv(temp_csv_path, index=False)
            
            # 간단하게 데이터 로깅 (스키마 명시 없이)
            try:
                mlflow.log_artifact(temp_csv_path, "test_data")
                logger.info(f"Test data logged as artifact")
            except Exception as e:
                logger.warning(f"Failed to log test data: {e}")
            
            y_test = df.iloc[:, 0].astype('int')    
            df.drop(df.columns[0], axis=1, inplace=True)
            
            X_test = df.values
            
            predictions_prob = model.predict(X_test)
            
            # if predictions_prob is greater than 0.5, it is 1 as a fruad, otherwise it is 0 as a non-fraud
            threshold = 0.5
            predictions = [1 if e >= 0.5 else 0 for e in predictions_prob ] 
            
            # 분류 보고서 생성
            class_report = classification_report(y_true=y_test, y_pred=predictions, output_dict=True)
            print(f"{classification_report(y_true=y_test, y_pred=predictions)}")
            
            # 혼동 행렬 생성
            cm = confusion_matrix(y_true=y_test, y_pred=predictions)    
            print(cm)
            
            # 메트릭 계산
            mse = mean_squared_error(y_test, predictions)
            std = np.std(y_test - predictions)
            
            # MLflow에 메트릭 기록
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("standard_deviation", std)
            
            # 분류 메트릭 기록
            for label, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        # 메트릭 이름에 특수문자 제거
                        safe_label = str(label).replace(".", "_")
                        safe_metric = str(metric_name).replace(".", "_")
                        mlflow.log_metric(f"{safe_label}_{safe_metric}", value)
                else:
                    safe_label = str(label).replace(".", "_")
                    mlflow.log_metric(f"overall_{safe_label}", metrics)
            
            # 결과 저장
            report_dict = {
                "regression_metrics": {
                    "mse": {
                        "value": mse,
                        "standard_deviation": std
                    },
                },
                "classification_report": class_report,
                "confusion_matrix": cm.tolist()
            }
            
            pathlib.Path(output_evaluation_dir).mkdir(parents=True, exist_ok=True)
            
            evaluation_path = f"{output_evaluation_dir}/evaluation.json"
            with open(evaluation_path, "w") as f:
                f.write(json.dumps(report_dict))
            
            # MLflow에 결과 파일 기록
            mlflow.log_artifact(evaluation_path, "evaluation_results")
            
            # 혼동 행렬 시각화 및 저장
            try:
                plt.figure(figsize=(10, 8))
                
                # seaborn이 설치되어 있으면 사용, 아니면 matplotlib만 사용
                try:
                    import seaborn as sns
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                except ImportError:
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.colorbar()
                    # 혼동 행렬에 숫자 표시
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")
                
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title('Confusion Matrix')
                
                cm_path = f"{output_evaluation_dir}/confusion_matrix.png"
                plt.savefig(cm_path)
                mlflow.log_artifact(cm_path, "evaluation_results")
                logger.info(f"Confusion matrix visualization saved to {cm_path}")
            except Exception as e:
                logger.warning(f"Could not create confusion matrix visualization: {e}")
            
            logger.info(f"evaluation_path \n: {evaluation_path}")                
            logger.info(f"report_dict \n: {report_dict}")