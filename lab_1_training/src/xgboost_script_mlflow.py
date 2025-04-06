###########################################################################################################################
'''
훈련 코드는 크게 아래와 같이 구성 되어 있습니다.
- 커맨드 인자로 전달된 변수 내용 확인
- 훈련 데이터 로딩 
- xgboost의 cross-validation(cv) 로 훈련
- 훈련 및 검증 데이터 세트의 roc-auc 값을 metrics_data 에 저장
    - 모델 훈련시 생성되는 지표(예: loss 등)는 크게 두가지 방식으로 사용 가능
        - 클라우드 워치에서 실시간으로 지표 확인
        - 하이퍼 파라미터 튜닝(HPO) 에서 평가 지표로 사용 (예: validation:roc-auc)
        - 참조 --> https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/training-metrics.html
        - 참조: XGBoost Framework 에는 이미 디폴트로 정의된 metric definition이 있어서, 정의된 규칙에 따라서 모델 훈련시에 print() 를 하게 되면, 
               metric 이 클라우드 워치 혹은 HPO에서 사용이 가능
           
Name                Regex
validation:auc	.*\[[0-9]+\].*#011validation-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*
train:auc	    .*\[[0-9]+\].*#011train-auc:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*

실제 코드에 위의 Regex 형태로 print() 반영
print(f"[0]#011train-auc:{train_auc_mean}")
print(f"[1]#011validation-auc:{validation_auc_mean}")
    
- 훈련 성능을 나타내는 지표를 저장 (metrics.json)
- 훈련이 모델 아티펙트를 저장

'''
###########################################################################################################################

import os
import sys
import pickle
import xgboost as xgb
import argparse
import pandas as pd
import json
import mlflow
import mlflow.xgboost
from datetime import datetime
import traceback

def train_sagemaker(args):
    """SageMaker 환경에서 경로 설정"""
    if os.environ.get('SM_MODEL_DIR') is not None:
        args.train_data_path = os.environ.get('SM_CHANNEL_INPUTDATA')
        args.model_dir = os.environ.get('SM_MODEL_DIR')
        args.output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
    return args

def main():
    """
    메인 함수: 모델 훈련 프로세스 실행
    """
    parser = argparse.ArgumentParser(description="XGBoost 모델 훈련 스크립트")

    ###################################
    ## 커맨드 인자 처리
    ###################################    
    
    # 하이퍼파라미터 정의
    parser.add_argument('--scale_pos_weight', type=int, default=50, 
                        help="양성 클래스 가중치 (불균형 데이터 처리용)")    
    parser.add_argument('--num_round', type=int, default=999, 
                        help="최대 부스팅 라운드 수")
    parser.add_argument('--max_depth', type=int, default=3, 
                        help="트리의 최대 깊이")
    parser.add_argument('--eta', type=float, default=0.2, 
                        help="학습률")
    parser.add_argument('--objective', type=str, default='binary:logistic', 
                        help="학습 목표 함수")
    parser.add_argument('--nfold', type=int, default=5, 
                        help="교차 검증 폴드 수")
    parser.add_argument('--early_stopping_rounds', type=int, default=10, 
                        help="조기 중단 라운드 수")
    parser.add_argument('--train_data_path', type=str, default='../../data/dataset', 
                        help="훈련 데이터 경로")

    # SageMaker 특정 인자
    parser.add_argument('--model-dir', type=str, default='../model', 
                        help="모델 저장 디렉토리")
    parser.add_argument('--output-data-dir', type=str, default='../output', 
                        help="출력 데이터 디렉토리")

    args = parser.parse_args()
    
    # SageMaker 환경 확인
    args = train_sagemaker(args)

    ###################################
    ## MLflow 설정
    ###################################
    parent_run = None
    child_run = None
    mlflow_setup_success = False
    
    try:
        parent_run, child_run = setup_mlflow()
        mlflow_setup_success = parent_run is not None
    except Exception as e:
        print(f"MLflow 설정 실패: {e}")
        print(traceback.format_exc())
    
    ###################################
    ## 데이터 세트 로딩 및 변환
    ###################################
    
    try:
        # 데이터 파일 찾기
        csv_path = find_csv_file(args.train_data_path)
        if csv_path:
            data = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No CSV files found in {args.train_data_path}")
            
        print(f"데이터 로드 완료: {data.shape}")
        
        # 특성과 레이블 분리
        train = data.drop('fraud', axis=1)
        label = pd.DataFrame(data['fraud'])

        if mlflow_setup_success:
            try:
                # 데이터셋 정보 로깅
                mlflow.log_param("dataset_shape", str(data.shape))
                mlflow.log_param("positive_samples", int(label.sum()))
                mlflow.log_param("negative_samples", int(len(label) - label.sum()))
            except Exception as e:
                print(f"데이터셋 정보 로깅 중 오류: {e}")
        
        # XGBoost DMatrix 생성
        dtrain = xgb.DMatrix(train, label=label)
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        if mlflow_setup_success:
            try:
                if child_run:
                    mlflow.end_run()
                if parent_run:
                    mlflow.end_run()
            except:
                pass
        raise
    
    ###################################
    ## 하이퍼파라미터 설정
    ###################################        
    
    params = {
        'max_depth': args.max_depth, 
        'eta': args.eta, 
        'objective': args.objective, 
        'scale_pos_weight': args.scale_pos_weight
    }
    
    num_boost_round = args.num_round
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds
    
    if mlflow_setup_success:
        try:
            # 중첩 실행에서는 하이퍼파라미터를 로깅해도 충돌이 발생하지 않음
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("num_boost_round", num_boost_round)
            mlflow.log_param("nfold", nfold)
            mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
        except Exception as e:
            print(f"하이퍼파라미터 로깅 중 오류: {e}")
    
    ###################################
    ## Cross-Validation으로 훈련하여, 훈련 및 검증 메트릭 추출
    ###################################            
    
    try:
        cv_results = xgb.cv(
            params = params,
            dtrain = dtrain,
            num_boost_round = num_boost_round,
            nfold = nfold,
            early_stopping_rounds = early_stopping_rounds,
            metrics = ('auc'),
            stratified = True,  # 레이블 (0,1) 의 분포에 따라 훈련, 검증 세트 분리
            seed = 0
        )
        
        print("cv_results: ", cv_results)
        
        # 최종 성능 지표 추출
        train_auc = cv_results.iloc[-1]['train-auc-mean']
        validation_auc = cv_results.iloc[-1]['test-auc-mean']
        best_iteration = len(cv_results)
        
        # SageMaker 호환 형식으로 출력
        print(f"[0]#011train-auc:{train_auc}")
        print(f"[1]#011validation-auc:{validation_auc}")

        if mlflow_setup_success:
            try:
                # MLflow에 메트릭 로깅
                mlflow.log_metric("train_auc", train_auc)
                mlflow.log_metric("validation_auc", validation_auc)
                mlflow.log_metric("train_auc_std", cv_results.iloc[-1]['train-auc-std'])
                mlflow.log_metric("validation_auc_std", cv_results.iloc[-1]['test-auc-std'])
                mlflow.log_metric("best_iteration", best_iteration)
                
                # 실제 사용된 num_boost_round 값 로깅 (early stopping으로 인해 변경될 수 있음)
                mlflow.log_param("actual_num_boost_round", best_iteration)
            except Exception as e:
                print(f"메트릭 로깅 중 오류: {e}")
        
        # SageMaker 호환성을 위한 메트릭 데이터
        metrics_data = {
            'classification_metrics': {
                'validation:auc': { 'value': validation_auc},
                'train:auc': {'value': train_auc}
            }
        }
        
        ###################################
        ## 오직 훈련 데이터 만으로 훈련하여 모델 생성
        ###################################            

        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=best_iteration)

        ###################################
        ## 모델 아티펙트 및 훈련/검증 지표를 저장
        ###################################            
        
        # 디렉토리 생성
        os.makedirs(args.output_data_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        
        # 파일 경로 설정
        metrics_location = os.path.join(args.output_data_dir, 'metrics.json')
        model_location = os.path.join(args.model_dir, 'xgboost-model')
        
        # 메트릭 저장
        with open(metrics_location, 'w') as f:
            json.dump(metrics_data, f)
        
        # 모델 저장
        model.save_model(model_location)
        
        # 모델 디렉토리 내용 확인
        print(f"모델 디렉토리 내용: {os.listdir(args.model_dir)}")
        
        # 모델 파일 크기 확인
        model_size = os.path.getsize(model_location)
        print(f"모델 파일 크기: {model_size} 바이트")
        
        # 모델을 pickle 형식으로도 저장 (선택 사항)
        pickle_location = os.path.join(args.model_dir, 'model.pkl')
        with open(pickle_location, 'wb') as f:
            pickle.dump(model, f)
        print(f"모델을 pickle 형식으로 {pickle_location}에 저장했습니다.")
        
        if mlflow_setup_success:
            try:
                # MLflow에 모델 로깅
                mlflow.xgboost.log_model(model, "xgboost-model")
                
                # 모델 아티팩트 경로 로깅
                mlflow.log_param("model_location", model_location)
                mlflow.log_param("metrics_location", metrics_location)
                
                # 모델 성능 요약 정보 로깅
                mlflow.set_tag("model_summary", f"XGBoost 모델 (깊이: {args.max_depth}, eta: {args.eta})")
                mlflow.set_tag("best_auc", f"{validation_auc:.4f}")
            except Exception as e:
                print(f"모델 로깅 중 오류: {e}")
        
        print(f"훈련 완료. 모델 저장 위치: {model_location}")
        print(f"메트릭 저장 위치: {metrics_location}")

    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        print(traceback.format_exc())
        # 오류 발생 시에도 MLflow 실행 종료
        if mlflow_setup_success:
            try:
                if child_run:
                    mlflow.end_run()
                if parent_run:
                    mlflow.end_run()
            except:
                pass
        raise
    
    # MLflow 실행 종료
    if mlflow_setup_success:
        try:
            # 중첩 실행이 있으면 먼저 종료
            if child_run:
                mlflow.end_run()
            # 부모 실행 종료
            if parent_run:
                mlflow.end_run()
        except Exception as e:
            print(f"MLflow 실행 종료 중 오류: {e}")


def setup_mlflow():
    """
    MLflow 추적 서버 연결 및 실험 설정
    
    Returns:
        tuple: (parent_run, child_run) - MLflow 실행 객체들
    """
    try:
        # 환경 변수에서 MLflow 추적 서버 URI 가져오기
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
        if tracking_uri:
            print(f"MLflow 추적 서버 URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
        else:
            print("MLFLOW_TRACKING_URI 환경 변수가 설정되지 않았습니다.")
            return None, None
        
        # 환경 변수에서 실험 이름 가져오기
        experiment_name = os.environ.get('MLFLOW_EXPERIMENT_NAME')
        if experiment_name:
            print(f"MLflow 실험 이름: {experiment_name}")
            mlflow.set_experiment(experiment_name)
        else:
            print("MLFLOW_EXPERIMENT_NAME 환경 변수가 설정되지 않았습니다.")
            return None, None
        
        # 환경 변수에서 실행 ID 가져오기
        parent_run_id = os.environ.get('MLFLOW_RUN_ID')
        
        # 현재 시간과 작업 ID를 사용하여 고유한 중첩 실행 이름 생성
        timestamp = datetime.now().strftime('%H%M%S')
        job_id = os.environ.get('TRAINING_JOB_NAME', '')
        if job_id:
            # SageMaker 훈련 작업 ID에서 마지막 부분만 추출
            job_suffix = job_id.split('-')[-1] if '-' in job_id else job_id
            child_run_name = f"training-{job_suffix}-{timestamp}"
        else:
            child_run_name = f"training-{timestamp}"
        
        if parent_run_id:
            print(f"MLflow 부모 실행 ID: {parent_run_id}")
            # 부모 실행 시작
            try:
                parent_run = mlflow.start_run(run_id=parent_run_id)
                # 중첩 실행 시작
                child_run = mlflow.start_run(run_name=child_run_name, nested=True)
                print(f"MLflow 중첩 실행 시작: {child_run.info.run_id}")
                return parent_run, child_run
            except Exception as e:
                print(f"부모 실행 시작 중 오류 발생: {e}")
                print(traceback.format_exc())
                # 부모 실행 없이 새 실행 시작
                run = mlflow.start_run(run_name=child_run_name)
                return run, None
        else:
            print("MLFLOW_RUN_ID 환경 변수가 설정되지 않았습니다. 새 실행을 시작합니다.")
            run = mlflow.start_run(run_name=child_run_name)
            return run, None
    except Exception as e:
        print(f"MLflow 설정 중 오류 발생: {e}")
        print(traceback.format_exc())
        return None, None

    
def find_csv_file(base_path):
    """
    주어진 경로에서 CSV 파일 찾기
    
    Args:
        base_path (str): 검색할 기본 경로
        
    Returns:
        str: 찾은 CSV 파일 경로 또는 None
    """
    import glob
    
    # 직접 경로 시도
    direct_path = os.path.join(base_path, 'train.csv')
    if os.path.exists(direct_path):
        print(f"Found direct path: {direct_path}")
        return direct_path
    
    # 재귀적으로 CSV 파일 검색
    csv_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)
    if csv_files:
        print(f"Found CSV in subdirectory: {csv_files[0]}")
        return csv_files[0]
    
    print(f"No CSV files found in {base_path}")
    return None


if __name__ == '__main__':
    main()