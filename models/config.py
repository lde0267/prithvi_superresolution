"""config.py - 경로 설정 (로컬/Colab 겸용)

final_SR.ipynb 와 models/*.py 가 공통으로 쓰는 프로젝트 경로를 한 곳에서 정의합니다.
"""
import os

try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def get_project_root() -> str:
    if IN_COLAB:
        from google.colab import drive as _drive
        _drive.mount('/content/drive')
        return '/content/drive/MyDrive'
    # 이 파일(models/config.py) 기준 상위 폴더 = PrithviSR/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = get_project_root()

# data_preprocessing 파이프라인(run_simulation.py -> run_mtf_simulation.py ->
# run_chip_extraction.py)이 만드는 학습용 HR/LR 칩 페어 위치.
# 구조: data/output/chips/{label}_{scene}/{base}_hr.tif(정답) + {base}_lr.tif(입력)
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data_preprocessing', 'data', 'output', 'chips')

# Prithvi 사전학습 인코더 가중치 + 체크포인트 저장 위치
PRITHVI_DIR = os.path.join(PROJECT_ROOT, 'models', 'prithvi')
ENCODER_FILE = os.path.join(PRITHVI_DIR, 'Prithvi_EO_V2_300M_TL.pt')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

os.makedirs(PRITHVI_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def print_paths() -> None:
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"dataset_root  : {DATASET_ROOT}  (존재: {os.path.exists(DATASET_ROOT)})")
    print(f"encoder_file  : {ENCODER_FILE}  (존재: {os.path.exists(ENCODER_FILE)})")
    print(f"checkpoint_dir: {CHECKPOINT_DIR}")


if __name__ == '__main__':
    print_paths()
