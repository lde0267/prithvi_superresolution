<div align="center">

# PrithviSR

### KOMPSAT-3A → Sentinel-2 초해상화(Super-Resolution) 전체 파이프라인

**KOMPSAT-3A 고해상도 위성영상을 Sentinel-2 방사 특성으로 모사한 학습 데이터를 만들고, NASA/IBM의 Prithvi-EO-2.0 파운데이션 모델을 LoRA로 파인튜닝해 초해상화 모델을 학습시키는 프로젝트입니다.**

</div>

---

## 📖 프로젝트 구성

이 저장소는 두 단계로 나뉩니다.

```
PrithviSR/
├── data_preprocessing/    # 1단계: 학습 데이터 생성 (git 저장소, 별도 관리)
├── models/                # 2단계: SR 모델 정의 + 학습 코드 (.py 모듈)
├── final_SR.ipynb         # 2단계 드라이버 노트북 (models/ 를 import)
├── checkpoints/           # 학습 체크포인트 (.gitignore)
└── .gitignore
```

| 단계 | 위치 | 역할 |
|---|---|---|
| **1. 데이터 전처리** | [`data_preprocessing/`](data_preprocessing/README.md) | 이미 공간정합된 KOMPSAT-3A/Sentinel-2 페어 → 방사모사 → MTF모사 → 학습용 HR/LR 칩 생성 |
| **2. 모델 학습** | [`models/`](models/SETUP.md), `final_SR.ipynb` | Prithvi-EO-2.0-300M 인코더(LoRA) + 커스텀 SR 디코더로 4× 초해상화 학습 |

---

## 🔭 전체 흐름

```mermaid
flowchart LR
    E[("data_preprocessing/data/examples/<br/>kompsat{N}.tif + sentinel{N}.tif")] --> S1[방사모사<br/>IR-MAD]
    S1 --> S2[MTF모사<br/>Gaussian σ]
    S2 --> S3[칩 추출]
    S3 --> C[("data_preprocessing/data/output/chips/<br/>{base}_hr.tif + {base}_lr.tif")]
    C --> D[models/dataset.py<br/>KOMPSATSRDataset]
    P[("models/prithvi/<br/>Prithvi_EO_V2_300M_TL.pt")] --> M[models/architecture.py<br/>Prithvi 인코더(LoRA) + SR 디코더]
    D --> T[models/train.py]
    M --> T
    T --> CK[("checkpoints/<br/>best_sr_model.pth")]
    CK --> V[models/evaluate.py<br/>PSNR/SSIM/SAM/Edge vs Bicubic]
    D -.-> V

    style E fill:#fffde7,stroke:#fbc02d
    style C fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style P fill:#e1f5ff,stroke:#0288d1
    style T fill:#f3e5f5,stroke:#8e24aa
    style V fill:#f3e5f5,stroke:#8e24aa
```

---

## 🚀 실행 순서

### 1단계 — 학습 데이터 생성 ([`data_preprocessing/`](data_preprocessing/README.md))
```bash
cd data_preprocessing
pip install rasterio numpy scipy scikit-image

# data/examples/ 에 kompsat{N}.tif + sentinel{N}.tif 배치 후:
python simulation/scripts/run_simulation.py
python simulation/scripts/run_mtf_simulation.py
python simulation/scripts/run_chip_extraction.py
```
자세한 내용(데이터 명세, 설계 결정, 검증 결과)은 [`data_preprocessing/README.md`](data_preprocessing/README.md) 참고.

### 2단계 — 모델 학습 ([`models/`](models/SETUP.md))
```bash
# 환경 설치 (Windows + MSVC 없는 환경의 상세 우회법은 models/SETUP.md 참고)
pip install rasterio numpy scipy scikit-image huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install timm peft terratorch  # + models/SETUP.md 의 우회 설치 순서

# Prithvi 사전학습 가중치 다운로드 (1.33GB, Apache-2.0)
python models/download_prithvi.py

# 학습
python models/train.py --max-epochs 200 --batch-size 8

# 평가 (PSNR/SSIM/SAM/Edge RMSE, Bicubic 베이스라인과 비교)
python models/evaluate.py --visualize 3
```
`final_SR.ipynb`를 Jupyter로 열어도 동일한 코드(`models/*.py`)를 그대로 불러와 실행됩니다.

---

## 📦 현재 상태 (2026-09-19)

### 데이터 전처리 (`data_preprocessing/`)
- 예시 3페어(`kompsat1~3`)로 방사모사 → MTF모사 → 칩추출 전체 파이프라인 검증 완료 (3/3 성공)
- 참고 정답 대비 상관관계: 2/3 페어 0.87~0.98 (정상), 1/3 페어는 소규모 칩 특유의 IR-MAD 불안정성으로 이상치 — 전체 씬 단위 재검증 필요. 자세한 수치는 [`data_preprocessing/README.md`](data_preprocessing/README.md)의 "주요 결과" 항목 참고.

### 모델 학습 (`models/`)
- **`final_SR.ipynb`의 학습(train) 섹션을 `models/*.py`로 분해** — `config.py`/`dataset.py`/`architecture.py`/`losses.py`/`checkpoint.py`/`train.py`/`download_prithvi.py`/`plot_history.py`
- **Prithvi-EO-2.0-300M-TL 실제 다운로드 완료** (HuggingFace `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`, 1.33GB, Apache-2.0)
- **전체 배선을 실제 가중치 + 실제 칩 데이터로 1-step 스모크 테스트하여 검증**:
  ```
  데이터: LR (2,4,224,224) → HR (2,4,896,896)
  모델 출력: (2,4,896,896)  ✓ GT와 shape 일치
  loss=0.2560 (l1=0.2137, sam=0.1282, edge=0.0766)
  forward + backward pass 성공
  ```
  CPU(MSVC 빌드 도구 없는 Windows) 환경에서 `terratorch`의 무거운 전이 의존성(`albumentations`/`torchgeo` 등)을 최소 스텁·버전 고정으로 우회한 설치 레시피는 [`models/SETUP.md`](models/SETUP.md)에 재현 가능하도록 정리되어 있습니다.
- 예시 데이터가 3칩(train=2/val=1/test=0)뿐이라 지금은 "배선이 맞다"는 스모크테스트 수준입니다. 의미 있는 학습에는 `data_preprocessing`에서 훨씬 많은 페어를 생성해야 합니다.
- **평가(`models/evaluate.py`, `models/metrics.py`) 재작성 완료** — 옛 노트북은 별도 `_baseline.tif` 폴더(`val_dataset_root`)를 요구했지만 실제로 존재한 적 없는 경로였습니다. 이제 Bicubic 베이스라인은 `F.interpolate`로 그 자리에서 계산하고, `data_preprocessing`이 만드는 `{base}_lr.tif`/`{base}_hr.tif`만 있으면 됩니다.
  ```bash
  python models/evaluate.py --visualize 3   # PSNR/SSIM/SAM/Edge RMSE, 모델 vs Bicubic
  ```
  실제 Prithvi 가중치로 스모크 테스트한 결과, **미학습 상태에서는 모델 출력이 Bicubic과 정확히 일치**했습니다 (`PrithviSRDecoder`의 `up_conv`가 zero-init이라 residual=0인 게 설계 의도 — 버그 아님, 학습되면서 점점 벌어짐).

### 알려진 제약
- GPU 없이 CPU로는 300M 파라미터 모델 특성상 배치 1개 forward+backward에도 수 분이 걸립니다. 실제 200 에폭 학습은 GPU 환경을 권장합니다.
- 테스트 분할이 현재 0개(칩 3개뿐)라 `evaluate.py`가 실제로 평가할 샘플이 없습니다 — 더 많은 페어가 있어야 의미 있는 지표가 나옵니다.

---

## 🙏 출처

- **KOMPSAT-3A**: [한국항공우주연구원 (KARI)](https://www.kari.re.kr/)
- **Sentinel-2**: [ESA Copernicus](https://dataspace.copernicus.eu/)
- **Prithvi-EO-2.0**: [ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) (NASA · IBM · Jülich Supercomputing Centre, Apache-2.0)

## 📁 라이선스

`models/`, `final_SR.ipynb`의 코드는 연구 목적으로 작성되었습니다. `data_preprocessing/`은 별도 git 저장소로 관리되며 자체 라이선스 조건을 따릅니다 (KOMPSAT-3A 원본·가공본 재배포는 KARI/제공처 협약 조건 확인 필요). Prithvi 가중치는 Apache-2.0.
