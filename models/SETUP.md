# 환경 설정 (Windows, MSVC Build Tools 없는 환경)

이 프로젝트는 CPU-only 환경, MSVC Build Tools 미설치 상태에서도 동작하도록 검증되었습니다.
`terratorch`의 일부 전이 의존성(`albumentations`→`albucore`→`stringzilla`)이 Windows 바이너리
휠이 없어 소스 빌드를 시도하며 MSVC 를 요구합니다 — 아래 순서대로 설치하면 이 문제를 피할 수
있습니다.

## 1. 기본 설치

```bash
pip install rasterio numpy scipy scikit-image huggingface_hub
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU 있으면 CUDA 버전으로
pip install timm peft
```

## 2. terratorch 관련 (핵심 — 순서 중요)

`albucore`(albumentations 의 고속 백엔드)가 최신 버전에서 Windows 휠이 없는 `stringzilla>=5.1.2`
를 요구합니다. 구버전 `albucore`로 고정해서 우회합니다:

```bash
pip install stringzilla        # 최신 버전(휠 있음, 5.1.1)을 먼저 깔아둠
pip install "albucore==0.2.12" # 이 버전까지는 stringzilla>=3.10.4 만 요구 (5.1.1로 충족됨)
pip install numkong             # albucore 0.2.12 의 또 다른 네이티브 의존성 (휠 있음)
pip install albumentations --no-deps   # albucore 를 강제로 최신으로 올리지 않도록 --no-deps
pip install pydantic

pip install terratorch
pip install einops segmentation_models_pytorch python-box
pip install torchgeo --no-deps
pip install geopandas requests lightning kornia lightly matplotlib pandas scikit-learn
```

**terratorch 가 `import terratorch.datasets` 시 관련 없는 다른 데이터셋(BioMassters, m_bigearthnet
등)까지 전부 로드하려 해서** 위 목록이 깁니다. 우리가 실제로 쓰는 건 `terratorch.registry.BACKBONE_REGISTRY`
(Prithvi 백본)뿐입니다.

## 3. GDAL(osgeo) 우회 (선택)

`shared/utils/proj_env.py` (data_preprocessing 쪽)가 `from osgeo import gdal` 을 요구하지만
실제로는 `gdal.UseExceptions()` 호출 하나만 씁니다. `pip install gdal` 은 MSVC 가 필요해 실패하므로,
아래처럼 최소 스텁을 만들어 `PYTHONPATH` 앞쪽에 두면 우회됩니다 (RPC 정사보정처럼 진짜 GDAL 기능이
필요한 곳에는 쓰지 마세요 — 이 프로젝트엔 더 이상 그런 부분이 없습니다):

```python
# osgeo_stub/osgeo/gdal.py
def UseExceptions():
    pass
```

## 4. Prithvi 가중치 다운로드

```bash
python models/download_prithvi.py
```
`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` (HuggingFace, Apache-2.0, 로그인 불필요) 에서
`Prithvi_EO_V2_300M_TL.pt` (약 1.33GB) 를 받아 `models/prithvi/` 에 저장합니다.

## 5. 검증

```bash
python models/train.py --max-epochs 1 --batch-size 2
```
데이터 스캔 → Prithvi 인코더(LoRA) 로드 → forward → loss → backward 까지 1 스텝 실행해 배선이
맞는지 확인합니다 (CPU 기준 배치 1개에 수 분 소요될 수 있음 — 300M 파라미터 모델이라 실제 학습은
GPU 를 권장합니다).

## 실제 검증 결과 (2026-09-19, CPU)

```
train=2 val=1 test=0
x_6ch: torch.Size([2, 6, 1, 224, 224])
lr_img: torch.Size([2, 4, 224, 224])
gt_img: torch.Size([2, 4, 896, 896])
model output shape: torch.Size([2, 4, 896, 896])
loss=0.2560  l1=0.2137  sam=0.1282  edge=0.0766
✅ 데이터 로딩 -> 인코더(LoRA) -> 디코더 -> 손실 -> 역전파까지 전부 정상 동작
```

## 6. 평가

```bash
python models/evaluate.py --visualize 3
```
테스트 분할(`get_data_split` 의 3번째 반환값)을 스캔해 `checkpoints/best_sr_model.pth` 로 추론하고,
Bicubic 베이스라인(`F.interpolate` 로 그 자리에서 계산 — 별도 `_baseline.tif` 파일 불필요)과
PSNR/SSIM/SAM/Edge RMSE 를 비교합니다. 체크포인트가 없으면 경고 후 초기 가중치로 평가합니다.

**val 분할로 스모크 테스트한 결과** (미학습 모델, 2026-09-19):
```
PSNR / SSIM / SAM / EDGE_RMSE 모두 Model == Bicubic (Gain +0.0000)
```
`PrithviSRDecoder.up_conv` 가 zero-init 이라 미학습 상태에서는 residual=0 → 모델 출력이 Bicubic 과
수학적으로 정확히 같은 게 설계 의도입니다 (버그 아님). 학습이 진행되면서 값이 벌어져야 정상입니다.
