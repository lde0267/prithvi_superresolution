# 프로젝트 코딩 가이드라인 (Rules)

# data_preprocessing 프로젝트 코딩 가이드라인


# 프로젝트 핵심

**KOMPSAT-3A 영상을 Sentinel-2 영상에 모사** — 이미 공간정합된 KOMPSAT-3A/Sentinel-2 페어(`data/examples/`)를 입력으로:
1. 방사모사 (IR-MAD + 선형회귀)
2. 상대적 MTF값 계산 (Gaussian σ 탐색)
3. 최종 방사·MTF 모사가 완료된 Simulated KOMPSAT-3A 영상 + 학습용 HR/LR 칩 생성


## 1. 프로젝트 구조 규칙 (Project Structure Rules)

### 1.1. 프로젝트 레이아웃
원본 다운로드(구 module1)·RPC 정사보정(구 module2)·시각화(구 module4)·SR 비교 데모(구 module5)는
모두 삭제되었고, 이제 **이미 공간정합된 KOMPSAT-3A/Sentinel-2 페어**를 `data/examples/`로
공급받아 방사·MTF 모사 + 칩 추출만 수행하는 단일 파이프라인이다.

```
data_preprocessing/
├── claude.md                      # 본 가이드라인
├── config/                        # paths.json / processing_params / sensor_specs
├── data/
│   ├── examples/   kompsat{N}.tif, sentinel{N}.tif, sim{N}.tif  # 입력 (이미 정합됨)
│   └── output/     {rad_sim, mtf_sim, chips}                    # 최종 산출물
├── simulation/                     # 구 module3_simulation
│   └── src/  ir_mad.py · linear_norm.py · pipeline.py · mtf.py · __init__.py
│       scripts/run_simulation.py · run_mtf_simulation.py · run_chip_extraction.py
└── shared/utils/  paths.py · proj_env.py   # simulation 공통 (GDAL/PROJ env, paths.json 로딩)
```

### 1.2. 코드 모듈화 규칙
- 하나의 파일은 500줄 이하로 작성.
- 주석: 파일 상단에 10줄 이내로 기능 요약.
- 타입 힌트: 모든 함수와 클래스에 필수 적용.
- `src/__init__.py` 에서 공개 API 재노출 + `shared.utils.proj_env` import 로 GDAL 환경 초기화 (side effect).

## 2. 언어 및 프레임워크 규칙

### 2.1. 백엔드 (Backend)
- **환경 변수**: `.env` 파일 사용, 절대 경로 저장 금지

### 2.3. 에이전트 (Agents)
- **프레임워크**: LangGraph 1.x
- **LLM**: OpenAI GPT-4o 또는 Claude Opus 4.5 사용
- **메모리**: `MessageMemory` 사용, 1000개 메시지 제한
- **도구**: `tool` 데코레이터 사용, 타입 힌트 필수

## 3. API 설계 규칙

### 3.1. RESTful API 명명 규칙
- **명사 사용**: `/users`, `/orders` 등
- **HTTP 메서드**: GET, POST, PUT, DELETE만 사용
- **버전 관리**: `/v1/` 접두사 사용
- **응답 형식**: JSON only, Content-Type: application/json

### 3.2. API 보안 규칙
- 모든 엔드포인트에 rate limiting 적용 (100회/분)
- SQL Injection 방지: 매개변수화된 쿼리 사용
- 입력값 검증: Joi 또는 Pydantic 사용
- CORS: 특정 도메인만 허용

## 4. 데이터베이스 규칙

### 4.1. 스키마 설계
-snake_case 사용
-모든 테이블에 soft delete 컬럼(`is_deleted`, `deleted_at`) 추가
-인덱스: 쿼리 100ms 초과 시 인덱스 추가
-데이터 타입: JSON 필드는 PostgreSQL JSONB 타입 사용

## 5. 성능 최적화 규칙

### 5.1. API 성능 규칙
- 응답 시간: 95 percentile 200ms 이하
- 페이징: limit 100, offset 0 기본값
- 캐싱: Redis 5분 캐싱 적용
- CDN: 모든 정적 파일은 CDN 사용

### 5.2. LLM 최적화 규칙
-temperature = 0.7 고정
-prompt 길이: 5000 토큰 이하
-비용 절감: 20k 토큰 초과 시 경고 로그

## 6. 보안 규칙

- 비밀번호: bcrypt 해시 사용
- 입력값 검증: 모두 필수
- 로깅: 보안 이벤트는 별도 DB에 저장

## 7. 알려진 이슈 / 주의사항 (Known Issues)

### 7.1. 방사·MTF 모사 (`simulation/`)

#### (1) IR-MAD 입력은 KOMPSAT-3A 를 10m 로 다운샘플 후 Sentinel-2 와 정합
- **방식**: `run_simulation.py::aggregate_4x4_mean` 으로 KOMPSAT-3A 2.5m → 10m 평균 다운샘플 후 IR-MAD 수행. PIF 기반 선형회귀로 (a,b) 를 학습한 뒤, **학습된 (a,b) 를 KOMPSAT-3A 2.5m 원본에 적용** 해 출력은 2.5m 유지.
- **이유**: IR-MAD 의 χ² CDF 는 두 영상이 동일 grid 일 것을 가정. 2.5m vs 10m 직접 입력은 차원 안 맞음.

#### (2) NoData=0 정책과 `aggregate_4x4_mean` 의 permissive 마스크
- 입력 `kompsat{N}.tif`/`sentinel{N}.tif` 는 `nodata=0` 정책을 따른다 (footprint 밖 + 어두운 valid 0 픽셀이 한 덩어리로 마스킹됨에 유의).
- `aggregate_4x4_mean` 은 4×4 블록 안에서 valid 픽셀(`!= 0`) 만 평균. 16 픽셀 모두 0 인 블록만 NoData 유지.
- **strict 마스크 (16 픽셀 전부 valid 요구) 금지**: NIR 의 어두운 픽셀이나 폴리곤 가장자리에서 너무 많은 블록을 떨궈 IR-MAD valid_mask 가 비게 됨.

#### (3) MTF — 가우시안 PSF σ 최적화 + 다단계 fallback (`mtf.py::estimate_relative_psf_per_band`)
순서대로:
1. **패치 탐색** (`find_best_patch`): Sentinel-2 10m 영상에서 100% valid 이고 분산(에지 대비) 최대인 500×500 패치 선택. KOMPSAT-3A 패치는 그 4× 영역(2000×2000 @2.5m).
2. **σ 최적화** (`scipy.optimize.minimize_scalar bounded`): KOMPSAT-3A 패치에 가우시안 블러 → 4×4 mean 다운샘플 → Sentinel-2 패치와 robust MSE. percentile cutoff `[95,90,85,80,75,70,60,50,40,30]` 단계적 완화 — 매 cutoff 마다 σ 가 boundary(`sigma_min+0.05` 또는 `sigma_max−0.05`) 에 수렴하면 다음 cutoff 로 재시도.
3. **위상정합 fallback**: 모든 cutoff 가 boundary 에 갇히면 `skimage.registration.phase_cross_correlation` 으로 KOMPSAT-3A↔Sentinel-2 sub-pixel shift 측정 → KOMPSAT-3A 패치를 물리적으로 shift 후 σ 재최적화. 이 경로는 정합 잔차가 σ 추정을 망가뜨리는 경우의 마지막 수단.
4. **전체영상 적용**: 찾은 σ 를 KOMPSAT-3A 2.5m 전체에 1회 적용 — 이때는 항상 `normalized_gaussian_filter` (NoData 블리딩 차단).

#### (4) `normalized_gaussian_filter` — NoData 블리딩 차단
- 일반 `gaussian_filter` 는 NoData(0) 를 valid 0 으로 취급해 블러 결과가 가장자리에서 어두워짐.
- 패턴: `V*mask` 와 `mask` 를 각각 가우시안 블러 후 나눈다 (`blurred_V / blurred_W`). 분모가 0 에 가까운 픽셀(`blurred_W <= 1e-4`) 은 출력 0.
- 패치 최적화 단계에서 100% valid 패치가 발견된 경우(`is_perfect=True`) 는 일반 `gaussian_filter` 사용 — 정규화 비용 회피.

#### (5) Status 사이드카 JSON
- `run_simulation.py` / `run_mtf_simulation.py` 는 모든 페어에 대해 결과 JSON 을 항상 기록 (success/skip/fail 무관). 각 디렉토리에 `_summary.json` / `_mtf_summary.json` 으로 집계.

#### (6) 칩 추출 (`run_chip_extraction.py`) — SR 학습용 페어 칩
- **칩 사양**: HR 896×896 @2.5m + LR 224×224 @10m, 동일 footprint(2240×2240 m), 4:1 비율.
- **HR 소스**: `data/examples/kompsat{label}.tif` (이미 공간정합된 원본, **방사·MTF 모사 안 됨**).
- **LR 소스**: `data/output/mtf_sim/{label}_*_simulated_final_2p5m.tif` 를 `aggregate_4x4_mean` 으로 4×4 mean → 10m. 즉 "Sentinel-2 처럼 보이도록 모사된 KOMPSAT-3A" 의 LR view.
- **타일링**: LR 격자에서 `stride_lr=150` (overlap=74/224 ≈ 33.0%, "최대 1/3 중첩" 한계). HR stride 는 자동으로 `stride_lr × 4 = 600`. 예시 데이터셋처럼 입력 자체가 이미 칩 크기(896×896)면 1개만 나와 사실상 pass-through.
- **valid 필터**: HR 칩 안에서 4밴드 모두 nonzero 인 픽셀 비율 < 0.8 이면 스킵 (7.1.(2) 의 nodata=0 정책상 footprint 밖과 dark valid 0 이 함께 묶이는 점 인지).
- **출력**: `data/output/chips/{label}_{scene}/{base}_{hr,lr}.tif` + `{base}_meta.json` (CRS, UTM bbox, transform, valid_ratio_hr 등). 디렉토리당 `_chip_summary.json` 집계.
- **결정성**: 칩 ID 는 HR 픽셀 좌표 `y{y:05d}_x{x:05d}` — 재실행 시 동일 칩이 같은 파일명. `--overwrite` 없으면 기존 칩 스킵.

### 7.2. 환경

- **Python 3.10+ 필수** — `float | None` 등 PEP 604 문법 사용.
- **`osgeo`(GDAL Python 바인딩) 불필요** — `shared/utils/proj_env.py` 가 필요로 하는 건 `gdal.UseExceptions()` 호출뿐이라 `rasterio`만으로 충분.
- **시스템 PROJ 충돌 경고** — PostgreSQL/PostGIS 등이 설치돼 있으면 GDAL이 그쪽 `proj.db`를 잘못 집어 경고가 뜰 수 있음. `shared/utils/proj_env.py`의 `PROJ_DATA`로 rasterio 번들 PROJ를 강제하면 완화됨.
