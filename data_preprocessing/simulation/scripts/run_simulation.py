"""run_simulation.py - 페어 단위 IR-MAD 방사모사 배치

흐름:
  1) data/examples/ 의 페어를 자동 탐지 (kompsat{N}.tif / sentinel{N}.tif)
  2) 페어별로 K3A 2.5m, S2 10m 4밴드 스택 로드 (각각 단일 멀티밴드 GeoTIFF)
  3) K3A 4×4 mean → 10m 다운샘플 (S2 와 동일 해상도)
  4) IR-MAD + PIF 선형회귀 fit (10m 에서)
  5) 학습된 (a, b) 를 K3A 2.5m 원본에 적용 → 2.5m simulated K3A
  6) data/output/rad_sim/ 에 GeoTIFF + 계수 JSON 사이드카 저장

입력 데이터는 이미 공간정합(coregistration)이 끝난 페어로 가정한다 — 즉 kompsat{N}
과 sentinel{N} 은 같은 origin/CRS 를 공유하고 K3A:S2 픽셀 비율이 정확히 4:1 이어야
한다 (module2_coregistration 이 하던 일은 더 이상 이 파이프라인의 일부가 아님 —
데이터는 이미 정합된 상태로 제공된다).

MTF 모사는 run_mtf_simulation.py 에서 별도 처리.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.src import (
    apply_linear_normalization,
    radiometric_simulation,
)
from shared.utils.proj_env import PROJ_DATA

_PATHS_CONFIG = json.loads((PROJECT_ROOT / "config" / "paths.json").read_text(encoding="utf-8"))


BAND_ORDER = ["Blue", "Green", "Red", "NIR"]

_PAIR_RE = re.compile(r"^kompsat(\d+)\.tif$")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def discover_pairs(examples_dir: Path) -> Dict[str, Dict[str, Path]]:
    """examples_dir 의 kompsat{N}.tif / sentinel{N}.tif 를 페어로 묶어 반환.

    파일명 규약:
      K3A(HR, 2.5m, 4밴드): "kompsat{N}.tif"
      S2 (LR, 10m, 4밴드):  "sentinel{N}.tif"
      (참고용 정답, 입력으로는 안 씀): "sim{N}.tif"

    Returns:
        {"1": {"k3a": Path, "s2": Path}, "2": {...}, ...}
    """
    pairs: Dict[str, Dict[str, Path]] = {}
    for tif in sorted(examples_dir.glob("kompsat*.tif")):
        m = _PAIR_RE.match(tif.name)
        if not m:
            continue
        label = m.group(1)
        s2_path = examples_dir / f"sentinel{label}.tif"
        if not s2_path.exists():
            logging.getLogger("rad_sim_batch").warning(
                "  짝이 되는 sentinel%s.tif 없음 → 페어 %s 스킵", label, label
            )
            continue
        pairs[label] = {"k3a": tif, "s2": s2_path}
    return pairs


def load_band_stack(path: Path, band_order: List[str]) -> Tuple[np.ndarray, dict]:
    """4밴드 멀티밴드 GeoTIFF 를 (bands, H, W) 로 로드. band_order 는 파일의 밴드 순서와
    일치한다고 가정 (Blue, Green, Red, NIR)."""
    with rasterio.open(path) as src:
        if src.count != len(band_order):
            raise ValueError(
                f"{path.name}: 밴드 수 {src.count} ≠ 기대값 {len(band_order)}"
            )
        data = src.read()
        profile = src.profile.copy()
    return data, profile


def aggregate_4x4_mean(arr: np.ndarray) -> np.ndarray:
    """K3A 2.5m → 10m 평균 다운샘플 (permissive).

    각 4×4 블록 안에서 valid 픽셀(!=0) 만 평균. 16 픽셀 모두 0 인 블록만 0
    (NoData) 유지. H/W 가 4의 배수가 아니면 우/하단 잉여를 잘라낸다.
    """
    n, h, w = arr.shape
    h2 = (h // 4) * 4
    w2 = (w // 4) * 4
    if (h, w) != (h2, w2):
        arr = arr[:, :h2, :w2]
        n, h, w = arr.shape

    blocked = arr.reshape(n, h // 4, 4, w // 4, 4).astype(np.float64)
    valid_blocked = (arr != 0).reshape(n, h // 4, 4, w // 4, 4)

    sum_valid = (blocked * valid_blocked).sum(axis=(2, 4))
    count_valid = valid_blocked.sum(axis=(2, 4))

    means = np.where(count_valid > 0, sum_valid / np.maximum(count_valid, 1), 0.0)
    return means.astype(arr.dtype)


def write_tif(out_path: Path, data: np.ndarray, profile: dict) -> None:
    """data: (bands, H, W) — profile 기반 multi-band TIF 저장."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(count=data.shape[0], dtype=data.dtype.name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def _save_record(record: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def _process_pair(
    label: str,
    pair: Dict[str, Path],
    out_dir: Path,
    bands: List[str],
    args,
    logger: logging.Logger,
) -> str:
    """페어 1개를 처리하고 status('success' / 'fail') 를 반환합니다.

    어떤 결과든 페어별 JSON ({label}_kompsat{label}_radsim.json) 을 항상 기록.
    """
    logger.info("\n[페어 %s] 시작", label)

    scene_stem = f"kompsat{label}"
    out_tif = out_dir / f"{label}_{scene_stem}_simulated_2p5m.tif"
    out_json = out_dir / f"{label}_{scene_stem}_radsim.json"

    record: dict = {
        "pair_label": label,
        "scene_stem": scene_stem,
        "bands": bands,
        "k3a_file": str(pair["k3a"]),
        "s2_file": str(pair["s2"]),
    }

    # 기존 성공 결과가 있으면 재처리 안 함 (--overwrite 로 강제 재실행 가능)
    if out_tif.exists() and out_json.exists() and not args.overwrite:
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("status") == "success":
                logger.info("  이미 성공 결과 존재 → 스킵 (%s)", out_tif.name)
                return "skip"
        except Exception:
            pass  # 기존 JSON 손상 등 → 재처리

    try:
        k3a_2p5m, k3a_profile = load_band_stack(pair["k3a"], bands)  # (n, H4, W4)
        s2_10m, _ = load_band_stack(pair["s2"], bands)                # (n, H, W)

        logger.info(
            "  K3A shape (2.5m): %s | S2 shape (10m): %s",
            k3a_2p5m.shape, s2_10m.shape,
        )

        k3a_10m = aggregate_4x4_mean(k3a_2p5m)

        # 두 영상은 공통 가상격자에서 나온 것으로 가정하지만, 혹시 1~2 픽셀
        # 어긋날 경우를 대비해 공통 부분만 사용.
        if k3a_10m.shape[1:] != s2_10m.shape[1:]:
            logger.warning(
                "  K3A 10m %s ≠ S2 10m %s → 공통 부분만 사용",
                k3a_10m.shape, s2_10m.shape,
            )
            h = min(k3a_10m.shape[1], s2_10m.shape[1])
            w = min(k3a_10m.shape[2], s2_10m.shape[2])
            k3a_10m = k3a_10m[:, :h, :w]
            s2_10m = s2_10m[:, :h, :w]

        valid_mask = ((k3a_10m != 0).all(axis=0)) & ((s2_10m != 0).all(axis=0))
        record["n_valid_at_10m"] = int(valid_mask.sum())
        record["grid_size_10m"] = list(s2_10m.shape[1:])

        result = radiometric_simulation(
            img_s2=s2_10m,
            img_k3a=k3a_10m,
            band_names=bands,
            mad_max_iter=args.max_iter,
            mad_epsilon=args.epsilon,
            pif_threshold=args.pif_threshold,
            nodata_value=0.0,
        )

        # 학습된 (a, b) 를 K3A 2.5m 원본에 적용 — 출력은 2.5m simulated K3A
        simulated_2p5m = apply_linear_normalization(k3a_2p5m, result.band_coefficients)

        # K3A nodata(0) 위치는 출력도 0 으로 유지
        nodata_mask = (k3a_2p5m == 0)
        simulated_2p5m[nodata_mask] = 0
        simulated_2p5m = np.clip(simulated_2p5m, 0, 65535).astype(np.uint16)

        write_tif(out_tif, simulated_2p5m, k3a_profile)

        record.update({
            "status": "success",
            "iterations": int(result.ir_mad_result.iterations),
            "n_pifs": int(result.ir_mad_result.pif_mask.sum()),
            "canonical_correlations": [
                float(x) for x in result.ir_mad_result.canonical_correlations
            ],
            "band_coefficients": [
                {
                    "band_name": c.band_name,
                    "slope": c.slope,
                    "intercept": c.intercept,
                    "r_squared": c.r_squared,
                    "rmse": c.rmse,
                    "n_pifs": c.n_pifs,
                }
                for c in result.band_coefficients
            ],
            "output_tif": str(out_tif),
        })
        _save_record(record, out_json)

        logger.info("  저장: %s", out_tif.name)
        logger.info("        %s", out_json.name)
        return "success"

    except Exception as e:
        logger.exception("  실패 (%s): %s", type(e).__name__, e)
        record.update({
            "status": "fail",
            "reason_class": type(e).__name__,
            "reason_detail": str(e),
        })
        _save_record(record, out_json)
        return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IR-MAD 기반 방사모사 배치 (K3A 2.5m → S2 방사 특성)",
    )
    parser.add_argument(
        "--examples-dir", default=None,
        help=f"kompsat{{N}}.tif / sentinel{{N}}.tif 페어 디렉토리 "
             f"(기본: config/paths.json examples_dir = {_PATHS_CONFIG.get('examples_dir')})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help=f"결과 저장 디렉토리 (기본: config/paths.json rad_sim_dir = {_PATHS_CONFIG['rad_sim_dir']})",
    )
    parser.add_argument(
        "--bands", nargs="+", default=BAND_ORDER,
        help="처리할 밴드 (기본: Blue Green Red NIR, 파일 내 밴드 순서와 일치해야 함)",
    )
    parser.add_argument("--max-iter", type=int, default=50, help="IR-MAD 최대 반복")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="IR-MAD 수렴 임계값")
    parser.add_argument("--pif-threshold", type=float, default=0.95, help="PIF 확률 임계값")
    parser.add_argument("--overwrite", action="store_true", help="기존 출력 덮어쓰기")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("rad_sim_batch")

    examples_dir = (
        Path(args.examples_dir).resolve()
        if args.examples_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["examples_dir"]).resolve()
    )
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (PROJECT_ROOT / _PATHS_CONFIG["rad_sim_dir"]).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not examples_dir.exists():
        logger.error("데이터 디렉토리를 찾을 수 없습니다: %s", examples_dir)
        sys.exit(1)

    bands = list(args.bands)
    pairs = discover_pairs(examples_dir)

    if not pairs:
        logger.error("페어를 찾지 못했습니다 (kompsat{N}.tif/sentinel{N}.tif): %s", examples_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("방사모사 배치 시작: 페어 %d개", len(pairs))
    logger.info("  examples_dir: %s", examples_dir)
    logger.info("  out_dir     : %s", out_dir)
    logger.info("  bands       : %s", bands)
    logger.info("=" * 60)

    success = 0
    skip = 0
    fail = 0

    # rasterio 호출 전체를 PROJ_DATA 로 격리 — system PROJ (PostgreSQL/PostGIS 등)
    # 가 잡혀 proj.db schema 버전 충돌이 나는 걸 막는다.
    env_kwargs = {"PROJ_LIB": PROJ_DATA} if PROJ_DATA else {}
    with rasterio.env.Env(**env_kwargs):
        for label in sorted(pairs.keys(), key=lambda s: int(s) if s.isdigit() else 0):
            outcome = _process_pair(label, pairs[label], out_dir, bands, args, logger)
            if outcome == "success":
                success += 1
            elif outcome == "skip":
                skip += 1
            else:
                fail += 1

    summary = {
        "success": success,
        "skip": skip,
        "fail": fail,
        "total": success + skip + fail,
        "out_dir": str(out_dir),
        "bands": bands,
    }
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("배치 종료 — 성공 %d, 스킵 %d, 실패 %d", success, skip, fail)
    logger.info("summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
