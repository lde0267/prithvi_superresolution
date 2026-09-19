"""download_prithvi.py - Prithvi EO v2 300M TL 사전학습 인코더 다운로드

출처: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
파일: Prithvi_EO_V2_300M_TL.pt (약 1.33GB, Apache-2.0, 로그인 불필요)

사용법:
    python models/download_prithvi.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ENCODER_FILE, PRITHVI_DIR  # noqa: E402

REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
FILENAME = "Prithvi_EO_V2_300M_TL.pt"


def download(force: bool = False) -> str:
    from huggingface_hub import hf_hub_download

    if os.path.exists(ENCODER_FILE) and not force:
        size_mb = os.path.getsize(ENCODER_FILE) / (1024 * 1024)
        print(f"✅ 이미 존재합니다: {ENCODER_FILE} ({size_mb:.1f} MB) — 다시 받으려면 --force")
        return ENCODER_FILE

    print(f"⬇️  다운로드 시작: {REPO_ID}/{FILENAME} → {PRITHVI_DIR}")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=PRITHVI_DIR,
    )

    # huggingface_hub 버전에 따라 심볼릭 링크/캐시 경로로 받아질 수 있어
    # 최종적으로 ENCODER_FILE 경로에 실제 파일로 존재하는지 확인.
    if os.path.abspath(downloaded_path) != os.path.abspath(ENCODER_FILE):
        import shutil
        shutil.copy2(downloaded_path, ENCODER_FILE)

    size_mb = os.path.getsize(ENCODER_FILE) / (1024 * 1024)
    print(f"✅ 다운로드 완료: {ENCODER_FILE} ({size_mb:.1f} MB)")
    return ENCODER_FILE


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prithvi EO v2 300M TL 체크포인트 다운로드")
    parser.add_argument('--force', action='store_true', help='이미 있어도 다시 다운로드')
    args = parser.parse_args()
    download(force=args.force)
