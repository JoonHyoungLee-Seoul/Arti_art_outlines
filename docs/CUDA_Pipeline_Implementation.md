# DexiNed + SwiftSketch CUDA Pipeline Implementation

## 개요 (Overview)
이 문서는 **CUDA 지원 NVIDIA GPU** 환경에서 다음을 구현하는 기술 가이드입니다:
- **배경**: **DexiNed**(엣지 검출, MIT License)로 **형태가 보이는 배경 윤곽** 생성
- **피사체**: **SwiftSketch** 결과(semantic/artistic outline) 활용  
- **최종**: 두 결과를 **자동으로 합성(merge)** 하는 완전 자동화 파이프라인

## 라이선스 요약
- **DexiNed**: **MIT License** (상업 이용 가능)
- **BiRefNet**: **MIT License**, GitHub Releases에 ONNX 제공
- **SwiftSketch**: 결과물 사용/배포 정책은 프로젝트 상황에 맞게 별도 검토
- **원본 예술작품**: 퍼블릭 도메인/라이선스 확인 필요

## 시스템 요구사항

### 하드웨어/소프트웨어
- Linux/Windows + **NVIDIA CUDA 지원 GPU**
- 최신 **NVIDIA 드라이버** (CUDA 12.x 호환)
- Python 3.9+ 권장
- 16GB+ RAM 권장

### CUDA 환경 설정
```bash
# PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime GPU 설치  
pip install onnxruntime-gpu

# 기타 의존성
pip install opencv-python-headless pillow numpy scipy scikit-image
```

### 설치 확인
```bash
python -c "
import torch, onnxruntime as ort
print('PyTorch CUDA:', torch.cuda.is_available())
print('ORT providers:', ort.get_available_providers())
"
```

## 파이프라인 구조

### 현재 구현된 폴더 구조
```
ARTI/
├── DexiNed/                          # DexiNed repository
│   └── checkpoints/BIPED/10/10_model.pth
├── image_clipping/                   # 메인 파이프라인
│   ├── models/BiRefNet-general-epoch_244.onnx
│   ├── images/                       # 입력 이미지 (7개 샘플)
│   ├── clipped_images_fg/           # 전경 추출 결과 (*_fg.png)
│   ├── clipped_images_bg/           # 배경 추출 결과 (*_bg.png)  
│   ├── bg_outlines_dexined/         # 배경 엣지 결과 (*_bg_edges.png)
│   ├── merged_outlines/             # 최종 병합 결과
│   ├── run_cutout.py               # BiRefNet 전경/배경 분리
│   ├── dexined_bg_cuda.py          # DexiNed 배경 엣지 검출
│   ├── merge_outlines.py           # 윤곽 병합
│   └── run_complete_pipeline.py    # 전체 파이프라인 실행
└── swiftsketch/ControlSketch/output_sketches/  # SwiftSketch 결과
```

## 파이프라인 단계별 실행

### 1단계: 전경/배경 분리 (BiRefNet CUDA)
```bash
cd image_clipping

# 단일 이미지
python run_cutout.py -i images/436332.jpg

# 배치 처리
python run_cutout.py -b images/

# 커스텀 출력 디렉토리
python run_cutout.py -i input.jpg --fg-dir ./fg/ --bg-dir ./bg/
```

**출력**: 
- `filename_fg.png` - 전경 (피사체만, 투명 배경)
- `filename_bg.png` - 배경 (피사체 부분 투명)

### 2단계: 배경 엣지 검출 (DexiNed CUDA)
```bash
# 단일 배경 이미지
python dexined_bg_cuda.py -i clipped_images_bg/436332_bg.png

# 모든 배경 이미지
python dexined_bg_cuda.py -b clipped_images_bg/

# 커스텀 출력
python dexined_bg_cuda.py -b clipped_images_bg/ -o ./edges/
```

**출력**:
- `filename_bg_edges.png` - 이진 엣지 맵 (흰색 엣지, 검은 배경)

### 3단계: 윤곽 병합
```bash
# 기본 병합
python merge_outlines.py \
  --swift swiftsketch/ControlSketch/output_sketches/436332/436332_32_strokes/final_sketch.png \
  --dexi bg_outlines_dexined/436332_bg_edges.png \
  --output merged_outlines/436332_final.png

# 커스텀 강도 설정
python merge_outlines.py \
  -s figure.png -d edges.png -o result.png \
  --bg-intensity 150 --figure-intensity 255
```

**출력**:
- 다중 강도 윤곽 이미지 (0=배경, 180=배경엣지, 255=피사체윤곽)

### 완전 자동화 실행
```bash
# 전체 파이프라인 원클릭 실행
python run_complete_pipeline.py \
  --sample 436332 \
  --input images/436332.jpg \
  --swift swiftsketch/ControlSketch/output_sketches/436332/436332_32_strokes/final_sketch.png \
  --output ./pipeline_results
```

## 기술적 세부사항

### BiRefNet 설정
- **모델**: BiRefNet-general-epoch_244.onnx
- **입력 크기**: 1024×1024 (자동 리사이징)
- **Providers**: CUDAExecutionProvider → CPUExecutionProvider
- **메모리 최적화**: 동적 아레나 할당

### DexiNed 설정  
- **모델**: 10_model.pth (BIPED 데이터셋 학습)
- **입력 정규화**: BGR 평균 차감 `[103.939, 116.779, 123.68]`
- **목표 크기**: 최대 1024px (32픽셀 정렬)
- **후처리**: 가우시안 블러 + Otsu/적응적 임계값

### 병합 알고리즘
- **우선순위 모드**: 피사체 윤곽이 배경 엣지를 덮어씀 (기본)
- **가산 모드**: 단순 결합 (더 많은 배경 디테일)
- **충돌 해결**: 형태학적 팽창 기반 마스킹

## 성능 벤치마크

| **작업** | **입력 크기** | **GPU 시간** | **CPU 시간** | **메모리** |
|----------|---------------|--------------|--------------|------------|
| BiRefNet 분할 | 4000×3000 | ~2초 | ~8초 | ~2GB |
| DexiNed 엣지 검출 | 4000×3000 | ~3초 | ~15초 | ~3GB |
| 윤곽 병합 | 4000×3000 | ~0.1초 | ~0.3초 | ~100MB |
| **전체 파이프라인** | **4000×3000** | **~5초** | **~23초** | **~3GB** |

*NVIDIA A100 80GB PCIe 기준*

## 트러블슈팅

### CUDA 관련 문제
```bash
# CUDA 설치 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# PyTorch CUDA 재설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### ONNX Runtime 문제
```bash
# ONNX Runtime GPU 확인
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 재설치
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

### 메모리 할당 오류
- 배치 크기나 입력 이미지 크기 줄이기
- CPU 폴백 사용: provider 목록에서 `CUDAExecutionProvider` 제거
- GPU 메모리 제한 늘리기

### 엣지 검출 품질 저하
- 입력 이미지에 충분한 배경 구조가 있는지 확인
- DexiNed 후처리의 임계값 조정
- 배경 이미지가 적절한 투명도를 가지는지 확인 (피사체 영역 투명)

## 상업적 사용 체크리스트
- **DexiNed**: MIT License (저작권 고지 유지 시 상업 이용 가능)
- **BiRefNet**: MIT License, GitHub Releases에 ONNX 제공
- **SwiftSketch 결과물**: 내부/외부 배포 정책 검토 필요
- **원본 작품 이미지**: 라이선스/퍼블릭 도메인 여부 확인 필수

---

*이 문서는 실제 구현된 CUDA 파이프라인을 기반으로 작성되었습니다. 최신 정보는 루트 디렉토리의 README.md를 참조하세요.*