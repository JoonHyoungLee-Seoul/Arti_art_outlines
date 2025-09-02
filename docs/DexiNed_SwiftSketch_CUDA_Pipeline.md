# 예술작품 **배경만** DexiNed(CUDA)로 윤곽화 + 피사체는 SwiftSketch → 최종 머지 (Windows/NVIDIA, 상업용 OK)

이 문서는 **CUDA 지원 NVIDIA GPU** 환경에서  
- **배경**: **DexiNed**(엣지 검출, MIT License)로 **형태가 보이는 배경 윤곽**을 만들고,  
- **피사체**: 이미 갖고 계신 **SwiftSketch** 결과(semantic/artistic outline)를 사용,  
- 두 결과를 **자동으로 합성(merge)** 하는 **끝판왕 파이프라인**을 처음부터 끝까지 제공합니다.

> 라이선스 요약  
> - **DexiNed** 저장소 라이선스: **MIT**(상업 이용 가능).  
> - **BiRefNet**(선택: 전경/배경 마스크 생성용) 라이선스: **MIT**, GitHub Releases에 ONNX 제공.  
> - **SwiftSketch**: 결과물 사용/배포 정책은 프로젝트 상황에 맞게 별도 검토.  
> - **원본 예술작품 이미지**: 퍼블릭 도메인/라이선스 확인 필요(예: 모나리자는 퍼블릭 도메인).

---

## 목차
1. [요구사항 / 폴더 구조](#요구사항--폴더-구조)  
2. [환경 셋업 (CUDA)](#환경-셋업-cuda)  
3. [코드/모델 받기](#코드모델-받기)  
4. [SwiftSketch 결과 준비](#swiftsketch-결과-준비)  
5. [(선택) 전경/배경 마스크 — BiRefNet + ORT CUDA](#선택-전경배경-마스크--birefnet--ort-cuda)  
6. [DexiNed(CUDA)로 배경 윤곽](#dexinedcuda로-배경-윤곽)  
7. [SwiftSketch(피사체) + DexiNed(배경) 머지](#swiftsketch피사체--dexined배경-머지)  
8. [원클릭 실행 (PowerShell)](#원클릭-실행-powershell)  
9. [스모크 테스트(10초 컷)](#스모크-테스트10초-컷)  
10. [품질/속도 팁](#품질속도-팁)  
11. [상업용 체크리스트](#상업용-체크리스트)  
12. [트러블슈팅](#트러블슈팅)

---

## 요구사항 / 폴더 구조

### 하드웨어/소프트웨어
- Windows 11/10 + **NVIDIA CUDA 지원 GPU**
- 최신 **NVIDIA 드라이버**(CUDA 12.x 호환)
- Python 3.10+ 권장
- (선택) Visual Studio Build Tools (일부 패키지 빌드 시)

### 권장 폴더 구조
```
project/
 ├─ env/                          # 가상환경(선택)
 ├─ DexiNed/                      # DexiNed repo
 ├─ models/
 │   ├─ dexined/                  # DexiNed 체크포인트(.pth)
 │   └─ birefnet/                 # BiRefNet ONNX(선택)
 ├─ inputs/
 │   └─ mona_lisa.jpg
 ├─ subject/
 │   └─ subject_outline.png       # SwiftSketch 출력(RGBA, 투명 배경)
 ├─ outputs/
 └─ pipeline/
     ├─ mask_birefnet_cuda.py     # (선택) 전경/배경 마스크(ORT CUDA)
     ├─ dexined_infer_cuda.py     # DexiNed 추론(CUDA)
     └─ merge_outlines.py         # 윤곽 병합(배경=회색, 피사체=검정)
```

---

## 환경 셋업 (CUDA)

### 1) 가상환경(선택)
PowerShell:
```powershell
py -m venv .\env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) 필수 패키지 설치

#### A. PyTorch (CUDA)
아래 커맨드는 **CUDA 12.6** 예시입니다. CUDA/버전은 [PyTorch 공식 페이지]에서 OS/패키지/Compute Platform을 선택한 뒤 제시되는 최신 커맨드를 사용하세요.
```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### B. ONNX Runtime (CUDA EP)
```powershell
pip install onnxruntime-gpu
```

#### C. 기타 유틸
```powershell
pip install opencv-python pillow numpy scipy scikit-image gdown onnx
```

### 3) 설치 확인
```powershell
python - << 'PY'
import torch, onnxruntime as ort
print("torch cuda available:", torch.cuda.is_available())
print("ort providers:", ort.get_available_providers())
PY
```
- `torch cuda available: True`  
- `ort providers`에 **CUDAExecutionProvider**가 포함되어 있으면 OK.

---

### 1) DexiNed 레포
```powershell
cd project
git clone https://github.com/xavysp/DexiNed.git 
```

### 2) DexiNed 체크포인트(.pth)
DexiNed **README**의 “Checkpoint Pytorch”(Google Drive) 링크를 통해 최신 체크포인트를 다운로드하세요(예: `10_model.pth`). 저장 위치:
```
models/dexined/10_model.pth
```

### 3) (선택) BiRefNet ONNX (전경/배경 마스크)
**GitHub Releases**에서 **ONNX** 파일을 내려받아 아래 위치에 저장하세요(예: `BiRefNet-general-epoch_244.onnx`):
```
models/birefnet/BiRefNet-general-epoch_244.onnx
```

---

## SwiftSketch 결과 준비
- 이미 갖고 계신 **SwiftSketch** 결과(`subject_outline.png`)를 사용합니다.  
- **RGBA(투명 배경)**, 선 색상은 **검정(#000)** 권장. (SVG라면 PNG로 내보내 주세요.)

---

## (선택) 전경/배경 마스크 — BiRefNet + ORT CUDA

`pipeline/mask_birefnet_cuda.py`:
```python
# pipeline/mask_birefnet_cuda.py
import onnxruntime as ort
import numpy as np, cv2, os, sys
from PIL import Image

def load_session(onnx_path):
    return ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

def letterbox_rgb(img, size=1024):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad = np.zeros((size, size, 3), dtype=np.uint8)
    pad[:nh, :nw] = resized
    x = pad.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # 1x3xHxW
    return x, (h, w), (nh, nw)

def crop_and_resize(mask_pad, orig_hw, resized_hw):
    h, w = orig_hw
    nh, nw = resized_hw
    mask = mask_pad[:nh, :nw]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

def run(onnx_path, image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sess = load_session(onnx_path)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x, (h, w), (nh, nw) = letterbox_rgb(bgr, 1024)

    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]  # (1,1,H,W) 가정
    y = y[0, 0]
    y = (y - y.min()) / max(1e-8, (y.max()-y.min()))
    mask_fg = (y > 0.5).astype(np.float32)
    mask_bg = 1.0 - mask_fg

    fg = (crop_and_resize(mask_fg, (h, w), (nh, nw)) * 255).astype(np.uint8)
    bg = (crop_and_resize(mask_bg, (h, w), (nh, nw)) * 255).astype(np.uint8)

    Image.fromarray(fg).save(os.path.join(out_dir, "mask_fg.png"))
    Image.fromarray(bg).save(os.path.join(out_dir, "mask_bg.png"))

if __name__ == "__main__":
    onnx_path, img_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    run(onnx_path, img_path, out_dir)
```

실행:
```powershell
python pipeline\mask_birefnet_cuda.py models\birefnet\BiRefNet-general-epoch_244.onnx inputs\mona_lisa.jpg outputs
```

---

## DexiNed(CUDA)로 배경 윤곽

핵심: **피사체를 흰색으로 채운 이미지**를 DexiNed에 넣어 **배경 구조의 엣지**만 얇고 선명하게 얻습니다.

`pipeline/dexined_infer_cuda.py`:
```python
# pipeline/dexined_infer_cuda.py
import os, sys, cv2, numpy as np
from PIL import Image
import torch

# DexiNed repo 경로
DEXI_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DexiNed"))
sys.path.append(DEXI_REPO)
from model import DexiNed   # DexiNed/model.py

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_bg_only(image_path, mask_fg_path=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if mask_fg_path and os.path.exists(mask_fg_path):
        m = cv2.imread(mask_fg_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.uint8)
        img = img.copy()
        img[m == 1] = 255  # 피사체는 흰색으로 지워 DexiNed 관심을 배경으로
    return img

def to_tensor(img_bgr):
    x = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return torch.from_numpy(x)

@torch.no_grad()
def run_dexined(pth_path, image_path, mask_fg_path, out_png):
    dev = device()
    net = DexiNed().to(dev).eval()
    sd = torch.load(pth_path, map_location="cpu")
    net.load_state_dict(sd, strict=False)

    bgr = prepare_bg_only(image_path, mask_fg_path)
    x = to_tensor(bgr).to(dev)

    outs = net(x)
    pred = outs[-1] if isinstance(outs, (list, tuple)) else outs  # 마지막 맵 사용
    pred = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()

    # 얇고 선명하게: 블러 후 Otsu + thinning(가능 시)
    pred = (pred * 255).astype(np.uint8)
    pred = cv2.GaussianBlur(pred, (3, 3), 0)
    _, bw = cv2.threshold(pred, 0, 255, cv2.THRESH_OTSU)
    try:
        from skimage.morphology import thin
        bw = thin((bw > 0)).astype(np.uint8) * 255
    except Exception:
        pass

    Image.fromarray(bw).save(out_png)
    return out_png

if __name__ == "__main__":
    # 사용법: python pipeline/dexined_infer_cuda.py <dexined.pth> <image> <mask_fg.png|None> <out_png>
    pth, img, mfg, outp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if mfg.lower() == "none": mfg = None
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    run_dexined(pth, img, mfg, outp)
```

실행:
```powershell
# 예: DexiNed 체크포인트가 models/dexined/10_model.pth라면
python pipeline\dexined_infer_cuda.py models\dexined\10_model.pth inputs\mona_lisa.jpg outputs\mask_fg.png outputs\bg_edges.png
```

(선택) 직선 구조 보강 — **OpenCV LSD** 예시:
```python
import cv2, numpy as np
from PIL import Image

edge = cv2.imread("outputs/bg_edges.png", cv2.IMREAD_GRAYSCALE)
img  = cv2.imread("inputs/mona_lisa.jpg", cv2.IMREAD_COLOR)

lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
lines, _, _, _ = lsd.detect(cv2.Canny(img, 60, 180))
canvas = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for l in lines:
        x1,y1,x2,y2 = l[0].astype(int)
        cv2.line(canvas, (x1,y1), (x2,y2), (255,255,255), 1)

cv2.imwrite("outputs/bg_edges_lsd.png", canvas[:, :, 0])
```

---

## SwiftSketch(피사체) + DexiNed(배경) 머지

`pipeline/merge_outlines.py`:
```python
# pipeline/merge_outlines.py
import os, sys, cv2, numpy as np
from PIL import Image

def load_rgba(path):  return np.array(Image.open(path).convert("RGBA"))
def load_gray(path):  return np.array(Image.open(path).convert("L"))

def colorize(gray, value=180):
    c = np.zeros((gray.shape[0], gray.shape[1], 4), np.uint8)
    c[..., :3] = value
    c[..., 3]  = gray
    return c

def expand_mask(mask, px=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
    return cv2.dilate(mask, k, 1)

def merge(subject_outline_png, bg_edges_png, mask_fg_png, out_png):
    S = load_rgba(subject_outline_png)     # SwiftSketch 출력 (선=검정, 배경=투명)
    B = load_gray(bg_edges_png)            # 0/255
    H, W = B.shape
    S = cv2.resize(S, (W, H), interpolation=cv2.INTER_NEAREST)

    # 피사체 주변 배경선 제거(겹침 방지)
    if mask_fg_png and os.path.exists(mask_fg_png):
        M = load_gray(mask_fg_png)
        M = cv2.resize(M, (W, H), interpolation=cv2.INTER_NEAREST)
        near = expand_mask((M > 127).astype(np.uint8) * 255, px=2)
        B[near > 0] = 0

    BG = colorize(B, value=180)  # 배경선=짙은 회색
    out = Image.alpha_composite(Image.fromarray(BG), Image.fromarray(S))
    out.save(out_png)
    return out_png

if __name__ == "__main__":
    subj, bg, mfg, outp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if mfg.lower() == "none": mfg = None
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    merge(subj, bg, mfg, outp)
```

실행:
```powershell
python pipeline\merge_outlines.py subject\subject_outline.png outputs\bg_edges.png outputs\mask_fg.png outputs\merged_outline.png
```

---

## 원클릭 실행 (PowerShell)

`run_all_cuda.ps1`:
```powershell
# 경로 설정
$IMG="inputs\mona_lisa.jpg"
$DEX="models\dexined\10_model.pth"                 # DexiNed 체크포인트
$BIR="models\birefnet\BiRefNet-general-epoch_244.onnx"  # 없으면 "None"
$SUB="subject\subject_outline.png"

# 1) (선택) 전경/배경 마스크
if (Test-Path $BIR) {
  python pipeline\mask_birefnet_cuda.py $BIR $IMG outputs | Out-Null
} else {
  $null | Out-File -FilePath outputs\mask_fg.png -Encoding ascii
}

# 2) DexiNed 배경 윤곽
python pipeline\dexined_infer_cuda.py $DEX $IMG outputs\mask_fg.png outputs\bg_edges.png

# 3) SwiftSketch + DexiNed 병합
python pipeline\merge_outlines.py $SUB outputs\bg_edges.png outputs\mask_fg.png outputs\merged_outline.png

Write-Host "DONE -> outputs\merged_outline.png"
```

---

## 스모크 테스트(10초 컷)

```powershell
# 1) CUDA/PyTorch/ORT 점검
python - << 'PY'
import torch, onnxruntime as ort
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("ort providers:", ort.get_available_providers())
PY

# 2) DexiNed 한 장 추론(마스크 없이)
python pipeline\dexined_infer_cuda.py models\dexined\10_model.pth inputs\mona_lisa.jpg none outputs\bg_edges.png

# 3) SwiftSketch 결과 + 배경 윤곽 병합
python pipeline\merge_outlines.py subject\subject_outline.png outputs\bg_edges.png none outputs\merged_outline.png
```

---

## 품질/속도 팁

- **텍스처 과민 시(붓터치/노이즈)**: DexiNed 입력 전에 `cv2.bilateralFilter`(σ=9~15)로 과잉 에지 감소.  
- **선 두께/톤**: `merge_outlines.py`에서 배경선(회색, 얇게), 피사체선(검정, 굵게) 대비 강화.  
- **해상도 트릭**: 초고해상도일 경우 DexiNed 입력을 1024~1536로 스케일 후 결과를 리스케일.  
- **(고급) DexiNed ONNX 내보내기**: PyTorch→ONNX export 후 `onnxruntime-gpu`로도 실행 가능(연산자 호환성 확인).

---

## 상업용 체크리스트

- **DexiNed**: **MIT License**(저작권 고지 유지 시 상업 이용 가능).  
- **BiRefNet**: **MIT**, Releases에 ONNX 제공.  
- **SwiftSketch 결과물**: 내부/외부 배포 정책 검토.  
- **원본 작품 이미지**: 라이선스/퍼블릭 도메인 여부 확인 필수.

---

## 트러블슈팅

- **`CUDAExecutionProvider`가 보이지 않음**  
  - `pip show onnxruntime-gpu` 확인, CUDA/cuDNN/드라이버 호환 맞추기.  
- **PyTorch에서 CUDA는 True인데 ORT만 실패**  
  - 환경 변수/DLL 경로 문제일 수 있음. `onnxruntime-gpu` 재설치 및 CUDA Toolkit/드라이버 재확인.  
- **DexiNed 체크포인트 경로 오류**  
  - `.pth`를 `models/dexined/`로 두고 스크립트 인자로 정확히 전달.  
- **마스크 없이 실행 시 피사체 에지가 섞임**  
  - BiRefNet 마스크 추가 사용 권장(`mask_fg.png`로 피사체 영역 흰색 채우기).  
- **결과가 너무 추상적/촘촘함**  
  - Otsu 대신 수동 임계, thinning 해제/erosion 적용 등으로 조절.

---

### 다음 단계
- 위 3개 스크립트를 하나로 묶은 **CLI(`pipeline_cuda.py`)** 제작(옵션 플래그: `--bg-thin`, `--near-dilate`, `--lsd`, `--threshold` 등).  
- SVG(피사체) + 래스터(배경) → **이중 벡터화**(Potrace 등) 파이프라인 추가.
