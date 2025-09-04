# ArtiTech — Styled Outline Generation (SwiftSketch Student Roadmap)

> **목표**: SwiftSketch의 “화풍이 있는 선화(스케치)”를 **즉시 서빙 가능한 Student 모델**로 재현하고, Tier‑A(진짜 SwiftSketch/ControlSketch 기반 Teacher)로 **품질 앵커**를 세운 뒤 **자동 승급**까지 지원하는 파이프라인을 구축한다.

---

## 0. 핵심 개념 한눈에
- **Tier‑A (Teacher/고품질)**  
  - **SwiftSketch/ControlSketch**로 만든 고품질 스케치.  
  - 사용자에게 직접 노출되기보다는 **Student 학습·앵커**로 사용. (필요 시 대표작은 서빙에 활용 가능)
- **Tier‑B (즉시 서빙)**  
  - **Student 모델**(우선) + **스타일라이저 폴백**.  
  - “날 엣지”는 사용자에게 **절대 노출하지 않음**. 언제나 **SwiftSketch풍 스타일**을 맞춘 결과만 노출.
- **StyleSpec**  
  - 스케치 스타일을 수치로 정의: **스트로크 수/길이·곡률 분포/실루엣 지배도/폐곡선율/선 두께 프로파일/교차율** 등.
- **자동 승급 로직**  
  - 동일 작품/프리셋에 Tier‑A가 들어오면 **Student 캐시를 자동 대체(승급)**.

---

## 1. 프로젝트 구조
```
art_outlines/
  configs/
    presets.yaml          # 프리셋(8/16/32/64 등) 및 후처리 규칙
    pipeline.yaml         # 전처리/후처리/추론 파이프라인 옵션
    stylespec_v0.1.md     # StyleSpec 설명 및 수치
  data/
    images/               # 원본 이미지
    meta.csv              # id,title,artist,rights,genre,tags...
  cache/
    outlines/{id}/{preset}/TIERA/*.svg|.png
    outlines/{id}/{preset}/STUDENT/*.svg|.png
  scripts/
    gen_tiera.py          # SwiftSketch/ControlSketch로 Teacher 생성
    parse_svg_stats.py    # SVG 통계(길이/곡률/폐곡선율/실루엣 등)
    train_student.py      # Student 학습
    export_onnx.py        # ONNX/INT8 내보내기
    stylizer_fallback.py  # 규칙기반 스타일라이저(폴백)
    serve_demo.py         # 서빙 데모(캐시/승급 로직 포함)
    eval_metrics.py       # 자동 지표 계산 및 리더보드
  eval/
    style_stats.json      # Teacher 기반 스타일 통계
    leaderboard.csv       # 자동 지표 집계
```

---

## 2. StyleSpec v0.1 (초기값 예시)
정확한 값은 D3의 Teacher 통계로 **보정**한다.

- **Stroke Count (프리셋별 최대)**  
  - Easy ≤ 120, Standard ≤ 300, Detailed ≤ 600
- **Silhouette Dominance (실루엣 지배도)**  
  - 전체 스트로크 길이 중 **실루엣 ≥ 70%**
- **Closed Path Ratio (폐곡선율)**  
  - ≥ 90%
- **Curvature Distribution (곡률 분포)**  
  - Teacher μ±σ 범위 안으로 KL/EMD ≤ 0.1
- **Intersections (교차율)**  
  - 교차점/스트로크 ≤ 3%
- **Line Width Profile (선 두께 프로파일)**  
  - 실루엣 2.5–3.0px, 내부선 1.5–2.0px
- **Silhouette‑First Rule**  
  - 실루엣 스트로크를 우선 생성/보존하고 내부선은 절제

> **목적**: Student/스타일라이저 결과가 **눈으로 보아도 SwiftSketch풍**이 되도록 정량 규제.

---

## 3. 데이터 준비
### 3.1 메타데이터 스키마
`data/meta.csv`
```
id,title,artist,rights,genre,tags,notes
000123,The Scream,Edvard Munch,PD,portrait,"face,bridge,sky",""
```
- **rights**는 *PD/CC0* 등 **퍼블릭 도메인만**.

### 3.2 해상도 규격
- 입력 리사이즈: 긴 변 512 또는 768 (학습은 512 권장, 서빙은 512→후보정)

---

## 4. Tier‑A(Teacher) 생성
### 4.1 대상 선정(초기 150–300장 권장)
- 버킷: **인물/건축/정물·동물** 균형으로 샘플링.
- 자동 점수: 에지 밀도↑, 대칭성↑, 얼굴/직선 구조 등.

### 4.2 SwiftSketch/ControlSketch 생성
- 우선순위: **SwiftSketch 32‑stroke** → 부족/실패 시 **ControlSketch**로 보강.
- 산출물: `cache/outlines/{id}/{preset}/TIERA/*.svg|.png`

### 4.3 통계 추출
```bash
python scripts/parse_svg_stats.py \
  --src cache/outlines/*/TIERA \
  --out eval/style_stats.json
```
- 길이/곡률/방향 히스토그램, 실루엣:내부 비율, 폐곡선율/교차율.

---

## 5. Student 모델
### 5.1 입출력
- **입력**: RGB(512) + 선택(Depth/Saliency/Seg/ROI)  
- **출력**: `K × (x0,y0,c1x,c1y,c2x,c2y,x3,y3,width,pen)`

### 5.2 아키텍처(권장 MVP)
- **Encoder**: 경량 CNN 또는 MobileViT(토큰 16×16/32×32)
- **Decoder**: Transformer Decoder가 **스트로크 시퀀스** 생성
- **Multi‑K**: 8/16/32/64 헤드 또는 가변‑K(top‑K 선택)

### 5.3 손실(증류 + 스타일 규제)
1) **Shape Loss**: 헝가리안 매칭 후 L1/L2  
2) **Rendering Loss**: pydiffvg 래스터 vs Teacher → Chamfer/LPIPS  
3) **Style Distribution Loss**: 길이/곡률/방향 히스토그램 KL/EMD  
4) **Constraints**: 실루엣 지배도 ≥τ, 폐곡선율 ≥90%, 교차율 ≤3%  
5) **Preset Loss**: K/최소 길이/곡률 상한/선 두께 프로파일

### 5.4 학습 커리큘럼
- 1단계: **K=32 고정**으로 안정화 →  
- 2단계: 8/16/32/64 **다중‑K** 확장 →  
- 3단계: **하드케이스 마이닝**(자동 하위 10% 가중↑) → 재학습

### 5.5 후처리
- RDP 단순화(곡률 가중), 스무딩, 폐곡선 강제, 짧은 path 제거, 선 두께 프로파일 적용.

### 5.6 경량화/배포
- ONNX export → ORT(DirectML/CPU/MPS/NNAPI)  
- QAT/INT8, 채널/토큰 축소, 연산 병합.

---

## 6. 스타일라이저(폴백)
- 입력 엣지/벡터를 **StyleSpec**으로 최적화:  
  - 실루엣 우선 선택 → 내부선 절제 → 분포(KL/EMD) 맞춤.  
- 목적: Student 실패 시에도 **SwiftSketch풍 결과**를 보장.

---

## 7. 서빙 파이프라인
### 7.1 캐시 키
```
(art_id, preset, tier, model_ver, params_hash)
# tier: TIERA | STUDENT
```
### 7.2 우선순위
1) **Tier‑A** 있으면 즉시 서빙  
2) 없으면 **Student** 결과 서빙  
3) Student 실패 → **스타일라이저 폴백**  
4) Tier‑A가 나중에 도착하면 **자동 승급**

### 7.3 UI 매핑
- 난이도 슬라이더 ↔ 8/16/32/64 (가까운 K로 매핑)  
- 옵션 토글: “주피사체만”, “선 두께” → **SVG 후처리로 즉시 반영**  
- 배지: `빠름(Student)` / `고급(Tier‑A)` / `자동 승급`

---

## 8. 자동 지표 & 휴먼 평가
### 8.1 자동 지표
- **Geom.** Chamfer, SSIM/LPIPS (Teacher 대비)  
- **Vector.** 평균 곡률, 길이/곡률 분포 KL, 폐곡선율, 교차율, 경로/노드 수  
- **Saliency.** 주피사체 커버리지(≥τ)
```bash
python scripts/eval_metrics.py --split val --out eval/leaderboard.csv
```

### 8.2 휴먼 평가(미니테스트)
- 10–15명, 5점 척도: 따라 그리기 쉬움 / 유사성 / 만족도(치유감)  
- 결과로 **프리셋/StyleSpec 파라미터** 조정.

---

## 9. 2주 로드맵(실행 체크리스트)

### Week 1 — 데이터/스펙 고정 & Student 베이스라인
**D1: 킥오프 & 뼈대**
- [ ] 리포/폴더 고정, `StyleSpec_v0.1.md`/`presets.yaml`/`meta.csv` 초안 커밋

**D2: Teacher 샘플 & 최소 생성**
- [ ] 장르 균형 150장 추출, SwiftSketch 32‑stroke(없으면 16) 생성
- [ ] 실패/성공 로그, 썸네일 저장 → `TIERA/`

**D3: Teacher 정제 & 통계**
- [ ] SVG 파서로 길이/곡률/폐곡선율/실루엣 통계 → `style_stats.json`
- [ ] StyleSpec 수치 보정(μ±σ, τ, 임계치)

**D4: Student 베이스라인 골격**
- [ ] 인코더/디코더, 데이터로더, 출력 텐서 Shape 검증

**D5: 손실(1차) & 미니 학습**
- [ ] Shape + Rendering 손실로 과적합 OK 수준까지 수렴 확인

**D6: 손실(2차) — 스타일 분포**
- [ ] 히스토그램 KL/EMD, 실루엣/폐곡선/교차 제약 추가
- [ ] 후처리(스무딩/짧은 path 제거) 결합

**D7: 다중‑K & 1차 벤치**
- [ ] 8/16/32/64 지원, 검증셋/지표 스크립트 완성, ckpt v0.1 저장

### Week 2 — 최적화/경량화 & 서빙 통합
**D8: 속도 최적화 & ONNX**
- [ ] 채널/토큰 축소, ONNX export, ORT 검증, INT8 시도

**D9: 스타일라이저 폴백**
- [ ] RDP/병합/폐곡선 강제 + StyleSpec 최적화 루프

**D10: 서빙 & 캐시 키**
- [ ] 캐시 키/승급 로직, 우선순위(Tier‑A > Student > 폴백), 데모 동작

**D11: 휴먼 평가**
- [ ] 10–15명 블라인드 비교, 프리셋/StyleSpec 미세 조정

**D12: 하드케이스 마이닝 & 재학습**
- [ ] 자동 하위 10% + 불만족 샘플 → Teacher 보강 후 재학습, ckpt v0.2

**D13: 문서화 & 배포 준비**
- [ ] README_Student, StyleSpec 확정본, scripts 사용법, 재현 절차

**D14: 스프린트 리뷰**
- [ ] 목표 지표 달성 여부 판단 → Go면 전 데이터 Student 캐시

---

## 10. 스크립트/CLI 계약

### 10.1 Teacher 생성
```bash
python scripts/gen_tiera.py \
  --input data/images \
  --meta data/meta.csv \
  --out cache/outlines \
  --preset 32 \
  --engine swiftsketch \
  --fallback controlskt \
  --jobs 4
```

### 10.2 스타일 통계
```bash
python scripts/parse_svg_stats.py \
  --src cache/outlines/*/TIERA \
  --out eval/style_stats.json
```

### 10.3 Student 학습
```bash
python scripts/train_student.py \
  --train_csv data/meta.csv \
  --tiera_dir cache/outlines \
  --img_size 512 \
  --k_list 8 16 32 64 \
  --loss shape+render+style \
  --epochs 20 --batch 16 \
  --out models/student_v0.1
```

### 10.4 ONNX 내보내기
```bash
python scripts/export_onnx.py \
  --ckpt models/student_v0.1/best.pt \
  --img_size 512 \
  --out models/student_v0.1.onnx \
  --int8
```

### 10.5 폴백 스타일라이저
```bash
python scripts/stylizer_fallback.py \
  --in_svg sample.svg \
  --stylespec configs/stylespec_v0.1.md \
  --out out.svg
```

### 10.6 서빙 데모
```bash
python scripts/serve_demo.py \
  --cache cache/outlines \
  --models models/student_v0.1.onnx \
  --presets configs/presets.yaml \
  --host 0.0.0.0 --port 8000
```

---

## 11. 완료 기준(허들)
- **스타일 유지**: 길이/곡률 분포 KL ≤ 0.1 (Teacher 대비)  
- **형상 유사**: Chamfer/LPIPS ≥ 90% (Teacher 대비)  
- **구조 품질**: 폐곡선율 ≥ 90%, 실루엣 지배도 ≥ 70%  
- **속도**: 512px, ONNX(DirectML/CPU/MPS/NNAPI)에서 즉시 서빙 가능  
- **휴먼 평가**: 따라 그리기/유사성 평균 점수 **Teacher − 0.3 이내**

---

## 12. 트러블슈팅
- **실루엣이 약함** → 실루엣 마스크 가중↑, 실루엣 우선 샘플링, 두께 프로파일 재조정  
- **선이 지저분함** → RDP 강도↑, 최소 길이/면적 임계치↑, 교차율 페널티↑  
- **속도 부족** → 채널/토큰 축소, INT8, 이미지 512 고정, ORT 세션 옵션 조정  
- **모델 불안정** → K=32 단일로 재안정화 후 다중‑K 재도입

---

## 13. 용어
- **SwiftSketch**: 벡터 스트로크 좌표 공간에서 확산으로 스케치 생성
- **ControlSketch**: Depth/Mask/ROI 제어가 가능한 Teacher 생성 도구
- **Student**: SwiftSketch풍 스타일을 빠르게 흉내 내는 경량 생성기
- **StyleSpec**: 스타일의 정량 규칙 모음(분포/제약)
- **자동 승급**: Tier‑A 도착 시 Student 캐시를 자동 교체

---

## 14. 라이선스 & 권리
- 입력 데이터는 **퍼블릭 도메인/적법 라이선스**만 사용.  
- 생성물 재배포 정책은 제품 정책을 따른다.

---

## 15. 다음 단계(스프린트+1 제안)
- 모바일( CoreML/NNAPI ) 벤치 및 UI 연결
- 하드케이스 자동 탐지기 고도화(불확실성 추정)
- 사용자 피드백 루프(“어려워요/이상해요” → Tier‑A 후보 편입)
