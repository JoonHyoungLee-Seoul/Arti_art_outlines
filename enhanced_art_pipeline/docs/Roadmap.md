목표: _SwiftSketch 풍의 선화 스타일_을 **즉시 서빙 가능한 Student 모델**로 재현하고, Tier-A(진짜 SwiftSketch)로 품질 앵커를 세운 뒤 **자동 승급**까지 연결.

# 0) 이번 스프린트 범위(변경점 요약)

- **Tier-B = Student 모델**(우선) + 규칙기반 스타일라이저(폴백). “날 엣지”는 유저에게 **미노출**.
    
- **StyleSpec v0.1** 확정: 스트로크 수/곡률/실루엣 지배도/폐곡선율/선 두께 프로파일의 **정량 규칙**.
    
- **Teacher(=Tier-A)**는 **소규모(예: 150–300장)**만 우선 생성해 증류 학습용으로 사용.
    
- **서빙 로직**: Tier-A 있으면 우선, 없으면 Student→(실패 시) 스타일라이저. 캐시 자동 승급.
    

---

# 1) 2주 로드맵 (D1–D14)

## Week 1 — 데이터/스펙 고정 & Student 베이스라인

### D1: 킥오프 & 뼈대

-  리포 구조 생성(`art_outlines/`), `configs/`, `scripts/`, `cache/` 폴더 고정
    
-  `StyleSpec_v0.1.md` 초안(아래 3)의 기본값 작성
    
-  `presets.yaml`(8/16/32/64) 초안 작성
    
-  `meta.csv` 스키마 확정(`id,title,artist,rights,genre,tags`)
    
- **DoD**: 리포 빌드 성공, 스펙 문서/설정 파일 커밋됨
    

### D2: Teacher 샘플 선정 & 최소 생성

-  장르별 3버킷(인물/건축/정물·동물)에서 **균형 샘플 150장** 추출
    
-  (가능한 자원에서) **SwiftSketch**로 32-stroke 우선 생성(없으면 16)
    
-  생성 실패/성공 로그 수집, 썸네일 저장
    
- **DoD**: `cache/outlines/{id}/TIERA/*`에 최소 100장 이상 생성
    

### D3: Teacher 정제 & 통계 추출

-  SVG 파서 작성(큐빅 베지어 파라미터/길이/곡률/폐곡선율 계산)
    
-  **스타일 통계** 산출: 길이·곡률 히스토그램, 실루엣:내부 비율, 선 교차율
    
-  `StyleSpec_v0.1.md`를 **Teacher 통계 기반**으로 수치 보정
    
- **DoD**: `eval/style_stats.json` 생성, StyleSpec 수치 확정
    

### D4: Student 베이스라인 모델(Encoder-Decoder) 골격

-  **입력**: RGB(512) + 옵션(Depth/Saliency toggle)
    
-  **출력 텐서**: `K×(x0,y0,c1x,c1y,c2x,c2y,x3,y3,width,pen)` (정규화 0–1)
    
-  데이터로더: (이미지, Teacher-SVG→텐서) 페어 로딩
    
- **DoD**: 단일 배치 forward/아웃풋 텐서 shape 검증 통과
    

### D5: 손실(1차) & 미니 학습

-  **좌표/형상 손실**: 매칭(헝가리안) + L1/L2
    
-  **렌더링 손실**: pydiffvg 래스터 → Chamfer/LPIPS
    
-  2–3h 미니 학습으로 수렴 확인(과적합 OK)
    
- **DoD**: Teacher 대비 Chamfer/SSIM 개선 추이 그래프 저장
    

### D6: 손실(2차) — **스타일 분포** 추가

-  길이/곡률/방향 히스토그램 → **KL/EMD 손실** 추가
    
-  실루엣 지배도(≥τ), 폐곡선율(≥90%) **제약** 추가
    
-  벡터 후처리(짧은 path 제거/스무딩) 모듈 추가
    
- **DoD**: 분포 손실이 적용된 학습 로그/지표 개선 확인
    

### D7: 다중-K 지원 & 1차 벤치

-  **다중 헤드(8/16/32/64)** 또는 가변-K(top-K) 구현
    
-  검증셋 고정(장르 분리), 자동 지표 스크립트 완성
    
-  **체크포인트 v0.1** 저장
    
- **DoD**: 8/16/32/64 모두 추론 성공, 기본 지표 표 생성
    

---

## Week 2 — 최적화/경량화 & 서빙 통합

### D8: 속도 최적화 & ONNX

-  모델 경량화(채널 축소/토큰 축소), 연산 병합
    
-  **ONNX export** + ORT 런타임 검증(CPU/DirectML/MPS 중 1개)
    
-  INT8(QAT or PTQ) 시도, 품질 손실 체크
    
- **DoD**: 512px 추론 지연 단축, ONNX 추론 스크립트 동작
    

### D9: 스타일라이저(폴백) 확정

-  RDP 단순화(곡률 가중), 경로 병합, 폐곡선 강제
    
-  StyleSpec 기준으로 **최적화 루프**(길이/곡률 분포 맞추기)
    
- **DoD**: 실패 시 Student 대신 스타일라이저로 품질 하락 최소화
    

### D10: 서빙 파이프라인 & 캐시 키

-  캐시 키 `(art_id,preset,tier,model_ver,params_hash)` 설계
    
-  **서빙 우선순위**: Tier-A > Student > 스타일라이저
    
-  “**더 깔끔하게**” 버튼 → Tier-A 있으면 즉시 교체
    
- **DoD**: 로컬 서버/노트북에서 워크플로 데모 가능
    

### D11: 휴먼 평가(미니테스트)

-  10–15명 대상 설문(따라 그리기 용이/유사성/만족도)
    
-  Tier-A vs Student vs 스타일라이저 **블라인드 비교**
    
- **DoD**: 휴먼 점수 수집 및 프리셋/StyleSpec 미세 조정
    

### D12: 하드케이스 마이닝 & 재학습

-  자동 하위 10%/휴먼 불만족 샘플 **리스트업**
    
-  Teacher 보강(가능하면 몇 장 추가 생성) → Student **리트레이닝**
    
- **DoD**: 하드케이스 지표 상승, 체크포인트 v0.2 저장
    

### D13: 문서화 & 배포 준비

-  `README_Student.md`(입출력/손실/지표/서빙 API)
    
-  `StyleSpec_v0.1.md` 확정본과 예시 이미지
    
-  `scripts/` 사용법(help/예시 커맨드) 정리
    
- **DoD**: 팀원이 그대로 실행 가능(버튼 3번 이내)
    

### D14: 스프린트 리뷰 & Go/No-Go

-  목표 지표 달성 확인(아래 2)
    
-  Go면: 전 데이터 Tier-B=Student로 **일괄 캐시**, Tier-A는 점진 증강
    
-  No-Go면: 병목/손실 비중/아키 수정 액션 아이템 작성
    
- **DoD**: 의사결정 및 다음 스프린트 백로그 생성
    

---

# 2) 완료 기준(지표/허들)

- **스타일 유지**: 길이/곡률 분포 KL ≤ 0.1 (Teacher 대비)
    
- **형상 유사**: Chamfer/LPIPS가 Teacher 대비 **≥90%** 수준
    
- **구조 품질**: 폐곡선율 ≥ 90%, 실루엣 지배도 ≥ 70%
    
- **속도**: 512px, ONNX CPU/DirectML/MPS에서 **실사용 응답** 확보
    
- **휴먼 평가**: 따라 그리기/유사성 평균 점수 **Teacher − 0.3 이내**
    

---

# 3) StyleSpec v0.1 (초기값 예시)

- **스트로크 수**: Easy≤120 / Standard≤300 / Detailed≤600
    
- **실루엣 지배도**(전체 길이 대비 실루엣 비중): ≥70%
    
- **폐곡선율**: ≥90%
    
- **평균 곡률 범위**: μ±σ 내(Teacher 통계 기반)
    
- **교차율**: 전체 교차점/스트로크 ≤ 3%
    
- **선 두께 프로파일**: 실루엣 2.5–3.0px / 내부 1.5–2.0px
    

---

# 4) 체크리스트(요약 버전)

### 데이터/Teacher

-  150–300장 균형 샘플 추출
    
-  Tier-A(16/32 우선) 생성 & 통계 산출
    
-  실패 로그/재시도 스크립트
    

### Student 모델

-  Encoder-Decoder 골격 + 다중-K 헤드
    
-  손실: 좌표/렌더링/스타일 분포/제약
    
-  후처리: 스무딩/폐곡선/짧은 path 제거
    
-  ONNX 변환 + 경량화(INT8)
    

### 스타일라이저(폴백)

-  RDP 단순화(곡률 가중)
    
-  분포 타게팅 최적화
    
-  API: Student 실패 시 자동 대체
    

### 평가/서빙

-  자동 지표 스크립트 + 리더보드
    
-  휴먼 테스트 폼/집계
    
-  서빙 우선순위 + 캐시 키/승급 로직
    
-  “더 깔끔하게” 버튼
    

---

# 5) 파일 구조 권장

```swift
art_outlines/
  configs/
    presets.yaml
    pipeline.yaml
    stylespec_v0.1.md
  data/
    images/
    meta.csv
  cache/
    outlines/{id}/{preset}/TIERA/*.svg|.png
    outlines/{id}/{preset}/STUDENT/*.svg|.png
  scripts/
    gen_tiera.py
    parse_svg_stats.py
    train_student.py
    export_onnx.py
    stylizer_fallback.py
    serve_demo.py
    eval_metrics.py
  eval/
    style_stats.json
    leaderboard.csv
```
