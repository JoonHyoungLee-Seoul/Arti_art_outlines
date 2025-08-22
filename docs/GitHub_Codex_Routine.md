# GitHub × Codex × Cloud Server 운용 가이드

이 문서는 혼자 개발하면서 **ChatGPT Codex (AI Agent)** 와 **GitHub** 및 **Cloud Server (elice.io 등)** 를 함께 사용할 때 필요한 기본 상식과 루틴을 정리한 것이다.

---

## 1. 기본 Git 워크플로우

### 파일 생성/수정 → 커밋 → 푸시

```bash
# 변경 확인
git status
git diff

# 파일 추가 (특정 파일만)
git add docs/Two_Week_Roadmap_TODOs.md

# 또는 모든 변경
git add .

# 커밋
git commit -m "docs: add Two_Week_Roadmap_TODOs.md"

# 푸시 (main 브랜치)
git push origin main
```

## 2. 커밋 메시지 규칙 (Conventional Commits 권장)

- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `refactor`: 리팩토링 (기능 변화 없음)
- `test`: 테스트 추가/수정
- `chore`: 설정/빌드/기타 유지보수

예시:
```text
feat: add outline scoring for Tier-A
fix: handle None input in gen_tiera.py
docs: add roadmap document
test: add case for empty input
```

## 3. 브랜치 전략

- `main`: 안정된 코드만 유지
- `feat/*`: 기능 개발
- `fix/*`: 버그 수정
- `chore/*`: 설정/환경 관련
- `exp/*` 또는 `wip/*`: 실험/미완성 코드

**브랜치 생성**
```bash
git checkout -b feat/outline-eval
```

## 4. 상황별 유용한 Git 명령어

### 잘못 스테이징 했을 때
```bash
git restore --staged <파일>
```

### 원격과 충돌할 때
```bash
git fetch origin
git rebase origin/main        # 히스토리 직선화 (권장)
# 충돌 해결 후:
git add <파일>
git rebase --continue
git push --force-with-lease   # 필요 시
```

```bash
# 단순 merge로 처리하려면
git pull --no-rebase origin main
```

### 로컬 변경을 버리고 원격만 따르기
```bash
# 단순 merge로 처리하려면
git fetch origin
git reset --hard origin/main
git submodule update --init --recursive
```

## 5. 서브모듈(swiftsketch) 관리

- **포함 방법**: 서브모듈로 추가
```bash
git submodule add git@github.com:JoonHyoungLee-Seoul/swiftsketch.git swiftsketch
```

- **업데이트 방법**: 특정 태그/커밋에 핀(pin)
```bash
cd swiftsketch
git fetch origin
git checkout <태그|커밋SHA|브랜치>
cd ..
git add swiftsketch
git commit -m "chore(submodule): bump swiftsketch to <ref>"
git push
```

- 클론할 때
```bash
git clone --recursive <repo-url>
git submodule update --init --recursive
```

## 6. Codex ↔ GitHub ↔ Cloud Server 루틴

### 1) 로컬/클라우드 서버에서 작업

- 기능/문서 추가 후 **테스트 통과 상태**에서 커밋 & 푸시
    
- Codex가 GitHub 최신 코드를 기준으로 작업하므로 **반드시 push** 필요
    
### 2) Codex 작업

- Codex는 GitHub 리포를 샌드박스 VM에 클론하여 작업
    
- 할 수 있는 일:
    
    - 테스트 실행
    - 코드 수정
    - 브랜치 생성
    - Pull Request(PR) 생성
        
**예시 프롬프트 (Codex Code 모드)**:
```javascript
브랜치 feat/tiera-eval
1) gen_tiera.py에 null 입력 테스트 추가
2) 실패 재현 후 수정
3) flake8, pytest -q 통과
4) PR 생성 (변경 요약 포함)
```

### 3) GitHub 업데이트
- Codex PR → CI 자동 실행 → 리뷰/merge → main 최신화
- CI 예시: GitHub Actions
```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive
    fetch-depth: 0
- uses: actions/setup-python@v5
  with:
    python-version: "3.10"
- run: pip install -r requirements.txt
- run: pytest -q
```

### 4) Cloud Server 동기화
- Codex/GitHub 변경을 서버에 적용
```bash
git fetch origin
git pull origin main
git submodule update --init --recursive
```

## 7. 푸시 전 체크리스트 ✅

-  `git status` 확인 → 원치 않는 파일이 스테이징되지 않았는가?
-  테스트/빌드 통과 여부 (`pytest -q`, `npm test` 등)
-  서브모듈 포인터(swiftsketch) 변경이 의도한 것인지
-  `.gitignore` 적용으로 시크릿/대용량 파일 제외 확인
    
---

## 8. 일일 운용 루프

1. **Cloud Server**: 개발 & 테스트 → `git commit` & `git push`
2. **Codex**: 작업 지시 (테스트 추가, 수정, PR 생성)
3. **GitHub**: PR 생성 & CI 확인 → merge
4. **Cloud Server**: `git pull`로 최신 코드 반영
5. (선택) 태그 릴리스 → changelog 업데이트

---

## 9. 좋은 습관

- 커밋은 **작업 단위별로** (하나의 목적만 담기)
- main은 항상 깨끗하게 (테스트 통과 보장)
- Codex에게 지시하기 전 항상 최신 코드를 push
- Codex PR은 CI 통과 후 merge
- 서브모듈은 특정 커밋/태그에 핀(pin) → 재현성 보장

