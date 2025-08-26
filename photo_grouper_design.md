# 📄 Photo Grouper – 설계 문서

## 1. 목표
- 로컬 사진을 **임베딩 + 코사인 유사도**로 그룹화
- 임계값 슬라이더 조절로 실시간 재그룹
- 클러스터별로 선택/내보내기(복사·이동·CSV)

---

## 2. 기술 스택
- **Python 3.10+**
- **PySide6** – 데스크톱 UI
- **PyTorch + torchvision** – ResNet50 임베딩
- **scikit-learn** – 코사인 유사도 계산
- **SQLite** – 임베딩 캐시
- **Pillow** – 이미지 로딩/썸네일
- (선택) **pillow-heif** – HEIC 지원
- (선택) **faiss-cpu** – 대용량 가속

---

## 3. 아키텍처

```
photo-grouper/
  app.py               # 엔트리포인트
  core/
    scanner.py         # 이미지 경로 수집
    embedder.py        # 임베딩 추출 + 캐시
    grouper.py         # 코사인 유사도 기반 그룹화
    export.py          # 선택 이미지 내보내기
    thumbs.py          # 썸네일 생성/캐시
  infra/
    cache_db.py        # SQLite 연결/스키마
    config.py          # 설정 로드/저장
  ui/
    main_window.py     # QMainWindow, 슬라이더, 진행률
    preview_panel.py   # 썸네일 그리드, 선택/체크
```

---

## 4. 데이터 흐름

1. **폴더 선택** → `scanner.scan_images()`로 파일 목록
2. **임베딩 계산** → `embedder.get_or_make_embedding()`
   - 캐시에서 읽거나 새로 계산
3. **유사도 계산** → `grouper.group_by_threshold()`
   - 타일링 방식으로 메모리 절약
4. **연결요소(Union-Find)** → 최종 그룹 생성
5. **UI 표시** → `main_window`에서 그룹별 대표 썸네일
6. **내보내기** → `export.copy_selected()` or `export.move_selected()`

---

## 5. 핵심 알고리즘
- **임베딩**: ResNet50 Global Average Pool → L2 Normalize
- **유사도 계산**: 코사인 유사도(내적), 타일링으로 메모리 절감
- **그룹화**: Union-Find 기반 연결 요소 탐지 → 연쇄 유사성 반영
- **캐싱**: `(path, mtime, sha256)`로 캐시 유효성 관리

---

## 6. UI 구성
- 폴더 선택 버튼
- 임계값 슬라이더 (0.70 ~ 0.99)
- 진행률 표시 바
- 그룹 목록 (썸네일 포함)
- 내보내기 버튼

---

## 7. 확장 아이디어
- 임베딩 모델 교체 (CLIP, EfficientNet 등)
- 예비 필터(pHash) 적용으로 속도 향상
- 그룹 병합/분할 기능
- 얼굴/장소 기반 필터링
- 자동 폴더 정리 (날짜/장소 기준)

---
