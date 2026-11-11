# Snowball 공통 모듈

Snowball 01~03 기능이 실제로 함께 사용하는 NiceGUI 전용 코드만 둡니다.

- `layout.py`: Snowball 헤더와 기능 선택 화면
- `styles.py`: Snowball NiceGUI 전용 CSS
- `backtest_results.py`: 개별·혼합 백테스트 공통 결과 화면
- `preset_store.py`: NiceGUI 사용자 저장소와 브라우저 localStorage 동기화
- `backtest_cache.py`: 수정시가·수정종가 기반 백테스트 전용 단기 캐시

기존 `functions/common`에 이름이나 목적이 비슷한 코드가 있어도 자동으로 합치지
않습니다. 데이터 기준, 캐시 수명, 조정가격 여부, UI 런타임이 동일하다고 테스트로
확인한 뒤에만 통합 후보로 제안합니다.

특히 다음 기능은 현재 의도적으로 분리합니다.

- 백테스트용 Yahoo 수정시가·수정종가 조회
- 계좌 평가용 Yahoo 일반 종가 조회
- Streamlit 상태 저장
- NiceGUI 사용자 저장소와 브라우저 localStorage

