# Snowball 기능 번호

기존 `functions/features/featureXX_*` 폴더 이름과 번호는 유지합니다. Snowball 기능은
별도의 접두어를 사용해 충돌 없이 합칠 수 있도록 구성합니다.

- `snowball01_individual_backtest`: 개별 동적자산배분 백테스트
- `snowball02_portfolio_mix`: 저장된 두 전략의 월별 비중 복원 혼합
- `snowball03_account_dashboard`: Google Sheet 읽기 전용 계좌 대시보드

`nicegui_app.py`는 위 페이지를 등록하고 서버를 시작하는 역할만 담당합니다.
계산 엔진은 UI 프레임워크와 독립적인 `snowball/` 패키지에 유지합니다.
