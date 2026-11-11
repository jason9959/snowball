"""Shared metric explanations and accessible comparison tables."""
import html
import io
import pandas as pd
import streamlit as st

HELP = {
    '원/달러': '미국 달러 1달러를 사는 데 필요한 원화 금액입니다. 조건 충족은 전략의 평균 환율 기준으로 판단합니다.',
    '달러지수': '주요 통화 대비 미국 달러의 상대적 가치를 나타내는 지수입니다.',
    '달러갭': '전략에서 원/달러 환율과 달러지수를 비교해 산출한 상대 수준 지표입니다.',
    '적정환율': '전략의 환율·달러지수 관계로 추정한 기준 환율이며 시장의 적정가격을 보장하지 않습니다.',
    '샤프': '무위험수익률을 0%로 두고, 일별 평균 수익률을 전체 변동성으로 나눈 연환산 지표입니다. 높을수록 같은 위험 대비 수익이 큽니다.',
    'Sortino': '목표수익률 0% 미만의 하락 편차를 전체 관측일 기준으로 계산합니다. 하락 위험 대비 수익을 나타내며, 하락 편차가 없으면 계산할 수 없습니다.',
    'Calmar': '연복리 수익률(CAGR)을 최대낙폭의 절댓값으로 나눈 값입니다. 최대낙폭이 0이면 계산할 수 없습니다.',
    'XIRR': '초기 투자금과 추가 납입금을 음수, 마지막 평가금액을 양수 현금흐름으로 두고 실제 날짜 간격을 반영한 연환산 수익률입니다. 미래 시뮬레이션에서는 가정한 일정에 따른 예상값입니다.',
    '회복': '최대낙폭 구간의 하락 직전 마지막 고점부터 그 수준을 처음 회복한 날까지의 달력 일수입니다. 하락 기간을 포함합니다. 종료일까지 회복하지 못하면 미회복, 하락이 없으면 0일입니다. 전체 기간의 가장 긴 침체 구간과는 다를 수 있습니다.',
    '총 예상 투입금 대비': '최종 예상 금액을 총 예상 투입금으로 나눈 뒤 1을 뺀 총수익률입니다. 연환산 수익률은 아닙니다.',
    'P5': '시뮬레이션 분포에서 5% 지점입니다. 이 값보다 작은 결과가 약 5%입니다.',
    'P50': '시뮬레이션 분포의 중앙값입니다.',
    'P95': '시뮬레이션 분포에서 95% 지점입니다. 이 값보다 큰 결과가 약 5%입니다.',
    '연환산 수익률': '분석 기간의 성과를 매년 일정하게 복리로 얻었다고 환산한 수익률(CAGR)입니다. 적립금이 있는 비교표에서는 납입 효과를 제외한 성과를 사용합니다.',
    '변동성': '수익률이 평균 주변에서 얼마나 흔들렸는지 나타냅니다. 일별 수익률의 표준편차에 √252를 곱해 연환산합니다.',
    '낙폭': '이전 고점 대비 하락률 중 가장 큰 손실입니다. 음수로 표시하며 0에 가까울수록 하락이 작습니다.',
    '수익률': '조회 기간의 투자 수익을 기준 투자금으로 나눈 비율입니다. 적립식의 총수익률은 (최종 평가금액 ÷ 총 납입금 − 1)이며 연환산 XIRR과 다릅니다.',
    '투입': '초기 투자금과 분석 기간에 추가로 납입한 금액의 합계입니다.',
    '납입': '초기 투자금과 분석 기간에 추가로 납입한 금액의 합계입니다.',
    '초기': '분석 시작 시점에 투자하는 금액입니다.',
    '최종': '분석 종료 시점의 보유자산 평가액과 현금의 합계입니다.',
    '투자 수익': '최종 자산에서 총 납입금을 뺀 손익 금액입니다.',
    '상승 확률': '선택한 주기로 집계한 과거 구간 중 수익률이 0%보다 컸던 비율입니다. 미래 상승을 보장하는 확률이 아닙니다.',
    '하락·보합 확률': '선택한 주기로 집계한 과거 구간 중 수익률이 0% 이하였던 비율입니다.',
    '평균 수익률': '선택한 주기별 수익률의 산술평균입니다. 연복리 수익률과 다릅니다.',
    '중앙값 수익률': '주기별 수익률을 크기순으로 정렬했을 때 가운데 위치한 값입니다.',
    '분위': '시뮬레이션 결과를 크기순으로 정렬한 위치입니다. P5/P50/P95는 각각 5%/50%/95% 지점이며 미래 결과를 보장하지 않습니다.',
    '빈도': '해당 수익률 구간에 포함된 관측 횟수입니다.',
    '비율': '포트폴리오에서 해당 종목이 차지하도록 설정한 투자 비중입니다.',
    '전환': '전략 규칙에 따라 보유 상태가 바뀐 횟수입니다. 매수·매도 전환은 각각 해당 방향의 횟수입니다.',
    '완료 사이클': '전략의 매수와 매도 과정을 마친 사이클 수입니다.',
    '성장': 'SPY 가격이 200일 단순이동평균보다 높은지 확인하는 성장 신호입니다.',
    '기대물가 수준': '5년 기대인플레이션 지표가 설정한 기준보다 높은지 확인합니다.',
    '기대물가 모멘텀': '기대인플레이션 지표의 60일 변화가 양수인지 확인합니다.',
    '자산 모멘텀': '자산 지표의 60일 회귀 기울기가 양수인지 확인합니다.',
    '자산': '기업이 보유하거나 통제하는 경제적 자원의 보고기간 말 장부금액입니다.',
    '부채': '기업이 향후 갚거나 이행해야 하는 의무의 보고기간 말 금액입니다.',
    '자본': '자산에서 부채를 뺀 순자산입니다.',
    '매출액': '해당 보고기간에 영업활동으로 발생한 매출입니다.',
    '영업이익': '매출에서 매출원가와 영업비용을 차감한 영업활동의 손익입니다.',
    '당기순이익': '영업외 손익과 법인세 등을 반영한 해당 기간의 최종 손익입니다.',
    '영업현금흐름': '주요 영업활동에서 발생한 현금 유입과 유출의 순액입니다.',
    '투자현금흐름': '설비와 금융자산 등의 취득·처분에서 발생한 현금흐름입니다.',
    '재무현금흐름': '차입·상환·증자·배당 등 자금 조달과 반환에서 발생한 현금흐름입니다.',
    '현금 및 현금성자산 증가(감소)': '보고기간 중 현금 및 현금성자산의 순증감입니다.',
    '기초 현금 및 현금성자산': '보고기간 시작 시점의 현금 및 현금성자산 잔액입니다.',
    '기말 현금 및 현금성자산': '보고기간 종료 시점의 현금 및 현금성자산 잔액입니다.',
}

def metric_help(label):
    label = str(label)
    aliases = {'CAGR':'연환산 수익률', '연복리':'연환산 수익률', 'MDD':'낙폭', 'MaxDD':'낙폭', 'Sharpe':'샤프', 'Volatility':'변동성'}
    if '회복' in label:
        return HELP['회복']
    if label in HELP:
        return HELP[label]
    for key, target in aliases.items():
        if key in label:
            return HELP[target]
    for key in sorted(HELP, key=len, reverse=True):
        if key in label:
            return HELP[key]
    return None

def metric(target, label, *args, **kwargs):
    explanation = metric_help(label)
    if explanation:
        kwargs['help'] = explanation
    return target.metric(label, *args, **kwargs)

def render_metric_table(columns, heading='지표'):
    frame = pd.DataFrame(columns)
    rows = []
    for label, values in frame.iterrows():
        explanation = metric_help(label)
        safe = html.escape(str(label))
        if explanation:
            safe = '<details class="metric-tip"><summary>' + safe + ' ⓘ</summary><span>' + html.escape(explanation) + '</span></details>'
        cells = ''.join('<td>' + html.escape('-' if pd.isna(v) else str(v)) + '</td>' for v in values)
        rows.append('<tr><th scope="row">' + safe + '</th>' + cells + '</tr>')
    headers = ''.join('<th scope="col">' + html.escape(str(c)) + '</th>' for c in frame.columns)
    st.html('''<style>
    .metric-scroll{max-width:100%;overflow:auto;padding-bottom:1rem}
    .metric-table{border-collapse:separate;border-spacing:0;width:100%;font-size:.95rem;border:1px solid #d9dde3;border-radius:8px;overflow:hidden;background:#fff}
    .metric-table th,.metric-table td{padding:.7rem .8rem;border-bottom:1px solid #e5e8eb;text-align:right;vertical-align:top}
    .metric-table thead th{background:#eeeeee;color:#4e5968;font-weight:700;border-bottom:1px solid #d9dde3}
    .metric-table tbody tr:nth-child(odd){background:#fafafa}
    .metric-table tbody tr:nth-child(even){background:#f5f6f7}
    .metric-table tbody tr:last-child th,.metric-table tbody tr:last-child td{border-bottom:0}
    .metric-table tbody th{color:#333d4b;font-weight:600}
    .metric-table th:first-child{text-align:left;min-width:180px}
    .metric-tip summary{cursor:pointer;list-style:none}
    .metric-tip span{display:none;max-width:360px;white-space:normal;font-weight:normal;padding:.6rem;border:1px solid #8886;border-radius:8px}
    .metric-tip:hover span,.metric-tip[open] span,.metric-tip:focus-within span{display:block}
    </style>''' + '<div class="metric-scroll"><table class="metric-table"><thead><tr><th>' + html.escape(heading) + '</th>' + headers + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')

def explain_metrics(labels):
    explanations = {str(label): metric_help(label) for label in labels if metric_help(label)}
    if explanations:
        with st.expander('지표 설명 ⓘ'):
            for label, explanation in explanations.items():
                st.markdown(f'**{label}** — {explanation}')

def metric_dataframe(target, data=None, *args, **kwargs):
    """Attach column-header help; a tap-open glossary also covers row labels."""
    frame = data if isinstance(data, pd.DataFrame) else getattr(data, 'data', None)
    if isinstance(frame, pd.DataFrame):
        # Keep calculation indices intact while presenting dates without a midnight time.
        frame = frame.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame.index = frame.index.strftime('%Y-%m-%d')
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].dt.strftime('%Y-%m-%d')
        data = frame
    if isinstance(frame, pd.DataFrame):
        config = dict(kwargs.get('column_config') or {})
        for column in frame.columns:
            explanation = metric_help(column)
            if explanation and column not in config:
                config[column] = st.column_config.Column(help=explanation)
        kwargs['column_config'] = config
    result = target.dataframe(data, *args, **kwargs)
    if isinstance(frame, pd.DataFrame):
        explain_metrics(list(frame.columns) + list(frame.index))
    return result

def table_download_bytes(tables):
    """Return CSV and Excel payloads for named display tables.

    ``tables`` is an ordered mapping of sheet/table names to DataFrames.
    Display-only date formatting is applied without mutating source frames.
    """
    prepared = {}
    for name, data in tables.items():
        frame = data.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame.index = frame.index.strftime('%Y-%m-%d')
        for column in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].dt.strftime('%Y-%m-%d')
        prepared[str(name)[:31]] = frame
    csv_data = next(iter(prepared.values())).to_csv(index=True).encode('utf-8-sig') if len(prepared) == 1 else None
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        for name, frame in prepared.items():
            frame.to_excel(writer, sheet_name=name, index=True)
    return csv_data, workbook.getvalue()

def render_table_downloads(tables, stem, key_prefix):
    """Render consistent CSV/Excel download buttons for result tables."""
    csv_data, excel_data = table_download_bytes(tables)
    left, right = st.columns(2)
    if csv_data is not None:
        left.download_button('CSV 다운로드', csv_data, f'{stem}.csv', 'text/csv', key=f'{key_prefix}_csv', use_container_width=True)
    right.download_button('Excel 다운로드', excel_data, f'{stem}.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=f'{key_prefix}_excel', use_container_width=True)
