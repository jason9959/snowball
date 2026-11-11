"""기업 재무 대시보드: API 연결 전 레이아웃 프로토타입."""
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

import plotly.graph_objects as go
import streamlit as st
from common.metric_ui import metric as render_metric, metric_dataframe
from .data_sources import DataSourceError, fetch_dart_companies, fetch_sec_companies, search_companies
from .financials import BALANCE, CASH, INCOME, load_financials
from .report_image import render_report

st.html("""
<style>
[data-testid="stMainBlockContainer"] {max-width:1180px; margin:auto; padding-top:3rem;}
h1 {letter-spacing:-1.3px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
.intro {color:#6B7684; margin-bottom:26px; font-size:17px;}
</style>
""")

COLORS = ["#3182F6", "#F59E42", "#20B89A", "#9A75DB", "#8294B0", "#234F81"]


def secret(name):
    try:
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_data(ttl=86_400, show_spinner=False)
def company_catalog(dart_api_key, sec_user_agent):
    """Load DART and SEC catalogs only when the user explicitly searches."""
    companies, errors = [], []
    jobs=[(fetch_dart_companies,dart_api_key),(fetch_sec_companies,sec_user_agent)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(loader,credential):loader for loader,credential in jobs}
        for future in as_completed(futures):
            try:
                companies.extend(future.result())
            except DataSourceError as error:
                errors.append(str(error))
    return companies, errors


@st.cache_data(ttl=3_600, show_spinner=False, max_entries=32)
def financial_result(company, start_year, start_quarter, end_year, end_quarter,
                     quarterly, dart_api_key, sec_user_agent):
    frame, unit = load_financials(
        company, start_year, end_year, quarterly, dart_api_key, sec_user_agent
    )
    if quarterly:
        allowed = {
            f"{year} Q{quarter}"
            for year in range(start_year, end_year + 1)
            for quarter in range(1, 5)
            if (start_year, start_quarter) <= (year, quarter) <= (end_year, end_quarter)
        }
        frame = frame.loc[[label for label in frame.index if label in allowed]]
    if frame.empty:
        raise DataSourceError("선택한 기간에 표시할 재무정보가 없습니다.")
    return frame, unit


def chart_section(title, description, columns, frame, unit):
    with st.container(border=True):
        st.subheader(title)
        st.caption(description)
        fig = go.Figure()
        for name, color in zip(columns, COLORS):
            fig.add_bar(x=frame.index.tolist(), y=frame[name].tolist(), name=name,
                        marker_color=color,
                        hovertemplate="%{x}<br>%{y:,.1f} " + unit + "<extra>%{fullData.name}</extra>")
        fig.update_layout(barmode="group", height=370, margin=dict(l=0, r=0, t=20, b=0),
                          legend=dict(orientation="h", y=1.18, x=0),
                          yaxis_title=unit, xaxis=dict(type="category"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(zeroline=True, zerolinecolor="#8294B0", gridcolor="#EEF1F4")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with st.expander("상세 수치 보기"):
            metric_dataframe(st, frame[columns].T, width="stretch")


@st.cache_data(show_spinner=False, max_entries=8)
def result_png(frame, name, frequency, unit):
    return render_report(frame, name, frequency, unit, [("01 Financial Position", BALANCE), ("02 Income", INCOME), ("03 Cash Flow", CASH)], COLORS)


def _goto(page: str) -> None:
    st.session_state["current_page"] = page


def _saved_defaults():
    return st.session_state.get("f08_saved", {})


def render(page: str) -> None:
    if page == "f08_results":
        render_results()
    else:
        render_conditions()


def render_conditions() -> None:
    st.title("🏢 기업 재무 대시보드")
    st.markdown('<div class="intro">기업의 자산, 실적, 현금 흐름을 기간별로 비교합니다.</div>', unsafe_allow_html=True)
    st.info("한국 기업은 DART, 미국 기업은 SEC EDGAR의 공식 공시 데이터를 사용합니다.")
    saved=_saved_defaults()

    if st.button("↺ 조건 초기화", key="reset_f08_conditions"):
        for key in list(st.session_state.keys()):
            if key.startswith("f08_w_") or key in {"f08_saved","f08_result","f08_matches","f08_search_query"}:
                st.session_state.pop(key,None)
        st.rerun()

    dart_api_key=secret("DART_API_KEY")
    sec_user_agent=secret("SEC_USER_AGENT")
    if not dart_api_key and not sec_user_agent:
        st.warning("Streamlit Secrets에 DART_API_KEY 또는 SEC_USER_AGENT를 설정해 주세요. 화면은 사용할 수 있지만 기업 검색은 실행되지 않습니다.")

    query=st.text_input("기업명 또는 종목코드",value=saved.get("query",""),placeholder="예: 삼성전자, 005930, Apple, AAPL",key="f08_w_query")
    c1,c2,c3=st.columns(3)
    with c1:
        frequency=st.radio("조회 기준",["연도별","분기별"],index=0 if saved.get("frequency","연도별")=="연도별" else 1,horizontal=True,key="f08_w_frequency")
    quarterly=frequency=="분기별"
    with c2:
        start_year=int(st.number_input("시작 연도",2015,date.today().year,int(saved.get("start_year",date.today().year-3)),key="f08_w_start_year"))
        start_quarter=st.selectbox("시작 분기",[1,2,3,4],index=int(saved.get("start_quarter",1))-1,format_func=lambda x:f"{x}분기",key="f08_w_start_q") if quarterly else 1
    with c3:
        end_year=int(st.number_input("종료 연도",2015,date.today().year,int(saved.get("end_year",date.today().year-1)),key="f08_w_end_year"))
        end_quarter=st.selectbox("종료 분기",[1,2,3,4],index=int(saved.get("end_quarter",4))-1,format_func=lambda x:f"{x}분기",key="f08_w_end_q") if quarterly else 4

    normalized=query.strip().casefold()
    searched_query=st.session_state.get("f08_search_query")
    matches=st.session_state.get("f08_matches",[]) if normalized and normalized==searched_query else []
    company=None
    if matches:
        company=st.selectbox("검색 결과",matches,format_func=lambda item:item.label,key="f08_w_company")
        st.caption(f"데이터 출처: {company.source}")
    else:
        st.caption("기업 목록은 화면 진입 시 불러오지 않습니다. 조건을 입력한 뒤 아래의 기업 검색 버튼을 눌러 주세요.")

    st.divider()
    left,right=st.columns(2)
    with left:
        back=st.button("뒤로",key="back_f08_conditions",use_container_width=True)
    with right:
        action_label="조회 실행" if matches else "기업 검색"
        run=st.button(action_label,key="run_f08_conditions",type="primary",use_container_width=True)

    if back:
        _goto("feature"); st.rerun(); return
    if not run:
        return
    if not normalized:
        st.error("기업명 또는 종목코드를 입력해 주세요."); return

    current_conditions={"query":query,"frequency":frequency,"start_year":start_year,"start_quarter":start_quarter,"end_year":end_year,"end_quarter":end_quarter}

    if not matches:
        try:
            with st.spinner("DART·SEC 기업 목록을 검색하고 있습니다..."):
                catalog,errors=company_catalog(dart_api_key,sec_user_agent)
            found=search_companies(catalog,normalized) if catalog else []
            if not found:
                detail=(" / ".join(errors)) if errors else "일치하는 기업을 찾지 못했습니다."
                st.error(detail); return
            st.session_state["f08_matches"]=found
            st.session_state["f08_search_query"]=normalized
            st.session_state["f08_saved"]=current_conditions
            st.rerun(); return
        except Exception as error:
            st.error(f"기업 목록을 불러오지 못했습니다: {error}"); return

    if (start_year,start_quarter)>(end_year,end_quarter):
        st.error("시작 기간을 종료 기간 이전으로 설정해 주세요."); return
    if not company:
        st.error("검색 결과에서 기업을 선택해 주세요."); return
    try:
        with st.spinner("공시 데이터를 불러오는 중입니다..."):
            frame,unit=financial_result(company,start_year,start_quarter,end_year,end_quarter,quarterly,dart_api_key,sec_user_agent)
        st.session_state["f08_saved"]=current_conditions
        st.session_state["f08_result"]={"company":company,"quarterly":quarterly,"frame":frame,"unit":unit}
        _goto("f08_results"); st.rerun()
    except DataSourceError as error:
        st.error(str(error))


def render_results() -> None:
    result=st.session_state.get("f08_result")
    if not result: _goto("f08_conditions"); st.rerun(); return
    company=result["company"]; quarterly=result["quarterly"]; frame=result["frame"]; unit=result["unit"]
    st.title("🏢 기업 재무 대시보드 결과")
    st.subheader(f"{company.name} · {company.symbol}")
    if company.source == 'DART' and not company.symbol:
        st.caption('종목코드가 없는 공시 기업입니다. DART에서 공시가 조회되더라도 이 앱이 사용하는 구조화 재무 API에는 해당 보고서가 없을 수 있습니다.')
    st.caption(f"{'분기별' if quarterly else '연도별'} · {frame.index[0]} ~ {frame.index[-1]} · {unit} · {company.source}")
    chart_section("01 재무상태","각 보고기간 말의 잔액을 비교합니다.",BALANCE,frame,unit)
    chart_section("02 손익","분기별 조회는 해당 분기 단독 실적, 연도별 조회는 연간 실적을 표시합니다.",INCOME,frame,unit)
    chart_section("03 현금흐름","기간 중 현금 흐름과 기초·기말 잔액을 비교합니다.",CASH,frame,unit)
    png=result_png(frame,company.name,"분기별" if quarterly else "연도별",unit)
    st.divider(); left,right=st.columns(2)
    with left:
        if st.button('뒤로',key="back_f08_results",use_container_width=True): _goto("f08_conditions"); st.rerun()
    with right:
        st.download_button("결과 저장",data=png,file_name=f"{company.name}_재무요약.png",mime="image/png",key="download_f08_results",use_container_width=True)
