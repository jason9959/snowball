import plotly.graph_objects as go

def price_chart(daily,trades,ticker):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=daily.index,y=daily['Close'],name=f'{ticker} 종가'))
    fig.add_trace(go.Scatter(x=daily.index,y=daily['avg_price'],name='평균단가'))
    if trades is not None and not trades.empty:
        for side,symbol in [('BUY','triangle-up'),('SELL','triangle-down')]:
            x=trades[trades.side==side]
            if not x.empty: fig.add_trace(go.Scatter(x=x.date,y=x.price,mode='markers',marker=dict(symbol=symbol,size=9),name=side))
    fig.update_layout(height=500,hovermode='x unified',title='가격 · 평균단가 · 매매'); return fig

def value_chart(daily):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=daily.index,y=daily['portfolio_value'],name='전략 자산'))
    fig.add_trace(go.Scatter(x=daily.index,y=daily['cash'],name='현금'))
    fig.update_layout(height=470,hovermode='x unified',title='자산 변화'); return fig

def t_chart(daily):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=daily.index,y=daily['T'],name='T값'))
    fig.add_hline(y=20,line_dash='dash'); fig.add_hline(y=39.1,line_dash='dot')
    fig.update_layout(height=360,title='T값 변화'); return fig
