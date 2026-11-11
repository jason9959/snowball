from io import BytesIO
import matplotlib.pyplot as plt

def create_png(ticker,daily,metrics):
    fig,axes=plt.subplots(2,1,figsize=(12,10),constrained_layout=True)
    axes[0].plot(daily.index,daily['Close'],label=ticker); axes[0].plot(daily.index,daily['avg_price'],label='Average cost'); axes[0].grid(alpha=.2); axes[0].legend()
    axes[1].plot(daily.index,daily['portfolio_value'],label='Strategy value'); axes[1].grid(alpha=.2); axes[1].legend()
    fig.suptitle(f"Laoer V2.2 Backtest · Final {metrics['final_value']:,.0f} · Return {metrics['return_pct']:.2f}% · MDD {metrics['mdd_pct']:.2f}%")
    out=BytesIO(); fig.savefig(out,format='png',dpi=160,bbox_inches='tight'); plt.close(fig); return out.getvalue()
