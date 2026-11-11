"""Snowball-specific NiceGUI styles kept separate from functions CSS."""

APP_CSS = r"""
    :root { --snow-bg: #f5f7fb; --snow-text: #20242d; }
    body { background: var(--snow-bg); color: var(--snow-text); }
    .app-header { background: rgba(255,255,255,.96); color: var(--snow-text);
      border-bottom: 1px solid #e7eaf0; backdrop-filter: blur(10px); }
    .page-container { max-width: 1040px; margin: 0 auto; padding: 92px 20px 36px; }
    .page-title { font-size: clamp(1.55rem, 4vw, 2.25rem); font-weight: 800; line-height: 1.2; }
    .page-subtitle { color: #687080; font-size: .95rem; }
    .page-back-button { align-self: flex-start; color: #4f5b70; margin-left: -10px; }
    .feature-card, .account-card, .migration-note { border-radius: 20px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.06); padding: 22px; }
    .feature-card { min-height: 260px; justify-content: space-between; }
    .feature-icon { font-size: 38px; color: #4059df; background: #eef1ff;
      padding: 10px; border-radius: 14px; }
    .feature-description { color: #687080; line-height: 1.65; min-height: 52px; }
    .primary-action { border-radius: 12px; min-height: 46px; }
    .migration-note { background: linear-gradient(135deg, #f6f8ff, #fff); }
    .status-text { color: #687080; font-size: .85rem; }
    .error-card { border: 1px solid #ffcdd2; background: #fff7f8; border-radius: 16px; }
    .warning-card { border: 1px solid #ffe0a3; background: #fffaf0; border-radius: 16px; }
    .empty-state { width: 100%; text-align: center; padding: 52px 12px; color: #687080; }
    .nav-active { color: #4059df; background: #eef1ff; }
    .date-input { min-width: 180px; }
    .section-title { font-size: 1.2rem; font-weight: 800; margin-top: 8px; }
    .metric-grid { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
    .metric-card { border-radius: 16px; border: 1px solid #e8ebf2; box-shadow: none;
      padding: 14px 16px; min-height: 72px; justify-content: center; }
    .metric-row { gap: 12px; }
    .metric-label { color: #737b8c; font-size: .86rem; line-height: 1.35;
      flex: 1 1 auto; }
    .metric-value { color: #222735; font-size: 1.08rem; font-weight: 800;
      text-align: right; white-space: nowrap; flex: 0 0 auto; }
    .dashboard-tabs { overflow-x: auto; flex-wrap: nowrap; }
    .dashboard-tabs .q-tab { min-width: 88px; padding: 0 10px; }
    .dashboard-panels { border-radius: 18px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.05); }
    .dashboard-chart { height: 430px; }
    .currency-asset-card { border-radius: 18px; border: 1px solid #e8ebf2;
      box-shadow: none; padding: 18px; gap: 12px; margin-bottom: 16px; }
    .currency-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    .account-table { max-width: 100%; overflow-x: auto; }
    .backtest-controls { border-radius: 20px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.06); padding: 22px; gap: 16px; }
    .strategy-expansion, .preset-library { border-radius: 20px; border: 1px solid #e8ebf2;
      background: white; box-shadow: 0 8px 26px rgba(32,47,79,.05); }
    .strategy-expansion .q-expansion-item__content { padding: 8px 18px 20px; }
    .editor-section { border-radius: 16px; border: 1px solid #eceef4;
      box-shadow: none; padding: 16px; gap: 12px; }
    .editor-fields, .backtest-field-grid {
      display: grid !important;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .editor-fields > *, .backtest-field-grid > * {
      width: 100% !important;
      min-width: 0 !important;
      max-width: none !important;
      flex: none !important;
    }
    .editor-section .q-field, .backtest-controls .q-field { width: 100%; }
    .preset-library { padding: 18px; gap: 10px; }
    .preset-upload { width: min(420px, 100%); }
    .diagnostic-download { align-self: flex-start; }
    .decision-select { width: min(360px, 100%); }
    .snapshot-code { max-height: 420px; overflow: auto; border-radius: 12px; }
    .control-grow { min-width: 180px; flex: 1; }
    .result-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    .rebalance-summary { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    @media (max-width: 700px) {
      .feature-grid { grid-template-columns: 1fr !important; }
      .page-container { padding: 82px 14px 28px; }
      .feature-card { min-height: 230px; }
      .app-header .nav-button { font-size: 0; }
      .metric-grid { grid-template-columns: 1fr !important; }
      .metric-card { min-height: 60px; padding: 12px 14px; }
      .metric-label { font-size: .88rem; }
      .metric-value { font-size: 1rem; }
      .date-input { min-width: calc(50% - 8px); flex: 1; }
      .strategy-expansion .q-expansion-item__content { padding: 6px 10px 14px; }
      .editor-section { padding: 12px; }
      .dashboard-chart { height: 360px; }
      .currency-metric-grid { grid-template-columns: 1fr !important; }
      .result-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .rebalance-summary { grid-template-columns: 1fr !important; }
      .editor-fields, .backtest-field-grid {
        grid-template-columns: 1fr !important;
        gap: 10px;
      }
      .backtest-date-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .editor-fields .control-grow, .backtest-field-grid .control-grow {
        min-width: 0 !important;
        width: 100% !important;
      }
    }
"""

