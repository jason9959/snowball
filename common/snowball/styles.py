"""Snowball-specific NiceGUI styles kept separate from functions CSS."""

APP_CSS = r"""
    :root { --snow-bg: #f5f7fb; --snow-text: #20242d; }
    body { background: var(--snow-bg); color: var(--snow-text); }
    .app-header { background: rgba(255,255,255,.96); color: var(--snow-text);
      border-bottom: 1px solid #e7eaf0; backdrop-filter: blur(10px);
      min-height: 56px; height: auto; padding: 8px 14px !important; flex-wrap: nowrap !important; }
    .app-brand-icon { font-size: 1.45rem; line-height: 1; }
    .app-brand-title { font-size: 1.05rem; line-height: 1.2; white-space: nowrap; }
    .page-container { max-width: 1040px; margin: 0 auto; padding: 24px 20px 36px; }
    .page-title { font-size: clamp(1.55rem, 4vw, 2.25rem); font-weight: 800; line-height: 1.2; }
    .page-subtitle { color: #687080; font-size: .95rem; }
    .page-back-button { align-self: flex-start; color: #4f5b70; margin-left: -10px; }
    .feature-card, .account-card, .migration-note { border-radius: 20px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.06); padding: 22px; }
    .feature-card { min-height: 112px; padding: 18px 20px; }
    .feature-card-row, .function-card-row { min-width: 0; }
    .feature-card-copy, .function-card-copy { flex: 1 1 auto; }
    .feature-icon { font-size: 38px; color: #4059df; background: #eef1ff;
      padding: 10px; border-radius: 14px; }
    .feature-description { color: #687080; line-height: 1.55; }
    .primary-action { border-radius: 12px; min-height: 46px; }
    .feature-row-action { flex: 0 0 auto; min-width: 150px; }
    .section-heading { margin-top: 8px; font-size: clamp(1.35rem, 3vw, 1.8rem); }
    .function-card { min-height: 84px; border-radius: 16px; border: 1px solid #e8ebf2;
      box-shadow: 0 5px 18px rgba(32,47,79,.045); padding: 14px 18px; }
    .function-icon { flex: 0 0 auto; font-size: 28px; color: #5966c9; background: #f0f2ff;
      padding: 8px; border-radius: 12px; }
    .function-description { color: #737b8c; font-size: .86rem; line-height: 1.5; }
    .function-action { flex: 0 0 auto; color: #4059df; min-height: 40px; min-width: 110px; }
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
      flex: 1 1 auto; white-space: nowrap; word-break: keep-all; }
    .metric-value { color: #222735; font-size: 1.08rem; font-weight: 800;
      text-align: right; white-space: nowrap; flex: 0 0 auto; }
    .dashboard-tabs { overflow-x: auto; flex-wrap: nowrap; }
    .dashboard-tabs .q-tab { min-width: 88px; padding: 0 10px; }
    .dashboard-panels { border-radius: 18px; border: 1px solid #e8ebf2;
      box-shadow: 0 8px 26px rgba(32,47,79,.05); }
    .result-tab-content { background: white; padding: 18px; min-height: 430px; }
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
    .period-date-grid {
      display: grid !important;
      grid-template-columns: minmax(105px, .7fr) repeat(2, minmax(150px, 1fr)) !important;
      gap: 12px;
      align-items: start;
    }
    .period-date-grid > * {
      width: 100% !important;
      min-width: 0 !important;
      max-width: none !important;
    }
    .ticker-search-grid {
      display: grid !important;
      grid-template-columns: minmax(100px, .55fr) minmax(170px, 1fr) minmax(220px, 1.5fr) auto;
      gap: 10px;
    }
    .ticker-list { border: 1px solid #eceef4; border-radius: 14px; padding: 8px; }
    .ticker-list-row { min-height: 52px; padding: 5px 7px; border-bottom: 1px solid #f0f1f5; gap: 10px; }
    .ticker-list-row:last-child { border-bottom: 0; }
    .ticker-index { width: 24px; color: #8991a2; text-align: center; flex: 0 0 auto; }
    .ticker-list-label { min-width: 0; flex: 1 1 auto; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; }
    .ticker-weight { width: 120px !important; flex: 0 0 120px; }
    .ticker-delete { flex: 0 0 auto; }
    .ticker-empty { width: 100%; padding: 14px; text-align: center; color: #8991a2; }
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
    .result-metric-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)) !important; }
    .monthly-heatmap { height: 510px; }
    .monthly-distribution { height: 340px; }
    .split-heatmap-scroll { overscroll-behavior-x: contain;
      -webkit-overflow-scrolling: touch; padding-bottom: 6px; }
    .rebalance-summary { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; }
    @media (max-width: 700px) {
      .feature-grid { grid-template-columns: 1fr !important; }
      .functions-grid { grid-template-columns: 1fr !important; }
      .page-container { padding: 18px 14px 28px; }
      .feature-card { min-height: 0; padding: 15px; }
      .function-card { min-height: 0; padding: 13px 14px; }
      .feature-row-action, .function-action { min-width: 44px; width: 44px; font-size: 0; padding: 0; }
      .feature-icon { font-size: 30px; padding: 8px; }
      .function-icon { font-size: 24px; padding: 7px; }
      .app-header .nav-button { font-size: 0; }
      .app-header { min-height: 52px; padding: 7px 10px !important; }
      .app-brand-title { font-size: 1rem; }
      .app-brand-icon { font-size: 1.3rem; }
      .metric-grid { grid-template-columns: 1fr !important; }
      .metric-card { min-height: 60px; padding: 12px 14px; }
      .metric-label { font-size: .88rem; }
      .metric-value { font-size: 1rem; }
      .date-input { min-width: calc(50% - 8px); flex: 1; }
      .strategy-expansion .q-expansion-item__content { padding: 6px 10px 14px; }
      .editor-section { padding: 12px; }
      .dashboard-chart { height: 360px; }
      .result-tab-content { padding: 10px; min-height: 360px; }
      .currency-metric-grid { grid-template-columns: 1fr !important; }
      .result-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .monthly-heatmap { height: 480px; }
      .monthly-distribution { height: 330px; }
      .rebalance-summary { grid-template-columns: 1fr !important; }
      .editor-fields, .backtest-field-grid {
        grid-template-columns: 1fr !important;
        gap: 10px;
      }
      .period-date-grid {
        grid-template-columns: minmax(82px, .65fr) repeat(2, minmax(0, 1fr)) !important;
        gap: 7px !important;
      }
      .period-date-grid .q-field__label { font-size: .74rem; }
      .period-date-grid .q-field__native,
      .period-date-grid .q-field__input { font-size: .82rem; padding-left: 0; padding-right: 0; }
      .ticker-search-grid { grid-template-columns: minmax(90px, .6fr) minmax(0, 1.4fr) auto !important; }
      .ticker-candidate { grid-column: 1 / -1; grid-row: 2; }
      .ticker-add { grid-column: 3; grid-row: 1; min-width: 78px; }
      .ticker-list-label { font-size: .86rem; }
      .ticker-weight { width: 92px !important; flex-basis: 92px; }
      .backtest-date-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      .editor-fields .control-grow, .backtest-field-grid .control-grow {
        min-width: 0 !important;
        width: 100% !important;
      }
    }
"""

