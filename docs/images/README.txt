Energy Trading & Portfolio Optimization System
Visual Documentation Assets

================================================================================
DIRECTORY STRUCTURE
================================================================================

docs/images/
├── dashboard/          # Streamlit dashboard screenshots
├── notebooks/          # Jupyter notebook visualizations
├── architecture/       # System architecture diagrams
└── results/            # Performance charts and comparison tables

================================================================================
IMAGE NAMING CONVENTIONS
================================================================================

Dashboard Screenshots:
- Format: {page_number}_{page_name}.png
- Example: 1_market_overview.png, 2_price_forecasts.png
- Size: 1920x1080 (Full HD) or 1280x720 (HD)
- Format: PNG (lossless compression)

Notebook Visualizations:
- Format: {notebook_name}_{chart_description}.png
- Example: data_exploration_correlation.png, price_forecast_comparison.png
- Size: 800x600 or 1200x800 (depending on chart complexity)
- Format: PNG for raster, SVG for vector graphics

Architecture Diagrams:
- Format: {diagram_type}.png or .svg
- Example: system_architecture.png, data_flow.svg
- Size: Variable (ensure readability)
- Format: SVG preferred (scalable), PNG as fallback

Results Charts:
- Format: {result_type}_{description}.png
- Example: backtest_summary.png, optimization_comparison.png
- Size: 1000x700 (standard chart size)
- Format: PNG

================================================================================
HOW TO GENERATE IMAGES
================================================================================

Dashboard Screenshots:
1. Run the dashboard: streamlit run src/dashboard/app.py
2. Navigate to each page (Market Overview, Price Forecasts, etc.)
3. Use browser screenshot tool (F12 → Ctrl+Shift+P → "Capture screenshot")
4. Save to docs/images/dashboard/ with appropriate name
5. Crop to remove browser chrome if needed

Notebook Visualizations:
1. Open Jupyter notebook: jupyter notebook notebooks/
2. Execute all cells in order
3. For Plotly figures: fig.write_image("docs/images/notebooks/chart_name.png")
4. For Matplotlib figures: plt.savefig("docs/images/notebooks/chart_name.png", dpi=150, bbox_inches='tight')
5. Ensure high resolution (dpi=150 or higher)

Architecture Diagrams:
1. Mermaid diagrams in README.md can be exported:
   - Use Mermaid Live Editor: https://mermaid.live/
   - Copy diagram code from README.md
   - Export as PNG or SVG
2. Save to docs/images/architecture/

Results Charts:
1. Generate from backtest/optimization results
2. Use reporting modules (BacktestReport, RiskReport)
3. Save figures: report.save_figures(output_dir="docs/images/results/")

================================================================================
IMAGE REQUIREMENTS
================================================================================

Quality:
- Minimum resolution: 800x600
- Recommended: 1200x800 or higher
- DPI: 150 for print quality, 96 for web
- Compression: PNG with moderate compression (balance size vs quality)

Content:
- Clear, readable text (minimum 10pt font)
- High contrast (dark text on light background or vice versa)
- Consistent color scheme (use COLORS from reporting.py)
- Include legends, axis labels, titles
- Remove unnecessary whitespace

Accessibility:
- Use colorblind-friendly palettes
- Include alt text in documentation
- Ensure sufficient contrast ratios

================================================================================
REFERENCING IMAGES IN DOCUMENTATION
================================================================================

Markdown Syntax:
![Alt Text](docs/images/category/image_name.png)

Example:
![Market Overview Dashboard](docs/images/dashboard/1_market_overview.png)

HTML Syntax (for more control):
<img src="docs/images/category/image_name.png" alt="Alt Text" width="800">

Relative Paths:
- From README.md: docs/images/dashboard/1_market_overview.png
- From docs/API.md: images/dashboard/1_market_overview.png
- From notebooks/: ../docs/images/notebooks/chart_name.png

================================================================================
IMAGE CHECKLIST
================================================================================

Before adding an image:
[ ] Image is relevant and adds value to documentation
[ ] Image is properly sized (not too large, not too small)
[ ] Image is optimized (compressed without quality loss)
[ ] Image has descriptive filename
[ ] Image is referenced in documentation with alt text
[ ] Image is stored in correct subdirectory

================================================================================
MAINTENANCE
================================================================================

When to update images:
- Dashboard UI changes
- New features added
- Performance improvements (update result charts)
- Bug fixes affecting visualizations

Version control:
- Commit images with descriptive messages
- Use Git LFS for large images (>1MB)
- Keep old versions if significant changes

================================================================================
PLACEHOLDER FILES
================================================================================

Dashboard Screenshots (to be generated):
- 1_market_overview.png
- 2_price_forecasts.png
- 3_trading_strategies.png
- 4_portfolio_optimization.png
- 5_risk_analytics.png

Notebook Visualizations (to be generated):
- data_exploration_correlation.png
- price_forecast_comparison.png
- strategy_equity_curves.png
- efficient_frontier.png
- renewable_complementarity.png
- curtailment_analysis.png

Architecture Diagrams (to be generated):
- system_architecture.png (export from README.md Mermaid diagram)
- data_flow.png
- module_dependencies.png

Results Charts (to be generated):
- backtest_summary.png
- optimization_comparison.png
- risk_decomposition.png
- stress_test_results.png

================================================================================
CONTACT
================================================================================

For questions about visual documentation:
- See main README.md for project contact information
- Refer to docs/API.md for technical details
- Check docs/ALGORITHMS.md for algorithm explanations

================================================================================
GENERATION INSTRUCTIONS
================================================================================

To populate this directory with actual images:

1. **Run Dashboard**:
   ```bash
   streamlit run src/dashboard/app.py
   ```
   Then take screenshots of each page.

2. **Execute Notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```
   Run all cells and save key visualizations.

3. **Export Mermaid Diagram**:
   - Copy the architecture diagram from README.md
   - Paste into https://mermaid.live/
   - Export as PNG/SVG
   - Save to docs/images/architecture/system_architecture.png

4. **Generate Performance Charts**:
   ```python
   from src.backtesting.reporting import BacktestReport
   from src.optimization.risk_reporting import RiskReport

   # After running backtests
   report = BacktestReport(result)
   figures = report.generate_full_report()
   for name, fig in figures.items():
       fig.write_image(f"docs/images/results/{name}.png")
   ```

================================================================================
