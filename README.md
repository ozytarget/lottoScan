# 🎰 Lottery Scanner

Statistical lottery analyzer for Mega Millions & Powerball. Analyzes historical draw data and suggests combinations based on frequency patterns and dispersion analysis.

## Features

- 📊 **Statistical Analysis** - Analyzes historical lottery data
- 🎯 **Smart Combinations** - Generates suggestions based on:
  - Number frequency patterns
  - Dispersion analysis (low/mid/high ranges)
  - Even/odd mix optimization
- 📈 **Visualizations** - Casino-themed interface with ball displays
- 🎲 **Least Frequent Numbers** - Shows numbers that haven't appeared recently
- 📅 **Draw History** - Reference table of all analyzed draws

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run lottoScan.py
```

Then open `http://localhost:8501` in your browser.

## How It Works

1. **Select Game** - Choose between Mega Millions or Powerball
2. **Set Parameters** - Adjust analysis window (days) and number of combinations
3. **Run Scanner** - Click the button to analyze historical data
4. **Review Results**:
   - Statistics card showing draw count and date range
   - Suggested combinations with ball visualization
   - Draw history for reference
   - Least frequent numbers that might be "due"

## Disclaimer

⚠️ **This is a statistical tool, NOT a prediction system**. Each lottery combination has equal mathematical probability regardless of historical patterns. This tool is for entertainment and analysis purposes only.

## Data Source

Historical lottery data sourced from Texas Lottery official records covering national Mega Millions and Powerball draws.

## License

MIT License
