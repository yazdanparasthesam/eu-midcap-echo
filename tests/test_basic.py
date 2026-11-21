# Minimal smoke tests
import numpy as np, pandas as pd

def test_metrics():
    from src.trading.metrics import sharpe, max_drawdown
    eq = pd.Series(np.linspace(100,150,252))
    assert max_drawdown(eq) == 0.0
    s = sharpe(eq.pct_change().fillna(0))
    assert s > 0

def test_sentiment():
    from src.features.sentiment_features import _lexicon_score
    assert _lexicon_score("Das Unternehmen steigt den Gewinn und erhöht die Prognose deutlich") > 0
    assert _lexicon_score("Gewinnwarnung und Prognose gesenkt, Verlust erwartet") < 0

def test_universe_yaml_loads():
    from src.config import load_universe
    u = load_universe()
    assert len(u["universe"]) >= 30
    assert "ADS.DE" in u["universe"]
