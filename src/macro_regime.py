import pandas as pd
import numpy as np
import joblib
import os
import logging
import warnings
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroCircuitBreaker:
    def __init__(self, model_dir="database/models", n_components=3):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        self.state_map = {}
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "macro_nifty_hmm.pkl")

    def train(self, macro_features: pd.DataFrame):
        """
        Trains the macro HMM and maps regimes autonomously based on return means.
        Saves model to disk.
        """
        X = macro_features[['log_returns', 'range_pct']].dropna().values
        if len(X) < 100:
            raise ValueError("Insufficient data to train Macro HMM (need at least 100 rows).")
        self.model.fit(X)

        # Sort hidden states by their mean log return
        means = self.model.means_[:, 0]
        sorted_indices = np.argsort(means)

        self.state_map = {
            sorted_indices[0]: -1,  # Lowest Mean -> BEARISH / CRASH
            sorted_indices[1]: 0,   # Median Mean -> SIDEWAYS
            sorted_indices[2]: 1    # Highest Mean -> BULLISH
        }

        joblib.dump({'model': self.model, 'state_map': self.state_map}, self.model_path)
        logger.info(f"Macro HMM trained and saved to {self.model_path}")

    def score(self, features: pd.DataFrame) -> float:
        """
        Returns the log-likelihood of the features under the saved model.
        Used to detect regime drift and trigger retraining.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Macro model not found. Run train() first.")
        data = joblib.load(self.model_path)
        X = features[['log_returns', 'range_pct']].dropna().values
        return data['model'].score(X)

    def predict(self, current_features: pd.DataFrame) -> pd.Series:
        """
        Loads the saved model and predicts the current macro regime.
        Returns a Series with the same index as the non-NaN rows of the input.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Macro model not found. Run train() first.")

        data = joblib.load(self.model_path)
        # Keep only rows with valid features
        valid_mask = current_features[['log_returns', 'range_pct']].notna().all(axis=1)
        X = current_features.loc[valid_mask, ['log_returns', 'range_pct']].values
        raw_states = data['model'].predict(X)

        # Map to -1,0,1
        mapped_states = [data['state_map'][state] for state in raw_states]
        return pd.Series(mapped_states, index=current_features.index[valid_mask], name="macro_regime")

if __name__ == "__main__":
    logger.info("Testing Phase 2: Macro Circuit Breaker...")

    # Create Dummy Nifty 50 Features
    dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
    dummy_macro = pd.DataFrame({
        'log_returns': np.random.normal(0.0005, 0.01, 500),
        'range_pct': np.random.uniform(0.01, 0.03, 500)
    }, index=dates)

    breaker = MacroCircuitBreaker()
    breaker.train(dummy_macro)
    logger.info(f"Macro model trained and saved to {breaker.model_path}")

    # Test scoring
    ll = breaker.score(dummy_macro.tail(100))
    logger.info(f"Log-likelihood on recent data: {ll:.2f}")

    # Test predictions
    predictions = breaker.predict(dummy_macro.tail(10))
    logger.info(f"Macro predictions for last 10 days. States: {predictions.unique()}")