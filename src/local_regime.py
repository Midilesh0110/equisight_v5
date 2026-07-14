import pandas as pd
import numpy as np
import joblib
import os
import logging
import warnings
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalRegimeEngine:
    def __init__(self, model_dir="database/models/local", n_components=3):
        self.n_components = n_components
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _train_single_ticker(self, ticker: str, features: pd.DataFrame):
        """
        Internal method to train and save an HMM for a single ticker.
        Returns the ticker name on success, or None if data insufficient.
        """
        X = features[['log_returns', 'range_pct']].dropna().values
        if len(X) < 100:
            logger.warning(f"{ticker}: insufficient data (rows={len(X)}). Skipping training.")
            return None

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        model.fit(X)

        means = model.means_[:, 0]
        sorted_indices = np.argsort(means)
        state_map = {
            sorted_indices[0]: -1,  # BEARISH
            sorted_indices[1]: 0,   # SIDEWAYS
            sorted_indices[2]: 1    # BULLISH
        }

        save_path = os.path.join(self.model_dir, f"{ticker}_hmm.pkl")
        joblib.dump({'model': model, 'state_map': state_map}, save_path)
        logger.info(f"Trained and saved: {ticker}")
        return ticker

    def train_all_parallel(self, ticker_data_dict: dict, n_jobs=4):
        """
        Trains all tickers simultaneously using parallel processing.
        n_jobs=4 is safe for 16GB RAM; adjust as needed.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._train_single_ticker)(ticker, df)
            for ticker, df in ticker_data_dict.items()
        )
        # Filter out None (failed tickers)
        successful = [r for r in results if r is not None]
        logger.info(f"Parallel training complete. {len(successful)}/{len(ticker_data_dict)} tickers trained.")
        return successful

    def score_ticker(self, ticker: str, features: pd.DataFrame) -> float:
        """
        Returns the log-likelihood of the features under the saved ticker model.
        Used to detect regime change and trigger retraining.
        """
        save_path = os.path.join(self.model_dir, f"{ticker}_hmm.pkl")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model for {ticker} not found.")
        data = joblib.load(save_path)
        X = features[['log_returns', 'range_pct']].dropna().values
        return data['model'].score(X)

    def predict_ticker(self, ticker: str, current_features: pd.DataFrame) -> pd.Series:
        """
        Predicts the current regime for a specific ticker using its saved model.
        Returns a Series aligned with the input index (NaN for missing data).
        """
        save_path = os.path.join(self.model_dir, f"{ticker}_hmm.pkl")
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model for {ticker} not found.")

        data = joblib.load(save_path)
        valid_mask = current_features[['log_returns', 'range_pct']].notna().all(axis=1)
        X = current_features.loc[valid_mask, ['log_returns', 'range_pct']].values
        raw_states = data['model'].predict(X)

        mapped_states = [data['state_map'][state] for state in raw_states]

        # Build a full-length Series with NaN for invalid rows
        result = pd.Series(index=current_features.index, dtype=float)
        result.loc[valid_mask] = mapped_states
        result.name = f"{ticker}_regime"
        return result

if __name__ == "__main__":
    logger.info("Testing Phase 2: Parallel Local Regime Engine...")

    dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
    dummy_dict = {
        'RELIANCE.NS': pd.DataFrame({
            'log_returns': np.random.normal(0, 0.01, 500),
            'range_pct': np.random.uniform(0.01, 0.03, 500)
        }, index=dates),
        'TCS.NS': pd.DataFrame({
            'log_returns': np.random.normal(0, 0.012, 500),
            'range_pct': np.random.uniform(0.01, 0.04, 500)
        }, index=dates),
        'HDFCBANK.NS': pd.DataFrame({
            'log_returns': np.random.normal(0, 0.015, 500),
            'range_pct': np.random.uniform(0.01, 0.02, 500)
        }, index=dates)
    }

    engine = LocalRegimeEngine()
    completed = engine.train_all_parallel(dummy_dict, n_jobs=2)  # 2 for testing
    logger.info(f"Successfully trained: {completed}")

    # Test scoring
    ll = engine.score_ticker('TCS.NS', dummy_dict['TCS.NS'].tail(100))
    logger.info(f"TCS.NS log-likelihood: {ll:.2f}")

    # Test prediction (returns full index)
    predictions = engine.predict_ticker('TCS.NS', dummy_dict['TCS.NS'].tail(10))
    logger.info(f"TCS.NS predictions:\n{predictions}")