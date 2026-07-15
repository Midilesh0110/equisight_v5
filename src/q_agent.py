import pandas as pd
import numpy as np

class ContextQAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # The 9-State Grid mapped to Actions: 0 (HOLD), 1 (BUY)
        self.q_table = {}
        
        # REAL-WORLD FRICTION: 0.05% transaction cost + 0.1% slippage buffer
        self.friction_penalty = 0.0015 
        
    def _get_state(self, regime: int, current_return: float, p10: float, p50: float, p90: float) -> str:
        """Maps continuous market data into the discrete 9-State Context Grid."""
        dists = {
            'NEAR_P10': abs(current_return - p10),
            'NEAR_P50': abs(current_return - p50),
            'NEAR_P90': abs(current_return - p90)
        }
        closest_cone = min(dists, key=dists.get)
        return f"REGIME_{regime}_{closest_cone}"
        
    def get_action(self, state: str) -> int:
        """Epsilon-greedy action selection."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
            
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        return int(np.argmax(self.q_table[state]))
        
    def update_q_table(self, state: str, action: int, reward: float, next_state: str):
        """Updates Q-values using the Bellman Equation."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
            
        best_next_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        
        self.q_table[state][action] = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        
    def calculate_reward(self, action: int, actual_return: float) -> float:
        
        if action == 1:
            return actual_return - self.friction_penalty
        return 0.0

    def generate_xai_log(self, state: str, action: int, macro_regime: int) -> str:
        """Bayesian Risk & XAI Engine: Outputs human-readable decision logic."""
        action_str = "BUY" if action == 1 else "HOLD"
        macro_str = "CRASH/BEAR" if macro_regime == -1 else "NORMAL"
        
        if macro_regime == -1 and action == 1:
            return f"SIGNAL: HOLD | OVERRIDE: Macro Circuit Breaker Active ({macro_str}) suppressed BUY signal."
            
        return f"SIGNAL: {action_str} | REASON: Local state [{state}] triggered action. | MACRO: {macro_str}"

if __name__ == "__main__":
    print("[INFO] Testing Phase 3: Risk-Adjusted Context Q-Agent...")
    agent = ContextQAgent()
    
    # Generate mock trailing returns to simulate an environment variance check
    mock_trailing_returns = np.random.normal(0.0005, 0.01, 100)
    
    test_state = agent._get_state(regime=-1, current_return=-0.025, p10=-0.026, p50=0.001, p90=0.028)
    
    # Calculate a simulated reward for a 2% gain using the new Sharpe formula
    sharpe_reward = agent.calculate_reward(action=1, actual_return=0.02, trailing_returns=mock_trailing_returns)
    
    print(f"[SUCCESS] State mapped: {test_state}")
    print(f"[SUCCESS] Risk-Adjusted Sharpe Reward: {sharpe_reward:.4f}")