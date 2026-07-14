import os
import sys
from flask import Flask, render_template, jsonify
import pandas as pd

# Ensure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.execution_db import ExecutionDatabase

app = Flask(__name__)

# Instantiate the database connection (WAL mode, same DB file)
# Create an absolute anchor to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "equisight_v5.db")

# Instantiate the database connection cleanly
db = ExecutionDatabase(db_path=DB_PATH)

@app.route('/')
def index():
    """Serve the main dashboard HTML page."""
    return render_template('index.html')

@app.route('/api/alpha_decay')
def alpha_decay():
    """Return the current Alpha Decay metric as JSON."""
    avg_days = db.get_alpha_decay_metrics()
    return jsonify({"avg_days_to_profit": avg_days})

@app.route('/api/recent_trades')
def recent_trades():
    """Return the last 20 trade outcomes for dashboard display."""
    conn = db._get_connection()
    query = """
        SELECT ticker, entry_date, exit_date, actual_profit, days_held, reason_for_exit
        FROM trade_outcomes
        ORDER BY exit_date DESC
        LIMIT 20
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/active_positions')
def active_positions():
    """Return current active positions (5-day lifecycle)."""
    conn = db._get_connection()
    query = "SELECT * FROM active_positions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)