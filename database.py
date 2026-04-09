# ═══════════════════════════════════════════════════════════════
#  database.py — SQLite setup + helpers
# ═══════════════════════════════════════════════════════════════

import sqlite3, json, os

DB_PATH = os.getenv("DB_PATH", "resumeai.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_snippet TEXT,
                target_role    TEXT,
                score          INTEGER,
                result_json    TEXT,
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    print("✅  SQLite database ready →", DB_PATH)

def save_analysis(snippet, role, score, result_json):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analyses (resume_snippet, target_role, score, result_json) VALUES (?,?,?,?)",
            (snippet, role, score, result_json)
        )
        conn.commit()

def get_all_analyses():
    with get_conn() as conn:
        rows = conn.execute("SELECT score, target_role, created_at FROM analyses ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def get_stats_summary():
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as total, AVG(score) as avg FROM analyses").fetchone()
    return {"total": row["total"], "avgScore": round(row["avg"] or 0, 1)}
