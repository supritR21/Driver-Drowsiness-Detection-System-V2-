from app.db.session import engine

try:
    conn = engine.connect()
    print("✅ DB Connected Successfully")
    conn.close()
except Exception as e:
    print("❌ DB Error:", e)