import sqlite3
conn = sqlite3.connect('cfpilot.db')
cur = conn.cursor()
tables = cur.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
with open('schema.txt', 'w') as f:
    for t in tables:
        if t[0]:
            f.write(t[0] + '\n\n')
