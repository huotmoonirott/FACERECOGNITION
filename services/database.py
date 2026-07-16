"""
database.py — SQLite layer for the Face Recognition System

v2 — generalized from "Smart Attendance" to a standalone face recognition
platform:
  - `attendance` table replaced by `recognition_events`: a generic log of
    every time a known (or unknown) face is recognised, with confidence,
    gender, and a liveness score attached. Unlike attendance, a person can
    have MANY events per day — this is a recognition/security log, not a
    once-daily check-in.
  - `students` table generalized to `people` (same shape, plus khmer_name)
    — still works fine for a classroom, but the language no longer assumes
    "student".
  - New `face_photos` table: supports MULTIPLE enrollment photos per
    person, each with its own stored embedding. encoder.py averages all of
    a person's embeddings into one robust identity vector at load time.

Legacy compatibility: if an older `students` / `attendance` schema exists
on disk, it's migrated into the new tables (people / recognition_events)
on first startup. The old function-name aliases themselves have since been
removed — see the bottom of this file for the current names to use.
"""

import sqlite3
import os
import logging
import pickle

log = logging.getLogger("FaceRecog.DB")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file now lives in services/)
DB_PATH = os.path.join(BASE_DIR, "face_recognition.db")


def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL UNIQUE,
                khmer_name    TEXT DEFAULT '',
                person_id     TEXT DEFAULT 'N/A',
                photo_path    TEXT,
                registered_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS face_photos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                photo_path  TEXT NOT NULL,
                embedding   BLOB,
                added_at    TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (person_name) REFERENCES people(name)
                    ON DELETE CASCADE ON UPDATE CASCADE
            );

            CREATE TABLE IF NOT EXISTS recognition_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                confidence  REAL DEFAULT 0,
                gender      TEXT,
                liveness    REAL,
                event_type  TEXT DEFAULT 'recognition'
            );

            CREATE INDEX IF NOT EXISTS idx_events_name_date
                ON recognition_events(name, date);
        """)
        _migrate_legacy_tables(con)
    log.info("Database initialized")


def _migrate_legacy_tables(con):
    """
    One-time migration from the old Smart Attendance schema
    (students / attendance) into the new schema (people /
    recognition_events), if the old tables exist and the new ones are
    still empty. Safe to call every startup.
    """
    tables = {row[0] for row in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if "students" in tables:
        people_count = con.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        if people_count == 0:
            try:
                cols = [r[1] for r in con.execute("PRAGMA table_info(students)").fetchall()]
                has_khmer = "khmer_name" in cols
                select_cols = "name, student_id, photo_path, registered_at"
                if has_khmer:
                    select_cols += ", khmer_name"
                rows = con.execute(f"SELECT {select_cols} FROM students").fetchall()
                for row in rows:
                    name, pid, photo, reg_at = row[0], row[1], row[2], row[3]
                    khmer = row[4] if has_khmer and len(row) > 4 else ""
                    con.execute(
                        "INSERT OR IGNORE INTO people "
                        "(name, khmer_name, person_id, photo_path, registered_at) "
                        "VALUES (?,?,?,?,?)",
                        (name, khmer or "", pid, photo, reg_at)
                    )
                    if photo:
                        con.execute(
                            "INSERT OR IGNORE INTO face_photos (person_name, photo_path) "
                            "VALUES (?,?)", (name, photo)
                        )
                log.info(f"Migrated {len(rows)} legacy student records into people/face_photos")
            except Exception as e:
                log.warning(f"Legacy students migration skipped: {e}")

    if "attendance" in tables:
        events_count = con.execute(
            "SELECT COUNT(*) FROM recognition_events").fetchone()[0]
        if events_count == 0:
            try:
                rows = con.execute("SELECT name, date, time FROM attendance").fetchall()
                for name, d, t in rows:
                    con.execute(
                        "INSERT INTO recognition_events (name, date, time, event_type) "
                        "VALUES (?,?,?, 'attendance')",
                        (name, d, t)
                    )
                log.info(f"Migrated {len(rows)} legacy attendance rows into recognition_events")
            except Exception as e:
                log.warning(f"Legacy attendance migration skipped: {e}")


# ── People (enrolled identities) ────────────────────────────────────────────

def register_person(name: str, person_id: str, photo_path: str, khmer_name: str = ""):
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO people (name, person_id, photo_path, khmer_name) "
            "VALUES (?,?,?,?)",
            (name, person_id, photo_path, khmer_name)
        )
        if photo_path:
            con.execute(
                "INSERT OR IGNORE INTO face_photos (person_name, photo_path) VALUES (?,?)",
                (name, photo_path)
            )
    log.info(f"Registered person: {name} (ID: {person_id})")


def add_face_photo(name: str, photo_path: str, embedding=None):
    """Add an additional enrollment photo for an existing person (multi-photo support)."""
    blob = pickle.dumps(embedding) if embedding is not None else None
    with _conn() as con:
        con.execute(
            "INSERT INTO face_photos (person_name, photo_path, embedding) VALUES (?,?,?)",
            (name, photo_path, blob)
        )
    log.info(f"Added enrollment photo for {name}: {photo_path}")


def set_photo_embedding(photo_path: str, embedding):
    with _conn() as con:
        con.execute(
            "UPDATE face_photos SET embedding=? WHERE photo_path=?",
            (pickle.dumps(embedding), photo_path)
        )


def get_face_photos(name: str):
    """Return [(photo_path, embedding_or_None), ...] for a person."""
    with _conn() as con:
        rows = con.execute(
            "SELECT photo_path, embedding FROM face_photos WHERE person_name=? ORDER BY id",
            (name,)
        ).fetchall()
    out = []
    for path, blob in rows:
        emb = pickle.loads(blob) if blob else None
        out.append((path, emb))
    return out


def get_all_face_photos():
    """Return {name: [(photo_path, embedding_or_None), ...]} for every enrolled person."""
    with _conn() as con:
        rows = con.execute(
            "SELECT person_name, photo_path, embedding FROM face_photos ORDER BY person_name, id"
        ).fetchall()
    result = {}
    for name, path, blob in rows:
        emb = pickle.loads(blob) if blob else None
        result.setdefault(name, []).append((path, emb))
    return result


def remove_face_photo(photo_path: str):
    with _conn() as con:
        con.execute("DELETE FROM face_photos WHERE photo_path=?", (photo_path,))
    log.info(f"Removed enrollment photo: {photo_path}")


def update_person(old_name: str, new_name: str, person_id: str, khmer_name: str,
                   photo_path: str = None):
    """
    Update an existing person's record. If new_name differs from old_name,
    recognition history and face_photos rows are re-keyed to new_name so
    logs/stats stay consistent. photo_path is only updated if a new path is
    supplied (caller is responsible for renaming the file on disk).
    """
    with _conn() as con:
        if photo_path is not None:
            con.execute(
                "UPDATE people SET name=?, person_id=?, khmer_name=?, photo_path=? "
                "WHERE name=?",
                (new_name, person_id, khmer_name, photo_path, old_name)
            )
        else:
            con.execute(
                "UPDATE people SET name=?, person_id=?, khmer_name=? WHERE name=?",
                (new_name, person_id, khmer_name, old_name)
            )
        if new_name != old_name:
            con.execute("UPDATE face_photos SET person_name=? WHERE person_name=?",
                        (new_name, old_name))
            con.execute("UPDATE recognition_events SET name=? WHERE name=?",
                        (new_name, old_name))
    log.info(f"Updated person: {old_name} -> {new_name} (ID: {person_id})")


def delete_person(name: str):
    """Delete a person, their enrollment photos, and their recognition history."""
    with _conn() as con:
        con.execute("DELETE FROM people WHERE name=?", (name,))
        con.execute("DELETE FROM face_photos WHERE person_name=?", (name,))
        con.execute("DELETE FROM recognition_events WHERE name=?", (name,))
    log.info(f"Deleted person and all records: {name}")


def get_all_people():
    with _conn() as con:
        people = con.execute(
            "SELECT name, person_id, registered_at, khmer_name FROM people ORDER BY name"
        ).fetchall()
        photo_counts = dict(con.execute(
            "SELECT person_name, COUNT(*) FROM face_photos GROUP BY person_name"
        ).fetchall())
    return [(name, pid, reg, khmer, photo_counts.get(name, 0))
            for name, pid, reg, khmer in people]


def get_person(name: str):
    """Return (name, person_id, khmer_name, photo_path, registered_at) for
    ONE person, fetched fresh from the DB. Use this whenever you need a
    person's full record (e.g. opening the Edit dialog) instead of trusting
    a GUI tree row's column order — that's what was causing khmer_name to
    silently go missing/scrambled on edit."""
    with _conn() as con:
        row = con.execute(
            "SELECT name, person_id, khmer_name, photo_path, registered_at "
            "FROM people WHERE name=?", (name,)
        ).fetchone()
    return row


def get_person_khmer_name(name: str) -> str:
    with _conn() as con:
        row = con.execute(
            "SELECT khmer_name FROM people WHERE name=?", (name,)
        ).fetchone()
    return row[0] if row and row[0] else ""


# ── Recognition events (generic log; replaces attendance) ──────────────────

def log_recognition(name: str, date: str, time: str, confidence: float = 0.0,
                     gender=None, liveness=None,
                     event_type: str = "recognition"):
    """Log a single recognition event. Unlike the old attendance table this
    allows MULTIPLE events per person per day — useful for access-control /
    security-style logging rather than once-daily attendance."""
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO recognition_events "
                "(name, date, time, confidence, gender, liveness, event_type) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, date, time, confidence, gender, liveness, event_type)
            )
    except Exception as e:
        log.error(f"log_recognition error: {e}")


def get_events(date_filter: str = None, name_filter: str = None, limit: int = 1000):
    with _conn() as con:
        query = ("SELECT id, name, date, time, confidence, gender, "
                  "liveness, event_type FROM recognition_events WHERE 1=1")
        params = []
        if date_filter:
            query += " AND date=?"
            params.append(date_filter)
        if name_filter:
            query += " AND name LIKE ?"
            params.append(f"%{name_filter}%")
        query += " ORDER BY date DESC, time DESC LIMIT ?"
        params.append(limit)
        rows = con.execute(query, params).fetchall()
    return rows


def get_weekly_summary():
    with _conn() as con:
        rows = con.execute("""
            SELECT date, COUNT(*) as cnt
            FROM recognition_events
            WHERE date >= date('now','-6 days')
            GROUP BY date
            ORDER BY date ASC
        """).fetchall()
    return rows


def get_person_event_summary():
    with _conn() as con:
        return con.execute("""
            SELECT name, COUNT(*) as total_events,
                   MIN(date) as first_seen, MAX(date) as last_seen,
                   AVG(confidence) as avg_confidence
            FROM recognition_events
            GROUP BY name
            ORDER BY total_events DESC
        """).fetchall()


def get_recent_events(limit: int = 25):
    """Most recent recognition events overall (for a live activity feed)."""
    with _conn() as con:
        return con.execute(
            "SELECT name, date, time, confidence, gender "
            "FROM recognition_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def get_total_stats():
    with _conn() as con:
        total_events = con.execute(
            "SELECT COUNT(*) FROM recognition_events").fetchone()[0]
        total_people = con.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        unique_dates = con.execute(
            "SELECT COUNT(DISTINCT date) FROM recognition_events").fetchone()[0]
        today_count = con.execute(
            "SELECT COUNT(*) FROM recognition_events WHERE date=date('now','localtime')"
        ).fetchone()[0]
        unknown_count = con.execute(
            "SELECT COUNT(*) FROM recognition_events WHERE name='Unknown'"
        ).fetchone()[0]
    return {
        "total_events": total_events,
        "total_people": total_people,
        "unique_dates": unique_dates,
        "today_count": today_count,
        "unknown_count": unknown_count,
    }


# ── Backwards-compatible aliases ─────────────────────────────────────────────
# Note: register_student/update_student/delete_student/get_all_students/
# get_student_khmer_name/get_attendance/get_student_attendance_summary/
# log_attendance have been removed — nothing in this codebase (gui.py,
# main.py) calls them anymore. If an external script still depends on the
# old names, import the new functions directly (register_person,
# update_person, delete_person, get_all_people, get_person_khmer_name,
# get_events, get_person_event_summary) and use event_type="attendance"
# with log_recognition() in place of log_attendance().