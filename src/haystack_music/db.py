"""SQLite schema and helpers (~/.local/share/haystack/discovery.db)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS local_tracks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,          -- short audio-content hash (mdat samples); fallback key, never mtime
    artist_raw TEXT, artists_json TEXT,  -- artists_json: normalised list (comma-joined strings split)
    artist_key TEXT,                     -- key of the first artist
    title_raw TEXT, title_base TEXT, title_key TEXT, version_suffix TEXT,
    album TEXT, genre TEXT, year TEXT,
    duration REAL,
    spotify_id TEXT,                     -- 22-char Spotify track id from the (mis-named) MusicBrainz Track Id tag
    spotify_artist_ids TEXT,
    isrc TEXT,
    present INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT, last_seen TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS local_tracks_hash ON local_tracks(content_hash);
CREATE INDEX IF NOT EXISTS local_tracks_sid ON local_tracks(spotify_id);

CREATE TABLE IF NOT EXISTS local_embeddings (
    content_hash TEXT PRIMARY KEY,
    status TEXT NOT NULL,                -- ok | error
    attempts INTEGER NOT NULL DEFAULT 0,
    n_windows INTEGER, file TEXT, error TEXT, seconds REAL, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS spotify_tracks (
    spotify_id TEXT PRIMARY KEY,
    name TEXT, artists_json TEXT, artist_key TEXT, title_key TEXT, version_suffix TEXT,
    duration REAL, isrc TEXT, album TEXT, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS likes (
    spotify_id TEXT PRIMARY KEY,
    added_at TEXT, first_seen TEXT, last_seen TEXT,
    removed_at TEXT                      -- set when no longer in Liked Songs on a full resync
);

CREATE TABLE IF NOT EXISTS plays (
    spotify_id TEXT NOT NULL, played_at TEXT NOT NULL, context TEXT,   -- context: a Spotify URI, or 'needle'
    PRIMARY KEY (spotify_id, played_at)
);

CREATE TABLE IF NOT EXISTS liked_map (
    spotify_id TEXT PRIMARY KEY,
    local_id INTEGER,                    -- NULL = not in library
    method TEXT, confidence TEXT, dt REAL, version TEXT, reason TEXT,
    review TEXT,                         -- NULL | pending | accepted | rejected
    use_local INTEGER NOT NULL DEFAULT 0,
    deezer_id INTEGER,                   -- preview vector source when use_local = 0
    vector_status TEXT,                  -- NULL | ok | not_found | error
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS deezer_vectors (
    deezer_id INTEGER PRIMARY KEY,
    e512 BLOB NOT NULL, e1280 BLOB, n_patches INTEGER, seconds REAL, created_at TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY,
    deezer_id INTEGER UNIQUE,
    artist TEXT, artist_key TEXT, title TEXT, title_key TEXT, version_suffix TEXT, song_key TEXT,
    duration REAL, isrc TEXT, mbid TEXT, deezer_artist_id INTEGER,
    source TEXT NOT NULL,                -- first source that produced it (like-rate attribution)
    harvested_at TEXT,
    vector_status TEXT,                  -- NULL | ok | no_preview | error
    spotify_id TEXT, spotify_status TEXT, -- NULL | ok | not_found
    state TEXT NOT NULL DEFAULT 'pool',  -- pool | offered | liked | negative | retired | excluded
    exclude_reason TEXT,
    unplayed_expiries INTEGER NOT NULL DEFAULT 0,
    score REAL, raw_cos REAL, nearest TEXT, scored_at TEXT
);
CREATE INDEX IF NOT EXISTS candidates_state ON candidates(state);
CREATE INDEX IF NOT EXISTS candidates_song ON candidates(song_key);

CREATE TABLE IF NOT EXISTS candidate_sources (
    candidate_id INTEGER NOT NULL, source TEXT NOT NULL, detail TEXT, seen_at TEXT,
    PRIMARY KEY (candidate_id, source, detail)
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY, created_at TEXT, size INTEGER, n_explore INTEGER, note TEXT
);

CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL, spotify_id TEXT NOT NULL, batch_id INTEGER,
    source TEXT, score REAL, exploration INTEGER NOT NULL DEFAULT 0,
    score_percentile REAL,               -- the pick's score percentile in its batch's scored pool (0.2.0+)
    added_at TEXT NOT NULL, removed_at TEXT,
    played INTEGER NOT NULL DEFAULT 0, first_played_at TEXT,
    outcome TEXT                         -- NULL (active) | liked | negative | unplayed | manual_removed
);
CREATE INDEX IF NOT EXISTS offers_active ON offers(removed_at);

-- Every track the bot ever put into one of its playlists. Hourly sync only removes rows here.
-- The pre-batch Discoveries clear removes every playlist track, then closes these rows.
CREATE TABLE IF NOT EXISTS playlist_adds (
    id INTEGER PRIMARY KEY,
    playlist TEXT NOT NULL,              -- discoveries | to_rip
    playlist_id TEXT NOT NULL, spotify_id TEXT NOT NULL,
    added_at TEXT NOT NULL, snapshot_id TEXT,
    removed_at TEXT, removed_reason TEXT
);
CREATE INDEX IF NOT EXISTS playlist_adds_active ON playlist_adds(playlist, removed_at);

-- Discoveries clear, written before the Spotify delete and removed after local bookkeeping.
-- A retry or the next sync finishes these rows once the track is gone, without a second verdict
-- and without reading the delete as a hand removal.
CREATE TABLE IF NOT EXISTS clear_pending (
    playlist_id TEXT NOT NULL,
    spotify_id TEXT NOT NULL,
    reason TEXT NOT NULL,                -- liked | unliked | recent
    verdict TEXT,                        -- 'no' or NULL (liked and recent record nothing)
    candidate_id INTEGER,
    added_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (playlist_id, spotify_id)
);

CREATE TABLE IF NOT EXISTS to_rip (
    spotify_id TEXT PRIMARY KEY, added_at TEXT, removed_at TEXT, local_id INTEGER, match_method TEXT
);

CREATE TABLE IF NOT EXISTS deezer_artists (
    name_key TEXT PRIMARY KEY, deezer_artist_id INTEGER, name TEXT, checked_at TEXT
);

-- What Deezer publishes about a track's album and artist (ADR 0017): re-ranker features and the card's label.
-- A row with fetched_at but no data: Deezer had no such album or artist when asked.
CREATE TABLE IF NOT EXISTS deezer_albums (
    album_id INTEGER PRIMARY KEY,
    label TEXT, release_date TEXT,       -- release_date: YYYY-MM-DD
    record_type TEXT,                    -- album | single | ep | compile
    genres_json TEXT, fans INTEGER, nb_tracks INTEGER, fetched_at TEXT
);
-- fan counts by Deezer artist id (deezer_artists above is a name -> id cache: a name holds one id)
CREATE TABLE IF NOT EXISTS deezer_artist_fans (
    artist_id INTEGER PRIMARY KEY, nb_fan INTEGER, fetched_at TEXT
);

-- Spotify track -> Deezer track whose preview stands in for it (on-demand Readings for tracks the server hadn't seen)
CREATE TABLE IF NOT EXISTS spotify_deezer (
    spotify_id TEXT PRIMARY KEY, deezer_id INTEGER,
    status TEXT,                         -- ok | no_preview | not_found | error
    method TEXT, checked_at TEXT
);

-- A Deezer track without a preview, looked up at a second provider (ADR 0020). found=0: not there when asked
-- (asked again after a month); found=1: its preview was embedded (deezer_vectors.provider says which).
CREATE TABLE IF NOT EXISTS preview_fallback (
    deezer_id INTEGER PRIMARY KEY, provider TEXT NOT NULL, found INTEGER NOT NULL, checked_at TEXT NOT NULL
);

-- the seeds each harvest source started from (ADR 0013): an artist is seeded at most once a week per source
CREATE TABLE IF NOT EXISTS harvest_seeds (
    at TEXT NOT NULL, source TEXT NOT NULL, artist_key TEXT NOT NULL,
    stars INTEGER,                       -- the seed's Rating, if any
    origin TEXT                          -- rating | discovery | like | library | broad (NULL: seed_policy "recent")
);
CREATE INDEX IF NOT EXISTS harvest_seeds_source ON harvest_seeds(source, at);

CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, job TEXT, started_at TEXT, finished_at TEXT, ok INTEGER, summary TEXT, error TEXT
);

-- every scoring-variant eval (ADR 0009): what it chose, what the batch used, and whether the variant switched
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY, at TEXT,
    trigger TEXT,                        -- weekly (a batch) | set (first batch, eval --choose) | report (eval)
    objective TEXT,                      -- the objective used: auc | mean_pct
    chosen TEXT, used TEXT, n_pos INTEGER, n_neg INTEGER,
    auc_json TEXT,                       -- {variant: AUC or null}
    switched INTEGER
);
-- the Calibrate queue (ADR 0015): pool candidates needle asks the listener about. The candidate stays in the
-- pool (a Verdict moves it on, as for an offer); judged_at is set when a Verdict for the track arrives, or the
-- like's time when the listener hearts it in Spotify without one.
CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL, spotify_id TEXT NOT NULL,
    picked_at TEXT NOT NULL,
    week TEXT NOT NULL,                  -- ISO week of the pick, e.g. 2026-W43 (calibrate.per_week)
    band TEXT,                           -- the stage-1 percentile band, e.g. 90-97
    percentile REAL, rerank_p REAL, disagreement REAL,   -- at the pick
    reason TEXT,                         -- the phrase the card shows
    rerank_reason TEXT,                  -- the re-ranker's reasons at the pick (JSON), when fitted
    judged_at TEXT
);
CREATE INDEX IF NOT EXISTS calibrations_open ON calibrations(judged_at);
CREATE INDEX IF NOT EXISTS calibrations_track ON calibrations(spotify_id);

CREATE TABLE IF NOT EXISTS rerank_runs (
    id INTEGER PRIMARY KEY, at TEXT,
    trigger TEXT,                        -- batch | fit (haystack rerank fit)
    variant TEXT,                        -- the stage-1 variant it re-ranks
    n_labels INTEGER, n_pos INTEGER, n_neg INTEGER,
    loo_auc_model REAL, loo_auc_stage1 REAL,
    used INTEGER, reason TEXT,
    model_json TEXT                      -- weights, scaler and feature names (ADR 0011)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120, isolation_level=None)  # autocommit; explicit BEGIN where needed
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=120000")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    _migrate(con)
    from . import playlists, traits, verdicts
    verdicts.ensure(con)
    traits.ensure(con)
    playlists.ensure(con)
    return con


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    """A connection that cannot write (no schema, no migrations), for `report` and `batch --plan`.
    FileNotFoundError when there is no database yet."""
    path = Path(path or config.DB_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"no database at {path} yet (run a harvest first)")
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=120, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=120000")
    return con


def _add_column(con, table: str, col: str, decl: str) -> None:
    if col in {r[1] for r in con.execute(f"PRAGMA table_info({table})")}:
        return
    try:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    except sqlite3.OperationalError as e:      # another process added it first
        if "duplicate column" not in str(e):
            raise


def _migrate(con) -> None:
    _add_column(con, "deezer_vectors", "style400", "BLOB")       # genre_discogs400 activations, f16
    _add_column(con, "candidates", "neg_penalty", "REAL")         # score drop from the negative penalty
    _add_column(con, "candidates", "nearest_negative", "TEXT")    # the rejected track that set it
    _add_column(con, "offers", "score_percentile", "REAL")         # the pick's percentile in its batch's pool
    _add_column(con, "candidates", "rerank_score", "REAL")         # the re-ranker's blend, in z units (ADR 0011)
    _add_column(con, "candidates", "rerank_p", "REAL")             # its chance the listener keeps it
    _add_column(con, "candidates", "rerank_reason", "TEXT")        # JSON list of up to two phrases
    _add_column(con, "rerank_runs", "n_soft", "INTEGER")           # soft labels in the fit (ADR 0016)
    _add_column(con, "plays", "ms_played", "INTEGER")              # needle plays: how long it played (0.5.0)
    _add_column(con, "plays", "completed", "INTEGER")              # and whether it reached the end
    for col in ("album_type", "release_date", "image_url", "image_url_small"):   # needle's track cards
        _add_column(con, "spotify_tracks", col, "TEXT")
    _add_column(con, "candidates", "album_id", "INTEGER")          # its Deezer album; 0: Deezer has none (0.6.0)
    _add_column(con, "candidates", "deezer_rank", "INTEGER")       # the track's Deezer rank (popularity)
    _add_column(con, "liked_map", "album_id", "INTEGER")           # deezer_id's album, as on candidates
    _add_column(con, "rerank_runs", "loo_auc_no_meta", "REAL")     # the model's AUC without Deezer's metadata
    _add_column(con, "deezer_vectors", "provider", "TEXT NOT NULL DEFAULT 'deezer'")   # whose preview (0.7.0)


def kv_get(con, k: str, default=None):
    r = con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def kv_set(con, k: str, v) -> None:
    con.execute("INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, None if v is None else str(v)))
