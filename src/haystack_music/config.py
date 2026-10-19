"""Paths and settings.

Defaults live here; ~/.config/haystack/config.toml (optional) overrides any key, e.g.

    [batch]
    size = 25          # songs per weekly batch (default 15)

HAYSTACK_DATA, HAYSTACK_STATE, HAYSTACK_CONFIG_DIR and HAYSTACK_CONFIG move the directories and the file (the
container sets them).
"""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

HOME = Path.home()


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or (HOME / default))


DATA_DIR = Path(os.environ.get("HAYSTACK_DATA") or _xdg("XDG_DATA_HOME", ".local/share") / "haystack")
STATE_DIR = Path(os.environ.get("HAYSTACK_STATE") or _xdg("XDG_STATE_HOME", ".local/state") / "haystack")
CONFIG_DIR = Path(os.environ.get("HAYSTACK_CONFIG_DIR") or _xdg("XDG_CONFIG_HOME", ".config") / "haystack")
CONFIG_FILE = Path(os.environ.get("HAYSTACK_CONFIG") or CONFIG_DIR / "config.toml")
DB_PATH = DATA_DIR / "discovery.db"
EMB_DIR = DATA_DIR / "embeddings"
STATUS_PATH = STATE_DIR / "status.json"

DEFAULTS: dict = {
    "library": {
        # Your music folder (MP4/M4A files), read-only. Empty: no local library, so your taste comes from
        # Liked Songs and Ratings alone. A root that is set but missing fails the scan instead of marking
        # every track gone (an unmounted disk).
        "root": "",
        "extensions": [".m4a"],
        "workers": 6,              # embedding processes (each runs at nice 19 inside the unit)
        "threads_per_worker": 3,   # TensorFlow intra-op threads per process
        "max_attempts": 3,         # give up on a file after this many embedding errors
    },
    "embedding": {
        "model": str(DATA_DIR / "models" / "discogs_track_embeddings-effnet-bs64-1.pb"),
        "windows": 9,              # ~one 30 s window every ~10 % of the track
        "window_seconds": 30.0,
    },
    # Where the secrets live (secrets.py): "files" (one 0600 file each in dir) or "1password" ([onepassword]).
    "secrets": {
        "backend": "files",
        "dir": str(CONFIG_DIR / "secrets"),
    },
    # The 1password backend. Items by id (or a name that is valid in an op:// reference) in vault.
    "onepassword": {
        "op_binary": "op",
        "vault": "",
        "token_file": "",          # the service-account token (0600), unless the unit passes LoadCredential=op-token
        "client_item": "",
        "client_id_field": "client id",
        "client_secret_field": "client secret",
        "token_item": "",          # the refresh token, with "issued at", "rotated at" and "scopes" next to it
        "token_field": "refresh token",
        "api_token_item": "",
        "api_token_field": "password",
        "alert_item": "",
        "alert_url_field": "url",
        "alert_key_field": "key",
    },
    "spotify": {
        # must be listed in the Spotify app's Redirect URIs; Spotify only accepts 127.0.0.1 for loopback
        "redirect_uri": "http://127.0.0.1:8518/spotify/callback",
        # user-library-modify: the server writes the heart that follows a verdict (ADR 0002)
        "scopes": ["user-library-read", "user-library-modify", "user-read-recently-played", "playlist-read-private",
                   "playlist-modify-private", "playlist-modify-public"],
        "discoveries_name": "Discoveries",
        # To Rip: a second playlist of the songs you like that aren't in the local library yet, to buy or rip
        "to_rip": False,
        "to_rip_name": "To Rip",
        "token_warn_days": 150,       # ~5 months after issue -> notify-send + status flag
        "token_lifetime_days": 182,   # ~6 months
        "full_likes_resync_hours": 24,
        "liked_preview_vectors_per_run": 60,
    },
    "batch": {
        "size": 15,                    # songs per weekly batch (`haystack batch --size N` for one run)
        "exploration_fraction": 0.2,   # the prior; each batch moves it by the outcomes (ADR 0010)
        "explore_band": [0.40, 0.75],  # score-percentile band the exploration picks come from
        "max_per_artist": 1,
        # Spread (ADR 0010): a top pick's score loses diversity * score sd * its closest cosine to a song already
        # picked (preview vectors centred on the pool); 0 picks by score alone. At most max_style_share of the top
        # picks share a top-1 Discogs style and at most max_per_nearest picks share a nearest liked song (0: no
        # cap). The caps give way when the pool can't fill the batch otherwise.
        "diversity": 0.5,
        "max_style_share": 0.4,
        "max_per_nearest": 2,
        # Exploration share: prior + 0.5 * (explore keep rate - top keep rate) over the last explore_window_batches
        # batches, clipped to exploration_bounds, once each side has explore_min_judged judged offers (0: off).
        "explore_window_batches": 8,
        "explore_min_judged": 8,
        "exploration_bounds": [0.1, 0.4],
        "expiry_days": 14,
        "max_unplayed_expiries": 2,
        "map_attempts_factor": 4,      # try mapping up to size*factor candidates to Spotify
        "min_embedded_fraction": 0.98, # real batches wait until the library embedding is (almost) complete
        # Before each batch, Discoveries is emptied. Outcomes are stored before the Spotify delete so the
        # hourly sync cannot mark them manual_removed. Unliked tracks at least this old get a normal 'no'.
        # Younger tracks: clear_recent "remove" (the weekly unit) takes them off as unplayed — back in the
        # pool, no negative — and "keep" leaves them in the playlist. clear_no_penalty closes every removed
        # offer except likes that same unplayed way, including tracks past the age cutoff.
        "clear_min_age_hours": 24,
        "clear_recent": "remove",
        "clear_no_penalty": False,
    },
    "scoring": {
        # additive bonus on raw 512-d cosine (p10..p90 nearest-other spread is only ~0.04)
        "bonus_library": 0.0,
        "bonus_like": 0.0,             # unused since verdicts: an Unrated like counts as 3 stars (ADR 0002)
        "bonus_discovery_like": 0.005,
        # Verdicts (ADR 0002): every Rating is a positive, weighted light (1 star) to heavy (5 stars), linear in
        # stars with 3 stars = bonus_discovery_like (a liked discovery was the strongest positive before);
        # an Unrated like counts as unrated_like_stars. Pass is a normal negative, Dislike the strongest, on the
        # negative_penalty strength scale below.
        "verdicts": {
            "star_bonus": [0.0, 0.0025, 0.005, 0.0075, 0.01],   # 1..5 stars
            "unrated_like_stars": 3,
            "pass_strength": 1.0,
            "dislike_strength": 2.0,
        },
        # raw | topk_mean | centred | csls | csls_centred | genre | genre_blend | auto
        # (auto = the variant the eval chose: at the first batch, then re-checked at every batch, ADR 0009)
        "variant": "auto",
        "csls_k": 10,                  # CSLS: neighbours in the reference set
        "topk": 10,                    # topk_mean / genre: k nearest positives
        "ref_size": 4000,              # CSLS: fixed random reference set of library + pool vectors
        "ref_seed": 20261008,
        # Rank candidates that sound like rejected tracks lower, whatever the variant:
        #   final = score - w * sd(score over the pool) * max_j(strength_j * excess_j)
        #   excess_j = max(0, sim_j - margin_j) / (1 - margin_j), sim in the variant's space (style vector for
        #   genre, average of embedding and style for genre_blend), margin_j = the pool's margin_quantile of
        #   similarity to negative j. Typical-of-the-pool similarity -> 0; an exact copy -> 1.
        # strength: Pass 1, Dislike 2 (scoring.verdicts); a Discoveries-clear no is a Pass (source 'playlist');
        # played-but-not-liked expiries and tracks removed from Discoveries by hand use the weights below.
        # w = 0 turns the penalty off.
        # Defaults: a near-copy of a strong no loses 1.5 score-sd (more than the top-1 -> top-15 gap of raw,
        # topk_mean, genre and genre_blend on the Oct 2026 1,500-track pool, about equal for centred), a normal no
        # 0.75 sd; 95% of the pool is untouched by any one negative.
        "negative_penalty": {
            "w": 0.75,
            "margin_quantile": 0.95,
            "weight_expired": 1.0,
            "weight_removed": 1.0,
        },
    },
    "eval": {
        "holdout": 0.2,                # share of liked songs held out per seed
        "seeds": [1, 2, 3, 4, 5],
        "recall_k": 15,
        "tie_tolerance": 0.5,          # mean-percentile points; ties go to the simpler variant
        # ADR 0009: choose by how a variant ranks the kept offers above the turned-down ones (AUC), every batch
        "objective": "auc",            # auc | mean_pct (the rule before 0.2: held-out liked songs only)
        "min_verdicts": 15,            # kept and turned-down offers each needed for auc; fewer -> the mean_pct rule
        "auc_tie": 0.02,               # AUC within this of the best is a tie; the simpler variant wins
        "switch_margin": 0.03,         # the weekly re-choice switches only for an AUC gain above this
        "chart_sources": ["deezer_chart", "deezer_editorial"],
        "preview_all_likes": True,     # sync also fetches Deezer previews for locally matched likes (eval queries)
        "min_queries": 20,             # preview-vector liked songs needed before the eval may choose a variant
        "min_style_coverage": 0.9,     # genre variants are evaluated only if this share of pool + queries has styles
    },
    "matching": {"duration_tolerance": 3.0, "confident_dt": 2.0},
    "harvest": {
        "sources": ["deezer_chart", "deezer_editorial", "deezer_new_releases",
                    "deezer_artist_radio", "deezer_related", "listenbrainz_similar"],
        "retired_sources": [],
        "max_new_per_source": 40,
        "seed_artists": 10,
        "seed_recordings": 10,
        "deezer_editorial_ids": [0, 106, 113, 116, 152, 464],
        "deezer_min_interval": 1.1,
        "listenbrainz_min_interval": 1.2,
        "listenbrainz_contact": "",    # an email or URL for the User-Agent, as MetaBrainz asks of API users
        "listenbrainz_algorithm": "session_based_days_7500_session_300_contribution_5_threshold_15_limit_50_skip_30",
        "chart_sources": ["deezer_chart", "deezer_editorial"],
        "chart_max_share": 0.2,        # chart + editorial-chart candidates: at most this share of the pool
        "broad_seeds": 150,            # `harvest --broad`: stratified seed artists (genre buckets)
        "broad_seed_rng": 20261008,    # fixed so parallel shards agree on the seed list
        "broad_radio_per_seed": 6,
        "broad_related_per_seed": 4,
        "broad_lb_per_seed": 3,
        "style_backfill_per_run": 150, # daily harvest: style vectors for older previews
        "traits_backfill_per_run": 150, # daily harvest: Readings for stored previews (Discoveries, likes first)
    },
    "sources": {"min_offers_before_retire": 30},
    # needle's private API (ADR 0004). The bearer token is the api-token secret (`haystack token new`).
    "api": {
        "host": "0.0.0.0",
        "port": 8517,
        # Loopback always gets in. The default is every private range; narrow it to the networks your clients
        # are on (a VPN's too, e.g. Tailscale's 100.64.0.0/10) and have the host firewall say the same.
        "allow_networks": ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"],
        "spotify_writes": True,        # dev only: false = record verdicts but never touch Spotify
    },
    # When `haystack serve` (the container) runs each job, in the server's local time: "hourly", "daily HH:MM",
    # "Mon HH:MM" (days Mon..Sun, comma-separated: "Mon,Thu 08:00") or "off" (scheduler.py). The systemd units
    # ignore this: their timers hold the times.
    "schedule": {
        "sync": "hourly",
        "library": "daily 03:15",
        "harvest": "daily 05:00",
        "batch": "Mon 08:00",
    },
    # Webhook alerts (alert.py): a JSON POST to the alert-webhook-url secret, with the alert-webhook-key secret
    # (if set) as the bearer token, when a job fails or the Spotify login needs renewing.
    "alert": {"enabled": False},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: Path | None = None) -> dict:
    path = path or CONFIG_FILE
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    return cfg


def ensure_dirs() -> None:
    for d in (DATA_DIR, STATE_DIR, EMB_DIR / "local"):
        d.mkdir(parents=True, exist_ok=True)
    for d in (DATA_DIR, STATE_DIR):
        os.chmod(d, 0o700)
