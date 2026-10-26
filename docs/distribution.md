# Distribution: options for making this an OSS package

Status: **option A in progress** (written 2026-10-15, updated 2026-10-19). Done: the container image on GHCR,
`haystack serve` with its in-process scheduler, and the paste-back Spotify sign-in. Decided so far: two fresh repos, haystack
(the server, AGPL-3.0) and needle (the desktop app, MIT); Deezer on by default with the README's Deezer notice.
The rest is open; turn each decision into an ADR.

## What is bespoke today

| Area | Today | What someone else would need |
|---|---|---|
| Topology | one server; needle reaches its API on a private network (`api.allow_networks` plus the host firewall) with a token file | a server URL and a pairing step in needle |
| Spotify auth | each user's own Spotify app; `haystack auth` prints the URL and takes the redirect address pasted back (or the loopback callback, through an SSH tunnel) | sign-in on the desktop (PKCE, below; on the README's roadmap) |
| Secrets | done: 0600 files by default, 1Password `op` as the other backend (`secrets.py`) | the OS keyring, maybe |
| Alerts | done: an optional generic webhook, or an outbox folder, off by default (`alert.py`) | ntfy or Apprise, maybe |
| Taste source | done: the local library is optional (`library.root`, unset by default); without one, Liked Songs and Ratings. A Spotify ID in the `MusicBrainz Track Id` tag is used when present | |
| To Rip | done: optional, off by default (`spotify.to_rip`) | |
| Candidates | Deezer previews + ListenBrainz; Deezer on by default, with the README's Deezer notice | pluggable sources |
| Wording | done: neutral wording ("the listener", "the server") | |

## Hard constraints

1. **Spotify developer quota.** Extended quota now needs a registered business with a large user base, and
   development-mode apps have a small user allowlist. A hosted service for others is effectively out. Each user
   registers their own Spotify app and enters its client ID (the usual model for third-party Spotify clients).
2. **Licensing.** Essentia (incl. `essentia-tensorflow`) is AGPL-3.0. The Essentia models are NonCommercial:
   models.html says CC BY-NC-SA 4.0, the LICENSE file in essentia.upf.edu/models/ names BY-NC-ND 4.0 (checked
   2026-10-17). `haystack models fetch` downloads them, nothing ships them. Decided: AGPL-3.0-or-later for the
   backend, with a `NOTICE` for Essentia, the models, Deezer, ListenBrainz and the Python packages.
   librespot (MIT) and Qt (LGPL) are fine for needle.
3. **Deezer previews.** Distributing the tool hands the terms risk to every user. Decided: on by default, with the
   README's Deezer notice; the 1 req / 1.1 s throttle and memory-only audio stay hard rules.
4. **librespot** needs Premium and is a grey area; spotifyd, ncspot and others ship it openly. Accepted risk,
   but say so in the README.
5. **Repo history** holds personal data and homelab references. Decided: fresh repos; run gitleaks on them before
   they go public. The Deezer covers (needle's `docs/mockups/src/assets/`) are gitignored and must stay out.

Selling point for the README: taste never comes from Spotify audio (local files and preview embeddings only),
which keeps clear of Spotify's rule against using its content for machine learning.

## Packaging options

### A. Self-hosted server + desktop client (Jellyfin / Navidrome model)

- Backend as a container image on GHCR. The systemd timers (`sync`, `batch`, backfills) become an in-process
  scheduler. Volumes: database, models, optional read-only music mount.
- needle via AUR (needle's `app/pkgbuild` exists), Flathub, AppImage. First-run screen: server URL + pairing code.
- The API needs TLS, or docs pointing at Tailscale / a reverse proxy. A bearer token over plain HTTP is only
  acceptable on a trusted LAN.
- Pro: closest to the current split (ADR 0004, 0006). An always-on server keeps the 15-minute play poll accurate;
  `recently-played` returns only the last 50 plays, so a sleeping host loses plays (ADR 0023 records them as gaps).
- Audience: homelab users.

### B. All-in-one desktop

- needle starts the backend as a local sidecar (systemd user timers on Linux, launchd on macOS). One install,
  no server.
- Cost: play sync is less accurate when the laptop sleeps; the first library embedding pins the CPU for hours.
- Audience: much wider. Realistic as a second mode on top of A, since the backend doesn't care where it runs.

### C. Split into library, CLI, GUI

- Core (scoring, Traits, embedding) on PyPI: `uv tool install haystack-music[embed]`; needle is one front end.
- Invites other front ends (TUI, web). More maintenance; only worth it if outside interest shows up.

### Ruled out: hosted SaaS

Spotify quota and Deezer terms (constraints 1 and 3).

## Portability work worth doing regardless

- **ONNX instead of essentia-tensorflow.** Discogs-EffNet is published as ONNX. onnxruntime brings ARM
  (Raspberry Pi, Apple Silicon) and Windows and drops TensorFlow. Catch: the mel-spectrogram front end must
  match Essentia's exactly. Check new embeddings against the existing `embeddings/local/*.npz` before switching.
  Dropping Essentia entirely would also remove the AGPL constraint.
- **Desktop-side PKCE sign-in.** needle opens the browser, runs Spotify's PKCE flow locally, sends the refresh
  token to the server. No client secret, no loopback callback on a headless box.
- **One config file** holding every host, path and playlist name, plus a `doctor` command that checks it
  (`status` / `fix`).
- **One user per instance.** Multi-user would touch every table and the Verdict log for little gain.

## Recommendation

Do A first:

1. Generalise config, secrets, alerts; neutral wording.
2. PKCE sign-in in needle and a pairing step.
3. Container image + Flatpak.

Then B as a "run locally" switch in needle. Spike ONNX in parallel; it decides how many platforms are reachable.
Settle the license (AGPL backend) and clean the history before anything goes public.
