# Changelog

## [0.15.0](https://github.com/kurtlieber/haystack/compare/v0.14.1...v0.15.0) (2026-11-12)


### Features

* **api:** keep generated songs and their verdicts in an optional store for spindle (dotfiles [#523](https://github.com/kurtlieber/haystack/issues/523)) ([#60](https://github.com/kurtlieber/haystack/issues/60)) ([5c1954c](https://github.com/kurtlieber/haystack/commit/5c1954c990b7e2d504487a1503c0edd89b0edb05))
* **api:** serve the Taste brief at GET /v1/taste for spindle (dotfiles [#523](https://github.com/kurtlieber/haystack/issues/523)) ([#59](https://github.com/kurtlieber/haystack/issues/59)) ([9ac9300](https://github.com/kurtlieber/haystack/commit/9ac93002c9d4ee11959695edc7ac437c7f5139eb))

## [0.14.1](https://github.com/kurtlieber/haystack/compare/v0.14.0...v0.14.1) (2026-11-10)


### Bug Fixes

* **batch:** retry a failed batch for a day before alerting ([#57](https://github.com/kurtlieber/haystack/issues/57)) ([bef1454](https://github.com/kurtlieber/haystack/commit/bef1454073f6b46a1e2abdf520d20b0e7308a3db))

## [0.14.0](https://github.com/kurtlieber/haystack/compare/v0.13.0...v0.14.0) (2026-11-05)


### Features

* **api:** list trait.harmony, so needle draws the harmonic movement scale (dotfiles [#450](https://github.com/kurtlieber/haystack/issues/450)) ([#55](https://github.com/kurtlieber/haystack/issues/55)) ([f61f56e](https://github.com/kurtlieber/haystack/commit/f61f56e57aa6120196baac08a8365333d910095c))

## [0.13.0](https://github.com/kurtlieber/haystack/compare/v0.12.1...v0.13.0) (2026-11-05)


### Features

* **traits:** a harmony Reading, harmonic movement across bars, for the re-ranker (dotfiles [#450](https://github.com/kurtlieber/haystack/issues/450)) ([#53](https://github.com/kurtlieber/haystack/issues/53)) ([a08ffc3](https://github.com/kurtlieber/haystack/commit/a08ffc3fe586802f73d13c6e67220087efcc231f))

## [0.12.1](https://github.com/kurtlieber/haystack/compare/v0.12.0...v0.12.1) (2026-11-05)


### Bug Fixes

* **api:** haystack-api starts again; report imports review as a module ([#51](https://github.com/kurtlieber/haystack/issues/51)) ([9f2b913](https://github.com/kurtlieber/haystack/commit/9f2b9135d2cd4133431b3d645683b7dd521290a3))

## [0.12.0](https://github.com/kurtlieber/haystack/compare/v0.11.0...v0.12.0) (2026-11-05)


### Features

* scene feeds replace the chart family, and a retired source's songs leave the batches ([#49](https://github.com/kurtlieber/haystack/issues/49)) ([88cf78e](https://github.com/kurtlieber/haystack/commit/88cf78ea557524ba53ce5ad7d1f088cc639e3e63))

## [0.11.0](https://github.com/kurtlieber/haystack/compare/v0.10.1...v0.11.0) (2026-11-05)


### Features

* a Song is a row of its own, keyed by ISRC, and every candidate joins one ([#47](https://github.com/kurtlieber/haystack/issues/47)) ([b2e5571](https://github.com/kurtlieber/haystack/commit/b2e5571d2fd50338a332b062f9cdc93ed5df8d75))

## [0.10.1](https://github.com/kurtlieber/haystack/compare/v0.10.0...v0.10.1) (2026-11-05)


### Documentation

* **roadmap:** what comes after 0.10.0 ([#45](https://github.com/kurtlieber/haystack/issues/45)) ([7b773d0](https://github.com/kurtlieber/haystack/commit/7b773d064d9acd746846938ef700c951980aec5c))

## [0.10.0](https://github.com/kurtlieber/haystack/compare/v0.9.0...v0.10.0) (2026-11-05)


### Features

* **sync:** plays are read every 15 minutes, so Spotify's last 50 never overflow: `haystack plays`, `schedule.plays = "every 15m"` (a new `every Nm` form) and `haystack-plays.timer` for the units; a poll that still gets a full page records a play gap, and `haystack status` shows the count and the latest (ADR 0023) ([ee0a7c0](https://github.com/kurtlieber/haystack/commit/ee0a7c00e6a4ad617cddb9a87e996f7455e54468))
* **clear:** a discovery nobody played is not a no: before each batch an unliked track played since it was added is a no as before, one never played goes back to the pool and counts toward `batch.max_unplayed_expiries` (the second retires it), and one whose time in Discoveries overlaps a play gap goes back without counting (ADR 0023) ([dc397ab](https://github.com/kurtlieber/haystack/commit/dc397ab0378f3d1a74fb0ff335e8c335a9052000))


### Bug Fixes

* **batch:** a candidate that is the same song as a like, a file or a judged track (by ISRC, else by artist, base title and length) is never offered or asked about, whichever release of it the harvest found (ADR 0021) ([0bc6924](https://github.com/kurtlieber/haystack/commit/0bc6924d8b1637dee6478adaaee6718b829c8a0a))
* **harvest:** `deezer_label` passes over the major labels, the big catalogue labels and the catalogue companies (`harvest.label_skip`, matched by whole words of the label's name) and the next label takes the slot: their album search returned a whole catalogue of every style (Columbia France's chanson for a listener of drum & bass); the run's stats count `majors_skipped` (ADR 0020 note) ([7c0709b](https://github.com/kurtlieber/haystack/commit/7c0709b262111989c428e73c1b32a8ec69a06941), [573ad9b](https://github.com/kurtlieber/haystack/commit/573ad9b844ec5db3bf13c0d2d63dc4437f04a894))
* **harvest:** a possible cover by its own title or artists (karaoke, a tribute, an instrumental) is no seed for artist radio, related artists, ListenBrainz or Last.fm, nor for the broad harvest: its similar tracks were more covers ([9f6daf3](https://github.com/kurtlieber/haystack/commit/9f6daf3dea0ba09007a6f0de50d341c92a1e8eb7))
* **cli:** `haystack genres` no longer fails with `TypeError: 'int' object is not subscriptable`: it passes the config, not the top count, to `liked_genres` (kurtlieber/dotfiles#350) ([3619a8e](https://github.com/kurtlieber/haystack/commit/3619a8ec0630ef750e3a3a655288f628caf7beff))
* `__version__`, and with it `/v1/health` and the Deezer and ListenBrainz User-Agent, said 0.8.0 in 0.9.0; a test now ties it to `pyproject.toml` ([8ecfebe](https://github.com/kurtlieber/haystack/commit/8ecfebec16aacd750b223e722a308d0372f0709a))

## [0.9.0](https://github.com/kurtlieber/haystack/compare/v0.8.0...v0.9.0) (2026-10-23)


### Features

* **api:** a possible cover is flagged and the listener may swap it for its original: each track is checked once, in the background, for a title marker (cover, karaoke, tribute, "originally performed by"), a live take, an ISRC Deezer files under another artist's track of the same title, or another artist's track of the same title three times as popular; a track card carries `original` and `original_check` (API feature `card.original`), and `POST /v1/swaps` takes back the cover's verdicts and puts the Original where the cover was in the listener's own playlists, Liked Songs and To Rip (ADR 0022; API feature `swaps`) ([50a3848](https://github.com/kurtlieber/haystack/commit/50a384876f62eb50d0ffb5ce1a2f3df007b56530))


### Bug Fixes

* **deps:** certifi 2026.10.19 or later, without the compromised eMudhra CA ([be9fd9e](https://github.com/kurtlieber/haystack/commit/be9fd9e8b1d49bf2261c5c47657545898a49f622))

## [0.8.0](https://github.com/kurtlieber/haystack/compare/v0.7.1...v0.8.0) (2026-10-22)


### Features

* **api:** a verdict counts for every release of a song: a track card carries `song`, its ISRC, and a playlist card's verdict, the playlist snapshot's verdicts and Rate your likes go by the Song, so needle doesn't ask again about a song rated on another album (ADR 0021; API feature `card.song`) ([4a5153f](https://github.com/kurtlieber/haystack/commit/4a5153f8b67f732748f4a262a8c7310857b8660b))


### Bug Fixes

* **traits:** drum & bass and jungle tempos are folded by style; Deezer's bpm is fetched where it was missing ([accce07](https://github.com/kurtlieber/haystack/commit/accce07d819e7bad0b8e2b4d6b8e7d4f8ab97ff2))
* **cli:** under `--json` the log goes to stderr, so stdout holds the JSON alone ([744b542](https://github.com/kurtlieber/haystack/commit/744b542c1d184951057a4cdb4350c654bf774e2b))

## [0.7.1](https://github.com/kurtlieber/haystack/compare/v0.7.0...v0.7.1) (2026-10-20)


### Features

* **llm:** `haystack llm judge` measures whether an optional language model's knowledge of artists, labels and scenes predicts your verdicts, blind and leave-one-out, against the re-ranker; `haystack llm suggest --dry-run` counts how many of its suggestions are real songs on Deezer. Off by default (`[llm]`, the `llm-api-key` secret, the optional extra `[llm]`); metadata only; providers `anthropic` and `openai_compatible` (ADR 0019) ([aa5d90a](https://github.com/kurtlieber/haystack/commit/aa5d90a845994719b1430aeb43a917a9d2e1af7b))

## [0.7.0](https://github.com/kurtlieber/haystack/compare/v0.6.0...v0.7.0) (2026-10-20)


### Features

* **harvest:** four more sources, idle until set up: Last.fm's similar songs and artists (`lastfm-api-key`), your own ListenBrainz recommendations (`harvest.listenbrainz_user`), ListenBrainz's LB Radio (`listenbrainz-token`) and the newest albums of the labels you keep most; `harvest.preview_fallback = "itunes"` (off by default) listens to Apple's preview of a song Deezer has none for (ADR 0020) ([11fb947](https://github.com/kurtlieber/haystack/commit/11fb94727cb636c2c29be3d27325ab414d8778cd))

## [0.6.0](https://github.com/kurtlieber/haystack/compare/v0.5.0...v0.6.0) (2026-10-20)


### Features

* **metadata:** what Deezer publishes about a track (rank, album label, release date, record type, fans) joins the re-ranker's features; `haystack metadata backfill` fills it in for tracks harvested before (ADR 0017; API feature `card.label`) ([b265975](https://github.com/kurtlieber/haystack/commit/b265975cd02ca490415b940038976f35ea820a66))

## [0.5.0](https://github.com/kurtlieber/haystack/compare/v0.4.0...v0.5.0) (2026-10-20)


### Features

* **alert:** an [alert] outbox folder instead of the webhook ([35faa49](https://github.com/kurtlieber/haystack/commit/35faa4986b7485dd80b7f51dce850cf223e2488c))
* **calibrate:** a third review queue, the candidates haystack is least sure of, for needle to ask about (ADR 0015; API feature `queue.calibrate`) ([456bb19](https://github.com/kurtlieber/haystack/commit/456bb1911939dcdfc7ba7c2f8e4ad16e45a7b41a))
* **plays:** a short, unfinished play is a soft no for the re-ranker, and plays grade an Unrated like (ADR 0016; API feature `plays.ms_played`) ([456bb19](https://github.com/kurtlieber/haystack/commit/456bb1911939dcdfc7ba7c2f8e4ad16e45a7b41a))
* **report:** which learning gates are open and what each still needs, and plays per week; status.json carries the gates ([456bb19](https://github.com/kurtlieber/haystack/commit/456bb1911939dcdfc7ba7c2f8e4ad16e45a7b41a))

## [0.4.0](https://github.com/kurtlieber/haystack/compare/v0.3.1...v0.4.0) (2026-10-19)


### Features

* **harvest:** seeds follow the Verdicts, styles are searched by name, caps follow the kept rate (ADR 0013) ([5d014c4](https://github.com/kurtlieber/haystack/commit/5d014c479cb3305e93f8875019f28f5fc14199eb))

## [0.3.1](https://github.com/kurtlieber/haystack/compare/v0.3.0...v0.3.1) (2026-10-19)


### Features

* **align:** map preview vectors into the library's space before scoring (ADR 0012) ([1782890](https://github.com/kurtlieber/haystack/commit/1782890e265eeebf1c180d310c566cc61586d0e8))

## [0.3.0](https://github.com/kurtlieber/haystack/compare/v0.2.0...v0.3.0) (2026-10-19)


### Features

* **rerank:** a second stage trained on the listener's Verdicts orders the top of the pool (ADR 0011; API features `rerank` and `why.reasons`) ([7b38866](https://github.com/kurtlieber/haystack/commit/7b388668394a8177c3d44a30495bc646a54d7fc8))

## [0.2.0](https://github.com/kurtlieber/haystack/compare/v0.1.0...v0.2.0) (2026-10-19)


### Features

* **api:** a server object with features on /v1/health and /v1/state ([f59b678](https://github.com/kurtlieber/haystack/commit/f59b678d5f0696e52bbe45379702c77a4f2fe2b4))
* **batch:** spread a batch across styles and liked songs, and let the exploration share follow the outcomes (ADR 0010; API feature `why.band`) ([003ad00](https://github.com/kurtlieber/haystack/commit/003ad00d7c55e900792b590532a82389d8ba56db))
* **eval:** choose the scoring variant by kept against turned-down offers, and re-check it every batch (ADR 0009) ([b049248](https://github.com/kurtlieber/haystack/commit/b049248c8ba3ee7732f2e515148d458a3fbf2666))
* batch --plan, the batch's picks with no Spotify calls and no writes ([d84c089](https://github.com/kurtlieber/haystack/commit/d84c089c3b45fbd8be815e408b0bb2ed456c2e04))
* haystack report, how discovery is doing on this installation ([34cade2](https://github.com/kurtlieber/haystack/commit/34cade27e4f07a32299ddb9867e8a155cc8a5b8e))

## [0.1.0](https://github.com/kurtlieber/haystack/commits/v0.1.0) (2026-10-19)


### Features

* container image on GHCR, and a README for first-time users ([c4690a0](https://github.com/kurtlieber/haystack/commit/c4690a0973e6fed3064d305003b002e0c0ad72f9))
* haystack serve (API + scheduled jobs) and a paste-back Spotify sign-in ([7b0655a](https://github.com/kurtlieber/haystack/commit/7b0655aa5c86947b4fa51f74aa0f5b1dfa85661c))


### Bug Fixes

* **serve:** the first batch also follows a good harvest run ([f574da3](https://github.com/kurtlieber/haystack/commit/f574da33c36e2489c0710235662a7adf855d8fb9))

---

0.1.0 is the first release from this repository. The code it started from was written in haystack's earlier
repository, so the work before it has no entries here.
