# Changelog

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
