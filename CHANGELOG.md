# Changelog

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
