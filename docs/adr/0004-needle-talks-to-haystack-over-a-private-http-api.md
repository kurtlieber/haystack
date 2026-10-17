# needle talks to haystack over a private HTTP API

needle sends Verdicts and plays to the server, and gets the Review queues, Readings and the Baseline back, through a small HTTP API that is a user service on the server. It accepts connections only from loopback and `api.allow_networks` (the private ranges by default; the host firewall should say the same), with a bearer token (the `api-token` secret). The server makes every Spotify write that follows a Verdict (the heart, Discoveries, To Rip), so needle never needs a Web API token in phase 1. needle keeps an outbox of Verdicts given while the server is unreachable and prefetches the Readings for both Review queues, so a Review works offline.

The API is the seam that a later move (to Cloudflare Workers, say) would keep.

**Considered Options**: SSH commands per call (no new port, but a handshake per call, SSH agent prompts mid-Review, and the GUI tied to a shell setup); a Syncthing-shared state directory (SQLite conflict copies, and no on-demand measuring).
