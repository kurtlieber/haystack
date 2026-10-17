# Security

## Reporting a vulnerability

Report it privately: on GitHub, open this repository's **Security** tab and choose **Report a vulnerability**.
Please don't open a public issue or pull request for it. haystack has one maintainer, so replies are best
effort; you'll hear back before anything about it is made public.

Only the latest release gets fixes.

## What haystack promises

A report that breaks one of these is a vulnerability:

* **Secrets are never logged or printed.** That covers the Spotify client secret and refresh token, needle's
  API token and the alert webhook. The one exception is `haystack token show`, which prints the API token to
  stdout when you ask. `haystack secret set` reads a value from a hidden prompt or stdin, never the command
  line, and `haystack check-secrets` shows only lengths.
* **The `files` backend keeps secrets private.** It writes one 0600 file per secret in a 0700 directory, each
  written whole and renamed into place. It refuses to read a secret file that others can read.
* **The `1password` backend never takes the service-account token from the environment.** It reads it from a
  systemd credential (`op-token`) or `onepassword.token_file` (0600), and passes it only to the `op` child
  process with every other `OP_*` variable removed. Field edits go to `op` on stdin, never on its command line.
* **needle's API admits only allowed callers.** Every request needs the bearer token (compared in constant
  time) and must come from loopback or a network in `api.allow_networks`.
* **Your files are only read.** haystack never writes to `library.root`.
* **Deezer preview audio never touches the disk.** It is decoded in memory through an ffmpeg pipe, and only its
  vectors are kept.
* The data and state directories are 0700.

## Known limits

These are by design, not vulnerabilities:

* The API speaks plain HTTP. Anyone who can watch the traffic on an allowed network can read the token and your
  Ratings. Keep `api.allow_networks` to networks you trust, or reach the server over a VPN, and have the host
  firewall allow only those networks too.
* The default `api.allow_networks` is every private range. Narrow it.
* Everyone with the token has full access to the API: there are no per-user accounts.
* `haystack auth` listens on a loopback port of the server during the Spotify sign-in only.
