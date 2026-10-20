# Setting up haystack: instructions for an AI agent

You are helping a person install and configure haystack, a self-hosted music discovery server for Spotify. Read
the [README](README.md) first for what it does; this file is the procedure. It has only been tested with Claude
Code, but any agent that can run shell commands on the target machine should be able to follow it.

If you are here to work on haystack's code instead, skip to [Working on the code](#working-on-the-code).

## Ground rules

- **Secrets stay out of the conversation.** Never ask the person to paste the Spotify client secret, a refresh
  token, the API token or a webhook key into the chat, and never put one on a command line, in a file you write,
  or in your output. `haystack secret set NAME` reads a hidden prompt; the person runs it in their own terminal.
  The client ID isn't secret, but it goes in the same way.
- **The Spotify sign-in is the person's.** You can't open their browser. They run `haystack auth`, open the link
  and paste the address back into their terminal, not into the chat.
- **Batch the person's steps.** Give every step only they can do as one numbered list, at the point where you
  need it, with the exact commands. Don't drip them out one at a time.
- **Don't touch their Spotify playlists** by hand, and don't change settings they didn't ask about. Every default
  is a reasonable one.
- **Docker is the default.** Use the systemd route only if the person asks for it or the machine can't run
  Docker (see [Without Docker](#without-docker)).

## 1. Check the machine

On the machine haystack will run on (over SSH if it's a server):

```sh
uname -m                     # must be x86_64: there is no ARM image
docker compose version       # Docker with the compose plugin
id -u; id -g                 # 1000/1000 needs nothing; anything else goes in compose.yaml's user:
df -h ~                      # about 2 GB for the image, plus the data
```

If `uname -m` isn't `x86_64`, stop and tell the person haystack doesn't run there yet. If Docker is missing,
offer to install it with their distribution's packages (or `https://get.docker.com`), and confirm before using
`sudo`. The user running haystack needs to be in the `docker` group, or every `docker` command needs `sudo`.

## 2. Ask the person

Ask these together, with the defaults filled in, so they can just say "defaults are fine":

1. **Folder for haystack.** Default `~/haystack`.
2. **Time zone.** Default: the machine's (`timedatectl show -p Timezone --value`, or `cat /etc/timezone`).
3. **Music folder?** A folder of `.m4a` files on this machine, or none. Tell them haystack treats every song in
   it as one they like, so it should hold music they love. Other formats are skipped: check what's there with
   `find <folder> -type f -iname '*.m4a' | wc -l`, and say so if it holds mostly MP3 or FLAC. If it's a network
   share, it must be mounted before haystack starts. The README's
   [Your music folder](README.md#your-music-folder-optional) section has the details.
4. **How many songs per batch, and when?** Default 15 songs, Mondays at 08:00.
5. **To Rip playlist?** A second playlist of songs they rate 3 stars or more that aren't in their music folder.
   Default off.
6. **needle?** Will they use the desktop app? If so, from which networks (home LAN range, a VPN such as
   Tailscale's `100.64.0.0/10`)?
7. **Contact for ListenBrainz.** An email or URL that MetaBrainz asks API users to send. Optional.
8. **Alerts?** A webhook URL (ntfy, a chat webhook, anything that takes a JSON POST) for failures and sign-in
   expiry. Optional.

Also tell them about the [Deezer notice](README.md#deezer-notice) and let them decide. If they want to avoid
Deezer, explain that removing the Deezer sources leaves only ListenBrainz, which still needs Deezer previews,
so in practice haystack depends on Deezer.

## 3. Lay out the files

```sh
mkdir -p ~/haystack/config ~/haystack/data && cd ~/haystack
curl -fsSLO https://raw.githubusercontent.com/kurtlieber/haystack/main/compose.yaml
curl -fsSL https://raw.githubusercontent.com/kurtlieber/haystack/main/config.example.toml -o config/config.toml
```

Create `config` and `data` as the person's user before the first `docker compose` command, or Docker creates
them as root. Edit `compose.yaml`:

- `TZ:` the time zone from step 2.
- If `id -u`/`id -g` weren't 1000: uncomment `user:` and set `"<uid>:<gid>"`.
- If there's a music folder: uncomment the `/music` volume line and put the host path on the left. Keep `:ro`.

## 4. Write config.toml

Edit `config/config.toml`. Keep the file short: a key left out keeps its default (`src/haystack_music/config.py`
has every default with comments). From the answers:

```toml
[library]
root = "/music"                  # only with a music folder; the path inside the container
# On a machine with few cores (check nproc), lower these so the first embedding pass doesn't starve it:
# workers = 2
# threads_per_worker = 2

[spotify]
to_rip = true                    # only if they want To Rip

[batch]
size = 15

[schedule]
batch = "Mon 08:00"              # "hourly", "daily HH:MM", "Mon HH:MM", "Mon,Thu HH:MM" or "off"; local time

[harvest]
listenbrainz_contact = "..."     # if they gave one

[api]
# Who besides the container's own loopback may reach needle's API. needle on the same machine arrives from
# Docker's bridge network, so keep 172.16.0.0/12 in that case.
allow_networks = ["192.168.1.0/24", "100.64.0.0/10", "172.16.0.0/12"]

[alert]
enabled = true                   # only with a webhook
```

Check it parses: `docker compose run --rm haystack status > /dev/null` fails with the line number on a TOML
error. (A first run also pulls the image, about 540 MB.) A bad `[schedule]` value only shows when the server
starts, in step 7: the container keeps restarting and its log names the value.

## 5. The person's steps

Give the person this list, filled in, and wait until they say it's done:

1. Create a Spotify app at https://developer.spotify.com/dashboard → **Create app**. Any name and description.
   Redirect URI exactly `http://127.0.0.1:8518/spotify/callback`. API: **Web API**. Save, then open
   **Settings** for the Client ID and Client secret.
2. In a terminal on the server, in `~/haystack`, enter the two keys (each prompt hides what you type):
   ```sh
   docker compose run --rm haystack secret set spotify-client-id
   docker compose run --rm haystack secret set spotify-client-secret
   ```
3. Sign in to Spotify:
   ```sh
   docker compose run --rm haystack auth
   ```
   Open the printed link in any browser and agree. The page it lands on won't load; that's expected. Copy the
   whole address from the address bar and paste it into the terminal within a few minutes.
4. Only with alerts: `docker compose run --rm haystack secret set alert-webhook-url` (and `alert-webhook-key` if
   the webhook wants a bearer token).

In Claude Code, the person can run these from the prompt with the `!` prefix only if it gives them an
interactive terminal; a separate SSH session to the server is the reliable way.

## 6. Finish the setup

These need no secrets from the person, so run them yourself:

```sh
cd ~/haystack
docker compose run --rm haystack check-secrets    # lengths only, never values; the client keys and the token must be set
docker compose run --rm haystack models fetch     # Essentia's models, into data/models
docker compose run --rm haystack token new        # only with needle: the API's bearer token
docker compose up -d
```

## 7. Check it works

```sh
docker compose ps                                   # running, not restarting
docker compose logs --since 10m                     # "needle API on 0.0.0.0:8517" and "schedule: ... next at" lines
docker compose exec haystack haystack status        # auth.ok true; the jobs' last runs
```

Expect, in the first minutes: `sync`, `library` and `harvest` start at once and exit 0, and the batch is
scheduled for its next time. `exit 3` means the Spotify sign-in is missing: repeat step 5.3. The first batch
runs after the first harvest (or, with a music folder, once it's embedded, which can take hours), and
`status` then shows its result. Look for the Discoveries playlist in the person's Spotify.

Then tell the person, briefly: when batches come, that liking a song is a yes and leaving it is a no, that they
sign in again about every six months (`docker compose exec haystack haystack auth`), and how to update
(`docker compose pull && docker compose up -d`).

## 8. Connect needle (if they use it)

On each computer with needle:

```sh
mkdir -p ~/.config/needle
(umask 077; ssh <server> 'cd ~/haystack && docker compose exec -T haystack haystack token show' > ~/.config/needle/token)
```

and in `~/.config/needle/config.toml`:

```toml
[haystack]
servers = ["http://<server address>:8517"]
```

Don't print the token. If needle can't connect: is the client's network in `api.allow_networks`, does the
server's firewall allow port 8517 from it, and does `docker compose logs` show a 403 (network) or 401 (token)?

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Permission denied` under `/config` or `/data` | The folders belong to another uid: set `user:` in compose.yaml, or `chown -R` them to 1000. |
| Jobs exit 3, `auth.ok` false | No or expired Spotify sign-in: the person runs `docker compose exec haystack haystack auth`. |
| Spotify says `INVALID_CLIENT: Invalid redirect URI` | The app's Redirect URI isn't exactly `http://127.0.0.1:8518/spotify/callback`. |
| "Essentia models missing" | `docker compose run --rm haystack models fetch`. |
| "needle's API is off" | No API token: `docker compose run --rm haystack token new`, then `docker compose restart`. |
| The library job fails at once | `library.root` is set but the folder isn't mounted, or is empty: check the volume line. |
| Machine sluggish for hours after the start | The first library embedding: lower `library.workers` and `threads_per_worker`, then restart. |

The full reference (every command, file locations, the scoring) is [docs/reference.md](docs/reference.md).

## Without Docker

Only if the person asks, or the machine can't run Docker: haystack runs as systemd user units, with Python 3.14
and ffmpeg. Follow [docs/reference.md, "systemd user units"](docs/reference.md#systemd-user-units-no-container).
The config lives in `~/.config/haystack/config.toml`, the commands are the same without the `docker compose ...`
prefix, `[schedule]` is ignored (the timers set the times), and the person's steps are the same, with `haystack
auth` run in an SSH session to the server.

## Working on the code

- Tests: `python -m pytest -q` in a venv with `pip install -e ".[embed,test]"`.
- How it fits together, the rules the code keeps (the library is read-only, the Deezer throttle, preview audio
  only in memory) and every command: [docs/reference.md](docs/reference.md). Terms: [GLOSSARY.md](GLOSSARY.md).
  Decisions: [docs/adr/](docs/adr/). needle's API: [docs/api.md](docs/api.md).
- `spotify.py` is the only module that talks to Spotify. Keep it that way.
- Never commit secrets, a `config.toml`, or Deezer audio or cover art.
- **The docs change with the code, in the same PR.** A change to what haystack does, a command, a setting or the
  API updates every doc that describes it: [README.md](README.md) (its settings table) and
  [config.example.toml](config.example.toml), [docs/how-it-works.md](docs/how-it-works.md) (plain language, no
  jargon), [docs/reference.md](docs/reference.md), [docs/api.md](docs/api.md) (a dated line in "Changes" for
  every `/v1` addition), [GLOSSARY.md](GLOSSARY.md) for a new term, and an ADR for a decision. ADR numbers are
  shared with needle: take the next number that is free in both repositories.
- **Every release checks the docs.** A PR that bumps `version` in `pyproject.toml` is a release. In it, add the
  version's section to [CHANGELOG.md](CHANGELOG.md) and read the docs above against every change since the last
  tag (`git log --oneline v<last>..`). Fix in that PR any line the release makes untrue, including older text
  that the new behaviour contradicts. After the merge, tag `v<version>` on `main`.
