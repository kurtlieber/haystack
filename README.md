<p align="center">
  <img alt="haystack: finds the needle" src="docs/brand/png/haystack-banner.png" width="800">
</p>

haystack finds new music for your Spotify account by **how it sounds**. It listens to the songs you already love
(your Liked Songs, your ratings and, if you have one, your own music folder), then looks through thousands of
candidates from Deezer and ListenBrainz. Once a week it puts the closest matches into a private **Discoveries**
playlist in your Spotify account.

You don't have to do anything special: just listen. Like a song in Spotify and haystack learns from it. Let one sit
in Discoveries and it counts as a no. The next week's picks get better.

haystack is a small server, with no screen of its own. It runs on an always-on machine (a home server, a NAS, a
cheap VPS) and you set it up once in a single settings file, `config.toml`. Its companion app,
[needle](#needle), is where you rate songs and play what haystack found.

- [The easiest setup: ask your AI agent](#the-easiest-setup-ask-your-ai-agent)
- [What you need](#what-you-need)
- [Install with Docker](#install-with-docker)
- [Settings: config.toml](#settings-configtoml)
- [Day to day](#day-to-day)
- [needle](#needle)
- [Deezer notice](#deezer-notice)
- [Roadmap](#roadmap)
- [License](#license)

## The easiest setup: ask your AI agent

If you use an AI coding agent that can run commands, such as Claude Code, point it at
[AGENTS.md](AGENTS.md):

> Set up haystack for me by following https://github.com/kurtlieber/haystack/blob/main/AGENTS.md

It asks a few questions (where your music is, when you want new songs, your time zone) and does the rest. A few
steps stay yours: creating the Spotify app, typing in its two keys, and signing in to Spotify. The agent gives you
those as a short list up front. AGENTS.md has only been tested with Claude Code, but any agent that can run shell
commands should work.

Prefer to do it yourself? Read on.

## What you need

- **An always-on Linux machine with Docker**, x86_64 (Intel or AMD; ARM boards like the Raspberry Pi aren't
  supported yet). It needs about 2 GB of disk for the image, plus room for its data. It should stay on: Spotify
  only shows haystack your last 50 plays, so it checks every hour.
- **A Spotify account** and a free Spotify developer app (step 2 below), so haystack can manage its playlists.
- **Optionally, your own music folder** (`.m4a` files). With one, haystack learns your taste from far more songs
  than your Spotify likes. It only ever reads the folder. The first pass over a big library takes hours of CPU.

## Install with Docker

**1. Get the two files.** Make a folder for haystack and put the compose file and your settings in it:

```sh
mkdir -p ~/haystack/config ~/haystack/data && cd ~/haystack
curl -fsSLO https://raw.githubusercontent.com/kurtlieber/haystack/main/compose.yaml
curl -fsSL https://raw.githubusercontent.com/kurtlieber/haystack/main/config.example.toml -o config/config.toml
```

Open `compose.yaml` and set `TZ` to your time zone (for example `America/New_York`). If you have a music folder,
uncomment its line there and set `library.root = "/music"` in `config/config.toml`. If `id -u` doesn't print
`1000`, uncomment `user:` and put your ids there, so the files haystack writes stay yours.

**2. Create a Spotify app.** On the [Spotify developer dashboard](https://developer.spotify.com/dashboard), choose
**Create app**:

- Name and description: anything, such as "haystack".
- Redirect URI: exactly `http://127.0.0.1:8518/spotify/callback`.
- Which API: **Web API**.

Save, open the app's **Settings**, and keep its **Client ID** and **Client secret** at hand.

**3. Give haystack the app's keys.** Each command asks for the value without showing it:

```sh
docker compose run --rm haystack secret set spotify-client-id
docker compose run --rm haystack secret set spotify-client-secret
```

**4. Download the sound models** (Essentia's; see [License](#license) for why they aren't in the image):

```sh
docker compose run --rm haystack models fetch
```

**5. Sign in to Spotify:**

```sh
docker compose run --rm haystack auth
```

It prints a link. Open it in any browser, on your laptop or your phone, and agree. Spotify then sends you to a
page that **won't load**. That's expected. Copy the whole address from the browser's address bar (it starts with
`http://127.0.0.1:8518/spotify/callback?code=`) and paste it into the terminal within a few minutes.

**6. If you'll use needle**, make its key now ([needle](#needle) says where it goes):

```sh
docker compose run --rm haystack token new
```

**7. Start it:**

```sh
docker compose up -d
docker compose logs -f      # watch it work; Ctrl-C stops watching, not haystack
```

haystack starts collecting candidates right away. Your first Discoveries batch comes when that first collection
is done; with a music folder, it waits until the folder has been processed. After that, a new batch comes every
Monday at 08:00, or whenever you set it to.

## Settings: config.toml

Everything haystack does is set in `config/config.toml`. The file you downloaded lists the common settings with
their defaults. Keep only what you change: anything you leave out keeps its default. After a change, run
`docker compose restart`.

The ones people change most:

| Setting | Default | What it does |
|---|---|---|
| `batch.size` | `15` | How many new songs each batch brings. |
| `schedule.batch` | `"Mon 08:00"` | When a new batch arrives. `"Fri 18:00"`, `"Mon,Thu 08:00"` or `"daily 07:00"` work too, in your `TZ`. |
| `library.root` | `""` (none) | Your music folder. In Docker this is `"/music"`, mounted in `compose.yaml`. |
| `library.workers`, `library.threads_per_worker` | `6`, `3` | How much CPU the first pass over your library gets. On a small VPS, try `1` and `2`. |
| `batch.expiry_days` | `14` | A discovery you don't like leaves Discoveries after this many days. |
| `batch.max_per_artist` | `1` | Songs per artist in one batch. |
| `spotify.discoveries_name` | `"Discoveries"` | The playlist's name. |
| `spotify.to_rip` | `false` | Also keep a **To Rip** playlist of songs you rate 3 stars or more that aren't in your music folder, to buy later. |
| `harvest.listenbrainz_contact` | `""` | Your email or website. ListenBrainz asks API users for one. |
| `harvest.retired_sources` | `[]` | Candidate sources to stop using. `haystack sources` shows how well each one does for you. |
| `api.allow_networks` | private ranges | Which networks may reach needle's API. Narrow it to your home network and VPN. In Docker, keep `172.16.0.0/12` if needle runs on the same machine. |
| `alert.enabled` | `false` | Send a webhook when a job fails or the Spotify sign-in is running out. Set the `alert-webhook-url` secret first. |

For example, 25 songs every Friday evening, with a music folder:

```toml
[batch]
size = 25

[schedule]
batch = "Fri 18:00"

[library]
root = "/music"
```

`[schedule]` also sets when the other jobs run: `sync` (hourly), `library` (daily 03:15) and `harvest` (daily
05:00). Any of them can be `"off"`. Every other setting, including the scoring options, is in
[docs/reference.md](docs/reference.md).

## Day to day

| To... | Run |
|---|---|
| See what haystack is doing | `docker compose exec haystack haystack status` |
| Read the log | `docker compose logs -f` |
| Get a batch right now | `docker compose exec haystack haystack batch` (add `--size 30` for a bigger one) |
| Update haystack | `docker compose pull && docker compose up -d` |
| Back up | copy the `config` and `data` folders (`config/secrets/` holds your keys) |

**Sign in again about every six months.** Spotify's sign-in lasts about six months. haystack tells you a month
ahead (in `status`, and as an alert if alerts are on). Run `docker compose exec haystack haystack auth` and
paste the address back, as in step 5.

**What counts as yes and no.** A Discoveries song you like in Spotify is a yes: it leaves Discoveries at the next
hourly check, and with `to_rip` on, it goes to To Rip if it isn't in your music folder. A song still in
Discoveries when the next batch comes is a no, played or not, because the playlist is meant to be listened
through. Songs added less than a day before are let off. Ratings in needle (1 to 5 stars, Pass, Dislike) count
too, and say more than a like.

Without Docker, haystack runs as systemd user services on any Linux machine with Python 3.14 and ffmpeg. See
[docs/reference.md](docs/reference.md#running-it), which also covers every command, how the scoring works and
where the files live.

## needle

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/svg/needle-and-haystack-on-dark.svg">
    <img alt="needle and haystack" src="docs/brand/svg/needle-and-haystack-on-light.svg" width="600">
  </picture>
</p>

[needle](https://github.com/kurtlieber/needle) is the desktop app that goes with haystack. It plays your
Discoveries and lets you rate each song in a keystroke. haystack doesn't need it, but ratings teach haystack much
faster than likes alone.

needle talks to haystack over your network, on port 8517. To connect a needle:

1. In needle's `~/.config/needle/config.toml`, set `servers = ["http://<your server>:8517"]` under `[haystack]`.
2. Copy haystack's key into needle's token file:

   ```sh
   (umask 077; mkdir -p ~/.config/needle
    ssh <your server> 'cd ~/haystack && docker compose exec -T haystack haystack token show' > ~/.config/needle/token)
   ```

Keep the API on your home network or a VPN such as Tailscale. It isn't meant to face the internet.

## Deezer notice

Every candidate's audio is its 30-second Deezer preview, and most candidates come from Deezer's charts, editorial
picks, artist radio and related artists, through Deezer's public API (no key). Deezer's terms may not allow this
kind of automated use, so decide for yourself before you run haystack. haystack keeps it gentle: at most one
request per 1.1 seconds, preview audio held only in memory (never written to disk), and only the derived numbers
kept, on your own machine. `harvest.sources` can drop the Deezer sources, but ListenBrainz candidates still need
their Deezer previews.

Your taste never comes from Spotify's audio: only your own files and the Deezer previews are analysed.

## Roadmap

- **More places to find music.** Deezer and ListenBrainz are the first candidate sources. More will be added.
- **needle and haystack stay two programs.** This is a deliberate design choice. haystack runs headless on one
  server (a VPS or a machine at home) and is the single source of truth for your ratings. needle runs on as many
  computers as you like, and they all share that one store. Merging them would mean one copy of your taste per
  computer, and plays missed whenever a laptop sleeps.
- **No GUI for haystack.** `config.toml` is how you set it up, and stays that way.
- **haystack's settings in needle.** Longer term, needle may get a screen for the common haystack settings.
- **Spotify sign-in from needle.** needle would sign in to Spotify on your computer (PKCE) and hand the sign-in
  to haystack, with no address to paste back.

## License

haystack is AGPL-3.0-or-later ([LICENSE](LICENSE)). It downloads Essentia's models on setup instead of shipping
them: they are for non-commercial use only (Creative Commons BY-NC), so commercial use needs a license from the
MTG. [NOTICE](NOTICE) lists what haystack uses and on which terms, Deezer's included. Security reports:
[SECURITY.md](SECURITY.md).
