# Embedding comparison

`embedding_compare.py` asks one question: does a self-supervised music model tell your yes from your no better
than the Discogs-EffNet vectors haystack uses today? It is an experiment, not a feature. It reads haystack's
database read-only, changes nothing in haystack, and writes only vectors into a folder of its own. Whether
haystack ever adopts another model is decided by the rule below, on two runs of this script (ROADMAP.md,
"Better ears").

## What it compares

| name | what it is | licence of the weights |
|---|---|---|
| `effnet512` | haystack's own 512-d vectors, read from the database and the library's npz files (no recomputation) | Essentia models: CC BY-NC-SA 4.0 / CC BY-NC-ND 4.0 (see NOTICE) |
| `effnet512-aligned` | the same, with previews mapped into the library's space as scoring does (only when the alignment is in use, ADR 0012) | as above |
| `mert-95m`, `mert-330m` | [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) and [-330M](https://huggingface.co/m-a-p/MERT-v1-330M), 24 kHz | CC BY-NC 4.0 |
| `muq-large` | [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter), 24 kHz | CC BY-NC 4.0 |
| `clap-music` | [laion/larger_clap_music](https://huggingface.co/laion/larger_clap_music), 48 kHz | Apache 2.0 |

Licences as published when this was written; check the model pages. Nothing is redistributed: the weights download
from the publisher into your Hugging Face cache, as haystack's Essentia models download at install. MERT runs model
code from the Hub (`trust_remote_code`), so the script pins every model to a fixed revision.

MERT and MuQ give one vector per layer. Each clip yields two: the last layer averaged over time (`mert-95m`) and the
mean of the middle third of the layers (`mert-95m-mid`). CLAP's feature extractor picks parts of a long clip at random
(its "fusion" mode), so its vectors differ very slightly from one run to the next.

## The tracks

`select` picks them once and writes `tracks.json`:

- the labelled tracks: the re-ranker's labels (`rerank.labels()`) plus the eval's kept and turned-down offers;
- your liked songs that have a Deezer preview;
- your liked songs' local files, as the same 30 s windows haystack embedded;
- 600 candidates from the pool, an equal number from each tenth of the stage-1 score.

The positives are the liked previews plus the liked files. Every score leaves out the track's own song (by Spotify
id and by artist and title), as `haystack eval` does.

## Setup

Use a venv of its own; torch does not belong in haystack's.

```sh
python3 -m venv ~/haystack-dev/venv-embed
~/haystack-dev/venv-embed/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio torchvision
~/haystack-dev/venv-embed/bin/pip install -r scripts/experiments/requirements.txt
~/haystack-dev/venv-embed/bin/pip install -e .      # haystack itself, for its database and Deezer code
```

Run it against a copy of your database rather than the live one: set `HAYSTACK_DATA`, `HAYSTACK_STATE` and
`HAYSTACK_CONFIG_DIR` to a scratch copy. It opens the database read-only either way. It also needs `ffmpeg`.

## Running it

```sh
py=~/haystack-dev/venv-embed/bin/python
$py scripts/experiments/embedding_compare.py --out ~/haystack-experiments select
nice -n 19 $py scripts/experiments/embedding_compare.py --out ~/haystack-experiments embed \
    --models mert-95m,muq-large,clap-music,mert-330m --pause 04:30-07:30
$py scripts/experiments/embedding_compare.py --out ~/haystack-experiments report
```

- `embed` runs the models one after the other and saves each model's vectors as `<name>.npz` every 50 clips. Stop
  it any time; it carries on where it left off. Run `report` whenever you like: it compares every model embedded
  so far on the tracks they all have. If a model has stopped too early for that to leave anything to compare, it
  says so and lists what each model has.
- On a CPU, plan for roughly 3 to 4.5 s per 30 s clip for MERT-95M and MuQ, 5 to 8 s for MERT-330M and well
  under a second for CLAP (measured at nice 19 on 8 threads), times a few thousand clips (one per preview, about
  nine per liked file). Batching does not help on a
  CPU, so clips go through one at a time; `--threads` sets torch's CPU threads.

## Deezer and your files

- The previews download from Deezer into memory, are embedded, and are dropped when the run ends. No audio is
  written to disk. Every model uses them, so they stay in memory for the whole run: about half a megabyte each,
  roughly 0.5 GB per thousand previews, on top of the model.
- At most one Deezer request every 1.2 s. The script uses haystack's own throttle file, so together with a running
  haystack it never goes faster than haystack's limit. That throttle file, in haystack's state folder, is the one
  file it touches outside `--out` (and the Hugging Face cache).
- `--pause 04:30-07:30` stops Deezer requests in that window, in UTC; set it to the hours around your daily
  harvest (`haystack-harvest.timer`), so the two never compete for Deezer.
- Your music files are only read.
- **`--out` holds your data**: `tracks.json` lists your liked songs, your verdicts and the paths of your files, and
  the npz files hold vectors of your music. Keep the folder private; do not commit it.

## The decision rule

A model goes further, towards the option ROADMAP.md describes under "Better ears", only if all of these hold, on
this run and again on a later run with about 200 labels:

- its AUC beats `effnet512`'s by at least 0.05 on both `raw` (the best cosine to any positive) and `topk_mean`
  (the mean of the best few);
- its recall@15 (the share of held-out liked songs that rank in the pool sample's top 15, ranked by `raw`) is
  not below `effnet512`'s;
- it costs at most 5 s per 30 s clip.

`report` prints a 90 % interval for each AUC gain (a paired bootstrap over the labels) and the AUC's standard
error at your label count. With a hundred labels that error is about 0.05, the size of the bar itself, which is
why one run is never enough. The self-retrieval column (how often a song's preview finds its own local file first)
shows how far apart a model hears a preview and a full file; a model that wins may need its own alignment map.
