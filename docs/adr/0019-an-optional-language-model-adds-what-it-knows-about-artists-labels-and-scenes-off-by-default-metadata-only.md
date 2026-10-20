# An optional language model adds what it knows about artists, labels and scenes; off by default; metadata only

The fingerprint hears how a song sounds, and Deezer's facts (ADR 0017) say where it comes from, but neither knows
that an act is mainstream rap rather than the listener's scene, which labels and scenes sit next to each other, or
that a remix is a cash-in. A language model can't hear, but it has read a great deal about music. From 0.7.1
haystack can ask one, by hand, and only to measure whether it helps: it has no say in a batch until the
measurements say it should.

1. **Off by default, by hand, read-only.** `[llm] enabled = false`. While it is false, `haystack llm judge` and
   `haystack llm suggest --dry-run` exit 2 before a key is read or a provider imported, and no job uses the model.
   `judge --dry-run` builds the requests and prints the first one without sending anything, so the listener can
   read exactly what would go. Both commands open the database read-only and need no Spotify sign-in.
2. **Metadata only.** For each song: artist, title, album, year, record label (ADR 0017), its top three Discogs
   styles with their scores, and the name of the source that found it. With each request, a taste profile: the 30
   artists with the most stars and likes (songs, mean stars), up to 50 artists of Passes and Dislikes, the 15 top
   styles by share of the liked songs, and the Traits' usual ranges (25th to 75th percentile). Never audio,
   Spotify or Deezer ids, account names or tokens: the songs are numbered 1 to 40 within each request.
3. **Blind and leave-one-out.** The judged set is the re-ranker's labels (`rerank.labels`, soft ones included,
   since the re-ranker trains on them), sorted and shuffled with a fixed seed into requests of 40. A verdict never
   leaves the host, and the profile sent with a request is computed without that request's songs and every other
   copy of them (the same song key), so the model can't read a song's verdict off the profile.
4. **What is measured.** The model answers against a JSON schema (`{"tracks": [{"id", "chance", "reason"}]}`, every
   object closed and every field required): a chance from 0 to 1 that the listener keeps the song (3 stars or
   more), and one short reason. haystack prints the AUC of the chance alone over the hard labels it answered for;
   the re-ranker's leave-one-out AUC over the same labels without and with the chance's logit (clipped to 0.01 to
   0.99 and standardised) as one more feature, with stage 1's AUC alongside; and the tokens and their cost at the
   listener's `price_in_per_mtok` and `price_out_per_mtok`. An unknown id, or a second answer for the same id, is
   ignored, a chance outside 0 to 1 is clamped, and a song with no answer is counted and left out of both AUCs.
5. **The decision rule, fixed before the first run.** Phase 2's rerank job (the chance as a re-ranker feature at
   batch time, under a weekly budget) is built only if the chance alone reaches AUC 0.70, or adds 0.05 to the
   re-ranker's leave-one-out AUC. The seeds job (the model names songs, each found on Deezer and offered as source
   `llm_suggest`) is built if the rerank bar is met, or on its own if more than 60 % of 100 suggestions resolve on
   Deezer. `haystack llm suggest --dry-run` measures that: one request with the whole profile, then one Deezer
   search per suggestion through the shared throttle and the harvest's own matching, counting what resolves, what
   has a preview and what haystack already knows. It offers and stores nothing.
6. **Two providers, imported lazily.** `anthropic`, through Anthropic's Python SDK as the optional extra `[llm]`
   (`anthropic>=1,<2`): model `claude-opus-5-5` by default, structured output through `output_config.format`, no
   `thinking` parameter (the model decides how much to think), streamed, `stop_reason` checked before the content
   is read, and the SDK's own retries. `openai_compatible`: a POST with `requests` to
   `{base_url}/chat/completions` with a strict JSON-schema `response_format`, for Ollama, llama.cpp's server, vLLM
   and other local servers, so a listener who wants no third party can still run it; a bearer key is sent only if
   one is set. The key is the `llm-api-key` secret (a file, or 1Password through `onepassword.llm_item`).
   `--debug` logs each request and answer; the key is never logged.
7. **A refused or cut-off request is missing, not sent elsewhere.** A reply that stops for `refusal` or
   `max_tokens` (`finish_reason` `length` on the other provider) leaves its songs without a chance. The run counts
   them, and the tokens spent still count towards its cost.

**Why the model reads metadata and not the preview.** The embedding already hears the preview, on the listener's
own machine and for free. What it can't know is what has been written about the artist, the label and the scene,
and that is what a text model is good at. Text is also small: about 3,000 tokens in for 40 songs.

**Why `max_tokens` is 16,000 and not about 8,000 as first sketched.** On Claude Opus 5.5 the model's thinking
counts against `max_tokens` and can't be switched off, and 40 answers with reasons take a couple of thousand tokens
on their own. A reply cut off at the limit is lost whole, since half a structured answer can't be read, so the limit
sits where a normal reply doesn't reach it. The bound is visible in money: 40 songs cost at most about $0.33 at
Opus 5.5's prices ($4 and $20 per million tokens in and out on 2026-10-20), and each run prints what it spent.

**Why no fallback model on a refusal.** The API can retry a refused request on another model. That would bill at
the other model's rates and send the listener's data to a model they didn't choose, and a refusal is unlikely on
song metadata. A refused request is counted and reported instead.

**Why the judged set is the re-ranker's labels.** The leave-one-out comparison can only be made on the re-ranker's
own labels, and the chance alone is then measured on the same songs, so the two numbers can be set side by side.
The eval's verdict queries (ADR 0009) are the same judged offers and Calibrate picks, behind a loader that scores
every variant first.

**Why a profile and not the listener's whole history.** A request has to fit, and cost, the same at 50 labels as
at 5,000. Artists with counts and mean stars, styles by share and Trait ranges summarise the taste in about 400
tokens, and they are the facts a reader of music would ask for. Song titles stay out of the profile, which also
keeps a judged song's own title from reaching the model with its verdict attached.

**Considered Options**:

- No language model. The re-ranker learns from the listener's verdicts alone, which takes months at a few a week,
  and never learns what it hasn't been told about an artist it hasn't met.
- A multimodal model that hears the preview. Rejected: it costs far more per song, sends audio off the machine, and
  hearing is the embedding's job (a separate experiment measures better embeddings, on the listener's own
  machine).
- A text-audio joint model such as MuQ-MuLan, run locally, that places songs and words in one space. Deferred: it
  could describe a song in words without sending anything, but it knows sounds, not scenes, and the embedding
  experiment comes first.
- An optional text model over metadata, by hand and measured before any job uses it, through Anthropic's SDK or
  any OpenAI-compatible server (chosen).

**Consequences**:

- A new optional extra, `[llm]` (`anthropic`). The core install and the Docker image are unchanged; the image gains
  the extra only if phase 2 is built. In the meantime `openai_compatible` needs nothing extra.
- A new config block, `[llm]` (`enabled`, `provider`, `model`, `base_url`, `min_profile_tracks`,
  `price_in_per_mtok`, `price_out_per_mtok`), and a new secret, `llm-api-key` (`onepassword.llm_item` and
  `llm_field`). Phase 2's settings (`jobs`, `rerank_top`, `seeds_per_week`, `weekly_budget_usd`, `hold_below`),
  its `llm_runs` table and the cards' `why.knowledge` come with phase 2, if it is built.
- No schema change and no `/v1` change.
- The re-ranker's design matrix is built by `rerank.design`, for both the fit and this measurement;
  `baseline.compute` can leave tracks out; the harvest's Deezer lookup is a module function, so the suggest dry
  run resolves names exactly as the harvest does.
- A run costs money at a hosted provider: at most about $0.33 per 40 judged songs with Claude Opus 5.5, much less
  with a smaller model, nothing with a local one. Phase 1 runs twice: now, and again at about 200 labels.
- A fresh installation behaves as 0.7.0 did: `llm.enabled` is false, so nothing is imported or sent, and even when
  it is on, `judge` sends nothing until there are `min_profile_tracks` (30) liked and rated songs.
