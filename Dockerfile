# haystack's server image: `haystack serve`, needle's API plus the scheduled jobs (README.md, "Install").
# linux/amd64 only: the pinned essentia-tensorflow ships a Linux wheel for x86_64 and CPython 3.14, nothing for ARM.
# The Essentia models are not in the image (they are NonCommercial, see NOTICE): `haystack models fetch` downloads
# them into the /data volume.
FROM python:3.14-slim

# ffmpeg decodes audio (embed.py calls /usr/bin/ffmpeg); tini reaps the jobs' children and passes signals on;
# tzdata makes TZ work, since [schedule] times are local time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir --root-user-action=ignore ".[embed]"

RUN useradd --uid 1000 --create-home haystack \
 && mkdir -p /config /data \
 && chown haystack:haystack /config /data
USER haystack

ENV HAYSTACK_CONFIG_DIR=/config \
    HAYSTACK_DATA=/data \
    HAYSTACK_STATE=/data/state \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3
VOLUME ["/config", "/data"]
EXPOSE 8517

ENTRYPOINT ["tini", "--", "haystack"]
CMD ["serve"]

LABEL org.opencontainers.image.title="haystack" \
      org.opencontainers.image.description="Self-hosted music discovery for Spotify, driven by how your music sounds" \
      org.opencontainers.image.source="https://github.com/kurtlieber/haystack" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"
