# PostgreSQL with CDC support (wal2json)
FROM postgres:15

# Install wal2json for CDC logical decoding
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-15-wal2json \
    && rm -rf /var/lib/apt/lists/*
