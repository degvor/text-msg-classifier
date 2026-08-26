#!/usr/bin/env bash
# Download the pinned jars into PySpark's own jars directory, verifying SHA-1.
# Living in $(pyspark)/jars means they are always on the classpath, so no
# --packages / --jars flag is needed and startup is fully offline.
set -euo pipefail

MANIFEST="${1:?usage: fetch-jars.sh <manifest> <dest>}"
DEST="${2:?usage: fetch-jars.sh <manifest> <dest>}"
MIRROR="${MAVEN_MIRROR:-https://repo1.maven.org/maven2}"

mkdir -p "$DEST"
while read -r sha path; do
    [[ -z "${sha:-}" || "$sha" == \#* ]] && continue
    name="$(basename "$path")"
    echo "==> $name"
    curl -fSL --retry 5 --retry-delay 2 --retry-connrefused \
         --max-time 300 -o "$DEST/$name" "$MIRROR/$path"
    echo "$sha  $DEST/$name" | sha1sum -c - >/dev/null \
        || { echo "CHECKSUM MISMATCH for $name" >&2; exit 1; }
done < "$MANIFEST"
echo "all jars verified into $DEST"
