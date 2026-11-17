#!/usr/bin/env bash
# Fetch a large file over many ranged connections at once.
#
# ftp.ebi.ac.uk throttles each connection to roughly 300-400 KiB/s but does not
# appear to cap a client in total: four extra streams each ran at full speed
# while the original download was still going. One stream puts the ChEMBL
# SQLite release nearly four hours away; this splits it.
#
#   scripts/parallel_fetch.sh <url> <output> [parts] [attempts]
#
# Each part is its own file and is appended to, so interrupting and re-running
# resumes rather than starting over. The parts are concatenated only once they
# are all complete, and the script refuses to assemble a short part.

set -u

URL="${1:?url}"
OUT="${2:?output path}"
PARTS="${3:-12}"
ATTEMPTS="${4:-8}"

dir=$(dirname "$OUT")
base=$(basename "$OUT")
mkdir -p "$dir"
cd "$dir" || exit 1

total=$(curl -sSLI "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
if [ -z "${total:-}" ] || [ "$total" -le 0 ] 2>/dev/null; then
  echo "could not read Content-Length for $URL" >&2
  exit 1
fi
echo "total $((total / 1024 / 1024)) MiB in $PARTS parts"

# An existing whole-file download seeds part 0, so nothing already fetched is
# thrown away.
chunk=$(( (total + PARTS - 1) / PARTS ))
if [ -f "$base" ] && [ ! -f "$base.part0" ]; then
  have=$(stat -c %s "$base")
  if [ "$have" -gt 0 ] && [ "$have" -le "$chunk" ]; then
    mv "$base" "$base.part0"
    echo "reusing $((have / 1024 / 1024)) MiB already downloaded as part 0"
  fi
fi

# Retrying is the outer loop's job, never curl's. `curl --retry` re-issues the
# same Range request and the shell appends the body a second time, so a dropped
# connection leaves a part nearly twice its length -- measured once, 919 MiB
# where 480 was wanted. Each pass here recomputes the offset from what is
# actually on disk, which makes resuming idempotent.
for attempt in $(seq 1 "$ATTEMPTS"); do
  pids=""
  for i in $(seq 0 $((PARTS - 1))); do
    start=$((i * chunk))
    end=$((start + chunk - 1))
    [ "$end" -ge "$total" ] && end=$((total - 1))
    part="$base.part$i"
    have=0
    [ -f "$part" ] && have=$(stat -c %s "$part")
    want=$((end - start + 1))
    if [ "$have" -gt "$want" ]; then
      echo "part $i is $have bytes, wanted $want - truncating and refetching" >&2
      : > "$part"
      have=0
    fi
    [ "$have" -eq "$want" ] && continue
    curl -sSL -r "$((start + have))-${end}" "$URL" >> "$part" &
    pids="$pids $!"
  done
  [ -z "$pids" ] && break
  for p in $pids; do wait "$p"; done
  echo "pass $attempt done"
done

# Assemble only when every part is exactly the length it should be.
short=0
for i in $(seq 0 $((PARTS - 1))); do
  start=$((i * chunk))
  end=$((start + chunk - 1))
  [ "$end" -ge "$total" ] && end=$((total - 1))
  want=$((end - start + 1))
  part="$base.part$i"
  have=0
  [ -f "$part" ] && have=$(stat -c %s "$part")
  if [ "$have" -ne "$want" ]; then
    echo "part $i: $have / $want bytes" >&2
    short=1
  fi
done
if [ "$short" -ne 0 ]; then
  echo "incomplete - re-run to resume" >&2
  exit 1
fi

cat $(for i in $(seq 0 $((PARTS - 1))); do echo "$base.part$i"; done) > "$base"
got=$(stat -c %s "$base")
if [ "$got" -ne "$total" ]; then
  echo "assembled $got bytes, expected $total" >&2
  exit 1
fi
rm -f "$base".part*
echo "done: $base ($((got / 1024 / 1024)) MiB)"
