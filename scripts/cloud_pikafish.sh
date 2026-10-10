#!/usr/bin/env bash
# Build the official corpus release on Linux; do not redistribute its binary.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_revision=4c17cee11f888ae1d48a9494f2e2239f019f0a1f
task_vendor="$task_root/vendor/Pikafish"
if [[ ! -d "$task_vendor" ]]; then
  mkdir -p "$task_root/vendor"
  git clone --depth 1 --branch Pikafish-2026-09-06 \
    https://github.com/official-pikafish/Pikafish.git "$task_vendor"
fi
if [[ "$(git -C "$task_vendor" rev-parse HEAD)" != "$task_revision" ]]; then
  printf '%s\n' 'Existing engine checkout differs from pinned official release; stopping.' >&2
  exit 2
fi
if [[ -n "$(git -C "$task_vendor" diff --name-only)" ]]; then
  printf '%s\n' 'Engine checkout has local source changes; stopping.' >&2
  exit 2
fi
make -C "$task_vendor/src" -j "${BUILD_JOBS:-2}" build ARCH="${PIKAFISH_ARCH:-x86-64}"
test -x "$task_vendor/src/pikafish"
test -f "$task_vendor/src/pikafish.nnue"
printf 'export PIKAFISH_PATH=%q\n' "$task_vendor/src/pikafish"
printf '%s\n' 'Use this path on this server. Cloud plan records exact binary and NNUE checksums.'
