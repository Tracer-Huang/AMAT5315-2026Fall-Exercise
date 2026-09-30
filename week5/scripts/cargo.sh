#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ -x .toolchains/nightly-2026-09-05/bin/rustc ]; then
 export RUSTC="$PWD/.toolchains/nightly-2026-09-05/bin/rustc"
 export DYLD_LIBRARY_PATH="$PWD/.toolchains/nightly-2026-09-05/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
 export RUSTDOC="$PWD/.toolchains/nightly-2026-09-05/bin/rustdoc"
 exec "$PWD/.toolchains/nightly-2026-09-05/bin/cargo" "$@"
fi
exec cargo +nightly-2026-09-05 "$@"
