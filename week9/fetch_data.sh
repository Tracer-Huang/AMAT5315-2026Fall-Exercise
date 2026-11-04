#!/bin/sh
set -eu
mkdir -p promedus
fetch() {
  dest=$1 expected=$2 url=$3
  if test -f "$dest"; then
    actual=$(shasum -a 256 "$dest" | cut -d ' ' -f 1)
    test "$actual" = "$expected" || { echo "Checksum mismatch: $dest; preserve and inspect it" >&2; exit 1; }
    return
  fi
  tmp=$(mktemp "${dest}.partial.XXXXXX")
  trap 'rm -f "$tmp"' EXIT HUP INT TERM
  curl --fail --silent --show-error --location --retry 3 --retry-all-errors --connect-timeout 10 --max-time 45 "$url" -o "$tmp"
  actual=$(shasum -a 256 "$tmp" | cut -d ' ' -f 1)
  test "$actual" = "$expected" || { echo "Checksum mismatch from $url" >&2; exit 1; }
  mv "$tmp" "$dest"
  trap - EXIT HUP INT TERM
}
asia=https://github.com/TensorBFS/TensorInference.jl/raw/935ed86f8e473f523cac9461c45a1f470e74f0f9/examples/asia-network/model.uai
base=https://github.com/mroavi/uai-2014-inference-competition/raw/22005844423f5bca62798548df808cc418fc0d7d/uai2014
fetch asia.uai 0d78314c96b1cbbfd94682f0e99f73461cb50a935a1da85cbc814c70e3fff32d "$asia"
fetch promedus/Promedus_14.uai f213b76a7f145a06f39daa9061408981dbd525e4ce2e5156edb83a7073d71349 "$base/PR/Promedus_14.uai"
fetch promedus/Promedus_14.uai.evid e345a4ebecf02cdd9fdaf42e7456afa5f6ecafa989b6a3c15ef525abe736a403 "$base/PR/Promedus_14.uai.evid"
fetch promedus/Promedus_14.uai.PR d52ca345f79e79f0385f7bd91a3c2e3521eff165b3dcb7b29be1e56ca5f96dfc "$base/PR/Promedus_14.uai.PR"
fetch promedus/Promedus_14.uai.MAR 06e3145bb59fd0a22af235af43bc7c45bee71a764647e152314bdded4f0d256f "$base/MAR/Promedus_14.uai.MAR"
echo 'All pinned input checksums verified.'
