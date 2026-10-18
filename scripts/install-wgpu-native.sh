#!/usr/bin/env bash
set -euo pipefail

# Pin the native C ABI used by the Java FFM binding. No system installation is modified.
if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "This installer is pinned to Linux x86_64; provide a matching 29.0.1.1 library explicitly on other platforms." >&2
  exit 1
fi
install_dir="${1:-$(pwd)/.native/wgpu-native}"
archive_dir="$(mktemp -d /tmp/jax4j-wgpu-native.XXXXXX)"
trap 'rm -rf -- "$archive_dir"' EXIT
url='https://github.com/gfx-rs/wgpu-native/releases/download/v29.0.1.1/wgpu-linux-x86_64-release.zip'
sha='95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a'
curl --fail --location --silent --show-error "$url" -o "$archive_dir/wgpu.zip"
printf '%s  %s\n' "$sha" "$archive_dir/wgpu.zip" | sha256sum --check --status
mkdir -p "$install_dir"
unzip -q -o "$archive_dir/wgpu.zip" -d "$install_dir"
printf 'Installed wgpu-native 29.0.1.1. Configure the library path:\nexport JAX4J_WGPU_NATIVE_PATH=%q\n' "$install_dir/lib/libwgpu_native.so"
