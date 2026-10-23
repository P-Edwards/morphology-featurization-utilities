#!/usr/bin/env bash
#
# install_openphenom.sh
#
# Optional. Downloads the model code of Recursion's OpenPhenom masked
# autoencoder from Hugging Face into openphenom/openphenom/, where the
# openphenom package installed by install.sh imports it from. It is needed
# only for the autoencoder features (morphology_featurization_tools.autoencoder
# and the "autoencoder" feature); everything else works without it.
#
# The OpenPhenom code and weights are distributed under Recursion's
# Non-Commercial End User License Agreement, not under the license of this
# repository. The script shows where to read it and asks you to accept it
# before downloading anything.
#
#   bash install_openphenom.sh                    # asks you to accept the license
#   bash install_openphenom.sh --accept-license   # for non-interactive use
#
# The files come from a fixed revision of the Hugging Face repository and are
# checked against the git object ids Hugging Face lists for them. The license
# is saved next to them as openphenom/openphenom/LICENSE. The pretrained
# weights (model.safetensors, about 710 MB) are not downloaded here; the
# autoencoder downloads them into the Hugging Face cache on first use.
#
# Optional variables:
#   OPENPHENOM_BASE_URL   download from a mirror of the repository at this
#                         revision instead (the files are checked all the same)
#
# Requirements: bash, curl, and sha1sum (Linux) or shasum (macOS).

set -euo pipefail

REPO="recursionpharma/OpenPhenom"
REVISION="0f92333685f6e9f031b804c70fe246f9b05ae90d"
LICENSE_URL="https://huggingface.co/$REPO/blob/$REVISION/LICENSE"
BASE_URL="${OPENPHENOM_BASE_URL:-https://huggingface.co/$REPO/resolve/$REVISION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/openphenom/openphenom"

# Each file and its git blob id at REVISION, as listed by
# https://huggingface.co/api/models/recursionpharma/OpenPhenom/tree/<REVISION>
MANIFEST="
LICENSE             df6212a2ba6e69b6bf0c3a2803a6da11b833086f
config.json         6ad508fd75283f449d557480fa45146aa9fab2ed
config.yaml         33596f63e55ea49a83e896c13dd26eaff98b7687
huggingface_mae.py  28becb04cf80d2777e7ce6784d50a61265622b4a
loss.py             c6b46d5547ad6cf61c1bc77ae139e1ba9c68a682
mae_modules.py      2afad0b07f235f9c3dc3c9f5ba921ad310efc858
mae_utils.py        77ef394c73ede10a3764da464b76a7a666e2dff0
masking.py          763767efb65bb7864fd0c276ee93a4c6e3b261ad
normalizer.py       3d47b5913f6fd20320251387b42563f0782885ca
vit.py              db5ebe7414c87f72158a72eb5566f6758a913075
vit_encoder.py      0e2ba74f0e9b6dd40dba86d534d124ec9ad14d91
"

info()  { echo "[openphenom] $*"; }
error() { echo "[openphenom] ERROR: $*" >&2; exit 1; }

accepted=0
for arg in "$@"; do
    case "$arg" in
        --accept-license) accepted=1 ;;
        -h|--help) sed -n '3,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) error "Unknown argument '$arg'. Usage: bash install_openphenom.sh [--accept-license]" ;;
    esac
done

if command -v sha1sum >/dev/null 2>&1; then
    sha1() { sha1sum | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha1() { shasum -a 1 | cut -d' ' -f1; }
else
    error "Neither sha1sum nor shasum was found."
fi

# The git blob id of a file: SHA-1 of "blob <size>\0" followed by its bytes.
blob_id() {
    local size
    size="$(wc -c < "$1" | tr -d '[:space:]')"
    { printf 'blob %s\0' "$size"; cat "$1"; } | sha1
}

all_present() {
    local name oid
    while read -r name oid; do
        [[ -z "$name" ]] && continue
        [[ -f "$TARGET/$name" && "$(blob_id "$TARGET/$name")" == "$oid" ]] || return 1
    done <<< "$MANIFEST"
}

if all_present; then
    info "The OpenPhenom model code at revision $REVISION is already in $TARGET."
    exit 0
fi

if [[ "$accepted" -ne 1 ]]; then
    echo "OpenPhenom is distributed under Recursion's Non-Commercial End User"
    echo "License Agreement, which you can read at"
    echo
    echo "    $LICENSE_URL"
    echo
    if [[ ! -t 0 ]]; then
        error "No terminal to ask for acceptance. After reading the license, run again with --accept-license."
    fi
    read -r -p "Do you accept this license? Type 'yes' to download: " reply
    if [[ "$reply" != "yes" ]]; then
        info "License not accepted; nothing was downloaded."
        exit 1
    fi
fi

command -v curl >/dev/null 2>&1 || error "curl was not found."

# Everything is downloaded and checked in a temporary directory first, so a
# failed download leaves openphenom/openphenom/ as it was.
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

while read -r name oid; do
    [[ -z "$name" ]] && continue
    info "Downloading $name"
    curl -fsSL --retry 3 -o "$staging/$name" "$BASE_URL/$name" \
        || error "Could not download $BASE_URL/$name"
    found="$(blob_id "$staging/$name")"
    [[ "$found" == "$oid" ]] \
        || error "$name does not match revision $REVISION (expected $oid, got $found)."
done <<< "$MANIFEST"

mkdir -p "$TARGET"
while read -r name oid; do
    [[ -z "$name" ]] && continue
    mv -f "$staging/$name" "$TARGET/$name"
done <<< "$MANIFEST"

info "Installed the OpenPhenom model code (revision $REVISION) in $TARGET."
info "The pretrained weights are downloaded from Hugging Face on first use."
