#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_LINK="$ROOT/.env"
ENV_FILE="$ENV_LINK"
if [[ -L "$ENV_LINK" ]]; then
  ENV_FILE="$(readlink -f "$ENV_LINK")"
fi

read -r -s -p "Tencent SecretId: " SECRET_ID
printf "\n"
read -r -s -p "Tencent SecretKey: " SECRET_KEY
printf "\n"
if [[ -z "$SECRET_ID" || -z "$SECRET_KEY" ]]; then
  echo "[FAIL] SecretId and SecretKey are required." >&2
  exit 2
fi

mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
TEMP_FILE="$(mktemp "$(dirname "$ENV_FILE")/.translation-env.XXXXXX")"
cleanup() {
  rm -f "$TEMP_FILE"
}
trap cleanup EXIT

awk '
  !/^TRANSLATION_PROVIDER=/ &&
  !/^TENCENTCLOUD_SECRET_ID=/ &&
  !/^TENCENTCLOUD_SECRET_KEY=/ &&
  !/^TENCENT_TRANSLATION_HOST=/ &&
  !/^TENCENT_TRANSLATION_ACTION=/ &&
  !/^TENCENT_TRANSLATION_VERSION=/ &&
  !/^TENCENT_TRANSLATION_SERVICE=/ &&
  !/^TENCENT_TRANSLATION_REGION=/ &&
  !/^TENCENT_TRANSLATION_PROJECT_ID=/
' "$ENV_FILE" >"$TEMP_FILE"

{
  printf "\nTRANSLATION_PROVIDER=tencent_tmt\n"
  printf "TENCENTCLOUD_SECRET_ID=%s\n" "$SECRET_ID"
  printf "TENCENTCLOUD_SECRET_KEY=%s\n" "$SECRET_KEY"
  printf "TENCENT_TRANSLATION_HOST=tmt.tencentcloudapi.com\n"
  printf "TENCENT_TRANSLATION_ACTION=TextTranslate\n"
  printf "TENCENT_TRANSLATION_VERSION=2018-03-21\n"
  printf "TENCENT_TRANSLATION_SERVICE=tmt\n"
  printf "TENCENT_TRANSLATION_REGION=ap-beijing\n"
  printf "TENCENT_TRANSLATION_PROJECT_ID=0\n"
} >>"$TEMP_FILE"

chmod 600 "$TEMP_FILE"
mv "$TEMP_FILE" "$ENV_FILE"
trap - EXIT
unset SECRET_ID SECRET_KEY
echo "[OK] Tencent translation credentials saved to the private environment file."
echo "Next: ./scripts/test_translation_provider.py"
