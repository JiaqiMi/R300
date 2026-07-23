#!/usr/bin/env bash
# Compatibility wrapper.  The standalone 1X entry is now start_1x.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_1x.sh" "$@"
