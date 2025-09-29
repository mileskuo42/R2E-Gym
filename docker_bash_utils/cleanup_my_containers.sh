#!/usr/bin/env bash
set -euo pipefail

# Stop and remove only containers launched by this project (by image pattern).
# Usage:
#   bash docker_bash_utils/cleanup_my_containers.sh            # stop+rm defaults
#   bash docker_bash_utils/cleanup_my_containers.sh -n          # dry-run (show only)
#   bash docker_bash_utils/cleanup_my_containers.sh patternA patternB ...
#     (override default image patterns with your own regex fragments)

DRY_RUN=0

usage() {
	echo "Usage: $0 [-n] [pattern1 pattern2 ...]" 1>&2
	exit 1
}

while getopts ":n" opt; do
	case "$opt" in
		n) DRY_RUN=1 ;;
		*) usage ;;
	esac
done
shift $((OPTIND - 1))

# Default image regex patterns used by this repo
if [ "$#" -gt 0 ]; then
	PATTERNS=("$@")
else
	PATTERNS=(
		"^slimshetty/swebench-verified"
		"^jyangballin/"
		"^namanjain12/.+new"
	)
fi

# Join patterns into a single extended regex separated by |
REGEX="$(IFS='|'; echo "${PATTERNS[*]}")"

# Collect matching container IDs
mapfile -t MATCHING_IDS < <(docker ps -a --format '{{.ID}} {{.Image}}' | awk -v re="$REGEX" '$2 ~ re {print $1}')

if [ "${#MATCHING_IDS[@]}" -eq 0 ]; then
	echo "No matching containers found for regex: $REGEX"
	exit 0
fi

echo "Matching containers (will target these):"
docker ps -a --format '{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' | awk -v re="$REGEX" '$2 ~ re {print}'

if [ "$DRY_RUN" -eq 1 ]; then
	echo "Dry run (-n): not stopping/removing containers."
	exit 0
fi

# Stop only the running subset first (graceful)
RUNNING_IDS=$(docker ps --format '{{.ID}} {{.Image}}' | awk -v re="$REGEX" '$2 ~ re {print $1}')
if [ -n "$RUNNING_IDS" ]; then
	echo "$RUNNING_IDS" | xargs -r docker stop
fi

# Remove all matched containers (stopped or exited)
printf '%s\n' "${MATCHING_IDS[@]}" | xargs -r docker rm

echo "Done: stopped and removed matching containers."





