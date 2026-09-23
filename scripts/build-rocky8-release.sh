#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build and export an x86-64 Rocky Linux 8-compatible gpugeno release.

Usage:
  scripts/build-rocky8-release.sh [OUTPUT_DIRECTORY]

Environment:
  GPUGENO_PODMAN_SUDO=1   run Podman through non-interactive sudo
  GPUGENO_RELEASE_IMAGE   override the local image tag
  GPUGENO_PODMAN_BUILD_ARGS
                           additional whitespace-separated podman build flags

The default output directory is dist/rockylinux8-x86_64.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${1:-"${repo_root}/dist/rockylinux8-x86_64"}
if [[ -z ${output_dir} || ${output_dir} == / ]]; then
    echo "error: refusing unsafe output directory ${output_dir@Q}" >&2
    exit 2
fi
image=${GPUGENO_RELEASE_IMAGE:-localhost/gpugeno-release:rockylinux8}
containerfile="${repo_root}/packaging/rockylinux8/Containerfile"

podman_cmd() {
    if [[ ${GPUGENO_PODMAN_SUDO:-0} == 1 ]]; then
        sudo -n podman "$@"
    else
        podman "$@"
    fi
}

if ! command -v podman >/dev/null 2>&1; then
    echo "error: podman is required" >&2
    exit 1
fi
if [[ ${GPUGENO_PODMAN_SUDO:-0} == 1 ]] && ! sudo -n true; then
    echo "error: GPUGENO_PODMAN_SUDO=1 requires passwordless or pre-authorized sudo" >&2
    exit 1
fi

# Deliberately permit simple extra flags such as --pull=always. Shell quoting
# cannot be represented in this environment variable; use an empty value or
# ordinary whitespace-separated Podman options.
build_args=()
if [[ -n ${GPUGENO_PODMAN_BUILD_ARGS:-} ]]; then
    read -r -a build_args <<<"${GPUGENO_PODMAN_BUILD_ARGS}"
fi

source_revision=$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf 'unknown')
source_state=clean
if [[ -n $(git -C "${repo_root}" status --porcelain --untracked-files=normal 2>/dev/null || true) ]]; then
    source_state=dirty
fi

podman_cmd build \
    "${build_args[@]}" \
    --platform linux/amd64 \
    --build-arg "GPUGENO_SOURCE_REVISION=${source_revision}" \
    --build-arg "GPUGENO_SOURCE_STATE=${source_state}" \
    --file "${containerfile}" \
    --tag "${image}" \
    "${repo_root}"

output_parent=$(dirname "${output_dir}")
mkdir -p "${output_parent}"
staging_dir=$(mktemp -d "${output_parent}/.gpugeno-rocky8-release.XXXXXX")
container_id=
cleanup() {
    if [[ -n ${container_id} ]]; then
        podman_cmd rm "${container_id}" >/dev/null 2>&1 || true
    fi
    if [[ -n ${staging_dir} ]]; then
        rm -rf "${staging_dir}"
    fi
}
trap cleanup EXIT

container_id=$(podman_cmd create "${image}")
podman_cmd cp "${container_id}:/opt/gpugeno-release/." "${staging_dir}/"
if [[ ${GPUGENO_PODMAN_SUDO:-0} == 1 ]]; then
    sudo -n chown -R "$(id -u):$(id -g)" "${staging_dir}"
fi

chmod 0755 "${staging_dir}/gpugeno"
(
    cd "${staging_dir}"
    sha256sum gpugeno > gpugeno.sha256
)

rm -rf "${output_dir}"
mv "${staging_dir}" "${output_dir}"
# The staging path has moved and must not be removed by cleanup.
staging_dir=

printf 'release executable: %s\n' "${output_dir}/gpugeno"
printf 'checksum:           %s\n' "${output_dir}/gpugeno.sha256"
printf 'build metadata:     %s\n' "${output_dir}/release-info.txt"
printf 'container image:    %s\n' "${image}"
