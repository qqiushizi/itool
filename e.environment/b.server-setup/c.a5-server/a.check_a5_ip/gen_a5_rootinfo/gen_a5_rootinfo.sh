#!/usr/bin/env bash
set -euo pipefail

ATTACH_DIR="${ATTACH_DIR:-configs}"
BEFORE_HCCL="${ATTACH_DIR}/default_hccl_rootinfo.json"
BEFORE_ATLAS="${ATTACH_DIR}/default_atlas_350_3.json"
NEW_HCCL="${ATTACH_DIR}/new_hccl_rootinfo.json"
NEW_ATLAS="${ATTACH_DIR}/new_atlas_350_3.json"

usage() {
  echo "Usage: $0 [check_a5_ip_output_file]" >&2
  echo "If no file is provided, the script runs: bash check_a5_ip.sh" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for file in "${BEFORE_HCCL}" "${BEFORE_ATLAS}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing input file: ${file}" >&2
    exit 1
  fi
done

if ! command -v jq >/dev/null 2>&1; then
  echo "Missing dependency: jq" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

check_output="${tmp_dir}/check_a5_ip.out"
ip_tsv="${tmp_dir}/npu_ip.tsv"
ip_json="${tmp_dir}/npu_ip.json"

if [[ $# -gt 1 ]]; then
  usage
  exit 1
elif [[ $# -eq 1 ]]; then
  if [[ ! -f "$1" ]]; then
    echo "Missing check output file: $1" >&2
    exit 1
  fi
  cp "$1" "${check_output}"
else
  if [[ ! -f "check_a5_ip.sh" ]]; then
    echo "Missing check_a5_ip.sh. Provide a saved output file, or place check_a5_ip.sh in current directory." >&2
    exit 1
  fi
  bash check_a5_ip.sh > "${check_output}"
fi

awk -F'|' '
function trim(s) {
  gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s)
  return s
}
$0 !~ /^\|/ { next }
{
  npu_id = trim($2)
  bond_ip = trim($10)
  if (npu_id ~ /^[0-9]+$/ && bond_ip ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\/[0-9]+$/) {
    sub(/\/.*/, "", bond_ip)
    print npu_id "\t" bond_ip
  }
}
' "${check_output}" | sort -n -k1,1 > "${ip_tsv}"

rank_count="$(jq -r '.rank_count' "${BEFORE_HCCL}")"
ip_count="$(wc -l < "${ip_tsv}" | tr -d ' ')"
if [[ "${ip_count}" != "${rank_count}" ]]; then
  echo "Expected ${rank_count} NPU IP rows, got ${ip_count}" >&2
  exit 1
fi

jq -Rn '
  [inputs | split("\t") | {key: .[0], value: .[1]}] | from_entries
' "${ip_tsv}" > "${ip_json}"

jq --slurpfile npu_ip "${ip_json}" '
  def instance_id($device_id):
    if $device_id < 4 then "1" else "2" end;

  .topo_file_path = "/etc/atlas_350_3.json"
  | .rank_list |= map(
      . as $rank
      | {
          device_id: $rank.device_id,
          local_id: $rank.local_id,
          device_port: $rank.device_port,
          level_list: [
            ($rank.level_list[0] | .net_instance_id = instance_id($rank.device_id)),
            {
              net_layer: 3,
              net_instance_id: "cluster",
              net_type: "CLOS",
              net_attr: "",
              rank_addr_list: [
                {
                  addr_type: "IPV4",
                  addr: (
                    $npu_ip[0][$rank.device_id | tostring]
                    // error("missing BOND_IP for NPU_ID " + ($rank.device_id | tostring))
                  ),
                  ports: ["d2h"],
                  plane_id: "plane_0"
                }
              ]
            }
          ]
        }
    )
' "${BEFORE_HCCL}" > "${NEW_HCCL}"

jq '
  .edge_list |= map(
    select(
      .net_layer != 0
      or .link_type != "PEER2NET"
      or .topo_type != "CLOS"
      or .topo_instance_id != 4
      or .local_a_ports != ["d2h"]
      or .protocols != ["PCIE"]
      or .position != "DEVICE"
    )
  )
  | .edge_count = (.edge_list | length)
' "${BEFORE_ATLAS}" > "${NEW_ATLAS}"

echo "Generated ${NEW_HCCL}"
echo "Generated ${NEW_ATLAS}"
