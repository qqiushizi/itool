#!/usr/bin/env bash
set -euo pipefail

FILTER_REGEX='Mellanox|ConnectX'

normalize_bus_id() {
  printf '%s' "${1#0000:}" | tr '[:lower:]' '[:upper:]'
}

normalize_pci_addr() {
  local bus_id
  bus_id="$(normalize_bus_id "$1")"
  printf '0000:%s' "$bus_id"
}

sysfs_pci_addr() {
  normalize_pci_addr "$1" | tr '[:upper:]' '[:lower:]'
}

normalize_slot_id() {
  normalize_bus_id "$1" | sed 's/\.[0-7]$//'
}

mapping='
0 03:00.0 04:00.0
1 2E:00.0 2F:00.0
2 1C:00.0 1B:00.0
3 16:00.0 15:00.0
4 9C:00.0 9D:00.0
5 BB:00.0 BC:00.0
6 B6:00.0 B5:00.0
7 84:00.0 83:00.0
'

declare -A nic_bus_to_ib_name
declare -A pci_to_net_iface
declare -A pci_to_operstate
declare -A pci_to_mac
declare -A slot_to_net_ifaces
declare -A slot_to_operstates
declare -A slot_to_macs
declare -A slot_to_bond
declare -A slot_to_bond_ip
declare -A slot_to_bond_eths

while read -r path; do
  [[ -z "${path:-}" ]] && continue

  ib_name="${path##*/}"
  nic_pci="$(grep -oE '0000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]' <<< "$path" | tail -n 1)"
  [[ -z "${nic_pci:-}" ]] && continue

  nic_bus="$(normalize_bus_id "$nic_pci")"
  nic_bus_to_ib_name["$nic_bus"]="$ib_name"
done < <(find /sys/devices -name "mlx[0-9]*_[0-9]*" 2>/dev/null)

command -v lspci >/dev/null 2>&1 || {
  echo "ERROR: lspci not found" >&2
  exit 1
}

mapfile -t pci_devs < <(
  lspci -t -v |
    awk -v re="$FILTER_REGEX" '
      {
        line = $0
        while (match(line, /\[[0-9a-fA-F]+\]/)) {
          current_bus = substr(line, RSTART + 1, RLENGTH - 2)
          line = substr(line, RSTART + RLENGTH)
        }
      }

      $0 ~ re {
        if (current_bus == "") {
          next
        }

        if (match($0, /[0-9a-fA-F]{2}\.[0-7][[:space:]]+.*$/)) {
          printf "0000:%s:%s\n", current_bus, substr($0, RSTART, 4)
        }
      }
    ' |
    sort -u
)

for pci in "${pci_devs[@]}"; do
  pci_key="$(normalize_pci_addr "$pci")"
  pci_sysfs="$(sysfs_pci_addr "$pci")"
  slot_key="$(normalize_slot_id "$pci")"
  net_dir="/sys/bus/pci/devices/${pci_sysfs}/net"

  if [[ ! -d "$net_dir" ]]; then
    pci_to_net_iface["$pci_key"]="NO_NET_DIR"
    pci_to_operstate["$pci_key"]="-"
    pci_to_mac["$pci_key"]="-"
    continue
  fi

  iface_found=0
  for iface_path in "$net_dir"/*; do
    [[ -e "$iface_path" ]] || continue

    iface_found=1
    iface="${iface_path##*/}"

    operstate="-"
    mac="-"
    [[ -r "/sys/class/net/${iface}/operstate" ]] && operstate="$(cat "/sys/class/net/${iface}/operstate")"
    [[ -r "/sys/class/net/${iface}/address" ]] && mac="$(cat "/sys/class/net/${iface}/address")"

    pci_to_net_iface["$pci_key"]="$iface"
    pci_to_operstate["$pci_key"]="$operstate"
    pci_to_mac["$pci_key"]="$mac"

    if [[ "$pci_key" == *.0 ]]; then
      if [[ -e "/sys/class/net/${iface}/master" ]]; then
        slot_to_bond["$slot_key"]="$(basename "$(readlink -f "/sys/class/net/${iface}/master")")"
      else
        slot_to_bond["$slot_key"]="None"
      fi
    fi
  done

  if [[ "$iface_found" -eq 0 ]]; then
    pci_to_net_iface["$pci_key"]="NO_IFACE"
    pci_to_operstate["$pci_key"]="-"
    pci_to_mac["$pci_key"]="-"
  fi
done

while read -r npu_id _ nic_bus; do
  [[ -z "${npu_id:-}" ]] && continue

  slot_key="$(normalize_slot_id "$nic_bus")"
  pci0="$(normalize_pci_addr "${slot_key}.0")"
  pci1="$(normalize_pci_addr "${slot_key}.1")"

  slot_to_net_ifaces["$slot_key"]="${pci_to_net_iface[$pci0]:---},${pci_to_net_iface[$pci1]:---}"
  slot_to_operstates["$slot_key"]="${pci_to_operstate[$pci0]:---},${pci_to_operstate[$pci1]:---}"
  slot_to_macs["$slot_key"]="${pci_to_mac[$pci0]:---},${pci_to_mac[$pci1]:---}"
  slot_to_bond["$slot_key"]="${slot_to_bond[$slot_key]:-None}"

  if [[ "${slot_to_bond[$slot_key]}" != "None" ]] && command -v ip >/dev/null 2>&1; then
    slot_to_bond_ip["$slot_key"]="$(ip -o -4 addr show dev "${slot_to_bond[$slot_key]}" 2>/dev/null | awk '{ips = ips ? ips "," $4 : $4} END {print ips}' || true)"
    slot_to_bond_ip["$slot_key"]="${slot_to_bond_ip[$slot_key]:-None}"
  else
    slot_to_bond_ip["$slot_key"]="None"
  fi

  if [[ "${slot_to_bond[$slot_key]}" != "None" && -r "/sys/class/net/${slot_to_bond[$slot_key]}/bonding/slaves" ]]; then
    slot_to_bond_eths["$slot_key"]="$(tr ' ' ',' < "/sys/class/net/${slot_to_bond[$slot_key]}/bonding/slaves")"
  else
    slot_to_bond_eths["$slot_key"]="None"
  fi
done <<< "$mapping"

printf "| NPU_ID | map_NPU_BUS_ID | map_NIC_BUS_ID | find_NET_IFACE | lspci_NPU_ETH | EHT_STATUS | ETH_MAC | ETH_IN_BOND | BOND_IP | BOND_ETHS |\n"
printf "|--|--|--|--|--|--|--|--|--|--|\n"

while read -r npu_id npu_bus nic_bus; do
  [[ -z "${npu_id:-}" ]] && continue

  npu_bus="$(normalize_bus_id "$npu_bus")"
  nic_bus="$(normalize_bus_id "$nic_bus")"
  slot_key="$(normalize_slot_id "$nic_bus")"

  ib_name="${nic_bus_to_ib_name[$nic_bus]:---}"
  net_iface="${slot_to_net_ifaces[$slot_key]:---}"
  net_in_bond="${slot_to_bond[$slot_key]:-None}"
  bond_ip="${slot_to_bond_ip[$slot_key]:-None}"
  bond_eths="${slot_to_bond_eths[$slot_key]:-None}"
  operstate="${slot_to_operstates[$slot_key]:---}"
  mac="${slot_to_macs[$slot_key]:---}"

  printf "|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|\n" \
    "$npu_id" "$npu_bus" "$nic_bus" "$ib_name" "$net_iface" "$operstate" "$mac" "$net_in_bond" "$bond_ip" "$bond_eths"
done <<< "$mapping"