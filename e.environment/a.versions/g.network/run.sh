#!/bin/bash
echo "========== Network / Cluster Info =========="
echo ""

echo "[Network Interfaces]"
ip addr 2>/dev/null | head -30 || ifconfig 2>/dev/null | head -30

echo ""
echo "[Network Routes]"
ip route 2>/dev/null || route -n 2>/dev/null || echo "route command failed"

echo ""
echo "[DNS Servers]"
cat /etc/resolv.conf 2>/dev/null | grep nameserver || echo "DNS config not found"

echo ""
echo "[NTP / Time Sync]"
ntpq -p 2>/dev/null || chronyc sources 2>/dev/null || echo "NTP not available"
systemctl status chronyd 2>/dev/null | head -5 || echo "chronyd not available"

echo ""
echo "[Hostname / Hosts]"
hostname
cat /etc/hosts 2>/dev/null | head -10

echo ""
echo "[Firewall Status]"
systemctl status firewalld 2>/dev/null | head -3 || echo "firewalld not available"
iptables -L 2>/dev/null | head -10 || echo "iptables not accessible"

echo ""
echo "================================================"
