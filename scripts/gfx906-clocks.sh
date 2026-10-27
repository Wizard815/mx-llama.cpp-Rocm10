#!/bin/bash
# Usage: gfx906-clocks.sh [high|auto] - set DPM performance level on every Vega20 (MI50/MI60) card.
level="${1:-high}"
for d in /sys/class/drm/card[0-9]*/device; do
  [ -f "$d/vendor" ] && [ "$(cat $d/vendor)" = 0x1002 ] || continue
  case "$(cat $d/device)" in 0x66a0|0x66a1|0x66a2|0x66a3|0x66a7|0x66af) ;; *) continue ;; esac
  echo "$level" > "$d/power_dpm_force_performance_level"
  echo "$d: $(cat $d/power_dpm_force_performance_level) sclk=$(grep '\*' $d/pp_dpm_sclk) mclk=$(grep '\*' $d/pp_dpm_mclk)"
done
