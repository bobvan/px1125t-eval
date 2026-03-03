#!/bin/bash
# Install udev rule for PX1125T and add current user to dialout group.
set -e

RULES_SRC="$(dirname "$0")/99-px1125t.rules"
RULES_DST=/etc/udev/rules.d/99-px1125t.rules

echo "Installing udev rule..."
sudo cp "$RULES_SRC" "$RULES_DST"
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=tty

# Verify symlink appeared
sleep 1
if [ -e /dev/ttyPX1125T ]; then
    echo "  /dev/ttyPX1125T → $(readlink -f /dev/ttyPX1125T)"
else
    echo "  WARNING: /dev/ttyPX1125T not found — try unplugging and replugging the device"
fi

# Add user to dialout so sudo isn't needed for serial access
if ! groups "$USER" | grep -qw dialout; then
    echo "Adding $USER to dialout group (re-login required)..."
    sudo usermod -aG dialout "$USER"
fi

echo "Done."
