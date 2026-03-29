#!/bin/bash
#
# Agent Grid Setup Wizard
# Detects Python and runs the interactive setup wizard.
#

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "Error: python3 is required but not found." >&2
        exit 1
    fi
fi

"$PYTHON_BIN" -m agentgrid.cli.setup_wizard "$@"
