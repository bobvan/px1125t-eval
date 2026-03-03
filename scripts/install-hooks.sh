#!/bin/bash
# Install git hooks for px1125t-eval.
# The post-commit hook runs check_shared.py so divergence from testAnt
# is caught immediately when either repo is committed to.
set -e
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/post-commit"
cat > "$HOOK" << 'HOOK_EOF'
#!/bin/sh
python3 "$(git rev-parse --show-toplevel)/scripts/check_shared.py" --gt-root ~/gt
HOOK_EOF
chmod +x "$HOOK"
echo "Installed post-commit hook at $HOOK"
