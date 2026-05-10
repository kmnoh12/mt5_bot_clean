# Security / Upload Notes

This repository was staged from the Desktop `mt5_bot_clean` folder.

Excluded before GitHub upload:
- Python bytecode/cache directories (`__pycache__`, `*.pyc`)
- runtime logs/state (`*.log`, runtime lock files)
- memory/daily runtime journal entries and reinforcement log
- legacy snapshot folder and legacy config file
- local credentials/secrets patterns via `.gitignore`

`mt5_bot/config.yaml` is sanitized for GitHub: broker login/server/password and notification/API credentials are placeholders. Put real credentials only in a local ignored config file or edit locally after clone.
