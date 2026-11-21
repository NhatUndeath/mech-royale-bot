# Mech Royale bot

Survival-first bot for the `mechfight/1` protocol. Python standard library only.

- `bot.py`: zone, looting, loadout, roof transitions, and combat decisions
- `nav.py`: time-sliced ground flow, local sightline costs, and low-roof connections built from `/arena`
- `mechfight.py`: the kit's protocol wrapper
- `telemetry.py`: optional match summaries on stderr

Build with `docker build -t my-bot .`. The bot defaults to the Practice policy.
In an optional `params.py`, set `PRACTICE = False` for the previous free-for-all
engagement policy, or set `ROOFS = False` to test the Practice policy without
active rooftop transitions.

For a local run, set `MECH_TELEMETRY` to a run-group name and capture stderr.
Summarize one or more logs with `python telemetry.py run1.log run2.log`.
The starter-kit checker replays recorded input. Use a separate telemetry group
for the checker because its final placement is part of the recording.
