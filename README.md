# Mech Royale bot

Survival-first bot for the `mechfight/1` protocol. Python standard library only.

- `bot.py`: decision logic (zone, combat, looting, upgrades)
- `nav.py`: navigation grid built from the map in `/arena`, time-sliced flow-field pathing
- `mechfight.py`: the kit's protocol wrapper

Build: `docker build -t my-bot .`
