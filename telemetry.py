"""Opt-in mechfight/1 measurements. Set MECH_TELEMETRY to a run group such as
'v5-practice' or 'v5-check-fixture'. Redirect stderr to a log and summarize
runs with: python telemetry.py path/to/log [...]. A replayed fixture's placement
belongs to its recorded bot. Use responsive matches for policy comparisons.

Exposure estimates use currently tracked mechs (including memory), base sensor
ranges, their aim and map geometry. Other opponents, upgraded sensors, foliage
and shared squad knowledge can differ from this estimate. Damage events cover
the last observed two seconds; the server sends no fatal-hit field in 'end'.
"""
import json
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path


PREFIX = "MECH_TELEMETRY "


class MatchTelemetry:
    def __init__(self, hello, bot, group):
        self.group = group
        self.nav = getattr(bot, "nav", None)
        cfg = hello["config"]
        self.vision = cfg["vision"]
        self.eye_height = cfg["mech"]["eye_height"]
        self.size_per_cube = cfg["cubes"]["size_per_cube"]
        self.match_seconds = cfg["match"]["duration_s"]
        self.mechs = int(cfg["match"]["mechs"])
        self.times_ns = []
        self.exposure = Counter()
        self.hits = deque()
        self.cubes = None
        self.upgrades = []
        self.picks = {}
        self.t = 0.0
        self.alive = None

    def reverse_sight(self, obs):
        me = obs["self"]
        tx, ty, tz = me["pos"]
        height = me["stats"]["height"]
        center = (tx, ty, tz + height / 2)
        top = (tx, ty, tz + height)
        enemies = [actor for actor in obs["mechs"] if actor["kind"] == "mech"]
        if not enemies:
            return "no_tracked_mech"
        if not self.nav or not self.nav.ready:
            return "unknown_no_map"
        # Diagnostics run after sending the action; cap optional ray work so the
        # next input isn't held up by a crowded observation.
        deadline = time.perf_counter_ns() + 1_000_000
        for enemy in enemies:
            ex, ey, ez = enemy["pos"]
            eye = (ex, ey, ez + self.eye_height * (1 + self.size_per_cube * enemy.get("cubes", 0)))
            distance = math.dist(eye, center)
            if distance > self.vision["range"]:
                continue
            if distance > self.vision["near_range"]:
                yaw_error = math.remainder(math.atan2(ty - ey, tx - ex) - enemy["aim"][0], math.tau)
                if abs(yaw_error) > math.radians(self.vision["fov_deg"] / 2):
                    continue
            for point in (center, top):
                if time.perf_counter_ns() >= deadline:
                    return "unknown_budget"
                if self.nav.los(eye, point):
                    return "estimated_visible"
        return "no_tracked_witness"

    def observe(self, obs, action, elapsed_ns):
        self.times_ns.append(elapsed_ns)
        self.exposure[self.reverse_sight(obs)] += 1
        self.t = obs["t"]
        self.alive = obs["alive"]
        me = obs["self"]
        self.cubes = me["cubes"]
        self.upgrades = me["upgrades"]
        offer = obs.get("offer")
        pick = action.get("pick")
        if offer and isinstance(pick, int) and 0 <= pick < len(offer["choices"]):
            self.picks[offer["id"]] = offer["choices"][pick]["id"]
        while self.hits and self.hits[0][0] < self.t - 2.0:
            self.hits.popleft()
        for event in obs["events"]:
            if event["type"] == "damaged":
                kind = event["weapon"]
                attacker = event.get("by")
                if attacker == me["id"]:
                    kind = "self"
                self.hits.append((self.t, kind, attacker))

    def terminal_context(self, place):
        if place is None:
            return "unfinished"
        if place == 1:
            return "first_place"
        if self.alive is not None and self.alive > 1 and self.t >= self.match_seconds - 0.11:
            return "possible_timeout"
        sources = set()
        attackers = set()
        for _, weapon, actor in self.hits:
            if weapon in ("zone", "npc", "self"):
                sources.add(weapon)
            elif actor is not None and 0 <= actor < self.mechs:
                sources.add("mech")
                attackers.add(actor)
            else:
                sources.add("unknown")
        if not sources:
            return "unknown"
        if len(sources) > 1:
            return "mixed"
        if sources != {"mech"}:
            return next(iter(sources))
        if len(attackers) >= 3:
            return "3plus_mech_attackers"
        return "one_mech_attacker" if len(attackers) == 1 else "two_mech_attackers"

    def finish(self, end):
        place = end.get("place")
        times = sorted(self.times_ns)
        def percentile(p):
            if times:
                return round(times[math.ceil(len(times) * p) - 1] / 1_000_000, 3)
            return None

        record = dict(group=self.group, place=place, kills=end.get("kills"),
                      placement_points=self.mechs - place if place is not None else None,
                      decisions=len(times), exposure_decisions=dict(self.exposure),
                      last_observed_cubes=self.cubes, applied_upgrades=self.upgrades,
                      upgrade_choices=list(self.picks.values()),
                      last_2s_damage_context=self.terminal_context(place),
                      last_observed_t=self.t, p95_ms=percentile(0.95), p99_ms=percentile(0.99))
        print(PREFIX + json.dumps(record, separators=(",", ":")), file=sys.stderr, flush=True)


def summarize(paths):
    groups = {}
    for path in paths:
        # Windows PowerShell 5 file redirection can produce UTF-16 text.
        with Path(path).open("rb") as file:
            bom = file.read(2)
        encoding = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        with Path(path).open(encoding=encoding, errors="replace") as file:
            for line in file:
                if PREFIX not in line:
                    continue
                # 'check.py' may prepend a prefix to forwarded bot stderr.
                record = json.loads(line.split(PREFIX, 1)[1])
                groups.setdefault(record["group"], []).append(record)
    if not groups:
        raise SystemExit("No MECH_TELEMETRY match summaries in these logs.")
    for group, records in groups.items():
        complete = [r for r in records if r["place"] is not None]
        exposure = Counter()
        for r in records:
            exposure.update(r["exposure_decisions"])
        average_place = sum(r["place"] for r in complete) / len(complete) if complete else None
        average_points = sum(r["placement_points"] for r in complete) / len(complete) if complete else None
        print(json.dumps(dict(group=group, completed=len(complete), unfinished=len(records) - len(complete),
                             mean_place=average_place, mean_placement_points=average_points,
                             exposure_decisions=dict(exposure),
                             terminal_contexts=Counter(r["last_2s_damage_context"] for r in records),
                             observed_cube_totals=[r["last_observed_cubes"] for r in records],
                             applied_upgrades=Counter(u for r in records for u in r["applied_upgrades"]),
                             choices=Counter(u for r in records for u in r["upgrade_choices"]),
                             worst_match_p95_ms=max((r["p95_ms"] for r in records if r["p95_ms"] is not None), default=None),
                             worst_match_p99_ms=max((r["p99_ms"] for r in records if r["p99_ms"] is not None), default=None)),
                         sort_keys=True))


if __name__ == "__main__":
    summarize(sys.argv[1:])
