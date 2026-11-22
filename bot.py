"""Survival-first mech bot.

Priorities, every decision:
1. Stay ahead of the zone (flow-field navigation on a grid built from the map).
2. Face whoever is shooting us (rear hits take x1.5), strafe while fighting.
3. Shoot back with lead; use laser / lightning / mortar when held.
4. Early and when safe: grab weapons and cubes, farm NPC drones and crates.
5. Retreat when losing a fight.
"""
import math
import os
import random
import sys
import time
from typing import NamedTuple

import mechfight as mf
from nav import Nav

UPGRADE_PREF = ["armor", "plating", "nanites", "gun_rate", "gun_damage", "laser_damage", "laser_charge",
                "servos", "gun_mag", "gun_velocity", "sensors", "gyros", "reactor", "capacitor", "jets"]
WEAPON_PREF = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
COMBAT_BUDGET_S = 0.009


class Shot(NamedTuple):
    kind: str
    slot: int
    aim: tuple[float, float]
    tolerance: tuple[float, float]
    ready_in: float
    impact_in: float
    finish_in: float

    def aligned(self, aim):
        return (abs(wrap(aim[0] - self.aim[0])) < self.tolerance[0]
                and abs(aim[1] - self.aim[1]) < self.tolerance[1])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def norm2(x, y):
    l = math.hypot(x, y)
    return (x / l, y / l) if l > 1e-6 else (0.0, 0.0)


class Bot:
    name = "survivor"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(hello.get("seed", 0))
        mdir = hello.get("map", {}).get("dir") or "/arena"
        if not os.path.isdir(mdir):
            for alt in ("/arena", os.path.join(os.path.dirname(__file__), "arena")):
                if os.path.isdir(alt):
                    mdir = alt
                    break
        self.nav = Nav(mdir)
        self.goal = None
        self.goal_key = None
        self.last_pos = None
        self.stuck_t = 0.0
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.threat = {}          # mech id -> (time, bearing)
        self.last_hit_t = -99.0
        self.last_hit_bearing = None
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.scan_yaw = 0.0
        self.sound_look = None    # (until, bearing)
        self.prev_t = 0.0
        self.combat_target = None
        self.cover_goal = None
        self.retreat_until = -1.0
        self.recover_until = -1.0
        self.laser_started = None

    # ------------------------------------------------------------------ helpers
    def weapon_slots(self, me):
        out = {}
        for i, w in enumerate(me.get("weapons") or []):
            if w:
                out[w["kind"]] = (i, w)
        return out

    def zone_goal(self, obs, me):
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        x, y = me["pos"][0], me["pos"][1]
        if key != self.goal_key or self.goal is None:
            d = math.dist((x, y), nc)
            r = 0.5 * nr
            if d <= r:
                g = (x, y)
            else:
                ux, uy = norm2(x - nc[0], y - nc[1])
                g = (nc[0] + ux * r, nc[1] + uy * r)
            if self.nav.ready:
                i, j = self.nav.nearest_free(g[0], g[1], 80, main=True)
                g = self.nav.center(i, j)
            self.goal, self.goal_key = g, key
        self.nav.request(self.goal)
        return self.goal

    def must_travel(self, obs, me):
        """True when the zone leaves no time for detours."""
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        if not me.get("in_zone", True):
            return True
        need = max(0.0, math.dist((x, y), z["next_center"]) - 0.75 * z["next_radius"])
        speed = me["stats"].get("walk_speed", 11.7) * 0.6
        avail = max(0.0, z.get("shrink_end", t) - t) * speed
        return need > avail - 40

    def lead(self, me, tgt, speed):
        mx, my, mz = me["pos"]
        mz += me["stats"].get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        h = 4.5 if tgt["kind"] == "mech" else 1.75
        if tgt["kind"] == "mech":
            h *= 1 + 0.05 * (tgt.get("cubes") or 0)
        tz += h
        vx, vy, vz = tgt.get("vel", [0, 0, 0])
        t = 0.0
        for _ in range(3):
            px, py, pz = tx + vx * t, ty + vy * t, tz + (vz * t if abs(vz) < 3 else 0)
            d = math.dist((mx, my, mz), (px, py, pz))
            t = d / speed + 0.05
        yaw = math.atan2(py - my, px - mx)
        pitch = math.atan2(pz - mz, math.hypot(px - mx, py - my))
        return yaw, pitch, math.hypot(px - mx, py - my)

    def mortar_aim(self, me, tgt, wstats):
        mx, my, mz = me["pos"]
        mz += me["stats"].get("height", 9) + 0.5
        tx, ty, tz = tgt["pos"]
        v, g = wstats.get("speed", 72), wstats.get("gravity", 22)
        x = math.hypot(tx - mx, ty - my)
        yv = tz + 1 - mz
        disc = v ** 4 - g * (g * x * x + 2 * yv * v * v)
        if disc < 0 or x < 1:
            return None
        ang = math.atan2(v * v - math.sqrt(disc), g * x)
        tof = x / (v * math.cos(ang))
        vx, vy, _ = tgt.get("vel", [0, 0, 0])
        tx2, ty2 = tx + vx * tof, ty + vy * tof
        x2 = math.hypot(tx2 - mx, ty2 - my)
        disc = v ** 4 - g * (g * x2 * x2 + 2 * yv * v * v)
        if disc < 0 or x2 < 1:
            return None
        ang = math.atan2(v * v - math.sqrt(disc), g * x2)
        return math.atan2(ty2 - my, tx2 - mx), ang, x2 / (v * math.cos(ang))

    def aligned(self, me, yaw, pitch, dist, tol_r=2.5):
        ay, ap = me["aim"]
        tol = math.atan2(tol_r, max(dist, 1.0)) + 0.01
        return abs(wrap(ay - yaw)) < tol and abs(ap - pitch) < tol + 0.03

    def mortar_clear(self, me, aim, flight, stats):
        """Check the low arc against solid cover and terrain before taking the shot."""
        x, y, z = me["pos"]
        yaw, pitch = aim
        speed, gravity = stats.get("speed", 72), stats.get("gravity", 22)
        vx = speed * math.cos(pitch) * math.cos(yaw)
        vy = speed * math.cos(pitch) * math.sin(yaw)
        vz = speed * math.sin(pitch)
        start = (x, y, z + me["stats"].get("height", 9) + 0.5)
        steps = max(3, math.ceil(flight / 0.16))
        a = start
        for i in range(1, steps + 1):
            dt = flight * i / steps
            b = (x + vx * dt, y + vy * dt, start[2] + vz * dt - 0.5 * gravity * dt * dt)
            if not self.nav.combat_los(a, b):
                return False
            a = b
        return True

    def shots(self, me, target, slots, distance, allow_charge, deadline):
        x, y, z = me["pos"]
        tx, ty, tz = target["pos"]
        size = 1 + self.cfg.get("cubes", {}).get("size_per_cube", 0.05) * (target.get("cubes") or 0)
        muzzle = (x, y, z + me["stats"].get("muzzle_height", 7))
        center = (tx, ty, tz + 4.5 * size)
        yaw = math.atan2(ty - y, tx - x)
        pitch = math.atan2(center[2] - muzzle[2], distance)
        stats = me["stats"]
        min_pitch = stats.get("min_pitch", -0.69)
        max_pitch = stats.get("max_pitch", 1.29)
        target_hp = target.get("hp", target.get("max_hp", 1400))
        scale = 1 + self.cfg.get("cubes", {}).get("damage_per_cube", 0.04) * min(
            me.get("cubes", 0), self.cfg.get("cubes", {}).get("stat_cap", 10))
        plans = []
        direct = None
        for kind, (slot, w) in slots.items():
            if time.perf_counter() >= deadline:
                break
            if kind == "laser" and not allow_charge and not w.get("charge"):
                continue
            ws = w["stats"]
            ready = w.get("cooldown", 0)
            flight = charge = reloads = 0.0
            aim = (yaw, pitch)
            tolerance = math.atan2(2 * size, max(distance, 1.0)) + 0.01
            yaw_tol, pitch_tol = tolerance, tolerance + 0.03
            damage = ws.get("damage", 45) * scale
            interval = ws.get("cooldown", 0.25)
            count = math.ceil(target_hp / max(1, damage))
            if kind == "gun":
                speed = ws.get("speed", 252)
                ay, ap, hd = self.lead(me, target, speed)
                if hd >= ws.get("range", 230):
                    continue
                end = (x + math.cos(ay) * hd, y + math.sin(ay) * hd, muzzle[2] + math.tan(ap) * hd)
                if not self.nav.combat_los(muzzle, end):
                    continue
                aim = (ay, ap)
                flight = math.dist(muzzle, end) / speed
                ammo, mag = w.get("ammo", ws.get("mag", 12)), max(1, ws.get("mag", 12))
                reload = ws.get("reload", 2.4)
                ready = max(ready, w.get("reload", 0), reload if ammo <= 0 and not w.get("reload") else 0)
                available = mag if ammo <= 0 or w.get("reload", 0) > 0 else ammo
                reloads = math.ceil(max(0, count - available) / mag) * reload
            elif kind == "mortar":
                if not 25 < distance < 220:
                    continue
                sol = self.mortar_aim(me, target, ws)
                if sol is None or not min_pitch <= sol[1] <= max_pitch:
                    continue
                aim, flight = sol[:2], sol[2]
                if not self.mortar_clear(me, aim, flight, ws):
                    continue
                yaw_tol = pitch_tol = 0.03
            elif kind in ("laser", "lightning"):
                if math.dist(muzzle, center) >= ws.get("range", 280 if kind == "laser" else 45):
                    continue
                if direct is None:
                    direct = self.nav.combat_los(muzzle, center)
                if not direct:
                    continue
                if kind == "laser":
                    full = ws.get("charge", 0.75)
                    charge = max(0, full - (w.get("charge") or 0))
                    interval += full
                else:
                    if me["energy"] < max(10, ws.get("energy", 3)):
                        continue
                    yaw_tol = math.radians(ws.get("cone_deg", 30)) * 0.45
                    pitch_tol = math.inf
                    aim = (yaw, max(min_pitch, min(max_pitch, pitch)))
            else:
                continue
            if not min_pitch <= aim[1] <= max_pitch:
                continue
            slew = max(abs(wrap(aim[0] - me["aim"][0])) / stats.get("turn_rate", 4),
                       abs(aim[1] - me["aim"][1]) / stats.get("pitch_rate", 4))
            impact = max(ready + charge, slew) + flight
            plans.append(Shot(kind, slot, aim, (yaw_tol, pitch_tol), ready,
                              impact, impact + max(0, count - 1) * interval + reloads))
        return plans

    def choose_target(self, me, mechs, slots, t, allow_charge, deadline):
        """Choose among feasible shot plans. A blocked contact still matters to movement."""
        x, y = me["pos"][:2]
        best, best_score = (None, math.inf, []), -1.0
        for target in sorted(mechs, key=lambda m: math.hypot(m["pos"][0] - x, m["pos"][1] - y)):
            if time.perf_counter() >= deadline:
                break
            distance = math.hypot(target["pos"][0] - x, target["pos"][1] - y)
            plans = self.shots(me, target, slots, distance, allow_charge, deadline)
            if not plans:
                continue
            incoming = self.threat.get(target["id"])
            threat = max(0.0, 1 - (t - incoming[0]) / 3.0) if incoming else 0.0
            finish = max(0.0, 1 - min(p.finish_in for p in plans) / 4)
            ready = max(0.0, 1 - min(p.impact_in for p in plans) / 2.0)
            near = max(0.0, 1 - distance / 280)
            score = 0.35 * threat + 0.30 * finish + 0.20 * ready + 0.15 * near
            if target["id"] == self.combat_target:
                score += 0.05
            if score > best_score:
                best, best_score = (target, distance, plans), score
        self.combat_target = best[0]["id"] if best[0] else None
        return best

    def firing_angles(self, me, mechs, t, deadline):
        """Return distinct attackers with fresh damage or a plausible current firing line."""
        x, y, z = me["pos"]
        center = (x, y, z + me["stats"].get("height", 9) * 0.5)
        angles = []
        seen = set()
        for enemy in mechs:
            eid = enemy["id"]
            seen.add(eid)
            tx, ty, tz = enemy["pos"]
            dist = math.hypot(tx - x, ty - y)
            bearing = math.atan2(ty - y, tx - x)
            hit = self.threat.get(eid)
            recent = hit is not None and t - hit[0] < 1.2
            facing = abs(wrap(enemy.get("aim", (bearing, 0))[0] - bearing - math.pi)) < 0.50
            weapons = enemy.get("weapons", ("gun",))
            in_range = any(kind != "lightning" or dist <= self.cfg["weapons"]["lightning"]["range"]
                           for kind in weapons)
            if recent or (dist < 145 and facing and in_range):
                muzzle = (tx, ty, tz + 7 * (1 + 0.05 * (enemy.get("cubes") or 0)))
                if time.perf_counter() < deadline and self.nav.combat_los(muzzle, center):
                    angles.append((bearing, muzzle))
                elif recent and (hit[2] == "mortar" or t - hit[0] < 0.20):
                    angles.append((hit[1], None))
        for eid, (when, bearing, _) in self.threat.items():
            if 0 <= eid <= 15 and eid not in seen and t - when < 1.2:
                angles.append((bearing, None))
        return angles

    def recovery_cover(self, me, mechs, deadline):
        """Return None when exposed, True under cover, or False with safe separation."""
        x, y, z = me["pos"]
        top = (x, y, z + me["stats"].get("height", 9) * 0.85)
        covered = False
        for enemy in mechs:
            tx, ty, tz = enemy["pos"]
            if math.hypot(tx - x, ty - y) > 160 or not enemy.get("weapons", ("gun",)):
                continue
            muzzle = (tx, ty, tz + 7 * (1 + 0.05 * (enemy.get("cubes") or 0)))
            if time.perf_counter() >= deadline or self.nav.combat_los(muzzle, top):
                return None
            covered = True
        return covered

    def cover_direction(self, obs, me, angles, away, deadline):
        """Find a short walk that breaks known firing lines without replacing the flow field."""
        if not me.get("grounded") or any(origin is None for _, origin in angles):
            return None
        x, y = me["pos"][:2]
        ax, ay = away
        if away == (0.0, 0.0):
            ax, ay = -math.cos(angles[0][0]), -math.sin(angles[0][0])
        zone = obs["zone"]
        candidates = [self.cover_goal] if self.cover_goal else []
        for distance in (12, 20):
            candidates.extend((x + dx * distance, y + dy * distance) for dx, dy in (
                (-ay, ax), (ay, -ax), (ax, ay), norm2(ax - ay, ay + ax), norm2(ax + ay, ay - ax)))
        for cx, cy in candidates:
            if time.perf_counter() >= deadline:
                break
            distance = math.hypot(cx - x, cy - y)
            if not 2 < distance <= 24 or math.dist((cx, cy), zone["center"]) >= zone["radius"] - 2:
                continue
            if (obs["t"] >= zone["shrink_start"]
                    and math.dist((cx, cy), zone["next_center"]) > max(
                        zone["next_radius"] - 2, math.dist((x, y), zone["next_center"]))):
                continue
            steps = math.ceil(distance / 2)
            if any(not self.nav.free(x + (cx - x) * i / steps, y + (cy - y) * i / steps)
                   for i in range(1, steps + 1)):
                continue
            behind = (cx, cy, self.nav.ground_h(cx, cy) + me["stats"].get("height", 9) * 0.75)
            if all(time.perf_counter() < deadline and not self.nav.combat_los(origin, behind)
                   for _, origin in angles):
                self.cover_goal = (cx, cy)
                return norm2(cx - x, cy - y)
        self.cover_goal = None
        return None

    def combat_fire(self, act, me, slots, plans, allow_charge, t):
        """Choose the shared torso command first, then gate every trigger against it."""
        fire = [False, False]
        laser = slots.get("laser")
        charge = (laser[1].get("charge") or 0) if laser else 0
        beam = next((p for p in plans if p.kind == "laser"), None)
        if charge:
            if beam is None:
                act["aim"] = list(me["aim"])
                return fire
            plan = beam
        else:
            plan = min((p for p in plans if allow_charge or p.kind != "laser"),
                       key=lambda p: p.impact_in + 0.35 * min(p.finish_in, 4), default=None)
        if plan is None:
            return fire
        act["aim"] = list(plan.aim)
        if plan.kind == "laser":
            ws = laser[1]["stats"]
            full = ws.get("charge", 0.75)
            if self.laser_started is None:
                self.laser_started = t - charge
            aligned = plan.aligned(me["aim"])
            hold = (allow_charge and plan.ready_in <= 0.05
                    and (aligned or charge > 0 or plan.impact_in <= full + 0.15)
                    and (charge < full or not aligned)
                    and t - self.laser_started < full + min(0.4, ws.get("hold_s", 2)))
            fire[plan.slot] = hold
        for other in plans:
            if (other.kind != "laser" and other.ready_in <= 0.10
                    and other.aligned(plan.aim) and other.aligned(me["aim"])):
                fire[other.slot] = True
        return fire

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        me = obs["self"]
        t = obs["t"]
        dt = max(0.05, t - self.prev_t) if self.prev_t else 0.1
        self.prev_t = t
        x, y, z = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act = {"tick": obs["tick"]}
        dbg = []

        # upgrades
        off = obs.get("offer")
        if off:
            ids = [c["id"] for c in off["choices"]]
            best = min(range(len(ids)), key=lambda i: UPGRADE_PREF.index(ids[i]) if ids[i] in UPGRADE_PREF else 99)
            act["pick"] = best

        # threats from events and sounds
        for ev in obs.get("events") or []:
            if (ev["type"] == "damaged" and ev.get("by") not in (None, me["id"])
                    and ev.get("bearing") is not None):
                self.threat[ev["by"]] = (t, ev["bearing"], ev.get("weapon"))
                self.last_hit_t, self.last_hit_bearing = t, ev["bearing"]
            elif ev["type"] == "kill":
                self.threat.pop(ev.get("target"), None)
            if (ev["type"] in ("damaged", "hit") and t < self.recover_until
                    and hp_frac < 0.9):
                self.recover_until = t + self.cfg["mech"].get("regen_delay", 5) + 1.5
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] == "near":
                if self.sound_look is None or self.sound_look[0] < t:
                    self.sound_look = (t + 1.0, s["bearing"])

        mechs = [m for m in obs.get("mechs") or [] if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs") or [] if m["kind"] == "npc"]
        slots = self.weapon_slots(me)
        laser = slots.get("laser")
        charging = bool(laser and laser[1].get("charge"))
        if not charging:
            self.laser_started = None

        def d2(p):
            return math.dist((x, y), (p[0], p[1]))

        # Keep the pressure that controls our legs independent of the selected shot.
        travel = self.must_travel(obs, me)
        zone = obs["zone"]
        urgent = travel and (not me.get("in_zone", True)
                             or d2(zone["next_center"]) > zone["next_radius"] - 2)
        angles = self.firing_angles(me, mechs, t, t_start + 0.003)
        crossfire = any(abs(wrap(a - b)) >= 0.55 for i, (a, _) in enumerate(angles)
                        for b, _ in angles[i + 1:])
        vulnerable = bool(angles) and (hp_frac < 0.53 or (hp_frac < 0.75 and me["energy"] < 12))
        allow_charge = not (urgent or crossfire or vulnerable or hp_frac < 0.35
                            or (angles and t < self.retreat_until))
        target, tdist, plans = self.choose_target(me, mechs, slots, t, allow_charge, t_start + 0.006)
        goal = self.zone_goal(obs, me)
        mode = "zone"
        move_to = goal

        # ---------------- farming / looting when safe
        nearest_mech = min((d2(m["pos"]) for m in mechs), default=1e9)
        safe = not travel and nearest_mech > 90 and hp_frac > 0.45
        farm_target = None
        if safe:
            empty = any(w is None for w in me["weapons"])
            picks = []
            for p in obs.get("pickups") or []:
                if p["kind"] in slots or p["kind"] == "gun":
                    continue
                gain = WEAPON_PREF.get(p["kind"], 0)
                if not empty:
                    worst = min((WEAPON_PREF.get(k, 0) for k in slots if k != "gun"), default=99)
                    if gain <= worst:
                        continue
                if abs(p["pos"][2] - self.nav.ground_h(p["pos"][0], p["pos"][1])) > 3:
                    continue
                if d2(p["pos"]) < 70:
                    picks.append(p)
            cubes = [c for c in obs.get("cubes") or [] if d2(c["pos"]) < 60 and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
            if picks:
                p = min(picks, key=lambda p: d2(p["pos"]))
                move_to, mode = (p["pos"][0], p["pos"][1]), "loot"
                if d2(p["pos"]) < self.cfg["loot"]["pickup_radius"] - 0.3:
                    act["take"] = p["id"]
                    if not empty:
                        worst_slot = min((s for k, (s, w) in slots.items() if k != "gun"),
                                         key=lambda s: WEAPON_PREF.get(me["weapons"][s]["kind"], 0))
                        act["drop"] = worst_slot
            elif cubes:
                c = min(cubes, key=lambda c: d2(c["pos"]))
                move_to, mode = (c["pos"][0], c["pos"][1]), "cube"
            if t < 260 and hp_frac > 0.6 and not mechs:
                npc_c = [n for n in npcs if d2(n["pos"]) < 70]
                crates = [c for c in obs.get("crates") or [] if d2(c["pos"]) < 70
                          and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
                if npc_c:
                    farm_target = dict(min(npc_c, key=lambda n: d2(n["pos"])))
                elif crates and mode == "zone":
                    c = min(crates, key=lambda c: d2(c["pos"]))
                    farm_target = {"kind": "crate", "pos": c["pos"], "vel": [0, 0, 0], "id": c["id"]}
        # supply drop: only if very close and nobody around
        if safe and mode == "zone" and not mechs:
            for dr in obs.get("drops") or []:
                if dr["lands_in"] <= 3 and d2(dr["pos"]) < 70 and (dr.get("holder") in (None, me["id"])):
                    move_to, mode = (dr["pos"][0], dr["pos"][1]), "drop"

        # ---------------- movement
        gx, gy = move_to
        dgoal = math.dist((x, y), (gx, gy))
        mv = (0.0, 0.0)
        if mode == "zone":
            if dgoal > 6:
                s = self.nav.steer(x, y, (gx, gy))
                mv = norm2(*(s if s else (gx - x, gy - y)))
        else:
            if dgoal > 1.0:
                s = None
                if dgoal > 15:
                    s = None  # direct is fine for short hops; walls slide us
                mv = norm2(gx - x, gy - y)
        boost = not me.get("in_zone", True) or travel and dgoal > 60

        # Briefly commit to exits and shots, with immediate interrupts for zone and crossfire.
        duel = target is not None and tdist < 160
        finish = min((p.finish_in for p in plans), default=math.inf) < 1.4
        danger = crossfire or (not finish and (vulnerable or (duel and hp_frac < 0.35)))
        if danger:
            self.retreat_until = t + 0.55
            if hp_frac < 0.9:
                self.recover_until = t + self.cfg["mech"].get("regen_delay", 5) + 1.5
        retreat = bool(danger or (angles and t < self.retreat_until))
        recovery = None
        if (not urgent and not retreat and not angles and hp_frac < 0.9
                and t < self.recover_until and me.get("grounded")):
            recovery = self.recovery_cover(me, mechs, t_start + COMBAT_BUDGET_S)
        recover = recovery is not None
        cover_move = None
        if recover:
            mode = "recover"
            if recovery:
                mv, boost = (0.0, 0.0), False
        elif not urgent and (retreat or duel):
            cover_angles = angles
            if not cover_angles:
                tx, ty, tz = target["pos"]
                cover_angles = [(math.atan2(ty - y, tx - x), (tx, ty, tz + 7))]
            ax, ay = norm2(-sum(math.cos(a) for a, _ in cover_angles),
                           -sum(math.sin(a) for a, _ in cover_angles))
            cover_move = self.cover_direction(obs, me, cover_angles, (ax, ay),
                                              t_start + COMBAT_BUDGET_S)
            if retreat:
                if (ax, ay) == (0.0, 0.0):
                    bearing = cover_angles[0][0]
                    ax, ay = -math.sin(bearing), math.cos(bearing)
                mv = cover_move or norm2(ax + mv[0] * 0.35, ay + mv[1] * 0.35)
                boost = bool(angles) and me.get("grounded") and me["energy"] > 25
                mode = "retreat"
            else:
                tyaw = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
                px, py = -math.sin(tyaw), math.cos(tyaw)
                if t > self.strafe_until:
                    self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
                    self.strafe_until = t + self.rng.uniform(0.6, 1.6)
                sx, sy = px * self.strafe_sign, py * self.strafe_sign
                if cover_move and sx * cover_move[0] + sy * cover_move[1] < 0:
                    sx, sy = -sx, -sy
                # Close on lightning range only with an exit and enough energy for it.
                want_close = ("lightning" in slots and 35 < tdist < 55 and cover_move
                              and me["energy"] > 30)
                fx, fy = (math.cos(tyaw), math.sin(tyaw)) if want_close else (0.0, 0.0)
                base = mv
                mv = norm2(sx + base[0] * 0.5 + fx * 0.6, sy + base[1] * 0.5 + fy * 0.6)
                if not self.nav.free(x + mv[0] * 4, y + mv[1] * 4):
                    self.strafe_sign = -self.strafe_sign
                    self.strafe_until = t + 1.0
                    if cover_move:
                        mv = cover_move
                    elif self.nav.free(x - sx * 4, y - sy * 4):
                        mv = (-sx, -sy)
                    else:
                        mv = base
                mode = "fight"
        else:
            self.cover_goal = None
        fighting = mode == "fight"
        # NPC avoidance when not farming them
        if not recovery and (farm_target is None or farm_target.get("kind") != "npc"):
            for n in npcs:
                dn = d2(n["pos"])
                if dn < 50:
                    ax, ay = norm2(x - n["pos"][0], y - n["pos"][1])
                    w = (50 - dn) / 50
                    mv = norm2(mv[0] + ax * w, mv[1] + ay * w)

        # stuck handling
        speed = math.hypot(me["vel"][0], me["vel"][1])
        if mv != (0.0, 0.0) and speed < 1.5 and not fighting:
            self.stuck_t += dt
        else:
            self.stuck_t = max(0.0, self.stuck_t - dt)
        if self.stuck_t > 0.8 and t > self.unstick_until:
            ang = math.atan2(mv[1], mv[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.8, 1.8)
            self.unstick_dir = (math.cos(ang), math.sin(ang))
            self.unstick_until = t + 0.8
            self.stuck_t = 0.0
        if t < self.unstick_until and not (recovery or cover_move or (urgent and (duel or angles))):
            mv = self.unstick_dir
            if me.get("grounded") and me["energy"] > 30:
                act["jump"] = True
            boost = me["energy"] > 20
        elif self.stuck_t > 0.4 and me.get("grounded"):
            act["jump"] = True
        if not me.get("grounded") and me["energy"] > 15 and self.stuck_t > 0.3:
            boost = True

        act["move"] = [round(mv[0], 3), round(mv[1], 3)]
        act["boost"] = bool(boost)

        # ---------------- aiming and firing
        fire = [False, False]
        shoot_at = farm_target if target is None and not (recover or charging) else None
        if not recover and (target is not None or charging):
            fire = self.combat_fire(act, me, slots, plans, allow_charge, t)

        if shoot_at is not None:
            gun = slots.get("gun")
            gspeed = gun[1]["stats"]["speed"] if gun else 252
            yaw, pitch, hd = self.lead(me, shoot_at, gspeed)
            act["aim"] = [yaw, max(-0.69, min(1.29, pitch))]
            aligned = self.aligned(me, yaw, pitch, hd, 1.4)
            eye = (x, y, z + me["stats"].get("muzzle_height", 7))
            tp = (shoot_at["pos"][0], shoot_at["pos"][1], shoot_at["pos"][2] + 3)
            clear = self.nav.los(eye, tp) if hd < 230 else False
            if gun and aligned and clear and hd < gun[1]["stats"].get("range", 230):
                if gun[1].get("reload", 0) == 0 and gun[1].get("ammo", 1) > 0:
                    fire[gun[0]] = True
            lt = slots.get("lightning")
            if lt:
                i, w = lt
                if hd < w["stats"].get("range", 45) * 0.95 and abs(wrap(me["aim"][0] - yaw)) < math.radians(w["stats"].get("cone_deg", 30)) * 0.45 \
                        and me["energy"] > 10 and clear:
                    fire[i] = True
        if "aim" not in act:
            # no target: look at threats, then sounds, else scan along travel
            look = None
            if self.last_hit_bearing is not None and t - self.last_hit_t < 3:
                look = self.last_hit_bearing
            elif self.sound_look and self.sound_look[0] > t:
                look = self.sound_look[1]
            if look is None:
                base = math.atan2(mv[1], mv[0]) if mv != (0.0, 0.0) else self.scan_yaw
                self.scan_yaw = wrap(self.scan_yaw + 1.6 * dt)
                look = wrap(base + 1.0 * math.sin(t * 0.9)) if mv != (0.0, 0.0) else self.scan_yaw
            act["aim"] = [look, 0.0]
            gun = slots.get("gun")
            if gun and gun[1].get("reload", 0) == 0 and gun[1].get("ammo", 99) < gun[1]["stats"].get("mag", 12) * 0.5:
                act["reload"] = True
        act["fire"] = fire

        self.nav.work(max(0.0, 0.015 - (time.perf_counter() - t_start)))
        mf.draw.label(f"{mode} hp{hp_frac:.0%} {'T' + str(target['id']) if target else ''}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
