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

import mechfight as mf
from nav import Nav

UPGRADE_PREF = ["armor", "plating", "nanites", "gun_rate", "gun_damage", "laser_damage", "laser_charge",
                "servos", "gun_mag", "gun_velocity", "sensors", "gyros", "reactor", "capacitor", "jets"]
WEAPON_PREF = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}


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
        self.laser_hold = 0.0

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
        if disc < 0:
            return None
        ang = math.atan2(v * v - math.sqrt(disc), g * x2)
        return math.atan2(ty2 - my, tx2 - mx), ang

    def aligned(self, me, yaw, pitch, dist, tol_r=2.5):
        ay, ap = me["aim"]
        tol = math.atan2(tol_r, max(dist, 1.0)) + 0.01
        return abs(wrap(ay - yaw)) < tol and abs(ap - pitch) < tol + 0.03

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
            if ev["type"] == "damaged" and ev.get("by") is not None and ev.get("bearing") is not None:
                self.threat[ev["by"]] = (t, ev["bearing"])
                self.last_hit_t, self.last_hit_bearing = t, ev["bearing"]
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] == "near":
                if self.sound_look is None or self.sound_look[0] < t:
                    self.sound_look = (t + 1.0, s["bearing"])

        mechs = [m for m in obs.get("mechs") or [] if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs") or [] if m["kind"] == "npc"]
        slots = self.weapon_slots(me)

        def d2(p):
            return math.dist((x, y), (p[0], p[1]))

        travel = self.must_travel(obs, me)
        goal = self.zone_goal(obs, me)
        mode = "zone"
        move_to = goal

        # An attacker may still be tracked after moving behind a wall.
        contacts = [(d2(m["pos"]), m) for m in mechs]
        attackers = {mid for mid, hit in self.threat.items() if 0 <= mid < 16 and t - hit[0] < 4}
        nearby = attackers | {m["id"] for distance, m in contacts if distance < 130}
        outnumbered = len(nearby) > 1
        low_energy = me["energy"] < 0.20 * me.get("max_energy", 100)
        muzzle = (x, y, z + me["stats"].get("muzzle_height", 7))

        def shot_clear(actor):
            px, py, pz = actor["pos"]
            return self.nav.los(muzzle, (px, py, pz + 3))

        ranked = []
        for distance, m in contacts:
            if distance >= 230:
                continue
            recent = m["id"] in attackers
            bearing = math.atan2(y - m["pos"][1], x - m["pos"][0])
            aim_error = abs(wrap(bearing - m["aim"][0]))
            defend = recent or distance < 55 or (distance < 160 and aim_error < 0.45)
            enemy_max_hp = max(1, m.get("max_hp", 1400))
            enemy_hp = m.get("hp", enemy_max_hp)
            weak = enemy_hp <= min(me["hp"], 0.45 * enemy_max_hp)
            advantage = me["hp"] > 1.10 * enemy_hp or (
                aim_error > math.pi / 2 and me["hp"] >= 0.85 * enemy_hp
            )
            attack = (not travel and not (nearby - {m["id"]}) and hp_frac >= 0.60 and not low_energy
                      and ((weak and distance < 180) or (advantage and distance < 160)))
            if defend or attack or (outnumbered and m["id"] in nearby):
                retreat = outnumbered or low_energy or hp_frac < 0.45 or me["hp"] < 0.70 * enemy_hp
                rank = (not recent, not defend, distance - 40 * (1 - enemy_hp / enemy_max_hp))
                ranked.append((rank, m, distance, defend or attack, retreat, attack))

        target = None
        tdist = 1e9
        engage = retreat = push = False
        mortar = slots.get("mortar")
        for _, m, distance, shoot, withdraw, advantage in sorted(ranked, key=lambda row: row[0])[:3]:
            # A ready mortar can still lob at a recent attacker without a direct ray.
            indirect = m["id"] in attackers and mortar and 25 < distance < 220 and mortar[1].get("cooldown", 0) <= 0.1
            if shot_clear(m) or indirect:
                target, tdist = m, distance
                engage, retreat, push = shoot, withdraw, advantage
                break

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

        # combat movement: strafe / retreat
        fighting = target is not None and me.get("in_zone", True) and (tdist < 160 or retreat)
        if fighting:
            ty_ = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            px, py = -math.sin(ty_), math.cos(ty_)
            if t > self.strafe_until:
                self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
                self.strafe_until = t + self.rng.uniform(0.6, 1.6)
            sx, sy = px * self.strafe_sign, py * self.strafe_sign
            if retreat:
                ax, ay = -math.cos(ty_), -math.sin(ty_)
                mv = norm2(ax * 0.7 + sx * 0.5 + mv[0] * 0.6, ay * 0.7 + sy * 0.5 + mv[1] * 0.6)
                boost = me["energy"] > 25
                mode = "retreat"
            else:
                zone = obs["zone"]
                want_close = (push and "lightning" in slots and 35 < tdist < 75
                              and math.dist(target["pos"][:2], zone["next_center"]) <= zone["next_radius"])
                fx, fy = (math.cos(ty_), math.sin(ty_)) if want_close else (0.0, 0.0)
                mv = norm2(sx + mv[0] * 0.5 + fx * 0.6, sy + mv[1] * 0.5 + fy * 0.6)
                mode = "fight"
            # do not strafe into walls
            if self.nav.ready and not self.nav.free(x + mv[0] * 4, y + mv[1] * 4):
                self.strafe_sign = -self.strafe_sign
                self.strafe_until = t + 1.0
        # NPC avoidance when not farming them
        if farm_target is None or farm_target.get("kind") != "npc":
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
        if t < self.unstick_until:
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
        shoot_at = target if engage else farm_target

        if shoot_at is not None:
            gun = slots.get("gun")
            gspeed = gun[1]["stats"]["speed"] if gun else 252
            yaw, pitch, hd = self.lead(me, shoot_at, gspeed)
            act["aim"] = [yaw, max(-0.69, min(1.29, pitch))]
            aligned = self.aligned(me, yaw, pitch, hd, 2.0 if shoot_at.get("kind") == "mech" else 1.4)
            clear = hd < 230 and shot_clear(shoot_at)
            if gun and aligned and clear and hd < gun[1]["stats"].get("range", 230):
                if gun[1].get("reload", 0) == 0 and gun[1].get("ammo", 1) > 0:
                    fire[gun[0]] = True
            lz = slots.get("laser")
            if lz:
                i, w = lz
                full = w["stats"].get("charge", 0.75)
                ch = w.get("charge", 0) or 0
                if shoot_at.get("kind") == "mech" and hd < w["stats"].get("range", 280) and w.get("cooldown", 0) <= 0.05:
                    if ch < full or not (aligned and clear):
                        fire[i] = ch < full + 1.5   # keep charging, release before auto-fire
                    else:
                        fire[i] = False             # release = fire
            lt = slots.get("lightning")
            if lt:
                i, w = lt
                if hd < w["stats"].get("range", 45) * 0.95 and abs(wrap(me["aim"][0] - yaw)) < math.radians(w["stats"].get("cone_deg", 30)) * 0.45 \
                        and me["energy"] > 10 and clear:
                    fire[i] = True
            mo = slots.get("mortar")
            if mo and shoot_at.get("kind") == "mech" and 25 < hd < 220:
                i, w = mo
                sol = self.mortar_aim(me, shoot_at, w["stats"])
                if sol:
                    if w.get("cooldown", 0) <= 0.1:
                        act["aim"] = [sol[0], max(-0.69, min(1.29, sol[1]))]
                        if abs(wrap(me["aim"][0] - sol[0])) < 0.03 and abs(me["aim"][1] - sol[1]) < 0.03:
                            fire[i] = True
        else:
            # no target: look at threats, then sounds, else scan along travel
            look = None
            if self.last_hit_bearing is not None and t - self.last_hit_t < 3:
                look = self.last_hit_bearing
            elif target is not None:
                look = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
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
        # never keep a laser charging without a target
        lz = slots.get("laser")
        if lz and shoot_at is None and (lz[1].get("charge") or 0) > 0:
            fire[lz[0]] = False
        act["fire"] = fire

        self.nav.work(max(0.0, 0.015 - (time.perf_counter() - t_start)))
        mf.draw.label(f"{mode} hp{hp_frac:.0%} {'T' + str(target['id']) if target else ''}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
