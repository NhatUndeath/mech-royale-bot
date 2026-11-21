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

UPGRADE_PREF = ["armor", "plating", "nanites", "servos", "gun_rate", "gun_damage", "laser_charge",
                "laser_damage", "lightning_range", "mortar_reload", "mortar_splash", "gun_mag",
                "gun_velocity", "sensors", "gyros", "reactor", "capacitor", "jets", "laser_pierce",
                "mortar_velocity", "lightning_chain", "lightning_efficiency"]
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
        self.nav = Nav(mdir, cfg["mech"])
        self.goal = None
        self.goal_key = None
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
        self.danger_until = -1.0
        self.cover_until = -1.0
        self.escape_goal = None
        self.escape_phase = None

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
            # Rotate to the near side with room for error; keep shorter routes in small circles.
            r = (0.72 if nr > 55 else 0.50) * nr
            if d <= r:
                g = (x, y)
            else:
                ux, uy = norm2(x - nc[0], y - nc[1])
                g = (nc[0] + ux * r, nc[1] + uy * r)
            if self.nav.ready and nr > 4:
                cover_radius = (0.82 if nr > 55 else 0.68) * nr
                g = self.nav.cover_spot(nc, cover_radius, g, self.rng) or g
            if self.nav.ready:
                i, j = self.nav.cell(*g)
                if not self.nav.main[j * self.nav.gx + i]:
                    g = self.nav.center(*self.nav.nearest_free(*g, 80, main=True))
            self.goal, self.goal_key = g, key
        return self.goal

    def must_travel(self, obs, me):
        """True when the zone leaves no time for detours."""
        z = obs["zone"]
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        if not me.get("in_zone", True):
            return True
        if z["next_radius"] <= 4 and (t >= z["shrink_start"] or z["radius"] <= 8):
            return True
        need = max(0.0, math.dist((x, y), z["next_center"]) - 0.75 * z["next_radius"])
        if need < 0.1:
            return False
        route = self.nav.route_length(x, y, self.goal)
        if route is not None:
            goal_margin = max(0, 0.75 * z["next_radius"] - math.dist(self.goal, z["next_center"]))
            need = max(need, route - goal_margin)
        speed = me["stats"].get("walk_speed", 11.7) * 0.6
        avail = max(0.0, z.get("shrink_end", t) - t) * speed
        return need > avail - 40

    def closing_mechs(self, me, mechs):
        """Nearby game actors that are already close or will be close in two match seconds."""
        nearby = []
        for actor in mechs:
            dx, dy = actor["pos"][0] - me["pos"][0], actor["pos"][1] - me["pos"][1]
            vx = actor["vel"][0] - me["vel"][0]
            vy = actor["vel"][1] - me["vel"][1]
            dv2 = vx * vx + vy * vy
            ahead = max(0, min(2, -(dx * vx + dy * vy) / dv2)) if dv2 else 0.0
            d = math.hypot(dx, dy)
            if d < 115 or math.hypot(dx + vx * ahead, dy + vy * ahead) < 100:
                nearby.append((d, actor))
        return [actor for _, actor in sorted(nearby, key=lambda item: item[0])]

    def escape_spot(self, obs, nearby, zone_goal):
        x, y, _ = obs["self"]["pos"]
        zone = obs["zone"]
        if not self.nav.ready or not nearby or zone["next_radius"] <= 4:
            return zone_goal
        if obs["t"] >= self.cover_until or self.escape_phase != self.goal_key:
            threats = [m["pos"] for m in nearby[:2]]
            positions = [(px, py, pz + self.cfg["mech"].get("eye_height", 8)) for px, py, pz in threats]
            ax, ay = norm2(x - threats[0][0], y - threats[0][1])
            near = (x + ax * 24, y + ay * 24)
            circle = (zone["center"], max(0, zone["radius"] - 2))
            if math.dist((x, y), zone["next_center"]) < zone["next_radius"]:
                circle = (zone["next_center"], max(0, zone["next_radius"] - 2))
            self.escape_goal = self.nav.cover_spot(
                (x, y), 42, near, self.rng, 64, threats=positions, me_pos=(x, y),
                safe_circle=circle, near_range=self.cfg["vision"]["near_range"]) or zone_goal
            self.cover_until = obs["t"] + 1.5
            self.escape_phase = self.goal_key
        return self.escape_goal

    @staticmethod
    def target_center(tgt):
        tx, ty, tz = tgt["pos"]
        h = 4.5 if tgt["kind"] == "mech" else 1.75
        if tgt["kind"] == "mech":
            h *= 1 + 0.05 * (tgt.get("cubes") or 0)
        return tx, ty, tz + h

    def lead(self, me, tgt, speed=None):
        mx, my, mz = me["pos"]
        mz += me["stats"].get("muzzle_height", 7)
        tx, ty, tz = self.target_center(tgt)
        vx, vy, vz = tgt.get("vel", [0, 0, 0])
        px, py, pz = tx, ty, tz
        if speed:
            for _ in range(3):
                t = math.dist((mx, my, mz), (px, py, pz)) / speed + 0.05
                px, py, pz = tx + vx * t, ty + vy * t, tz + (vz * t if abs(vz) < 3 else 0)
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

    def fire_control(self, act, me, target, slots, escaping):
        """The two game slots share a torso. Pick its direction before setting either trigger."""
        x, y, z = me["pos"]
        muzzle = (x, y, z + me["stats"].get("muzzle_height", 7))
        min_pitch = me["stats"].get("min_pitch", -0.7)
        max_pitch = me["stats"].get("max_pitch", 1.3)
        direct = self.lead(me, target)
        yaw, pitch, hd = direct
        current_point = self.target_center(target)
        distance = math.dist(muzzle, current_point)

        def clear_gun_ray(solution):
            sy, sp, horizontal = solution
            end = (x + math.cos(sy) * horizontal, y + math.sin(sy) * horizontal,
                   muzzle[2] + math.tan(sp) * horizontal)
            return self.nav.los(muzzle, end)

        fire = [False, False]
        gun = slots.get("gun")
        gun_aim = self.lead(me, target, gun[1]["stats"]["speed"]) if gun else direct
        aim = gun_aim[:2]
        laser = slots.get("laser")
        lightning = slots.get("lightning")
        mortar = slots.get("mortar")
        if laser and target["kind"] == "mech" and not escaping:
            i, w = laser
            stats = w["stats"]
            if (distance <= stats["range"] and min_pitch <= pitch <= max_pitch
                    and w.get("cooldown", 0) <= 0.05 and self.nav.los(muzzle, current_point)):
                aim = (yaw, pitch)
                on_target = self.aligned(me, yaw, pitch, hd)
                # Releasing a charged laser fires it along the current torso aim.
                charge = w.get("charge", 0) or 0
                fire[i] = (charge > 0 or on_target) and (charge < stats["charge"] or not on_target)
        elif lightning:
            i, w = lightning
            stats = w["stats"]
            if distance <= stats["range"] and me["energy"] >= stats.get("energy", 3) + (15 if escaping else 0):
                aim = (yaw, pitch)
                if (abs(wrap(me["aim"][0] - yaw)) <= math.radians(stats["cone_deg"])
                        and self.nav.los(muzzle, current_point)):
                    fire[i] = w.get("cooldown", 0) <= 0.05
        elif mortar and target["kind"] == "mech" and not escaping:
            i, w = mortar
            if (hd > max(25, 2 * w["stats"].get("splash_radius", 7))
                    and w.get("cooldown", 0) <= 0.1 and self.nav.los(muzzle, current_point)):
                sol = self.mortar_aim(me, target, w["stats"])
                if sol and min_pitch <= sol[1] <= max_pitch:
                    aim = sol
                    fire[i] = self.aligned(me, *sol, hd, tol_r=1.5)
        act["aim"] = [aim[0], max(min_pitch, min(max_pitch, aim[1]))]
        if gun:
            i, w = gun
            gy, gp, gh = gun_aim
            if (min_pitch <= gp <= max_pitch and gh <= w["stats"]["range"] * math.cos(gp)
                    and w.get("reload", 0) == 0 and w.get("ammo", 1) > 0
                    and self.aligned(me, gy, gp, gh) and self.aligned(act, gy, gp, gh)
                    and clear_gun_ray(gun_aim)):
                fire[i] = True
        return fire

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        me = obs["self"]
        t = obs["t"]
        dt = max(0.0, min(0.25, t - self.prev_t)) if self.prev_t else 0.1
        self.prev_t = t
        x, y, z = me["pos"]
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        act = {"tick": obs["tick"]}

        # upgrades
        off = obs.get("offer")
        if off and off.get("choices"):
            ids = [c["id"] for c in off["choices"]]
            best = min(range(len(ids)), key=lambda i: UPGRADE_PREF.index(ids[i]) if ids[i] in UPGRADE_PREF else 99)
            act["pick"] = best

        # threats from events and sounds
        for ev in obs.get("events") or []:
            if ev["type"] == "damaged" and ev.get("by") is not None and ev.get("bearing") is not None:
                self.threat[ev["by"]] = (t, ev["bearing"])
                self.last_hit_t, self.last_hit_bearing = t, ev["bearing"]
                if ev.get("weapon") != "npc":
                    self.danger_until = max(self.danger_until, t + 2.5)
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] == "near":
                if self.sound_look is None or self.sound_look[0] < t:
                    self.sound_look = (t + 1.0, s["bearing"])
                if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar"):
                    self.danger_until = max(self.danger_until, t + 1.0)

        mechs = [m for m in obs.get("mechs") or [] if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs") or [] if m["kind"] == "npc"]
        slots = self.weapon_slots(me)
        nearby = self.closing_mechs(me, mechs)
        crowded = len(nearby) >= 2
        if crowded:
            self.danger_until = t + 2.5

        def d2(p):
            return math.dist((x, y), (p[0], p[1]))

        # ---------------- target choice
        target = None
        engage = False
        if mechs:
            def score(m):
                d = d2(m["pos"])
                s = d
                if m["id"] in self.threat and t - self.threat[m["id"]][0] < 4:
                    s -= 60
                s -= 40 * (1 - (m.get("hp", 1400) / max(1, m.get("max_hp", 1400))))
                attacked = m["id"] in self.threat and t - self.threat[m["id"]][0] < 5
                weak = m.get("hp", 1400) < 0.35 * max(1, m.get("max_hp", 1400))
                engageable = attacked or d < 60 or (weak and hp_frac > 0.6 and d < 150 and not crowded)
                return (not engageable, s)
            priority, target = min(((score(m), m) for m in mechs), key=lambda row: row[0])
            engage = not priority[0]
        tdist = d2(target["pos"]) if target else 1e9

        goal = self.zone_goal(obs, me)
        travel = self.must_travel(obs, me)
        mode = "zone"
        move_to = goal

        # ---------------- farming / looting when safe
        nearest_mech = min((d2(m["pos"]) for m in mechs), default=1e9)
        wants_cover = crowded or (bool(nearby) and hp_frac < 0.5)
        safe = not (travel or wants_cover) and t >= self.danger_until and nearest_mech > 90 and hp_frac > 0.45
        def affordable_detour(pos):
            zone = obs["zone"]
            xy = pos[:2]
            if math.dist(xy, zone["center"]) > zone["radius"]:
                return False
            back = max(0, math.dist(xy, zone["next_center"]) - 0.75 * zone["next_radius"])
            if back == 0:
                return True
            return d2(pos) + back < max(0, zone["shrink_end"] - t - 2) * me["stats"]["walk_speed"] * 0.65
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
                if d2(p["pos"]) < (120 if t < 200 else 70) and affordable_detour(p["pos"]):
                    picks.append(p)
            cubes = [c for c in obs.get("cubes") or [] if d2(c["pos"]) < 60 and affordable_detour(c["pos"])
                     and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
            if picks:
                p = min(picks, key=lambda p: d2(p["pos"]) / (1 + 0.3 * WEAPON_PREF.get(p["kind"], 0)))
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
            # farm NPC drones (2 cubes) and crates (1 cube + maybe a weapon) while it is quiet
            if t < 260 and hp_frac > 0.6 and nearest_mech > 150 and not engage:
                npc_c = [n for n in npcs if d2(n["pos"]) < 70]
                crates = [c for c in obs.get("crates") or [] if d2(c["pos"]) < 70
                          and abs(c["pos"][2] - self.nav.ground_h(c["pos"][0], c["pos"][1])) < 3]
                if npc_c:
                    farm_target = dict(min(npc_c, key=lambda n: d2(n["pos"])))
                elif crates and mode == "zone":
                    c = min(crates, key=lambda c: d2(c["pos"]))
                    farm_target = {"kind": "crate", "pos": c["pos"], "vel": [0, 0, 0], "id": c["id"]}
                if farm_target and mode == "zone" and affordable_detour(farm_target["pos"]):
                    move_to, mode = tuple(farm_target["pos"][:2]), "farm"
        # supply drop: only if very close and nobody around
        if safe and mode == "zone" and not mechs:
            for dr in obs.get("drops") or []:
                if (dr["lands_in"] <= 3 and d2(dr["pos"]) < 70 and dr.get("holder") in (None, me["id"])
                        and affordable_detour(dr["pos"])):
                    move_to, mode = (dr["pos"][0], dr["pos"][1]), "drop"

        # ---------------- movement
        if not travel and wants_cover:
            move_to, mode = self.escape_spot(obs, nearby, goal), "cover"
        elif (not travel and t < min(self.danger_until, self.cover_until + 1) and self.escape_goal
              and self.escape_phase == self.goal_key):
            move_to, mode = self.escape_goal, "cover"
        seeking_cover = mode == "cover"
        direct_detour = mode != "zone" and self.nav.walk_clear(x, y, move_to)
        self.nav.request(goal if direct_detour else move_to)
        gx, gy = move_to
        dgoal = math.dist((x, y), (gx, gy))
        mv = (0.0, 0.0)
        stop_dist = min(6.0, max(0.1, obs["zone"]["next_radius"] * 0.10)) if mode == "zone" else 1.0
        if mode == "farm":
            stop_dist = 50 if farm_target["kind"] == "npc" else 18
        if dgoal > stop_dist or not me["in_zone"] and dgoal > 0.01:
            offset = self.nav.steer(x, y, (gx, gy)) if dgoal > 4 and not direct_detour else None
            mv = norm2(*(offset or (gx - x, gy - y)))
        boost = travel and dgoal > (5 if not me["in_zone"] else 60)

        # combat movement: strafe / retreat
        if target is not None:
            if not engage and tdist < 180 and not (travel or seeking_cover):
                # stay out of it: move away from them, keep facing them
                ax, ay = norm2(x - target["pos"][0], y - target["pos"][1])
                w = (180 - tdist) / 180
                mv = norm2(mv[0] + ax * w * 1.5, mv[1] + ay * w * 1.5)
                mode = "avoid"
        fighting = engage and tdist < 160
        if fighting and not (travel or seeking_cover):
            ty_ = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            px, py = -math.sin(ty_), math.cos(ty_)
            if t > self.strafe_until:
                self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
                self.strafe_sign = self.nav.strafe_side(x, y, px, py, self.strafe_sign)
                self.strafe_until = t + self.rng.uniform(0.6, 1.6)
            sx, sy = px * self.strafe_sign, py * self.strafe_sign
            their = target.get("hp", 1400) / max(1, target.get("max_hp", 1400))
            retreat = hp_frac < 0.5 and their > hp_frac - 0.05
            if retreat:
                ax, ay = -math.cos(ty_), -math.sin(ty_)
                mv = norm2(ax * 0.7 + sx * 0.5 + mv[0] * 0.6, ay * 0.7 + sy * 0.5 + mv[1] * 0.6)
                boost = me["energy"] > 25
                mode = "retreat"
            else:
                want_close = "lightning" in slots and tdist > slots["lightning"][1]["stats"]["range"] * 0.82
                fx, fy = (math.cos(ty_), math.sin(ty_)) if want_close else (0.0, 0.0)
                mv = norm2(sx + mv[0] * 0.5 + fx * 0.6, sy + mv[1] * 0.5 + fy * 0.6)
                mode = "fight"
        # NPC avoidance when not farming them
        if not (travel or seeking_cover) and (farm_target is None or farm_target.get("kind") != "npc"):
            for n in npcs:
                dn = d2(n["pos"])
                if dn < 70:
                    ax, ay = norm2(x - n["pos"][0], y - n["pos"][1])
                    w = 1.5 * (70 - dn) / 70
                    mv = norm2(mv[0] + ax * w, mv[1] + ay * w)

        # stuck handling
        speed = math.hypot(me["vel"][0], me["vel"][1])
        if mv != (0.0, 0.0) and speed < 1.5 and (mode != "zone" or dgoal > 2):
            self.stuck_t += dt
        else:
            self.stuck_t = max(0.0, self.stuck_t - dt)
        if self.stuck_t > 0.8 and t > self.unstick_until:
            ang = math.atan2(mv[1], mv[0]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.8, 1.8)
            self.unstick_dir = (math.cos(ang), math.sin(ang))
            self.unstick_until = t + 0.8
            self.stuck_t = 0.0
        forward_escape = self.unstick_dir[0] * mv[0] + self.unstick_dir[1] * mv[1] > 0
        if t < self.unstick_until and (mode != "zone" or dgoal > 2) and (not travel or forward_escape):
            mv = self.unstick_dir
            if me.get("grounded") and me["energy"] > 30:
                act["jump"] = True
            boost = me["energy"] > 20
        elif self.stuck_t > 0.4 and me.get("grounded"):
            act["jump"] = True
        if not me.get("grounded") and me["energy"] > 15 and self.stuck_t > 0.3:
            boost = True

        zone = obs["zone"]
        if not travel and me["in_zone"]:
            step = 0.3 * me["stats"]["walk_speed"]
            ahead = (x + step * mv[0], y + step * mv[1])
            if math.dist(ahead, zone["center"]) > zone["radius"]:
                offset = self.nav.steer(x, y, goal)
                mv = norm2(*(offset or (zone["center"][0] - x, zone["center"][1] - y)))
        mv = self.nav.safe_step(x, y, mv, self.strafe_sign, min(6.0, dgoal) if mode == "zone" else 6.0)
        if mode == "zone" and obs["zone"]["next_radius"] < 4:
            scale = min(1.0, dgoal / max(1.0, me["stats"]["walk_speed"] * 0.2))
            mv = (mv[0] * scale, mv[1] * scale)
        act["move"] = [mv[0], mv[1]]
        act["boost"] = bool(boost)

        # ---------------- aiming and firing
        fire = [False, False]
        shoot_at = None
        if target is not None and engage:
            shoot_at = target
        elif farm_target is not None:
            shoot_at = farm_target
        elif target is not None:
            yaw, _, _ = self.lead(me, target)
            act["aim"] = [yaw, 0.0]

        if shoot_at is not None:
            fire = self.fire_control(act, me, shoot_at, slots, travel or seeking_cover)
        elif target is None:
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

        self.nav.work(max(0.0, min(0.006, 0.009 - (time.perf_counter() - t_start))))
        mf.draw.label(f"{mode} hp{hp_frac:.0%} {'T' + str(target['id']) if target else ''}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
