"""Placement-first bot for Mech Royale. Practice defaults target the house-bot lobby."""
import math
import os
import random
import sys
import time

import mechfight as mf
from nav import Nav

try:
    import params as P
except ModuleNotFoundError as exc:
    if exc.name != "params":
        raise
    P = None


def opt(name, default):
    return getattr(P, name, default)


# The protocol does not identify practice versus finals. Set PRACTICE=False
# in params.py to select the existing free-for-all engagement and loot policy.
PRACTICE = opt("PRACTICE", True)
EXPOSURE_NAV = opt("EXPOSURE_NAV", PRACTICE)
ROOFS = opt("ROOFS", PRACTICE)
GATE = opt("GATE", True)
RETREAT = opt("RETREAT", "v2")
COVER = opt("COVER", True)
FARM = opt("FARM", True)
DODGE = opt("DODGE", False)
BREAK_LOS = opt("BREAK_LOS", True)
EDGE = opt("EDGE", None)
LOOT_EARLY = opt("LOOT_EARLY", 70 if PRACTICE else 120)

UPGRADE_PREF = ["armor", "plating", "nanites", "servos", "gun_rate", "gun_damage", "laser_charge",
                "laser_damage", "lightning_range", "mortar_reload", "mortar_splash", "gun_mag",
                "gun_velocity", "sensors", "gyros", "reactor", "capacitor", "jets", "laser_pierce",
                "mortar_velocity", "lightning_chain", "lightning_efficiency"]
WEAPON_PREF = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
SURVIVAL_UPGRADES = ("armor", "servos", "plating", "nanites", "reactor", "capacitor", "jets", "sensors")
PRACTICE_UPGRADES = SURVIVAL_UPGRADES + tuple(u for u in UPGRADE_PREF if u not in SURVIVAL_UPGRADES)


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
        self.pickup_retry = {}
        self.roof_route = None
        self.next_roof_search = 0.0
        self.roof_started = 0.0
        self.roof_launched = False
        self.roof_exit = None

    # ------------------------------------------------------------------ helpers
    def weapon_slots(self, me):
        out = {}
        for i, w in enumerate(me.get("weapons") or []):
            if w:
                out[w["kind"]] = (i, w)
        return out

    def pickup_value(self, kind, late):
        # Keep the flexible gun during the economy phase. Lightning's energy
        # cost and short range matter less once the circle is small.
        return ({"gun": 3.0, "mortar": 3.3, "laser": 2.4, "lightning": 4.2} if late else
                {"gun": 4.0, "mortar": 3.1, "laser": 2.6, "lightning": 2.3}).get(kind, 0.0)

    def pickup_swap(self, kind, me, slots, late):
        if kind not in self.cfg["weapons"] or kind in slots:
            return False, None
        if any(w is None for w in me["weapons"]):
            return True, None
        if not PRACTICE:
            preference = WEAPON_PREF
            candidate = min(((preference[k], i) for k, (i, w) in slots.items()
                             if k != "gun" and not w.get("charge")), default=None)
            if candidate and preference.get(kind, 0) > candidate[0]:
                return True, candidate[1]
            return False, None
        gain = self.pickup_value(kind, late)
        equipped = [(self.pickup_value(k, late), i) for k, (i, w) in slots.items() if not w.get("charge")]
        worst = min(equipped, default=None)
        if worst and gain >= worst[0] + 0.45:
            return True, worst[1]
        return False, None

    @staticmethod
    def reserve(me, urgent=False):
        return me["max_energy"] * (0.12 if urgent else 0.30)

    def roof_action(self, obs, safe, mode, raised_loot, mechs, deadline):
        """Keep one short transition active until landing, danger, or timeout."""
        me, zone, t = obs["self"], obs["zone"], obs["t"]
        x, y, z = me["pos"]
        xy = (x, y)
        if self.roof_route is None:
            if (not safe or not me["grounded"] or mode != "zone" or t < self.next_roof_search
                    or zone["next_radius"] < 25 or time.perf_counter() >= deadline):
                return None
            self.next_roof_search = t + 1.5
            self.roof_route = self.nav.roof_transition(
                me, zone, mechs, self.cfg["vision"], min(deadline, time.perf_counter() + 0.003),
                reserve=self.reserve(me), size_per_cube=self.cfg["cubes"]["size_per_cube"],
                target=raised_loot)
            self.roof_started = t
            self.roof_launched = False
        route = self.roof_route
        if route is None:
            return None

        launch, landing, top = route["launch"], route["landing"], route["top"]
        since_start = t - self.roof_started
        if (since_start > (4.5 if self.roof_launched else 4.0) or not safe or
                me["grounded"] and self.roof_launched and z < top - 0.5):
            self.roof_route = None
            self.next_roof_search = t + 7.0
            if not me["grounded"] and since_start <= 4.0:
                # Land without spending more energy. Past the edge, finish the
                # crossing. Before clearance, retreat toward the launch.
                dest = landing if z >= top or math.dist(xy, landing) < math.dist(xy, launch) else launch
                return (norm2(dest[0] - x, dest[1] - y), False, False, "roof-abort")
            return None
        if me["grounded"] and abs(z - top) <= 0.5 and self.roof_launched:
            self.roof_exit = (top, route["escape"])
            self.roof_route = None
            self.next_roof_search = t + 7.0
            return (0.0, 0.0), False, False, "roof-landed"
        if me["grounded"] and math.dist(xy, launch) > 1.2:
            return norm2(launch[0] - x, launch[1] - y), False, False, "roof-approach"

        if me["grounded"] and me["energy"] < self.reserve(me) + route["energy_cost"]:
            self.roof_route = None
            self.next_roof_search = t + 5.0
            return None
        if me["grounded"]:
            if math.hypot(*me["vel"][:2]) > 1.5:
                return (0.0, 0.0), False, False, "roof-approach"
            self.roof_started = t
            self.roof_launched = True
        dx, dy = landing[0] - x, landing[1] - y
        remaining = math.hypot(dx, dy)
        vx, vy = me["vel"][:2]
        toward = max(0, (vx * dx + vy * dy) / max(remaining, 0.01))
        braking = toward * toward / (2 * me["stats"]["air_accel"])
        # Brake only after the cylinder is inside the roof. At crossing
        # speeds that leaves space to land without drifting off the far edge.
        if remaining <= 1.2 or (remaining < 3 and remaining <= braking + 0.5):
            return (0.0, 0.0), False, False, "roof-land"
        burn = me["stats"]["jet_drain"] * self.cfg["match"]["decide_every"] / self.cfg["match"]["tick_hz"]
        jet = not me["grounded"] and me["energy"] >= self.reserve(me) + burn
        if z < top + 0.6:
            # Climb alongside the edge before moving over it. If we cannot
            # sustain the climb, release thrust and let the landing abort it.
            return (0.0, 0.0), jet, me["grounded"], "roof-climb"
        return norm2(landing[0] - x, landing[1] - y), jet and z < top + 3.0, False, "roof-cross"

    def zone_goal(self, obs, me, deadline=None):
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0], 1), round(nc[1], 1), round(nr, 1))
        x, y = me["pos"][0], me["pos"][1]
        if key != self.goal_key or self.goal is None:
            d = math.dist((x, y), nc)
            # Rotate to the near side with room for error; keep shorter routes in small circles.
            edge = EDGE if EDGE is not None else (0.50 if PRACTICE or nr <= 55 else 0.72)
            r = edge * nr
            if d <= r:
                g = (x, y)
            else:
                ux, uy = norm2(x - nc[0], y - nc[1])
                g = (nc[0] + ux * r, nc[1] + uy * r)
            if COVER and self.nav.ready and nr > 4:
                cover_edge = (0.82 if nr > 55 else 0.68) if EDGE is None else min(0.95, edge + 0.05)
                if PRACTICE and EDGE is None:
                    cover_edge = 0.55
                g = self.nav.cover_spot(nc, cover_edge * nr, g, self.rng, deadline=deadline) or g
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

    def escape_spot(self, obs, nearby, zone_goal, deadline=None):
        x, y, _ = obs["self"]["pos"]
        zone = obs["zone"]
        if not BREAK_LOS or not self.nav.ready or not nearby or zone["next_radius"] <= 4:
            return zone_goal
        phase = (self.goal_key, tuple(m["id"] for m in nearby[:2]))
        if obs["t"] >= self.cover_until or self.escape_phase != phase:
            threats = [m["pos"] for m in nearby[:2]]
            positions = [(px, py, pz + self.cfg["mech"].get("eye_height", 8)) for px, py, pz in threats]
            ax, ay = norm2(x - threats[0][0], y - threats[0][1])
            near = (x + ax * 24, y + ay * 24)
            circle = (zone["center"], max(0, zone["radius"] - 2))
            if math.dist((x, y), zone["next_center"]) < zone["next_radius"]:
                circle = (zone["next_center"], max(0, zone["next_radius"] - 2))
            self.escape_goal = self.nav.cover_spot(
                (x, y), 42, near, self.rng, 64, threats=positions, me_pos=(x, y),
                safe_circle=circle, near_range=self.cfg["vision"]["near_range"], deadline=deadline) or zone_goal
            self.cover_until = obs["t"] + 1.5
            self.escape_phase = phase
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
        return math.atan2(ty2 - my, tx2 - mx), ang, x2 / (v * math.cos(ang))

    def aligned(self, me, yaw, pitch, dist, tol_r=2.5):
        ay, ap = me["aim"]
        tol = math.atan2(tol_r, max(dist, 1.0)) + 0.01
        return abs(wrap(ay - yaw)) < tol and abs(ap - pitch) < tol + 0.03

    def fire_control(self, act, me, target, slots, escaping, defensive_mortar=False, deadline=None, urgent=False):
        """The two game slots share a torso. Pick its direction before setting either trigger."""
        x, y, z = me["pos"]
        muzzle = (x, y, z + me["stats"].get("muzzle_height", 7))
        min_pitch = me["stats"].get("min_pitch", -0.7)
        max_pitch = me["stats"].get("max_pitch", 1.3)
        direct = self.lead(me, target)
        yaw, pitch, hd = direct
        current_point = self.target_center(target)
        distance = math.dist(muzzle, current_point)

        def confirmed_ray(start, end):
            clear = self.nav.los(start, end, deadline)
            return clear and (deadline is None or time.perf_counter() < deadline)

        def clear_gun_ray(solution):
            sy, sp, horizontal = solution
            end = (x + math.cos(sy) * horizontal, y + math.sin(sy) * horizontal,
                   muzzle[2] + math.tan(sp) * horizontal)
            return confirmed_ray(muzzle, end)

        fire = [False, False]
        gun = slots.get("gun")
        gun_aim = self.lead(me, target, gun[1]["stats"]["speed"]) if gun else direct
        aim = gun_aim[:2]
        laser = slots.get("laser")
        lightning = slots.get("lightning")
        mortar = slots.get("mortar")
        target_is_mech = target["kind"] == "mech"
        safe_direct_pitch = min_pitch <= pitch <= max_pitch
        cadence = self.cfg["match"]["decide_every"] / self.cfg["match"]["tick_hz"]

        # Prepare only usable alternatives. A special slot on cooldown or
        # outside its envelope does not prevent using the other slot.
        arc = None
        if mortar and target_is_mech and (not escaping or defensive_mortar):
            i, w = mortar
            stats = w["stats"]
            if hd > max(24, 2 * stats["splash_radius"]) and w["cooldown"] < cadence:
                solution = self.mortar_aim(me, target, stats)
                if solution and min_pitch <= solution[1] <= max_pitch:
                    myaw, mpitch, flight = solution
                    if flight < stats.get("ttl", 7):
                        speed = stats["speed"] * math.cos(mpitch)
                        vx, vy = speed * math.cos(myaw), speed * math.sin(myaw)
                        vz = stats["speed"] * math.sin(mpitch)
                        start_z = z + me["stats"]["height"] + 0.5
                        prev = (x, y, start_z)
                        for part in range(1, 7):
                            dt = flight * part / 6
                            point = (x + vx * dt, y + vy * dt,
                                     start_z + vz * dt - 0.5 * stats["gravity"] * dt * dt)
                            if not confirmed_ray(prev, point):
                                break
                            prev = point
                        else:
                            self_xy = (x + me["vel"][0] * flight, y + me["vel"][1] * flight)
                            clearance = stats["splash_radius"] + me["stats"]["radius"]
                            if math.dist(self_xy, prev[:2]) > clearance:
                                arc = (i, (myaw, mpitch))

        electric = False
        if lightning and not defensive_mortar:
            i, w = lightning
            stats = w["stats"]
            move_drain = 0
            if act.get("boost"):
                move_drain = cadence * me["stats"]["boost_drain" if me["grounded"] else "jet_drain"]
            electric = (distance <= stats["range"] and safe_direct_pitch and w["cooldown"] < cadence
                        and me["energy"] - stats["energy"] - move_drain >= self.reserve(me, urgent)
                        and confirmed_ray(muzzle, current_point))

        beam = False
        if laser and target_is_mech and not (escaping or defensive_mortar):
            i, w = laser
            stats = w["stats"]
            beam = (distance <= stats["range"] and safe_direct_pitch and w["cooldown"] < cadence
                    and confirmed_ray(muzzle, current_point))

        charging = laser and laser[1].get("charge", 0) > 0
        if beam and charging:
            aim = (yaw, pitch)
        elif electric:
            # Lightning only tests a yaw cone. Aim the gun/laser when its
            # solution still keeps the lightning target inside that cone.
            stats = lightning[1]["stats"]
            if (not gun and not beam) or abs(wrap(aim[0] - yaw)) > math.radians(stats["cone_deg"]) * 0.8:
                aim = (yaw, pitch)
            if not gun and beam:
                aim = (yaw, pitch)
        elif arc:
            aim = arc[1]
        elif beam:
            aim = (yaw, pitch)

        if arc and self.aligned(me, *arc[1], hd, tol_r=1.5):
            if self.aligned({"aim": aim}, *arc[1], hd, tol_r=1.5):
                fire[arc[0]] = True
        if electric:
            cone = math.radians(lightning[1]["stats"]["cone_deg"])
            if abs(wrap(me["aim"][0] - yaw)) <= cone and abs(wrap(aim[0] - yaw)) <= cone:
                fire[lightning[0]] = True
        if beam and self.aligned({"aim": aim}, yaw, pitch, hd):
            i, w = laser
            on_target = self.aligned(me, yaw, pitch, hd)
            charge = w.get("charge", 0)
            fire[i] = (charge > 0 or on_target) and (charge < w["stats"]["charge"] or not on_target)
        act["aim"] = [aim[0], max(min_pitch, min(max_pitch, aim[1]))]
        if gun and not defensive_mortar:
            i, w = gun
            gy, gp, gh = gun_aim
            if (min_pitch <= gp <= max_pitch and gh <= w["stats"]["range"] * math.cos(gp)
                    and w["cooldown"] < cadence and w["reload"] == 0 and w["ammo"] > 0
                    and self.aligned(me, gy, gp, gh) and self.aligned(act, gy, gp, gh)
                    and clear_gun_ray(gun_aim)):
                fire[i] = True
        return fire

    # ------------------------------------------------------------------ main
    def act(self, obs):
        t_start = time.perf_counter()
        # Keep headroom for protocol serialization and the runner. Optional work
        # normally stops near 9 ms; the 22 ms deadline also bounds pressure paths.
        deadline = t_start + 0.022
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
            priorities = PRACTICE_UPGRADES if PRACTICE else UPGRADE_PREF
            best = min(range(len(ids)), key=lambda i: priorities.index(ids[i]) if ids[i] in priorities else 99)
            act["pick"] = best

        # threats from events and sounds
        for ev in obs.get("events") or []:
            if ev["type"] == "take_failed":
                self.pickup_retry[ev["id"]] = t + (2.0 if ev.get("reason") == "gone" else 0.6)
            if ev["type"] == "damaged" and ev.get("by") is not None and ev.get("bearing") is not None:
                self.threat[ev["by"]] = (t, ev["bearing"])
                self.last_hit_t, self.last_hit_bearing = t, ev["bearing"]
                if ev.get("weapon") != "npc":
                    self.danger_until = max(self.danger_until, t + 2.5)
        for s in obs.get("sounds") or []:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] == "near":
                if self.sound_look is None or self.sound_look[0] < t:
                    self.sound_look = (t + 1.0, s["bearing"])
                if PRACTICE or s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar"):
                    self.danger_until = max(self.danger_until, t + 1.0)

        mechs = [m for m in obs.get("mechs") or [] if m["kind"] == "mech"]
        npcs = [m for m in obs.get("mechs") or [] if m["kind"] == "npc"]
        slots = self.weapon_slots(me)
        zone = obs["zone"]
        late = zone["next_radius"] <= 70 or obs["alive"] <= 6
        nearby = self.closing_mechs(me, mechs)
        crowded = len(nearby) >= 2
        if crowded:
            self.danger_until = t + 2.5

        def d2(p):
            return math.dist((x, y), (p[0], p[1]))

        # ---------------- target choice
        target = None
        engage = False
        defensive_mortar = False
        ordered = sorted(mechs, key=lambda m: d2(m["pos"])) if PRACTICE else nearby
        if PRACTICE and ordered:
            target = ordered[0]
            distance = d2(target["pos"])
            engage = distance <= 30
            if late and not crowded and hp_frac > 0.48:
                engage = distance <= (90 if obs["alive"] <= 4 else 65)
            if not engage and crowded and "mortar" in slots and 30 < distance <= 65:
                dx, dy = target["pos"][0] - x, target["pos"][1] - y
                vx, vy = target["vel"][:2]
                closing = dx * vx + dy * vy < -2 * distance
                hit_by_target = t - self.threat.get(target["id"], (-99, 0))[0] < 1.5
                defensive_mortar = closing or hit_by_target
                engage = defensive_mortar
        elif mechs:
            def score(m):
                d = d2(m["pos"])
                s = d
                if m["id"] in self.threat and t - self.threat[m["id"]][0] < 4:
                    s -= 60
                s -= 40 * (1 - (m.get("hp", 1400) / max(1, m.get("max_hp", 1400))))
                attacked = m["id"] in self.threat and t - self.threat[m["id"]][0] < 5
                weak = m.get("hp", 1400) < 0.35 * max(1, m.get("max_hp", 1400))
                selective_target = attacked or d < 60 or (weak and hp_frac > 0.6 and d < 150 and not crowded)
                return (GATE and not selective_target, s)
            priority, target = min(((score(m), m) for m in mechs), key=lambda row: row[0])
            engage = not priority[0]
        tdist = d2(target["pos"]) if target else 1e9

        goal = self.zone_goal(obs, me, deadline)
        travel = self.must_travel(obs, me)
        mode = "zone"
        move_to = goal

        # ---------------- farming / looting when safe
        nearest_mech = min((d2(m["pos"]) for m in mechs), default=1e9)
        retreat_hp = 0.35 if RETREAT == "v1" else 0.50
        wants_cover = crowded or (bool(nearby) and (PRACTICE or hp_frac < retreat_hp))
        loot_clearance = self.cfg["vision"]["range"] if PRACTICE else 90
        safe = not (travel or wants_cover) and t >= self.danger_until and nearest_mech > loot_clearance and hp_frac > 0.45
        def affordable_detour(pos):
            zone = obs["zone"]
            xy = pos[:2]
            if math.dist(xy, zone["center"]) > zone["radius"]:
                return False
            back = max(0, math.dist(xy, zone["next_center"]) - 0.75 * zone["next_radius"])
            if back == 0:
                return True
            return d2(pos) + back < max(0, zone["shrink_end"] - t - 2) * me["stats"]["walk_speed"] * 0.65

        on_ground_surface = abs(z - self.nav.ground_h(x, y)) < 3

        def loot_on_surface(pos):
            if on_ground_surface:
                return abs(pos[2] - self.nav.ground_h(*pos[:2])) < 3
            return (ROOFS and abs(pos[2] - z) < 3 and
                    self.nav.roof_walk(me, pos, deadline) is not None)

        farm_target = None
        raised_loot = None
        if safe:
            picks = []
            for p in obs.get("pickups") or []:
                if t < self.pickup_retry.get(p["id"], -1):
                    continue
                takeable, swap_slot = self.pickup_swap(p["kind"], me, slots, late)
                if not takeable:
                    continue
                if d2(p["pos"]) < (LOOT_EARLY if t < 200 else 70) and affordable_detour(p["pos"]):
                    if loot_on_surface(p["pos"]):
                        picks.append((p, swap_slot))
                    elif ROOFS and d2(p["pos"]) < 30 and (raised_loot is None or d2(p["pos"]) < d2(raised_loot)):
                        raised_loot = p["pos"]
            cubes = []
            for cube in obs.get("cubes") or []:
                p = cube["pos"]
                if d2(p) >= 60 or not affordable_detour(p):
                    continue
                if loot_on_surface(p):
                    cubes.append(cube)
                elif ROOFS and d2(p) < 30 and (raised_loot is None or d2(p) < d2(raised_loot)):
                    raised_loot = p
            if cubes and min(d2(c["pos"]) for c in cubes) <= 10:
                picks = []
            if picks:
                p, swap_slot = min(picks, key=lambda item: d2(item[0]["pos"]) /
                                   (1 + 0.25 * self.pickup_value(item[0]["kind"], late)))
                move_to, mode = (p["pos"][0], p["pos"][1]), "loot"
                if d2(p["pos"]) < self.cfg["loot"]["pickup_radius"] - 0.3 and abs(p["pos"][2] - z) < 3.8:
                    act["take"] = p["id"]
                    if swap_slot is not None:
                        act["drop"] = swap_slot
            elif cubes:
                c = min(cubes, key=lambda c: d2(c["pos"]))
                move_to, mode = (c["pos"][0], c["pos"][1]), "cube"
            # farm NPC drones (2 cubes) and crates (1 cube + maybe a weapon) while it is quiet
            if PRACTICE and FARM and mode == "zone" and obs["zone"]["next_radius"] >= 25 and not mechs:
                # Favor crates we can walk to and collect; a visible crate across
                # a courtyard wall is a poor reason to broadcast gunfire.
                ordered_crates = sorted(obs.get("crates") or [], key=lambda c: d2(c["pos"]))
                for crate in ordered_crates[:6]:
                    p = crate["pos"]
                    if d2(p) > 45 or time.perf_counter() >= deadline:
                        break
                    if not affordable_detour(p):
                        continue
                    if (loot_on_surface(p) and (not on_ground_surface or self.nav.walk_clear(x, y, p[:2]))
                            and self.nav.los((x, y, z + me["stats"]["muzzle_height"]), (p[0], p[1], p[2] + 1.75), deadline)):
                        farm_target = {"kind": "crate", "id": crate["id"], "pos": p, "vel": (0, 0, 0)}
                        move_to, mode = tuple(p[:2]), "farm"
                        break
                    if ROOFS and d2(p) < 30 and p[2] > z + 3 and (raised_loot is None or d2(p) < d2(raised_loot)):
                        raised_loot = p
            elif not PRACTICE and FARM and t < 260 and hp_frac > 0.6 and nearest_mech > 150 and not engage:
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
        if not PRACTICE and safe and mode == "zone" and not mechs:
            for dr in obs.get("drops") or []:
                if (dr["lands_in"] <= 3 and d2(dr["pos"]) < 70 and dr.get("holder") in (None, me["id"])
                        and affordable_detour(dr["pos"])):
                    move_to, mode = (dr["pos"][0], dr["pos"][1]), "drop"

        # ---------------- movement
        if not travel and wants_cover:
            move_to, mode = self.escape_spot(obs, nearby, goal, deadline), "cover"
        elif (not travel and t < min(self.danger_until, self.cover_until + 1) and self.escape_goal
              and self.escape_phase and self.escape_phase[0] == self.goal_key):
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
        if fighting and not (travel or seeking_cover or PRACTICE):
            ty_ = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
            px, py = -math.sin(ty_), math.cos(ty_)
            if t > self.strafe_until:
                self.strafe_sign = -self.strafe_sign if self.rng.random() < 0.7 else self.strafe_sign
                self.strafe_sign = self.nav.strafe_side(x, y, px, py, self.strafe_sign)
                self.strafe_until = t + self.rng.uniform(0.6, 1.6)
            sx, sy = px * self.strafe_sign, py * self.strafe_sign
            their = target.get("hp", 1400) / max(1, target.get("max_hp", 1400))
            retreat_margin = 0.0 if RETREAT == "v1" else 0.05
            retreat = hp_frac < retreat_hp and their > hp_frac - retreat_margin
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
        if PRACTICE and seeking_cover and nearby:
            threat = nearby[0]
            away = norm2(x - threat["pos"][0], y - threat["pos"][1])
            if mv != (0.0, 0.0) or self.nav.los(
                    (x, y, z + me["stats"]["eye_height"]), self.target_center(threat)):
                # Follow cover with a lateral component; never wait in the open
                # to align a shot. Choose the side already leading toward cover.
                side = self.strafe_sign if abs(mv[0] * away[1] - mv[1] * away[0]) < 0.05 else (
                    1 if mv[1] * away[0] > mv[0] * away[1] else -1)
                lateral = 0.8 if crowded else 0.45
                mv = norm2(mv[0] + 0.35 * away[0] - lateral * side * away[1],
                           mv[1] + 0.35 * away[1] + lateral * side * away[0])
                boost = crowded and me["energy"] > 35
        if PRACTICE and late and not travel and self.nav.ready and zone["radius"] > 10:
            for shot in (obs.get("projectiles") or [])[:24]:
                px, py, pz = shot["pos"]
                vx, vy, vz = shot["vel"]
                speed_sq = vx * vx + vy * vy
                if speed_sq < 1 or math.hypot(px - x, py - y) > 100:
                    continue
                eta = ((x - px) * vx + (y - py) * vy) / speed_sq
                if not 0 < eta < 0.35 or abs(pz + eta * vz - z - 4.5) > 6:
                    continue
                if math.hypot(px + eta * vx - x, py + eta * vy - y) > me["stats"]["radius"] + 2:
                    continue
                sx, sy = norm2(-vy, vx)
                if sx * mv[0] + sy * mv[1] < 0:
                    sx, sy = -sx, -sy
                mv = norm2(mv[0] + 1.4 * sx, mv[1] + 1.4 * sy)
                boost = True
                # Briefly jump for a low straight projectile if the next step
                # is open. A ground evasion remains available at low energy.
                if (shot["kind"] == "gun" and pz + eta * vz < z + 2.5 and me["grounded"]
                        and me["energy"] > self.reserve(me) + 15
                        and self.nav.walk_clear(x, y, (x + 5 * sx, y + 5 * sy))):
                    act["jump"] = True
                break
        if DODGE and not (travel or seeking_cover):
            for m in mechs:
                if not m.get("charging"):
                    continue
                bearing = math.atan2(y - m["pos"][1], x - m["pos"][0])
                if abs(wrap(m["aim"][0] - bearing)) < 0.15:
                    side = 1 if int(t * 2) % 2 == 0 else -1
                    mv = norm2(mv[0] - 2 * side * math.sin(bearing), mv[1] + 2 * side * math.cos(bearing))
                    boost = me["energy"] > 10
                    mode = "dodge"
                    break

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

        if not travel and me["in_zone"]:
            step = 0.3 * me["stats"]["walk_speed"]
            ahead = (x + step * mv[0], y + step * mv[1])
            if math.dist(ahead, zone["center"]) > zone["radius"]:
                offset = self.nav.steer(x, y, goal)
                mv = norm2(*(offset or (zone["center"][0] - x, zone["center"][1] - y)))
        mv = self.nav.safe_step(x, y, mv, self.strafe_sign, min(6.0, dgoal) if mode == "zone" else 6.0)
        if EXPOSURE_NAV and not travel and me.get("grounded") and time.perf_counter() < deadline:
            vision = self.cfg["vision"]
            size_scale = self.cfg["cubes"]["size_per_cube"]
            if (mode in ("zone", "cover", "avoid") and
                    self.nav.can_hold_roof(me, zone, ordered, vision, deadline, size_per_cube=size_scale)):
                mv, boost, mode = (0.0, 0.0), False, "roof"
                act.pop("jump", None)
            elif mv != (0.0, 0.0) and ordered and zone["next_radius"] > 4:
                mv = self.nav.stealth_step(me, mv, move_to, ordered, vision, deadline, zone=zone,
                                           size_per_cube=size_scale)
        roof_movement = None
        if ROOFS:
            if mode in ("loot", "cube", "farm") and me["grounded"] and dgoal > stop_dist:
                roof_walk = self.nav.roof_walk(me, move_to, deadline)
                if roof_walk is not None:
                    mv = roof_walk
            elif self.roof_exit and travel:
                top, landing_ground = self.roof_exit
                if z >= top - 0.5 and math.dist((x, y), landing_ground) > 2:
                    mv = norm2(landing_ground[0] - x, landing_ground[1] - y)
                    boost = False
                    act.pop("jump", None)
                else:
                    self.roof_exit = None
            roof_movement = self.roof_action(obs, safe, mode, raised_loot, ordered, deadline)
            if roof_movement:
                mv, boost, jump, mode = roof_movement
                self.stuck_t = 0.0
                self.unstick_until = -1.0
                if jump:
                    act["jump"] = True
                else:
                    act.pop("jump", None)
                act.pop("take", None)
                act.pop("drop", None)
        if mode == "zone" and obs["zone"]["next_radius"] < 4:
            scale = min(1.0, dgoal / max(1.0, me["stats"]["walk_speed"] * 0.2))
            mv = (mv[0] * scale, mv[1] * scale)
        act["move"] = [mv[0], mv[1]]
        dt_action = self.cfg["match"]["decide_every"] / self.cfg["match"]["tick_hz"]
        burn = me["stats"]["boost_drain"] if me["grounded"] else me["stats"]["jet_drain"]
        act["boost"] = bool(boost and me["energy"] >= self.reserve(me, travel) + burn * dt_action)

        # ---------------- aiming and firing
        fire = [False, False]
        shoot_at = None
        if target is not None and engage and roof_movement is None:
            shoot_at = target
        elif farm_target is not None and roof_movement is None:
            shoot_at = farm_target
        elif target is not None:
            yaw, _, _ = self.lead(me, target)
            act["aim"] = [yaw, 0.0]

        if shoot_at is not None and time.perf_counter() < deadline:
            fire = self.fire_control(act, me, shoot_at, slots, travel or (seeking_cover and not (late and not crowded)),
                                     defensive_mortar, deadline, urgent=travel)
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
        if gun and act.get("drop") != gun[0] and not fire[gun[0]]:
            state = gun[1]
            if state["reload"] == 0 and (state["ammo"] == 0 or
                    (not engage and farm_target is None and state["ammo"] < state["stats"]["mag"] * 0.5)):
                act["reload"] = True
        if "drop" in act:
            fire[act["drop"]] = False
        act["fire"] = fire

        budget = min(0.006, t_start + 0.009 - time.perf_counter())
        if budget > 0:
            self.nav.work(budget)
        if time.perf_counter() >= deadline:
            return act
        secondary = me["weapons"][1]
        weapon_name = secondary["kind"] if secondary else "-"
        mf.draw.label(f"{mode} hp{me['hp']:.0f}/{me['max_hp']:.0f} e{me['energy']:.0f} {weapon_name} "
                      f"T{target['id'] if target else '-'}@{tdist if target else 0:.0f} "
                      f"n{len(mechs)} st{self.stuck_t:.1f}")
        if self.goal:
            mf.draw.line([x, y, z + 2], [self.goal[0], self.goal[1], z + 2], "#40c0ff")
        if shoot_at is not None:
            tx, ty, tz = shoot_at["pos"]
            mf.draw.line([x, y, z + 7], [tx, ty, tz + 4], "#ff5040")
        hits = [event for event in obs.get("events") or [] if event["type"] == "damaged"]
        if hits:
            mf.draw.text(" ".join(f"{e['weapon']}:{e['amount']:.0f}{'R' if e.get('rear') else ''}<{e.get('by')}"
                                  for e in hits[:6]))
        return act


if __name__ == "__main__":
    mf.run(Bot)
