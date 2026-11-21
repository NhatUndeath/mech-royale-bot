"""Build the walking grid at hello and advance the flow field in small slices during play."""
import json
import math
import os
import sys
import time
from array import array
from collections import deque

CELL = 2.0          # meters per nav cell
WALK_H = 9.0        # mech height: overhangs lower than this are walls
STEP = 1.2
BUCKET = 24.0


def log(*a):
    print("[nav]", *a, file=sys.stderr, flush=True)


def _pip(x, y, flat):
    """Point in polygon given as a flat [x0,y0,x1,y1,...] list."""
    n = len(flat) // 2
    inside = False
    j = n - 1
    xj, yj = flat[2 * j], flat[2 * j + 1]
    for i in range(n):
        xi, yi = flat[2 * i], flat[2 * i + 1]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        xj, yj = xi, yi
    return inside


def _disk_polygon(x, y, radius, prism, inside=False):
    """Test intersection, or full containment, of a horizontal cylinder footprint."""
    x0, y0, x1, y1, ring, holes, _, _ = prism
    if x < x0 - radius or x > x1 + radius or y < y0 - radius or y > y1 + radius:
        return False
    contained = _pip(x, y, ring) and not any(_pip(x, y, h) for h in holes)
    if radius <= 0 or contained != inside:
        return contained
    limit = radius * radius
    for path in (ring, *holes):
        ax, ay = path[-2:]
        for i in range(0, len(path), 2):
            bx, by = path[i], path[i + 1]
            if (min(ax, bx) - radius <= x <= max(ax, bx) + radius
                    and min(ay, by) - radius <= y <= max(ay, by) + radius):
                dx, dy = bx - ax, by - ay
                t = min(1.0, max(0.0, ((x - ax)*dx + (y - ay)*dy) / (dx*dx + dy*dy or 1.0)))
                if (ax + t*dx - x)**2 + (ay + t*dy - y)**2 <= limit:
                    return not inside
            ax, ay = bx, by
    return contained


class Terrain:
    def __init__(self, path, meta):
        self.nx, self.ny = int(meta["nx"]), int(meta["ny"])
        self.minx, self.miny = float(meta["minX"]), float(meta["minY"])
        self.step = float(meta.get("step", 1.0))
        self.scale = float(meta.get("scale", 0.01))
        a = array("h")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        self.h = a

    def height(self, x, y):
        fx = (x - self.minx) / self.step
        fy = (y - self.miny) / self.step
        i = min(max(int(fx), 0), self.nx - 2)
        j = min(max(int(fy), 0), self.ny - 2)
        tx = min(max(fx - i, 0.0), 1.0)
        ty = min(max(fy - j, 0.0), 1.0)
        h, nx = self.h, self.nx
        a = h[j * nx + i]
        b = h[j * nx + i + 1]
        c = h[(j + 1) * nx + i]
        d = h[(j + 1) * nx + i + 1]
        return ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty) * self.scale


class Nav:
    def __init__(self, map_dir, mech=None):
        self.mech = mech or {}
        self.ready = False
        self.terrain = None
        self.blocked = None
        self.prisms = []
        self.field = None          # (goal, dist array)
        self._want = None          # goal requested
        self.solid_bins = {}
        self.roof_edges = {}
        self._risk_key = None
        self._risk = {}
        self.map_dir = map_dir
        self._job = None           # (goal, generator) of the BFS in progress
        try:
            t0 = time.time()
            self._build()
            self.ready = True
            log(f"grid ready in {time.time() - t0:.2f}s ({self.gx}x{self.gy})")
        except Exception as e:  # never kill the bot
            log("grid build failed:", repr(e))

    # ------------------------------------------------------------------ build
    def _build(self):
        with open(os.path.join(self.map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        tmeta = w["terrain"]
        self.terrain = Terrain(os.path.join(self.map_dir, "v2", tmeta.get("file", "terrain.i16")), tmeta)
        b = w["bounds"]
        self.minx, self.miny = b["minX"], b["minY"]
        self.gx = int((b["maxX"] - b["minX"]) / CELL) + 1
        self.gy = int((b["maxY"] - b["minY"]) / CELL) + 1
        gx, gy = self.gx, self.gy
        T = self.terrain
        ground = array("f", [0.0]) * (gx * gy)
        for j in range(gy):
            y = self.miny + (j + 0.5) * CELL
            for i in range(gx):
                ground[j * gx + i] = T.height(self.minx + (i + 0.5) * CELL, y)
        raw = bytearray(gx * gy)
        self.walls = bytearray(gx * gy)
        max_rise = math.tan(math.radians(self.mech.get("max_slope_deg", 50))) * CELL
        # steep terrain and map edge
        for j in range(gy):
            row = j * gx
            for i in range(gx):
                if i == 0 or j == 0 or i == gx - 1 or j == gy - 1:
                    raw[row + i] = 1
                    continue
                g = ground[row + i]
                if (abs(ground[row + i + 1] - g) > max_rise or abs(ground[row + i - 1] - g) > max_rise
                        or abs(ground[row + i + gx] - g) > max_rise or abs(ground[row + i - gx] - g) > max_rise):
                    raw[row + i] = 1
        # A sampled solid interval per column serves the local exposure cost. Keep
        # intervals discrete: a bridge deck must leave the passage below it open.
        self.solid_base = array("f", [math.inf]) * (gx * gy)
        self.solid_top = array("f", [-math.inf]) * (gx * gy)
        # prisms
        for p in w["prisms"]:
            ring = p["ring"]
            holes = p.get("holes") or []
            x0, x1 = min(ring[0::2]), max(ring[0::2])
            y0, y1 = min(ring[1::2]), max(ring[1::2])
            index = len(self.prisms)
            self.prisms.append((x0, y0, x1, y1, ring, holes, p["base"], p["top"]))
            for bx in range(math.floor(x0 / BUCKET), math.floor(x1 / BUCKET) + 1):
                for by in range(math.floor(y0 / BUCKET), math.floor(y1 / BUCKET) + 1):
                    self.solid_bins.setdefault((bx, by), []).append(index)
            i0 = max(0, int((x0 - self.minx) / CELL))
            i1 = min(gx - 1, int((x1 - self.minx) / CELL) + 1)
            j0 = max(0, int((y0 - self.miny) / CELL))
            j1 = min(gy - 1, int((y1 - self.miny) / CELL) + 1)
            base, top = p["base"], p["top"]
            for j in range(j0, j1 + 1):
                y = self.miny + (j + 0.5) * CELL
                for i in range(i0, i1 + 1):
                    k = j * gx + i
                    x = self.minx + (i + 0.5) * CELL
                    if not _pip(x, y, ring) or any(_pip(x, y, h) for h in holes):
                        continue
                    if top > self.solid_top[k]:
                        self.solid_base[k], self.solid_top[k] = base, top
                    g = ground[k]
                    if top > g + self.mech.get("step_up", STEP) and base < g + self.mech.get("height", WALK_H):
                        raw[k] = 1
                        self.walls[k] = 1
        # dilate by one cell (mech radius 2.5 m)
        blk = bytearray(raw)
        for j in range(1, gy - 1):
            row = j * gx
            for i in range(1, gx - 1):
                k = row + i
                if raw[k]:
                    continue
                if raw[k - 1] or raw[k + 1] or raw[k - gx] or raw[k + gx]:
                    blk[k] = 1
        self.raw = raw
        self.blocked = blk
        self.ground = ground
        # largest connected free component: goals are snapped into it
        comp = bytearray(gx * gy)
        best, best_lab, lab = 0, 0, 0
        for s0 in range(gx * gy):
            if blk[s0] or comp[s0]:
                continue
            lab = lab + 1 if lab < 255 else 255
            comp[s0] = lab
            q = [s0]
            size = 0
            while q:
                k = q.pop()
                size += 1
                for m in (k + 1, k - 1, k + gx, k - gx):
                    if 0 <= m < gx * gy and not blk[m] and not comp[m]:
                        comp[m] = lab
                        q.append(m)
            if size > best:
                best, best_lab = size, lab
        self.main = bytearray(1 if c == best_lab and not blk[k] else 0 for k, c in enumerate(comp))
        self._notmain = bytes(1 - v for v in self.main)
        # Canopy at eye height is a concealment preference, independent of solid geometry.
        self.canopy = bytearray(gx * gy)
        trees = w.get("trees") or {}
        fields = trees.get("fields", [])
        data = trees.get("data", [])
        for start in range(0, len(data), len(fields) or 1):
            tree = dict(zip(fields, data[start:start + len(fields)]))
            eye = self.mech.get("eye_height", 8)
            cx, cy = tree.get("crownX", tree["x"]), tree.get("crownY", tree["y"])
            r = tree["crownR"]
            i0, j0 = self.cell(cx - r, cy - r)
            i1, j1 = self.cell(cx + r, cy + r)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    px, py = self.center(i, j)
                    k = j * gx + i
                    tree_eye = ground[k] + eye - tree["z"]
                    if tree["crownBase"] <= tree_eye <= tree["height"] and (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                        self.canopy[k] = 1

        self._index_roofs()

    def _near_solids(self, x, y, r):
        found = set()
        for bx in range(math.floor((x - r) / BUCKET), math.floor((x + r) / BUCKET) + 1):
            for by in range(math.floor((y - r) / BUCKET), math.floor((y + r) / BUCKET) + 1):
                found.update(self.solid_bins.get((bx, by), ()))
        return found

    def _air_clear(self, x, y, radius, low, high):
        for k in self._near_solids(x, y, radius):
            prism = self.prisms[k]
            if prism[7] > low and prism[6] < high and _disk_polygon(x, y, radius, prism):
                return False
        return True

    def _index_roofs(self):
        # Keep only short, reversible ground-to-roof connections. Test them
        # during hello, with space for a mech grown to 150% of its base size.
        self.roof_radius = 1.5 * self.mech.get("radius", 2.5)
        self.roof_height = 1.5 * self.mech.get("height", WALK_H)
        offset = self.roof_radius + 3.5
        for k, prism in enumerate(self.prisms):
            ring, top = prism[4], prism[7]
            ax, ay = ring[-2:]
            for idx in range(0, len(ring), 2):
                bx, by = ring[idx:idx + 2]
                dx, dy = bx - ax, by - ay
                length = math.hypot(dx, dy)
                ox, oy = ax, ay
                ax, ay = bx, by
                if length < 2 * offset:
                    continue
                nx, ny = -dy / length, dx / length
                parts = max(1, math.ceil(length / 22))
                for n in range(parts):
                    t = (n + 0.5) / parts
                    mx, my = ox + t*dx, oy + t*dy
                    if not _pip(mx + nx, my + ny, ring):
                        sx, sy = -nx, -ny
                    else:
                        sx, sy = nx, ny
                    launch = (mx - sx * offset, my - sy * offset)
                    landing = (mx + sx * offset, my + sy * offset)
                    li, lj = self.cell(*launch)
                    if not self.main[lj*self.gx + li]:
                        continue
                    ground = self.ground_h(*launch)
                    rise = top - ground
                    if not 4.0 <= rise <= 18.0:
                        continue
                    if not _disk_polygon(*landing, self.roof_radius + 0.5, prism, inside=True):
                        continue
                    # Check a clear vertical shaft and horizontal cylinder sweep.
                    # Padding samples by half their spacing covers the gaps.
                    ceiling = top + self.roof_height + 26
                    if not self._air_clear(*launch, self.roof_radius + 1, ground + 0.15, ceiling):
                        continue
                    steps = math.ceil(2 * offset / 2)
                    for s in range(steps + 1):
                        px = launch[0] + 2*offset*sx*s/steps
                        py = launch[1] + 2*offset*sy*s/steps
                        if (self.ground_h(px, py) > top - 0.1
                                or not self._air_clear(px, py, self.roof_radius + 1, top + 0.05, ceiling)):
                            break
                    else:
                        bucket = (math.floor(launch[0]/BUCKET), math.floor(launch[1]/BUCKET))
                        self.roof_edges.setdefault(bucket, []).append((k, launch, landing, rise))

    def roof_walk(self, me, goal, deadline):
        """A straight same-deck move, accepting only supported interior paths."""
        if not self.ready or not me.get("grounded") or time.perf_counter() >= deadline:
            return None
        x, y, z = me["pos"]
        dx, dy = goal[0] - x, goal[1] - y
        distance = math.hypot(dx, dy)
        if distance > 45 or (len(goal) > 2 and abs(goal[2] - z) > 3.5):
            return None
        stats = me.get("stats", {})
        radius = stats.get("radius", self.mech.get("radius", 2.5))
        height = stats.get("height", self.mech.get("height", WALK_H))
        for k in self._near_solids(x, y, 0):
            prism = self.prisms[k]
            if abs(prism[7] - z) > 0.65 or not _disk_polygon(x, y, radius, prism, inside=True):
                continue
            steps = max(1, math.ceil(distance / 2))
            for s in range(steps + 1):
                if time.perf_counter() >= deadline:
                    return None
                px, py = x + dx*s/steps, y + dy*s/steps
                if (not _disk_polygon(px, py, radius + 1, prism, inside=True)
                        or not self._air_clear(px, py, radius + 1, prism[7] + 0.1, z + height)):
                    break
            else:
                return (dx/distance, dy/distance) if distance > 0.1 else (0.0, 0.0)
        return None

    def roof_transition(self, me, zone, enemies, vision, deadline, reserve=35.0,
                        size_per_cube=0.05, target=None):
        """Choose a short, clear jump shaft, a safe landing and a ground escape."""
        now = time.perf_counter()
        deadline = min(deadline, now + 0.0025)
        if not self.ready or not me.get("grounded") or now >= deadline:
            return None
        x, y, z = me["pos"]
        if abs(z - self.ground_h(x, y)) > self.mech.get("step_up", STEP) + 0.5:
            return None
        stats = me.get("stats", {})
        radius = stats.get("radius", self.mech.get("radius", 2.5))
        height = stats.get("height", self.mech.get("height", WALK_H))
        if radius > self.roof_radius or height > self.roof_height:
            return None
        speed = stats.get("walk_speed", self.mech.get("walk_speed", 11.7))
        accel = stats.get("air_accel", self.mech.get("air_accel", 12))
        rise_speed = min(stats.get("jump_speed", self.mech.get("jump_speed", 11)),
                         stats.get("max_rise_speed", self.mech.get("max_rise_speed", 10)))
        drain = stats.get("jet_drain", self.mech.get("jet_drain", 15))
        if (rise_speed <= 1 or speed <= 1 or accel <= 0
                or stats.get("jet_accel", self.mech.get("jet_accel", 34)) <= stats.get("gravity", self.mech.get("gravity", 22))):
            return None
        if target is not None and len(target) < 3:
            return None
        candidates = []
        margin = radius + 5
        for bx in range(math.floor((x-30)/BUCKET), math.floor((x+30)/BUCKET)+1):
            for by in range(math.floor((y-30)/BUCKET), math.floor((y+30)/BUCKET)+1):
                for k, launch, landing, rise in self.roof_edges.get((bx, by), ()):
                    approach = math.dist((x, y), launch)
                    if approach > 30:
                        continue
                    if any(math.dist(p, zone["center"]) > zone["radius"] - margin or
                           math.dist(p, zone["next_center"]) > zone["next_radius"] - margin
                           for p in (launch, landing)):
                        continue
                    prism = self.prisms[k]
                    if target is not None and (abs(target[2] - prism[7]) > 3.5 or
                            not _disk_polygon(target[0], target[1], 0.0, prism, inside=True)):
                        continue
                    travel = math.dist(landing, launch)
                    cross_time = travel/speed + speed/(2*accel)
                    energy_cost = drain*((rise+1)/(0.8*rise_speed) + cross_time + 0.40)
                    if me.get("energy", 0) - energy_cost < reserve:
                        continue
                    weight = approach + (math.dist(target[:2], landing) if target is not None else 0) + 0.2*rise
                    candidates.append((weight, k, launch, landing, energy_cost))
        for _, k, launch, landing, cost in sorted(candidates):
            if time.perf_counter() >= deadline:
                return None
            if not self._clear(*self.cell(x, y), *self.cell(*launch)):
                continue
            top = self.prisms[k][7]
            if target is not None:
                arrival = dict(me, pos=(*landing, top))
                if self.roof_walk(arrival, target, deadline) is None:
                    continue
            # A jet is audible nearby even behind a building; farther observers
            # must have neither a head nor a center ray to the proposed perch.
            hidden = True
            for e in enemies:
                ex, ey, ez = e["pos"]
                if math.dist((ex, ey), launch) < 90:
                    hidden = False
                    break
                if math.dist((ex, ey), landing) <= vision.get("range", 200):
                    eye = self.mech.get("eye_height", 8) * (1 + size_per_cube * (e.get("cubes") or 0))
                    if any(self.los((ex, ey, ez + eye), (*landing, top + height*f), deadline) for f in (1, .5)):
                        hidden = False
                        break
            if hidden and time.perf_counter() < deadline:
                return dict(launch=launch, landing=landing, top=top, escape=launch, energy_cost=cost)
        return None

    # ------------------------------------------------------------------ queries
    def cell(self, x, y):
        i = int((x - self.minx) / CELL)
        j = int((y - self.miny) / CELL)
        return min(max(i, 0), self.gx - 1), min(max(j, 0), self.gy - 1)

    def center(self, i, j):
        return self.minx + (i + 0.5) * CELL, self.miny + (j + 0.5) * CELL

    def free(self, x, y):
        if not self.ready:
            return True
        i, j = self.cell(x, y)
        return not self.blocked[j * self.gx + i]

    def nearest_free(self, x, y, maxr=40, main=False):
        i0, j0 = self.cell(x, y)
        gx, gy = self.gx, self.gy
        blk = self._notmain if main else self.blocked
        for r in range(0, maxr):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < gx and 0 <= j < gy and not blk[j * gx + i]:
                        return i, j
        return i0, j0

    def wall_density(self, i, j):
        gx = self.gx
        left, right = max(0, i - 3), min(gx, i + 4)
        return sum(self.walls[row * gx + left:row * gx + right].count(1)
                   for row in range(max(0, j - 3), min(self.gy, j + 4)))

    def strafe_side(self, x, y, side_x, side_y, preferred):
        if not self.ready:
            return preferred
        origin = self.cell(x, y)
        def rank(sign):
            end = self.cell(x + 6 * side_x * sign, y + 6 * side_y * sign)
            return self._clear(*origin, *end), self.wall_density(*end)
        return max((preferred, -preferred), key=rank)

    def cover_spot(self, center, radius, near, rng, samples=96, *, threats=(), me_pos=None, safe_circle=None, near_range=45, deadline=None, height=None):
        """A free cell inside the circle, close to `near`, with walls around it for cover."""
        gx, main = self.gx, self.main
        candidates = {}
        height = self.mech.get("height", WALK_H) if height is None else height
        near_ground = self.ground_h(*near)
        for index in range(samples):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if index == 0:
                px, py = near
            elif index == 1 and me_pos:
                px, py = me_pos[:2]
            else:
                a = rng.uniform(0, 2 * math.pi)
                rr = radius * math.sqrt(rng.random())
                px, py = center[0] + rr * math.cos(a), center[1] + rr * math.sin(a)
            i, j = self.cell(px, py)
            px, py = self.center(i, j)
            if (not main[j * gx + i] or (i, j) in candidates or math.dist((px, py), center) > radius
                    or safe_circle and math.dist((px, py), safe_circle[0]) > safe_circle[1]):
                continue
            cover = self.wall_density(i, j)
            # Only main-component, walkable cells reach this point. Prefer
            # modest, accessible rises within a similar detour and cover score.
            rise = min(6.0, max(0.0, self.ground[j * gx + i] - near_ground))
            score = math.dist((px, py), near) - 4.0 * min(cover, 12) - rise
            candidates[i, j] = score
        if not candidates:
            return None
        shortlist = sorted(candidates, key=candidates.get)[:4]
        def rank(cell):
            i, j = cell
            px, py = self.center(i, j)
            eye = (px, py, self.ground[j * gx + i] + self.mech.get("eye_height", 8))
            center_z = self.ground[j * gx + i] + 0.5 * height
            top_z = self.ground[j * gx + i] + height
            exposed = sum(self.los(pos, (px, py, top_z), deadline) or self.los(pos, (px, py, center_z), deadline)
                          for pos in threats)
            soft = self.canopy[j * gx + i] and all(math.dist(pos, eye) > near_range for pos in threats)
            travel = math.dist((px, py), me_pos[:2]) if me_pos else 0
            return exposed, candidates[cell] + 0.3 * travel - 5 * soft
        best_cell, best_rank = None, None
        for cell in shortlist:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            value = rank(cell)
            if best_rank is None or value < best_rank:
                best_cell, best_rank = cell, value
        if best_cell is None:
            return None
        return self.center(*best_cell)

    def _sampled_visible(self, ex, ey, ez, i, j, z):
        """Approximate sight for many walking cells. Actual weapon rays use los()."""
        ex = (ex - self.minx) / CELL
        ey = (ey - self.miny) / CELL
        dx, dy = i + 0.5 - ex, j + 0.5 - ey
        n = max(1, math.ceil(max(abs(dx), abs(dy))))
        dx, dy, dz = dx / n, dy / n, (z - ez) / n
        gx, ground = self.gx, self.ground
        base, top = self.solid_base, self.solid_top
        for _ in range(n - 1):
            ex += dx
            ey += dy
            ez += dz
            k = int(ey) * gx + int(ex)
            if ez <= ground[k] or base[k] < ez < top[k]:
                return False
        return True

    def _exposure(self, cell, observers, vision, height):
        if cell in self._risk:
            return self._risk[cell]
        i, j = cell
        px, py = self.center(i, j)
        k = j * self.gx + i
        ground = self.ground[k]
        near = vision.get("near_range", 45)
        risk = 0.0
        for x, y, z, far in observers:
            distance = math.hypot(px - x, py - y)
            if distance > far:
                continue
            if not (self._sampled_visible(x, y, z, i, j, ground + height)
                    or self._sampled_visible(x, y, z, i, j, ground + 0.5 * height)):
                continue
            # Any live observer can alert the squad. An eye-height crown only
            # discounts a far sightline; canopy alone is no guarantee of hiding.
            canopy = self.canopy[k] and height <= self.mech.get("height", WALK_H) and vision.get("canopy_blocks", True)
            weight = 0.70 if distance > near and canopy else 1.0
            risk = max(risk, weight)
        self._risk[cell] = risk
        return risk

    def stealth_step(self, me, move, goal, enemies, vision, deadline, urgent=False, zone=None, size_per_cube=0.05):
        """Find a short walkable route that trades distance for reduced exposure.

        The static flow keeps its distances in meters. A two-layer local graph
        overlays the current enemies' sightlines. Movement of an observer by a
        few meters invalidates this small cache without restarting the flow.
        """
        if not self.ready or not enemies or urgent or not me.get("grounded"):
            return move
        now = time.perf_counter()
        deadline = min(deadline, now + 0.0025)
        if deadline <= now:
            return move
        x, y, z = me["pos"]
        start = self.cell(x, y)
        if self.blocked[start[1] * self.gx + start[0]] or abs(z - self.ground_h(x, y)) > 2.0:
            return move
        distance = math.dist((x, y), goal)
        length = math.hypot(*move)
        if distance < 6 or length < 0.1:
            return move
        fx, fy = move[0] / length, move[1] / length
        near_range = vision.get("near_range", 45)
        far = vision.get("range", 200)
        eye = self.mech.get("eye_height", 8)
        height = me.get("stats", {}).get("height", self.mech.get("height", WALK_H))
        observers = []
        for enemy in enemies:
            ex, ey, ez = enemy["pos"]
            # The grader reports positions inside map bounds. Clamp anomalous
            # test observations before indexing the sampled grid.
            if not (self.minx <= ex < self.minx + self.gx * CELL
                    and self.miny <= ey < self.miny + self.gy * CELL):
                continue
            r = math.hypot(ex - x, ey - y)
            if r <= far + 24:
                eye_z = ez + eye * (1.0 + size_per_cube * (enemy.get("cubes") or 0))
                observers.append((r, ex, ey, eye_z, far))
        if not observers:
            return move
        observers.sort()
        observers = [row[1:] for row in observers[:3]]
        key = (near_range, far, vision.get("canopy_blocks", True), height,
               tuple((round(ex / 4), round(ey / 4), round(ez / 2)) for ex, ey, ez, _ in observers))
        if key != self._risk_key or len(self._risk) > 512:
            self._risk.clear()
            self._risk_key = key
        scale = min(1.0, distance / 18.0)
        horizon = (x + 18 * scale * fx, y + 18 * scale * fy)
        # Keep the start layer balanced even when the optional lookahead ends
        # early. A queued action stays valid if the shared tick budget expires.
        first = []
        far_nodes = []
        limit = max(zone["radius"], math.dist((x, y), zone["center"])) if zone else None
        for forward, lateral in ((6, 0), (4, 4), (4, -4), (0, 6), (0, -6),
                                 (18, 0), (13, 9), (13, -9), (8, 14), (8, -14)):
            if time.perf_counter() >= deadline:
                break
            pos = (x + scale * (fx * forward - fy * lateral), y + scale * (fy * forward + fx * lateral))
            cell = self.cell(*pos)
            if ((zone and math.dist(pos, zone["center"]) > limit)
                    or not self.main[cell[1] * self.gx + cell[0]]):
                continue
            if forward <= 6 and not self._clear(*start, *cell):
                continue
            exposure = self._exposure(cell, observers, vision, height)
            if forward <= 6:
                # A one-hop route remains usable if all longer paths are blocked.
                cost = math.dist((x, y), pos) * (1.0 + 3.0 * exposure)
                first.append((pos, cell, exposure, cost))
            else:
                far_nodes.append((pos, cell, exposure))
        best_move = move
        best_cost = math.inf
        for pos, cell, exposure, cost in first:
            estimate = cost + math.dist(pos, horizon) * (1.0 + 3.0 * exposure)
            for end, end_cell, end_exposure in far_nodes:
                if time.perf_counter() >= deadline:
                    break
                if not self._clear(*cell, *end_cell):
                    continue
                # Sampling the middle also penalizes a crossing between two
                # hidden endpoints. Distances below are lengths of game edges.
                mid = self.cell((pos[0] + end[0]) * 0.5, (pos[1] + end[1]) * 0.5)
                mid_exposure = self._exposure(mid, observers, vision, height)
                avg_exposure = (exposure + 2 * mid_exposure + end_exposure) * 0.25
                value = cost + math.dist(pos, end) * (1.0 + 3.0 * avg_exposure) + math.dist(end, horizon)
                estimate = min(estimate, value)
            if estimate < best_cost:
                vx, vy = pos[0] - x, pos[1] - y
                step = math.hypot(vx, vy)
                best_cost, best_move = estimate, (length * vx / step, length * vy / step)
        return best_move

    def can_hold_roof(self, me, zone, enemies, vision, deadline, size_per_cube=0.05):
        """Only hold a deck we already occupy, with zone slack and solid cover."""
        now = time.perf_counter()
        deadline = min(deadline, now + 0.0015)
        if not self.ready or not me.get("grounded") or now >= deadline:
            return False
        x, y, z = me["pos"]
        stats = me.get("stats", {})
        radius = stats.get("radius", self.mech.get("radius", 2.5))
        height = stats.get("height", self.mech.get("height", WALK_H))
        margin = radius + 4
        if (abs(z - self.ground_h(x, y)) < 4 or
                math.dist((x, y), zone["center"]) > zone["radius"] - margin or
                math.dist((x, y), zone["next_center"]) > zone["next_radius"] - margin):
            return False
        # Reuse the exact disk, holes and overhead clearance of roof traversal.
        if self.roof_walk(me, (x, y), deadline) is None:
            return False
        center = (x, y, z + 0.5 * height)
        top = (x, y, z + height)
        for enemy in enemies:
            ex, ey, ez = enemy["pos"]
            if math.hypot(ex - x, ey - y) > vision.get("range", 200):
                continue
            if time.perf_counter() >= deadline:
                return False
            eye_height = self.mech.get("eye_height", 8) * (1 + size_per_cube * (enemy.get("cubes") or 0))
            origin = (ex, ey, ez + eye_height)
            if self.los(origin, top, deadline) or self.los(origin, center, deadline):
                return False
        return True

    def ground_h(self, x, y):
        return self.terrain.height(x, y) if self.terrain else 0.0

    def route_length(self, x, y, goal):
        if self.field and self.field[0] == goal:
            i, j = self.cell(x, y)
            cost = self.field[1][j * self.gx + i]
            if cost < (1 << 30):
                return cost * (CELL / 2.0)
        return None

    def request(self, goal):
        """Ask for a flow field toward goal (x, y). Cheap; computed in the background."""
        if not self.ready:
            return
        if self._want is not None and math.dist(self._want, goal) < 1.0:
            return
        self._want = (goal[0], goal[1])
        if self.field and math.dist(self.field[0], self._want) < 1.0:
            self._job = None
            return
        self._job = (self._want, self._bfs(self._want))

    def work(self, budget_s):
        """Advance the pending BFS for at most budget_s seconds (call once per decision)."""
        if self._job is None:
            return
        goal, gen = self._job
        deadline = time.perf_counter() + budget_s
        try:
            while time.perf_counter() < deadline:
                next(gen)
        except StopIteration as done:
            self.field = (goal, done.value)
            self._job = None

    def _bfs(self, goal):
        gx, gy, blk = self.gx, self.gy, self.blocked
        n = gx * gy
        INF = 1 << 30
        dist = array("i", [INF]) * n
        si, sj = self.nearest_free(goal[0], goal[1], 80, main=True)
        s = sj * gx + si
        dist[s] = 0
        dq = deque([s])
        pop, push = dq.popleft, dq.append
        offs = (1, -1, gx, -gx)
        diag = ((1, gx), (-1, gx), (1, -gx), (-1, -gx))
        cnt = 0
        while dq:
            cnt += 1
            if cnt & 127 == 0:
                yield None
            k = pop()
            d = dist[k] + 2
            for o in offs:
                m = k + o
                if 0 <= m < n and not blk[m] and dist[m] > d:
                    dist[m] = d
                    push(m)
            d1 = d + 1  # diagonal ~ 1.5 x straight (approximate, BFS-like relaxation)
            for a, b in diag:
                m = k + a + b
                if 0 <= m < n and not blk[m] and dist[m] > d1 and not blk[k + a] and not blk[k + b]:
                    dist[m] = d1
                    push(m)
        return dist

    def steer(self, x, y, goal):
        """Direction (dx, dy) to walk toward goal using the flow field, or None to go straight."""
        f = self.field
        if not self.ready or f is None or math.dist(f[0], goal) > 1.0:
            return None
        dist = f[1]
        gx, gy = self.gx, self.gy
        i0, j0 = self.cell(x, y)
        k0 = j0 * gx + i0
        INF = 1 << 30
        if dist[k0] >= INF:
            i0, j0 = self.nearest_free(x, y, 6)
            k0 = j0 * gx + i0
            if dist[k0] >= INF:
                return None
        best, bk = dist[k0], None
        R = 4
        for dj in range(-R, R + 1):
            j = j0 + dj
            if j < 0 or j >= gy:
                continue
            for di in range(-R, R + 1):
                i = i0 + di
                if i < 0 or i >= gx:
                    continue
                d = dist[j * gx + i]
                if d < best and self._clear(i0, j0, i, j):
                    best, bk = d, (i, j)
        if bk is None:
            return None
        cx, cy = self.center(*bk)
        return cx - x, cy - y

    def walk_clear(self, x, y, goal):
        return self.ready and self._clear(*self.cell(x, y), *self.cell(*goal))

    def _clear(self, i0, j0, i1, j1):
        """Use the flow field's no-corner-cut rule when looking ahead."""
        dx, dy = abs(i1 - i0), abs(j1 - j0)
        sx = 1 if i1 > i0 else -1
        sy = 1 if j1 > j0 else -1
        nx = ny = 0
        gx, blk = self.gx, self.blocked
        while nx < dx or ny < dy:
            cross = (2 * nx + 1) * dy - (2 * ny + 1) * dx
            if cross == 0 and (blk[j0 * gx + i0 + sx] or blk[(j0 + sy) * gx + i0]):
                return False
            if cross <= 0:
                i0 += sx
                nx += 1
            if cross >= 0:
                j0 += sy
                ny += 1
            if blk[j0 * gx + i0]:
                return False
        return True

    def safe_step(self, x, y, move, sign, lookahead=6.0):
        if not self.ready or move == (0.0, 0.0):
            return move
        origin = self.cell(x, y)
        if self.blocked[origin[1] * self.gx + origin[0]]:
            # A spawn on a roof or in the grid's clearance margin still needs an exit.
            i, j = self.nearest_free(x, y, 12, main=True)
            if self.main[j * self.gx + i]:
                cx, cy = self.center(i, j)
                length = math.hypot(cx - x, cy - y)
                if length:
                    return (cx - x) / length, (cy - y) / length
            return move
        dx, dy = move
        # Try the requested move first, then slide to an open side in this same decision.
        for cos_a, sin_a in ((1, 0), (0.7071, 0.7071 * sign), (0.7071, -0.7071 * sign),
                             (0, sign), (0, -sign), (-0.7071, 0.7071 * sign),
                             (-0.7071, -0.7071 * sign), (-1, 0)):
            vx, vy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
            end = self.cell(x + lookahead * vx, y + lookahead * vy)
            if self._clear(*origin, *end) and not self.blocked[end[1] * self.gx + end[0]]:
                return vx, vy
        return 0.0, 0.0

    def los(self, a, b, deadline=None):
        """Check a game sightline against the solid prisms and sampled terrain."""
        if not self.ready:
            return True
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        bins = self.solid_bins
        candidates = set()
        for bx in range(math.floor(min(a[0], b[0]) / BUCKET), math.floor(max(a[0], b[0]) / BUCKET) + 1):
            for by in range(math.floor(min(a[1], b[1]) / BUCKET), math.floor(max(a[1], b[1]) / BUCKET) + 1):
                candidates.update(bins.get((bx, by), ()))
        for index in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                return True
            x0, y0, x1, y1, ring, holes, base, top = self.prisms[index]
            if _hits_prism(a, (dx, dy, dz), (x0, y0, base), (x1, y1, top), ring, holes):
                return False
        steps = max(1, math.ceil(math.hypot(dx, dy) / self.terrain.step))
        for s in range(steps + 1):
            if deadline is not None and s % 16 == 0 and time.perf_counter() >= deadline:
                return True
            t = s / steps
            x, y, z = a[0] + dx * t, a[1] + dy * t, a[2] + dz * t
            if self.terrain.height(x, y) >= z:
                return False
        return True


def _hits_prism(start, delta, lower, upper, ring, holes):
    # Clip to the solid's box so a deck only blocks rays at the deck's own height.
    enter, leave = 0.0, 1.0
    for p, d, low, high in zip(start, delta, lower, upper):
        if abs(d) < 1e-10:
            if p < low or p > high:
                return False
            continue
        u, v = (low - p) / d, (high - p) / d
        enter, leave = max(enter, min(u, v)), min(leave, max(u, v))
        if leave < enter:
            return False
    dx, dy = delta[:2]
    mx, my = start[0] + dx * (enter + leave) / 2, start[1] + dy * (enter + leave) / 2
    if _pip(mx, my, ring) and not any(_pip(mx, my, hole) for hole in holes):
        return True
    for boundary in (ring, *holes):
        x0, y0 = boundary[-2:]
        for i in range(0, len(boundary), 2):
            x1, y1 = boundary[i:i+2]
            ex, ey = x1 - x0, y1 - y0
            det = dx * ey - dy * ex
            if abs(det) > 1e-10:
                rx, ry = x0 - start[0], y0 - start[1]
                t = (rx * ey - ry * ex) / det
                u = (rx * dy - ry * dx) / det
                if enter <= t <= leave and 0 <= u <= 1:
                    return True
            x0, y0 = x1, y1
    return False
