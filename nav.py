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
                    if raw[k]:
                        continue
                    g = ground[k]
                    if top <= g + STEP or base >= g + WALK_H:
                        continue  # walkable over or under
                    x = self.minx + (i + 0.5) * CELL
                    if _pip(x, y, ring) and not any(_pip(x, y, h) for h in holes):
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
            if not tree["crownBase"] <= eye <= tree["height"]:
                continue
            cx, cy = tree.get("crownX", tree["x"]), tree.get("crownY", tree["y"])
            r = tree["crownR"]
            i0, j0 = self.cell(cx - r, cy - r)
            i1, j1 = self.cell(cx + r, cy + r)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    px, py = self.center(i, j)
                    if (px - cx) ** 2 + (py - cy) ** 2 < r * r:
                        self.canopy[j * gx + i] = 1

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

    def cover_spot(self, center, radius, near, rng, samples=96, *, threats=(), me_pos=None, safe_circle=None, near_range=45):
        """A free cell inside the circle, close to `near`, with walls around it for cover."""
        gx, main = self.gx, self.main
        candidates = {}
        for index in range(samples):
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
            score = math.dist((px, py), near) - 4.0 * min(cover, 12)
            candidates[i, j] = score
        if not candidates:
            return None
        shortlist = sorted(candidates, key=candidates.get)[:4]
        def rank(cell):
            i, j = cell
            px, py = self.center(i, j)
            eye = (px, py, self.ground[j * gx + i] + self.mech.get("eye_height", 8))
            exposed = sum(self.los(pos, eye) for pos in threats)
            soft = self.canopy[j * gx + i] and all(math.dist(pos, eye) > near_range for pos in threats)
            travel = math.dist((px, py), me_pos[:2]) if me_pos else 0
            return exposed, candidates[cell] + 0.3 * travel - 5 * soft
        return self.center(*min(shortlist, key=rank))

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

    def los(self, a, b):
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
            x0, y0, x1, y1, ring, holes, base, top = self.prisms[index]
            if _hits_prism(a, (dx, dy, dz), (x0, y0, base), (x1, y1, top), ring, holes):
                return False
        steps = max(1, math.ceil(math.hypot(dx, dy) / self.terrain.step))
        for s in range(steps + 1):
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
