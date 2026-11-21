"""Navigation grid built from the map files, and a flow field (BFS distance to a goal).

Everything heavy runs in a background thread so act() stays well under the 50 ms budget.
"""
import json
import math
import os
import sys
import threading
import time
from array import array
from collections import deque

CELL = 2.0          # meters per nav cell
WALK_H = 9.0        # mech height: overhangs lower than this are walls
STEP = 1.2
MAX_SLOPE = math.tan(math.radians(46))


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
    def __init__(self, map_dir):
        self.ready = False
        self.terrain = None
        self.blocked = None
        self.prisms = []
        self.field = None          # (goal, dist array)
        self._want = None          # goal requested
        self._lock = threading.Lock()
        self._ev = threading.Event()
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
        # steep terrain and map edge
        for j in range(gy):
            row = j * gx
            for i in range(gx):
                if i == 0 or j == 0 or i == gx - 1 or j == gy - 1:
                    raw[row + i] = 1
                    continue
                g = ground[row + i]
                if (abs(ground[row + i + 1] - g) > MAX_SLOPE * CELL or abs(ground[row + i - 1] - g) > MAX_SLOPE * CELL
                        or abs(ground[row + i + gx] - g) > MAX_SLOPE * CELL or abs(ground[row + i - gx] - g) > MAX_SLOPE * CELL):
                    raw[row + i] = 1
        # prisms
        for p in w["prisms"]:
            ring = p["ring"]
            holes = p.get("holes") or []
            self.prisms.append((min(ring[0::2]), min(ring[1::2]), max(ring[0::2]), max(ring[1::2]), ring, holes, p["base"], p["top"]))
            x0, x1 = min(ring[0::2]), max(ring[0::2])
            y0, y1 = min(ring[1::2]), max(ring[1::2])
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
        blk = self.blocked
        if main:
            blk = bytes(1 - v for v in self.main) if not hasattr(self, "_notmain") else self._notmain
            self._notmain = blk
        for r in range(0, maxr):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < gx and 0 <= j < gy and not blk[j * gx + i]:
                        return i, j
        return i0, j0

    def ground_h(self, x, y):
        return self.terrain.height(x, y) if self.terrain else 0.0

    def request(self, goal):
        """Ask for a flow field toward goal (x, y). Cheap; computed in the background."""
        if not self.ready:
            return
        if self._want is not None and math.dist(self._want, goal) < 1.0:
            return
        self._want = (goal[0], goal[1])
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
            if cnt & 2047 == 0:
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

    def _clear(self, i0, j0, i1, j1):
        """No blocked cell on the straight line between two cells."""
        n = max(abs(i1 - i0), abs(j1 - j0))
        gx, blk = self.gx, self.blocked
        for s in range(1, n + 1):
            i = i0 + (i1 - i0) * s // n
            j = j0 + (j1 - j0) * s // n
            if blk[j * gx + i]:
                return False
        return True

    def los(self, a, b):
        """Rough line of sight between two 3D points against building prisms (for shots)."""
        if not self.ready:
            return True
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        L = math.hypot(dx, dy)
        steps = max(1, int(L / 3.0))
        for s in range(1, steps):
            t = s / steps
            x, y, z = a[0] + dx * t, a[1] + dy * t, a[2] + dz * t
            i, j = self.cell(x, y)
            if self.raw[j * self.gx + i]:
                for (x0, y0, x1, y1, ring, holes, base, top) in self.prisms:
                    if x0 <= x <= x1 and y0 <= y <= y1 and base <= z <= top and _pip(x, y, ring):
                        if not any(_pip(x, y, h) for h in holes):
                            return False
        return True
