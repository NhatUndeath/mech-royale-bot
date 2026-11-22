from array import array
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bot import Bot
from nav import Nav


class Arena:
    """Flat arena with rectangular solid cover, sampled like the local nav grid."""

    ready = False

    def __init__(self, walls=()):
        self.walls = walls

    def free(self, x, y):
        return all(not (x0 <= x <= x1 and y0 <= y <= y1)
                   for x0, y0, x1, y1, _ in self.walls)

    def los(self, start, end):
        steps = max(2, math.ceil(math.dist(start, end)))
        for step in range(1, steps):
            x, y, z = (a + (b - a) * step / steps for a, b in zip(start, end))
            if any(x0 <= x <= x1 and y0 <= y <= y1 and z <= top
                   for x0, y0, x1, y1, top in self.walls):
                return False
        return True

    def ground_h(self, x, y):
        return 0.0

    combat_los = los

    def request(self, goal):
        pass

    def steer(self, x, y, goal):
        return None

    def work(self, budget):
        pass


WEAPONS = {
    "gun": dict(speed=252, range=230, damage=45, cooldown=0.24, mag=12, reload=2.4),
    "mortar": dict(speed=72, gravity=22, damage=200, cooldown=2.6, splash_radius=7),
    "laser": dict(range=280, damage=315, cooldown=1.65, charge=0.75, min_charge=0.15,
                  min_frac=0.35, hold_s=2),
    "lightning": dict(range=45, damage=60, cooldown=0.25, cone_deg=30, energy=3),
}
CROSS_FIRE_WALLS = ((9, -22, 17, -3, 16), (-22, 9, -3, 17, 16))


def weapon(kind, **state):
    w = dict(kind=kind, stats=WEAPONS[kind], cooldown=0)
    if kind == "gun":
        w.update(ammo=12, reload=0)
    elif kind == "laser":
        w["charge"] = 0
    return w | state


def enemy(actor_id, x, y, *, hp=1400, aim=0, vel=(0, 0, 0)):
    return dict(id=actor_id, kind="mech", pos=(x, y, 0), vel=vel, hp=hp,
                max_hp=1400, cubes=0, aim=(aim, 0), weapons=["gun"])


def hit_by(actor_id, bearing):
    return dict(type="damaged", by=actor_id, bearing=bearing, amount=45, weapon="gun")


class CoverDuels(unittest.TestCase):
    def setUp(self):
        # Geometry and decisions should give the same result on a slow emulator.
        self.enterContext(patch("bot.time.perf_counter", return_value=0.0))
        self.zone = dict(center=(0, 0), radius=560, next_center=(0, 0), next_radius=380,
                         shrink_start=45, shrink_end=90, dps=20)
        self.mech = dict(id=0, pos=(0, 0, 0), vel=(0, 0, 0), aim=(0, 0), hp=1400,
                         max_hp=1400, grounded=True, in_zone=True, energy=100, cubes=0,
                         weapons=[weapon("gun"), None],
                         stats=dict(muzzle_height=7, height=9, walk_speed=11.7, min_pitch=-0.69,
                                    max_pitch=1.29, turn_rate=5, pitch_rate=4))

    def bot(self, walls=()):
        with patch("bot.Nav", return_value=Arena(walls)):
            return Bot({"config": {"mech": {"regen_delay": 5}, "weapons": WEAPONS,
                                   "loot": {"pickup_radius": 4}}})

    def observe(self, bot, enemies=(), *, t=10, events=(), zone=None, **mech):
        return bot.act(dict(type="obs", tick=round(t * 20), t=t, alive=16,
                            self=self.mech | mech, zone=zone or self.zone,
                            mechs=enemies, events=events))

    def test_blocked_recent_attacker_does_not_hide_clear_finisher(self):
        bot = self.bot(((13, -14, 17, 14, 20),))
        opponents = [enemy(1, 52, 0), enemy(2, 0, 72, hp=40)]
        action = self.observe(bot, opponents, aim=(math.pi / 2, -0.035), events=[hit_by(1, 0)])
        self.assertAlmostEqual(action["aim"][0], math.pi / 2)
        self.assertTrue(action["fire"][0])

    def test_ready_gun_can_finish_while_laser_is_on_cooldown(self):
        bot = self.bot()
        action = self.observe(bot, [enemy(1, 62, 0, hp=40)], aim=(0, -0.04),
                              weapons=[weapon("gun"), weapon("laser", cooldown=1.3)])
        self.assertEqual(action["fire"], [True, False])

    def test_laser_uses_current_bearing_when_releasing(self):
        bot = self.bot()
        action = self.observe(bot, [enemy(1, 80, 0, vel=(0, 16, 0))], aim=(0, -0.031),
                              weapons=[None, weapon("laser", charge=0.75)])
        self.assertAlmostEqual(action["aim"][0], 0)
        self.assertEqual(action["fire"], [False, False])

    def test_orphaned_laser_release_does_not_pick_the_mortar_aim(self):
        bot = self.bot(((32, -15, 39, 15, 6),))
        actual_aim = (0.35, -0.05)
        action = self.observe(bot, [enemy(1, 78, 0, hp=125)], aim=actual_aim,
                              weapons=[weapon("laser", charge=0.75), weapon("mortar")])
        self.assertEqual(action["aim"], list(actual_aim))
        self.assertEqual(action["fire"], [False, False])

    def test_laser_can_engage_beyond_gun_range(self):
        bot = self.bot()
        action = self.observe(bot, [enemy(1, 248, 0, hp=100)], aim=(0, -0.01),
                              weapons=[weapon("gun", ammo=0, reload=1.7), weapon("laser")])
        self.assertTrue(action["fire"][1])

    def test_new_laser_charge_cannot_slow_zone_escape(self):
        bot = self.bot()
        action = self.observe(bot, [enemy(1, 80, 0)], in_zone=False,
                              zone=self.zone | dict(center=(-400, 0), next_center=(-400, 0)),
                              weapons=[None, weapon("laser")])
        self.assertFalse(action["fire"][1])
        self.assertLess(action["move"][0], -0.9)

    def test_zone_escape_selects_gun_over_ineligible_laser_finisher(self):
        bot = self.bot()
        action = self.observe(bot, [enemy(1, 248, 0, hp=1), enemy(2, 0, 80)],
                              aim=(math.pi / 2, -0.031), in_zone=False,
                              zone=self.zone | dict(center=(-400, 0), next_center=(-400, 0)),
                              weapons=[weapon("gun"), weapon("laser")])
        self.assertEqual(action["fire"], [True, False])
        self.assertLess(action["move"][0], -0.9)

    def test_expired_combat_budget_keeps_zone_movement_and_valid_tick(self):
        bot = self.bot()
        start = iter([0.0])
        with patch("bot.time.perf_counter", side_effect=lambda: next(start, 0.020)):
            action = self.observe(bot, [enemy(1, 80, 0)], in_zone=False,
                                  zone=self.zone | dict(center=(-400, 0), next_center=(-400, 0)),
                                  weapons=[weapon("gun"), weapon("laser")])
        self.assertEqual(action["tick"], 200)
        self.assertEqual(action["fire"], [False, False])
        self.assertLess(action["move"][0], -0.9)

    def test_gun_trigger_respects_the_final_aim(self):
        bot = self.bot()
        target = enemy(1, 80, 0)
        aim = bot.lead(self.mech, target, 252)[:2]
        action = self.observe(bot, [target], aim=aim,
                              weapons=[weapon("gun"), weapon("mortar")])
        if action["fire"][0]:
            self.assertAlmostEqual(action["aim"][1], aim[1], delta=0.04)
        if action["fire"][1]:
            self.assertFalse(action["fire"][0])
        self.assertTrue(any(action["fire"]))

    def test_mortar_respects_solid_cover_along_arc(self):
        target = enemy(1, 78, 0, hp=125)
        for wall_height, expected in ((6, True), (25, False)):
            with self.subTest(wall_height=wall_height):
                bot = self.bot(((32, -15, 39, 15, wall_height),))
                aim = bot.mortar_aim(self.mech, target, WEAPONS["mortar"])[:2]
                action = self.observe(bot, [target], aim=aim, weapons=[None, weapon("mortar")])
                self.assertEqual(action["fire"][1], expected)

    def test_isolated_duel_strafes_toward_a_cover_exit(self):
        bot = self.bot((CROSS_FIRE_WALLS[0],))
        action = self.observe(bot, [enemy(1, 65, 0, hp=85, aim=math.pi)],
                              aim=(0, -0.038), events=[hit_by(1, 0)])
        self.assertLess(action["move"][1], 0)
        self.assertTrue(action["fire"][0])

    def test_blocked_duel_strafe_uses_checked_diagonal_escape(self):
        bot = self.bot(((-2, -14, 2, -3, 15), (12, -7, 16, -5, 15)))
        action = self.observe(bot, [enemy(1, 65, 0, aim=math.pi)], aim=(0, -0.038))
        self.assertLess(action["move"][0], -0.2)
        self.assertLess(action["move"][1], -0.2)

    def test_crossfire_retreats_before_critical_hp(self):
        bot = self.bot(CROSS_FIRE_WALLS)
        opponents = [enemy(1, 65, 0, aim=math.pi), enemy(2, 0, 60, aim=-math.pi / 2)]
        action = self.observe(bot, opponents, hp=1000,
                              weapons=[weapon("gun"), weapon("laser")],
                              events=[hit_by(1, 0), hit_by(2, math.pi / 2)])
        self.assertLess(action["move"][0], -0.2)
        self.assertLess(action["move"][1], -0.2)
        self.assertFalse(action["fire"][1])

    def test_out_of_range_lightning_does_not_count_as_crossfire(self):
        bot = self.bot()
        opponents = [enemy(1, 65, 0, aim=math.pi),
                     enemy(2, 0, 80, aim=-math.pi / 2) | {"weapons": ["lightning"]}]
        action = self.observe(bot, opponents, hp=1000, aim=(0, -0.038))
        self.assertTrue(action["fire"][0])
        self.assertFalse(action["boost"])

    def test_recovery_stays_in_cover_and_reloads_after_crossfire(self):
        bot = self.bot(CROSS_FIRE_WALLS)
        opponents = [enemy(1, 65, 0, aim=math.pi), enemy(2, 0, 60, aim=-math.pi / 2)]
        self.observe(bot, opponents, hp=730, events=[hit_by(1, 0), hit_by(2, math.pi / 2)])
        action = self.observe(bot, opponents, t=12, pos=(-12, -12, 0), hp=730,
                              weapons=[weapon("gun", ammo=2), None])
        self.assertEqual(action["move"], [0, 0])
        self.assertEqual(action["fire"], [False, False])
        self.assertTrue(action["reload"])

    def test_recovery_requires_cover_when_enemy_is_near(self):
        bot = self.bot()
        opponents = [enemy(1, 65, 0, aim=math.pi), enemy(2, 0, 60, aim=-math.pi / 2)]
        self.observe(bot, opponents, hp=730, events=[hit_by(1, 0), hit_by(2, math.pi / 2)])
        action = self.observe(bot, [enemy(1, 65, 0, aim=math.pi + 0.75)], t=11.3,
                              hp=730, aim=(0, -0.038))
        self.assertNotEqual(action["move"], [0, 0])
        self.assertTrue(action["fire"][0])

    def test_target_inertia_expires_immediately_when_shot_is_blocked(self):
        bot = self.bot()
        a, b = enemy(1, 60, -14), enemy(2, 60, 16)
        self.assertLess(self.observe(bot, [a, b])["aim"][0], 0)
        b = enemy(2, 60, 13)
        self.assertLess(self.observe(bot, [a, b], t=10.1)["aim"][0], 0)
        bot.nav.walls = ((21, -15, 31, -3, 25),)
        self.assertGreater(self.observe(bot, [a, b], t=10.2)["aim"][0], 0)

    def test_farming_and_upgrade_still_use_baseline_actions(self):
        bot = self.bot()
        action = bot.act(dict(type="obs", tick=200, t=10, alive=16,
                              self=self.mech | {"aim": (0, math.atan2(-5.25, 30))},
                              mechs=[], zone=self.zone, crates=[dict(id=1000, pos=(30, 0, 0), hp=300)],
                              offer={"choices": [{"id": "gun_rate"}, {"id": "armor"}]}))
        self.assertEqual(action["pick"], 1)
        self.assertEqual(action["fire"], [True, False])
        self.assertEqual(action["move"], [0, 0])


class CombatGeometry(unittest.TestCase):
    def test_bridge_and_terrain_occlude_only_the_matching_height(self):
        with tempfile.TemporaryDirectory() as path:
            arena = Path(path) / "v2"
            arena.mkdir()
            world = {
                "bounds": {"minX": 0, "minY": 0, "maxX": 48, "maxY": 40},
                "terrain": {"minX": 0, "minY": 0, "nx": 49, "ny": 41, "step": 1},
                "prisms": [{"base": 12, "top": 18, "ring": [12, 10, 16, 10, 16, 24, 12, 24]}],
            }
            (arena / "world.json").write_text(json.dumps(world), encoding="utf-8")
            terrain = array("h", [0]) * (49 * 41)
            for y in range(8):
                for x in range(18, 21):
                    terrain[y * 49 + x] = 600
            (arena / "terrain.i16").write_bytes(terrain.tobytes())
            nav = Nav(path)
            ray = getattr(nav, "combat_los", nav.los)
            self.assertTrue(ray((2, 20, 6), (34, 20, 6)), "pass under the bridge")
            self.assertFalse(ray((2, 20, 14), (34, 20, 14)), "bridge deck blocks the shot")
            self.assertFalse(ray((2, 4, 3), (34, 4, 3)), "terrain ridge blocks the shot")


if __name__ == "__main__":
    unittest.main()
