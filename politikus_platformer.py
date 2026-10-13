"""
Politikus Platformer - a smooth, modern-styled platformer starring Hungarian
politicians. (Anti-aliased vector rendering - NOT chunky 8-bit pixels.)

Controls:
    Title : <- / -> (or A/D) pick a politician,  ENTER / SPACE to start
            (or press 1-6 to jump straight in)
    Play  : A / D or <- / ->  move
            SPACE / W / UP    jump (hold = higher, release = short hop)
            R                 restart level
            ESC               quit

Goal: collect votes, dodge the press (microphones), grab the debate shield,
      reach the flag. Survive both levels!
"""

import array
import math
import sys

try:
    import pygame
except ImportError:
    sys.exit("pygame is required:  pip install pygame   (or: pip install pygame-ce)")

# ---------- palette ----------
SKIN = (238, 199, 168)
HAIR = {"H": (74, 46, 26), "L": (206, 178, 120), "b": (232, 200, 120),
        "r": (196, 92, 54), "g": (158, 158, 168),
        "SP": (150, 150, 160),   # salt-and-pepper (grey)
        "GB": (150, 108, 60),    # golden brown
        "DK": (44, 30, 20),      # dark brown / near black
        "MB": (96, 64, 38)}      # medium brown
EYE = {"B": (90, 70, 48), "L": (110, 150, 170), "G": (130, 165, 175)}  # brown / blue / grey-blue
SUIT = {"B": (34, 52, 120), "A": (60, 110, 200), "N": (30, 32, 44)}
TIE = (206, 42, 52)
WHITE = (245, 246, 250)
YELL = (255, 210, 54)
GREEN = (64, 178, 84)
DARK = (26, 28, 40)

# Appearance based on real photos:
#  Orbán      - salt-and-pepper slicked-back hair, clean-shaven, blue-grey eyes
#  Karácsony  - brown forward-swept hair, black rectangular glasses, light stubble
#  M. Magyar  - golden-brown up-swept quiff, light stubble, strong jaw
#  Szijjártó  - dark hair, receding at the temples, clean-shaven
#  Berki      - light brown hair, clean-shaven (no modern portrait available)
#  Dömötör    - dark gelled slicked-back hair, clean-shaven, round full face
POLITICIANS = [
    {"name": "Orban",     "hair": "SP", "suit": "B", "eyes": "L", "slick": True},
    {"name": "Karacsony", "hair": "MB", "suit": "A", "eyes": "G", "glasses": True, "stubble": True},
    {"name": "M. Magyar", "hair": "GB", "suit": "B", "eyes": "G", "quiff": True, "stubble": True},
    {"name": "Szijjarto", "hair": "DK", "suit": "N", "eyes": "B", "recede": True},
    {"name": "Berki",     "hair": "MB", "suit": "A", "eyes": "B"},
    {"name": "Domotor",   "hair": "DK", "suit": "B", "eyes": "B", "slick": True, "round": True},
]

VIEW_W, VIEW_H = 960, 540
GROUND_Y = 470
PLAYER_W, PLAYER_H = 46, 60


# ---------- render helpers ----------
def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def vgradient(w, h, top, bottom):
    s = pygame.Surface((w, h))
    for y in range(h):
        pygame.draw.line(s, lerp(top, bottom, y / max(1, h - 1)), (0, y), (w, y))
    return s


def soft_cloud(scale=1):
    c = pygame.Surface((150 * scale, 70 * scale), pygame.SRCALPHA)
    pts = [(40, 45, 30), (75, 35, 40), (110, 45, 30), (90, 52, 34), (55, 52, 28)]
    for x, y, r in pts:
        pygame.draw.circle(c, (255, 255, 255, 215), (int(x * scale), int(y * scale)), int(r * scale))
    return c


def make_shadow(w):
    s = pygame.Surface((w, int(w * 0.22)), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0, 0, 0, 90), s.get_rect())
    return s


def synth_sound(freq, ms, vol=110):
    rate = 22050
    n = int(rate * ms / 1000)
    buf = array.array("h")
    for i in range(n):
        t = i / rate
        v = vol if (int(t * freq) % 2 == 0) else -vol
        buf.append(int(v * (1.0 - (i / n) * 0.6)))
    return pygame.mixer.Sound(buffer=buf.tobytes())


def build_sounds():
    try:
        return {"jump": synth_sound(330, 90), "vote": synth_sound(880, 70),
                "hit": synth_sound(140, 160), "shield": synth_sound(660, 140),
                "win": synth_sound(523, 220)}
    except Exception:
        return {}


# ---------- level ----------
def build_level(level):
    ground_y = GROUND_Y
    if level == 1:
        plats = [(300 + i * 250, 340 if i % 2 == 0 else 298, 130, 16) for i in range(15)]
        votes = [(x + 65, y - 48) for (x, y, w, h) in plats]
        votes += [(200, 440), (1200, 440), (2600, 440), (3800, 440)]
        mic_idx = {2, 5, 8, 11, 13}
        mics = [(plats[i][0] + 55, plats[i][1] - 52, None) for i in mic_idx]
        mics += [(1600, ground_y - 54, None), (3200, ground_y - 54, None)]
        shields = [(plats[4][0] + 65, plats[4][1] - 46), (plats[9][0] + 65, plats[9][1] - 46)]
    else:
        plats = [(260 + i * 230, 330 if i % 2 == 0 else 288, 120, 16) for i in range(18)]
        votes = [(x + 60, y - 48) for (x, y, w, h) in plats]
        votes += [(150, 440), (1100, 440), (2200, 440), (3300, 440), (4200, 440)]
        mics = []
        move = {3: (1.6, 24), 6: (2.0, 22), 10: (1.8, 28), 14: (2.0, 24)}
        for i, (x, y, w, h) in enumerate(plats):
            cx = x + 55
            spd, amp = move.get(i, (1.4, 20))
            if i % 2 == 1 or i in move:
                mics.append((cx, y - 52, (spd, cx, amp)))
        mics += [(1400, ground_y - 54, None), (2800, ground_y - 54, None), (4000, ground_y - 54, None)]
        shields = [(plats[5][0] + 60, plats[5][1] - 46), (plats[12][0] + 60, plats[12][1] - 46)]

    platforms = [pygame.Rect(x, y, w, h) for (x, y, w, h) in plats]
    votes = [pygame.Rect(x - 20, y - 20, 40, 40) for (x, y) in votes]
    mic_mv = [mv if mv else None for (x, y, mv) in mics]
    mics = [pygame.Rect(x - 20, y, 40, 54) for (x, y, mv) in mics]
    shields = [pygame.Rect(x - 24, y - 24, 48, 48) for (x, y) in shields]
    flag_x = plats[-1][0] + plats[-1][2] + 220
    return ground_y, platforms, votes, mics, mic_mv, shields, flag_x


# ---------- character drawing (smooth) ----------
def draw_player(surf, x, y, w, h, pol, facing=1, walk=0.0, t=0.0):
    cx = x + w / 2
    suit = SUIT[pol["suit"]]
    hair = HAIR[pol["hair"]]
    skin = SKIN
    moving = abs(walk) > 0.05

    # legs
    leg_h = h * 0.34
    leg_w = w * 0.24
    swing = math.sin(walk) * (w * 0.16) if moving else 0
    for sx, off in ((-0.27, swing), (0.03, -swing)):
        lx = cx + w * sx
        pygame.draw.rect(surf, (34, 36, 48),
                         (lx, y + h - leg_h, leg_w, leg_h), border_radius=int(leg_w * 0.4))
        pygame.draw.rect(surf, (16, 16, 22),
                         (lx - 2, y + h - 8, leg_w + 4, 8), border_radius=3)

    # body (suit)
    body_top = y + h * 0.30
    body_h = leg_h - h * 0.02
    pygame.draw.rect(surf, suit, (cx - w * 0.42, body_top, w * 0.84, body_h),
                     border_radius=int(w * 0.22))
    # shirt
    pygame.draw.rect(surf, WHITE, (cx - w * 0.16, body_top, w * 0.32, body_h * 0.92),
                     border_radius=4)
    # tie
    pygame.draw.polygon(surf, TIE, [(cx, body_top + 3), (cx - w * 0.13, body_top + w * 0.28),
                                    (cx, body_top + body_h * 0.72), (cx + w * 0.13, body_top + w * 0.28)])

    # head
    hr = w * 0.40
    if pol.get("round"):
        hr = w * 0.44
    hcy = y + h * 0.17
    hx = int(cx)
    hy = int(hcy)
    pygame.draw.circle(surf, skin, (hx, hy), int(hr))

    # hair
    if pol.get("recede"):
        # receding: hair only on the sides, bare forehead
        pygame.draw.arc(surf, hair, (hx - hr, hy - hr, hr * 2, hr * 2), math.radians(150), math.radians(390), int(hr * 0.5))
        pygame.draw.circle(surf, hair, (hx - int(hr * 0.62), hy + int(hr * 0.05)), int(hr * 0.34))
        pygame.draw.circle(surf, hair, (hx + int(hr * 0.62), hy + int(hr * 0.05)), int(hr * 0.34))
    elif pol.get("quiff"):
        # base cap + up-swept top
        pygame.draw.circle(surf, hair, (hx, hy - int(hr * 0.12)), int(hr * 1.0))
        pygame.draw.circle(surf, skin, (hx, hy + int(hr * 0.34)), int(hr * 0.80))
        for k in range(-1, 3):
            pygame.draw.ellipse(surf, hair,
                                (hx + k * int(hr * 0.34) - int(hr * 0.2), hy - int(hr * 1.15),
                                 int(hr * 0.5), int(hr * 0.7)))
    elif pol.get("slick"):
        # slicked back: cap + a swept-back highlight, full hairline
        pygame.draw.circle(surf, hair, (hx, hy - int(hr * 0.14)), int(hr * 1.03))
        pygame.draw.circle(surf, skin, (hx, hy + int(hr * 0.30)), int(hr * 0.80))
        # sweep lines
        for k in range(-2, 3):
            pygame.draw.line(surf, lerp(hair, (255, 255, 255), 0.15),
                             (hx + k * int(hr * 0.28), hy - int(hr * 0.72)),
                             (hx + k * int(hr * 0.28) + int(facing * 4), hy - int(hr * 1.0)), 2)
    else:
        # default: simple forward-swept cap
        pygame.draw.circle(surf, hair, (hx, hy - int(hr * 0.16)), int(hr * 1.02))
        pygame.draw.circle(surf, skin, (hx, hy + int(hr * 0.28)), int(hr * 0.80))

    # stubble (light, on jawline)
    if pol.get("stubble"):
        pygame.draw.arc(surf, (180, 160, 150), (hx - int(hr * 0.85), hy - int(hr * 0.1),
                                                hr * 1.7, hr * 1.7), math.radians(20), math.radians(160), max(2, int(hr * 0.18)))

    # eyebrows
    eb = lerp(hair, (0, 0, 0), 0.25)
    ey = hy - int(hr * 0.24)
    for ex in (-1, 1):
        pygame.draw.line(surf, eb,
                         (hx + int(facing * hr * 0.1) + ex * int(hr * 0.18), ey),
                         (hx + int(facing * hr * 0.1) + ex * int(hr * 0.5), ey - int(hr * 0.06)), max(2, int(hr * 0.14)))

    # eyes (offset by facing, per-character color)
    eo = int(facing * hr * 0.12)
    er = max(2, int(hr * 0.16))
    eye_col = EYE[pol.get("eyes", "B")]
    for ex in (-1, 1):
        px = hx + eo + ex * int(hr * 0.42)
        py = hy + int(hr * 0.02)
        pygame.draw.circle(surf, (255, 255, 255), (px, py), er + 1)
        pygame.draw.circle(surf, eye_col, (px + int(facing * 1), py), max(1, er - 1))
        pygame.draw.circle(surf, (20, 20, 26), (px + int(facing * 1), py), max(1, er - 3))

    # glasses
    if pol.get("glasses"):
        gc = (24, 26, 34)
        for ex in (-1, 1):
            px = hx + eo + ex * int(hr * 0.42)
            py = hy + int(hr * 0.02)
            pygame.draw.rect(surf, gc, (px - int(hr * 0.34), py - int(hr * 0.30),
                                        int(hr * 0.68), int(hr * 0.52)), 2, border_radius=3)
        pygame.draw.line(surf, gc, (hx + eo - int(hr * 0.42), hy), (hx + eo + int(hr * 0.42), hy), 2)
        # temples
        for ex in (-1, 1):
            pygame.draw.line(surf, gc,
                             (hx + eo + ex * int(hr * 0.42) + ex * int(hr * 0.34), hy),
                             (hx + ex * int(hr * 0.9), hy - int(hr * 0.1)), 2)

    # nose
    pygame.draw.circle(surf, (226, 178, 148), (hx + eo, hy + int(hr * 0.30)), max(2, int(hr * 0.14)))

    # mouth
    pygame.draw.arc(surf, (170, 90, 84), (hx - int(hr * 0.24) + eo, hy + int(hr * 0.36),
                                          int(hr * 0.48), int(hr * 0.28)), 0, math.pi, 2)


class Game:
    MAX_LEVEL = 2
    JUMP_V = -14.0
    JUMP_BUFFER = 0.15
    COYOTE = 0.12

    def __init__(self):
        try:
            pygame.mixer.pre_init(22050, -16, 1, 512)
        except Exception:
            pass
        pygame.init()
        pygame.display.set_caption("Politikus Platformer")
        self.screen = pygame.display.set_mode((VIEW_W, VIEW_H))
        self.font = pygame.font.Font(None, 22)
        self.bigfont = pygame.font.Font(None, 54)
        self.clock = pygame.time.Clock()
        self.sounds = build_sounds()
        self.sky = vgradient(VIEW_W, VIEW_H, (150, 200, 245), (80, 150, 220))
        self.clouds = [soft_cloud(1.0), soft_cloud(0.7), soft_cloud(1.3)]
        self.shadows = {34: make_shadow(46), 40: make_shadow(52), 48: make_shadow(64)}
        self.state = "title"
        self.selected = 0
        self.t = 0.0
        self.particles = []
        self.reset_level(1)

    def play(self, name):
        s = self.sounds.get(name)
        if s:
            try:
                s.play()
            except Exception:
                pass

    def reset_level(self, level):
        (self.ground_y, self.platforms, self.votes, self.mics,
         self.mic_mv, self.shields, self.flag_x) = build_level(level)
        self.level = level
        self.player = pygame.Rect(70, self.ground_y - PLAYER_H, PLAYER_W, PLAYER_H)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.facing = 1
        self.walk = 0.0
        self.score = 0
        self.lives = 3
        self.camera_x = 0.0
        self.win = False
        self.shield_time = 0.0
        self.inv = 0.0
        self.mic_t = 0.0
        self.jump_buffer = 0.0
        self.coyote = 0.0
        self.particles = []

    # ---------- events ----------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if self.state == "play" and e.type == pygame.KEYUP:
            if e.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP) and self.vy < -5:
                self.vy = -5
            return True
        if e.type != pygame.KEYDOWN:
            return True
        k = e.key
        if k == pygame.K_ESCAPE:
            return False
        if self.state == "title":
            n = len(POLITICIANS)
            if k in (pygame.K_LEFT, pygame.K_a, pygame.K_DOWN, pygame.K_s):
                self.selected = (self.selected - 1) % n
            elif k in (pygame.K_RIGHT, pygame.K_d, pygame.K_UP, pygame.K_w):
                self.selected = (self.selected + 1) % n
            elif k in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = "play"; self.reset_level(1)
            elif pygame.K_1 <= k <= pygame.K_6:
                self.selected = k - pygame.K_1
                self.state = "play"; self.reset_level(1)
        elif self.state == "play":
            if k in (pygame.K_SPACE, pygame.K_w, pygame.K_UP):
                self.jump_buffer = self.JUMP_BUFFER
            if k == pygame.K_r:
                self.reset_level(self.level)
        elif self.state == "win":
            if k == pygame.K_r:
                self.reset_level(1); self.state = "play"
            elif k in (pygame.K_RETURN, pygame.K_SPACE):
                if self.level < self.MAX_LEVEL:
                    self.level += 1; self.reset_level(self.level); self.state = "play"
                else:
                    self.state = "title"
        elif self.state == "dead":
            if k == pygame.K_r:
                self.reset_level(self.level); self.state = "play"
            if k in (pygame.K_RETURN, pygame.K_SPACE):
                self.state = "title"
        return True

    # ---------- update ----------
    def _spawn(self, x, y, n, color, spd=3.0, life=0.6):
        for i in range(n):
            a = random_angle()
            self.particles.append({"x": x, "y": y,
                                   "vx": math.cos(a) * spd * (0.5 + random01()),
                                   "vy": math.sin(a) * spd * (0.5 + random01()) - 1.5,
                                   "life": life * (0.6 + random01()), "max": life,
                                   "color": color, "size": 3 + int(random01() * 3)})

    def update(self, dt):
        self.t += dt
        # particles always
        for p in self.particles[:]:
            p["x"] += p["vx"]; p["y"] += p["vy"]; p["vy"] += 0.18
            p["life"] -= dt
            if p["life"] <= 0:
                self.particles.remove(p)
        if self.state != "play" or self.win:
            return
        if self.shield_time > 0:
            self.shield_time = max(0.0, self.shield_time - dt)
        if self.inv > 0:
            self.inv = max(0.0, self.inv - dt)

        keys = pygame.key.get_pressed()
        speed = 6.5
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.vx = -speed; self.facing = -1
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.vx = speed; self.facing = 1
        else:
            self.vx = 0.0
        if abs(self.vx) > 0 and self.on_ground:
            self.walk += dt * 11

        self.vy += 0.9
        if self.vy > 22:
            self.vy = 22

        self.player.x += int(round(self.vx))
        if self.player.x < 0:
            self.player.x = 0
        if self.player.x > self.flag_x + 260:
            self.player.x = self.flag_x + 260

        was_ground = self.on_ground
        self.player.y += int(round(self.vy))
        self.on_ground = False
        if self.player.bottom >= self.ground_y and self.vy >= 0:
            self.player.bottom = self.ground_y; self.vy = 0; self.on_ground = True

        for p in self.platforms:
            if self.vy >= 0 and self.player.colliderect(p):
                prev_bottom = self.player.y - int(round(self.vy)) + self.player.h
                if prev_bottom <= p.top + 8:
                    self.player.bottom = p.top; self.vy = 0; self.on_ground = True

        if was_ground and not self.on_ground and self.vy > 0:
            pass
        if not was_ground and self.on_ground:
            self._spawn(self.player.centerx, self.player.bottom, 8, (180, 180, 170), 2.5, 0.4)

        if self.on_ground:
            self.coyote = self.COYOTE
        else:
            self.coyote = max(0.0, self.coyote - dt)
        if self.jump_buffer > 0:
            self.jump_buffer = max(0.0, self.jump_buffer - dt)
        if self.jump_buffer > 0 and (self.on_ground or self.coyote > 0):
            self.vy = self.JUMP_V
            self.on_ground = False
            self.coyote = 0.0
            self.jump_buffer = 0.0
            self.play("jump")
            self._spawn(self.player.centerx, self.player.bottom, 6, (210, 210, 200), 2.0, 0.35)

        self.mic_t += dt
        for m, mv in zip(self.mics, self.mic_mv):
            if mv:
                spd, center, amp = mv
                phase = (self.mic_t * spd) % 2.0
                m.centerx = int(center + (amp * phase if phase < 1 else -amp * (phase - 1)))

        target = self.player.centerx - VIEW_W // 2
        self.camera_x += (max(0, min(target, self.flag_x + 300 - VIEW_W)) - self.camera_x) * 0.12

        for v in self.votes[:]:
            if self.player.colliderect(v):
                self.score += 1; self.votes.remove(v); self.play("vote")
                self._spawn(v.centerx, v.centery, 12, YELL, 3.5, 0.5)

        for s in self.shields[:]:
            if self.player.colliderect(s):
                self.shield_time = 8.0; self.shields.remove(s); self.play("shield")
                self._spawn(s.centerx, s.centery, 14, (120, 180, 255), 4.0, 0.6)

        if self.inv <= 0:
            for m in self.mics:
                if self.player.colliderect(m):
                    if self.shield_time > 0:
                        self.shield_time = 0.0; self.inv = 1.2
                        self.play("hit"); self._knockback()
                        self._spawn(self.player.centerx, self.player.centery, 16, (120, 180, 255), 4.5, 0.6)
                    else:
                        self.lives -= 1; self.play("hit")
                        self._spawn(self.player.centerx, self.player.centery, 16, (220, 60, 60), 4.5, 0.6)
                        if self.lives <= 0:
                            self.state = "dead"
                        else:
                            self._knockback()
                    break

        if self.player.right >= self.flag_x:
            self.state = "win"; self.win = True; self.play("win")

    def _knockback(self):
        self.player.x = 70
        self.player.y = self.ground_y - PLAYER_H
        self.vx = 0.0; self.vy = 0.0
        self.camera_x = 0.0
        self.inv = 1.2
        self.jump_buffer = 0.0
        self.coyote = 0.0

    # ---------- draw ----------
    def draw_bg(self):
        self.screen.blit(self.sky, (0, 0))
        off = -int(self.camera_x)
        # distant hills
        for i in range(-2, 14):
            bx = i * 220 + (off % 220) - 220
            pygame.draw.ellipse(self.screen, (120, 178, 120),
                                (bx, self.ground_y - 70, 300, 140))
        # clouds
        for i, c in enumerate(self.clouds):
            cx = (i * 320 - int(self.camera_x * 0.3)) % (VIEW_W + 300) - 150
            self.screen.blit(c, (cx, 40 + i * 40))

    def draw_world(self):
        self.draw_bg()
        off = -int(self.camera_x)
        # ground
        gtop = pygame.draw.rect
        grad = vgradient(1, self.screen.get_height(), (70, 150, 70), (96, 66, 40))
        self.screen.blit(grad, (0, self.ground_y, VIEW_W, VIEW_H - self.ground_y))
        pygame.draw.rect(self.screen, (90, 190, 96), (0, self.ground_y, VIEW_W, 8))
        pygame.draw.rect(self.screen, (50, 120, 58), (0, self.ground_y + 8, VIEW_W, 3))

        for p in self.platforms:
            r = p.move(off, 0)
            if r.right < -20 or r.left > VIEW_W + 20:
                continue
            pygame.draw.rect(self.screen, (96, 66, 40), r, border_radius=6)
            pygame.draw.rect(self.screen, (74, 50, 30),
                             (r.x, r.y + r.h - 4, r.w, 4), border_radius=3)
            pygame.draw.rect(self.screen, (90, 190, 96), (r.x, r.y, r.w, 6), border_radius=6)

        # flag
        fx = self.flag_x + off
        if -80 < fx < VIEW_W + 80:
            self._draw_flag(fx, self.ground_y)

        # votes
        for v in self.votes:
            r = v.move(off, 0)
            if -40 < r.centerx < VIEW_W + 40:
                self._draw_vote(r.centerx, r.centery)

        for s in self.shields:
            r = s.move(off, 0)
            if -40 < r.centerx < VIEW_W + 40:
                self._draw_shield(r.centerx, r.centery)

        for m in self.mics:
            r = m.move(off, 0)
            if -60 < r.centerx < VIEW_W + 60:
                self._draw_mic(r.centerx, r.bottom)

        # particles
        for p in self.particles:
            a = int(255 * (p["life"] / p["max"]))
            s = pygame.Surface((p["size"] * 2, p["size"] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (p["color"][0], p["color"][1], p["color"][2], max(0, min(255, a))),
                               (p["size"], p["size"]), p["size"])
            self.screen.blit(s, (int(p["x"]) + off, int(p["y"])))

        # player shadow + player
        sh = self.shadows[40]
        shx = self.player.centerx + off
        self.screen.blit(sh, (shx - sh.get_width() // 2, self.ground_y - sh.get_height() + 2))
        if self.shield_time > 0:
            pygame.draw.circle(self.screen, (120, 180, 255, 0), (0, 0), 1)
            glow = pygame.Surface((120, 120), pygame.SRCALPHA)
            pulse = int(120 + 20 * math.sin(self.t * 8))
            pygame.draw.circle(glow, (120, 180, 255, pulse), (60, 60), 52)
            pygame.draw.circle(glow, (180, 220, 255, pulse), (60, 60), 40, 4)
            self.screen.blit(glow, (self.player.centerx + off - 60, self.player.centery - 60))
        if self.inv <= 0 or int(self.inv * 20) % 2 == 0:
            draw_player(self.screen, self.player.x + off, self.player.y,
                        PLAYER_W, PLAYER_H, POLITICIANS[self.selected], self.facing, self.walk, self.t)

    def _draw_flag(self, x, gy):
        pygame.draw.rect(self.screen, (210, 210, 220), (x, gy - 130, 6, 130), border_radius=3)
        pygame.draw.circle(self.screen, (255, 210, 54), (x + 3, gy - 132), 7)
        wave = math.sin(self.t * 4) * 4
        for i, col in enumerate((188, 26, 32, 255, 255, 255, 38, 144, 42)):
            pass
        # tricolor (green white red) waving
        cols = [(38, 144, 42), (255, 255, 255), (188, 26, 32)]
        for j, col in enumerate(cols):
            pygame.draw.polygon(self.screen, col, [
                (x + 6, gy - 128 + j * 26), (x + 80 + wave, gy - 128 + j * 26),
                (x + 80 + wave, gy - 102 + j * 26), (x + 6, gy - 102 + j * 26)])

    def _draw_vote(self, cx, cy):
        bob = math.sin(self.t * 5 + cx * 0.05) * 4
        cy += bob
        glow = pygame.Surface((70, 70), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 210, 54, 60), (35, 35), 32)
        self.screen.blit(glow, (cx - 35, cy - 35))
        pygame.draw.circle(self.screen, (255, 196, 40), (int(cx), int(cy)), 18)
        pygame.draw.circle(self.screen, (255, 230, 120), (int(cx), int(cy)), 18, 3)
        pygame.draw.circle(self.screen, (255, 240, 180), (int(cx) - 5, int(cy) - 6), 5)
        # check mark
        pygame.draw.lines(self.screen, (120, 90, 10), False,
                          [(cx - 6, cy), (cx - 1, cy + 5), (cx + 8, cy - 5)], 3)

    def _draw_shield(self, cx, cy):
        bob = math.sin(self.t * 4 + cx * 0.05) * 4
        glow = pygame.Surface((90, 90), pygame.SRCALPHA)
        pygame.draw.circle(glow, (120, 180, 255, 70), (45, 45), 42)
        self.screen.blit(glow, (cx - 45, cy - 45 + bob))
        pygame.draw.circle(self.screen, (90, 150, 230), (int(cx), int(cy + bob)), 20)
        pygame.draw.circle(self.screen, (190, 220, 255), (int(cx), int(cy + bob)), 20, 3)
        pygame.draw.lines(self.screen, (255, 255, 255), False,
                          [(cx - 6, cy + bob), (cx - 1, cy + 5 + bob), (cx + 8, cy - 6 + bob)], 3)

    def _draw_mic(self, cx, bottom):
        # stand
        pygame.draw.rect(self.screen, (40, 42, 54), (cx - 4, bottom - 40, 8, 40), border_radius=3)
        pygame.draw.rect(self.screen, (30, 32, 44), (cx - 18, bottom - 6, 36, 8), border_radius=4)
        # head
        pygame.draw.ellipse(self.screen, (28, 30, 40), (cx - 16, bottom - 72, 32, 36))
        # grille
        for i in range(4):
            pygame.draw.line(self.screen, (90, 96, 120), (cx - 13, bottom - 68 + i * 7), (cx + 13, bottom - 68 + i * 7), 1)
        pygame.draw.ellipse(self.screen, (120, 128, 150), (cx - 16, bottom - 72, 32, 36), 2)
        # red recording light
        pygame.draw.circle(self.screen, (255, 60, 60), (int(cx + 9), int(bottom - 70)), 3)

    def draw_hud(self):
        name = POLITICIANS[self.selected]["name"]
        panel = pygame.Surface((360, 40), pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 24, 40, 170), panel.get_rect(), border_radius=12)
        self.screen.blit(panel, (12, 12))
        line = f"Lv{self.level}   {name}   {self.score} votes   {self.lives} lives"
        if self.shield_time > 0:
            line += f"   SHIELD {int(self.shield_time)}s"
        self.screen.blit(self.font.render(line, True, (255, 255, 255)), (26, 22))

    def _center_text(self, lines, sub=None, starty=180):
        y = starty
        for ln in lines:
            s = self.bigfont.render(ln, True, (255, 255, 255))
            s2 = self.bigfont.render(ln, True, (10, 14, 30))
            self.screen.blit(s2, ((VIEW_W - s2.get_width()) // 2 + 2, y + 2))
            self.screen.blit(s, ((VIEW_W - s.get_width()) // 2, y))
            y += 56
        if sub:
            s = self.font.render(sub, True, (255, 224, 120))
            self.screen.blit(s, ((VIEW_W - s.get_width()) // 2, y + 6))

    def draw_title(self):
        self.screen.fill((18, 22, 44))
        # subtle vertical accent bars
        for i in range(0, VIEW_W, 40):
            pygame.draw.rect(self.screen, lerp((18, 22, 44), (30, 38, 70), (i / VIEW_W)), (i, 0, 40, VIEW_H))
        s = self.bigfont.render("POLITIKUS", True, (255, 255, 255))
        self.screen.blit(self.bigfont.render("POLITIKUS", True, (10, 14, 30)), ((VIEW_W - s.get_width()) // 2 + 3, 70 + 3))
        self.screen.blit(s, ((VIEW_W - s.get_width()) // 2, 70))
        s2 = self.bigfont.render("PLATFORMER", True, (120, 200, 255))
        self.screen.blit(self.bigfont.render("PLATFORMER", True, (10, 14, 30)), ((VIEW_W - s2.get_width()) // 2 + 3, 128 + 3))
        self.screen.blit(s2, ((VIEW_W - s2.get_width()) // 2, 128))
        hint = self.font.render("<- / ->  pick      ENTER  start      (1-6 jump in)", True, (255, 224, 120))
        self.screen.blit(hint, ((VIEW_W - hint.get_width()) // 2, 196))

        n = len(POLITICIANS)
        card_w, card_h = 150, 200
        gap = 18
        total = n * card_w + (n - 1) * gap
        x0 = (VIEW_W - total) // 2
        for i in range(n):
            x = x0 + i * (card_w + gap)
            y = 232
            sel = i == self.selected
            col = (40, 48, 80) if sel else (28, 32, 54)
            card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(card, (*col, 230), card.get_rect(), border_radius=14)
            if sel:
                pygame.draw.rect(card, (120, 200, 255), card.get_rect(), 3, border_radius=14)
            self.screen.blit(card, (x, y))
            draw_player(self.screen, x + card_w / 2 - 26, y + 34, 52, 72, POLITICIANS[i], 1, 0, self.t)
            nm = self.font.render(POLITICIANS[i]["name"], True, (255, 255, 255))
            self.screen.blit(nm, (x + (card_w - nm.get_width()) // 2, y + card_h - 34))

    def draw(self):
        if self.state == "title":
            self.draw_title()
        elif self.state == "play":
            self.draw_world(); self.draw_hud()
        elif self.state == "win":
            self.draw_world()
            ov = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
            ov.fill((10, 14, 30, 150))
            self.screen.blit(ov, (0, 0))
            if self.level < self.MAX_LEVEL:
                self._center_text([f"SZINT {self.level} KESZ!"],
                                  f"{self.score} votes  -  ENTER: next level,  R: restart")
            else:
                self._center_text(["TE GYOZTEL!"],
                                  f"Final: {self.score} votes  -  R: play again")
        elif self.state == "dead":
            self.draw_world()
            ov = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
            ov.fill((60, 10, 10, 150))
            self.screen.blit(ov, (0, 0))
            self._center_text(["JATSZVEGE"], "The press got you.  R: retry,  ENTER: menu")

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                running = self.handle_event(e)
            dt = self.clock.tick(60) / 1000.0
            self.update(dt)
            self.draw()
        pygame.quit()


def random01():
    import random
    return random.random()


def random_angle():
    import random
    import math
    return random.uniform(0, 2 * math.pi)


def main():
    Game().run()


if __name__ == "__main__":
    main()
