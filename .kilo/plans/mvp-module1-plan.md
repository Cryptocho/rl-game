# mvp.ipynb 下一步方案：完成模块 1「最小游戏环境」

## 总体目标

将当前单人移动 demo 升级为**完整的 1v1 Gym 风格环境**，包含规则 Bot、战斗系统、标准化观测/动作接口，使其可以直接对接模块 2 的 PPO 训练管线。

---

## 新增 Cell 结构

以下每个 `##` 为一个独立的 Jupyter cell（markdown 或 code），按顺序追加到现有 mvp.ipynb 之后。

---

### Cell 1: markdown — 规则 Bot 设计说明

```
## 规则 Bot 对手

Bot 继承 Player 的物理属性，使用脚本化策略：

1. **追逐**：每帧计算 Bot 到目标的距离，若 > 攻击距离，则向目标方向移动
2. **攻击**：距离 <= 攻击距离时自动攻击（有冷却）
3. **决策频率**：每帧都重新计算方向（模拟实时反应）

Bot 的属性：
- hp, max_hp
- attack_range（攻击判定距离）
- attack_cooldown（攻击间隔，帧数）
- attack_damage（单次伤害）
- attack_cooldown_remaining（当前冷却剩余帧数）
```

### Cell 2: code — 实现 Bot 数据类与行为逻辑

```python
@dataclass
class Bot:
    pos: np.ndarray          # [x, y]
    vel: np.ndarray          # [vx, vy]
    size: int = 20
    color: tuple = (255, 80, 80)  # 红色
    accel: float = 0.6
    friction: float = 0.90
    max_speed: float = 5.0

    hp: float = 100.0
    max_hp: float = 100.0
    attack_range: float = 30.0     # 攻击判定距离（含双方半径）
    attack_damage: float = 8.0
    attack_cooldown: int = 30      # 30 帧 = 0.5s @60fps
    attack_cooldown_remaining: int = 0

    def get_action(self, target_pos: np.ndarray) -> np.ndarray:
        """返回加速度方向向量（归一化），用于 physics_update"""
        direction = target_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        return direction / dist * self.accel

    def should_attack(self, target_pos: np.ndarray) -> bool:
        """距离 <= attack_range 且冷却完毕时返回 True"""
        dist = np.linalg.norm(target_pos - self.pos)
        return dist <= self.attack_range and self.attack_cooldown_remaining == 0

    def step_cooldown(self):
        """每帧调用，递减冷却"""
        if self.attack_cooldown_remaining > 0:
            self.attack_cooldown_remaining -= 1

    def take_damage(self, dmg: float):
        self.hp = max(0.0, self.hp - dmg)

    def is_dead(self) -> bool:
        return self.hp <= 0.0

    @property
    def rect(self) -> pygame.Rect:
        screen_x = int(FIELD_OFFSET[0] + self.pos[0] - self.size / 2)
        screen_y = int(FIELD_OFFSET[1] + self.pos[1] - self.size / 2)
        return pygame.Rect(screen_x, screen_y, self.size, self.size)

    @property
    def center(self) -> np.ndarray:
        return self.pos.copy()
```

### Cell 3: markdown — Bot 物理更新复用说明

```
## Bot 与 Player 共用物理更新

已有的 `physics_update()` 函数直接用于 Bot，但当前 `physics_update` 的加速度来源是 `get_accel_from_keys(keys)`，Bot 无法使用。

解决方案：将 `physics_update` 改造为接受 `accel` 数组参数，人类玩家和 Bot 统一调用。
```

### Cell 4: code — 重构 physics_update，分离加速度计算与物理更新

```python
def physics_update_entity(
    pos: np.ndarray, vel: np.ndarray, accel: np.ndarray,
    friction: float, max_speed: float, size: int,
    dt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """纯物理更新，返回 (new_pos, new_vel)"""
    vel = vel + accel * dt
    vel *= friction

    speed = np.linalg.norm(vel)
    if speed > max_speed:
        vel = vel / speed * max_speed

    pos = pos + vel * dt

    half = size / 2
    for axis in (0, 1):
        limit = FIELD_W if axis == 0 else FIELD_H
        if pos[axis] - half < 0:
            pos[axis] = half
            vel[axis] = abs(vel[axis]) * 0.5
        elif pos[axis] + half > limit:
            pos[axis] = limit - half
            vel[axis] = -abs(vel[axis]) * 0.5

    return pos, vel


def player_accel_from_keys(keys, accel_magnitude: float = 0.8) -> np.ndarray:
    """人类玩家按键 -> 加速度向量（已有函数改名保留）"""
    accel = np.array([0.0, 0.0], dtype=np.float32)
    for key, direction in KEY_MAP.items():
        if keys[key]:
            accel += direction
    norm = np.linalg.norm(accel)
    if norm > 1e-6:
        accel = accel / norm * accel_magnitude
    return accel


def action_to_accel(move_idx: int, accel_magnitude: float = 0.8) -> np.ndarray:
    """RL 动作的移动索引 -> 加速度向量

    move_idx: 0~8, 对应 MOVE_ACTIONS = ["STOP","W","A","S","D","WA","WD","SA","SD"]
    """
    MOVE_VECTORS = {
        0: (0.0, 0.0),   1: (0.0, -1.0),  2: (-1.0, 0.0),
        3: (0.0, 1.0),   4: (1.0, 0.0),   5: (1.0, -1.0),
        6: (1.0, 1.0),   7: (-1.0, 1.0),  8: (-1.0, -1.0),
    }
    dx, dy = MOVE_VECTORS[move_idx]
    v = np.array([dx, dy], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 1e-6:
        v = v / norm * accel_magnitude
    return v
```

### Cell 5: markdown — 战斗系统设计

```
## 战斗系统

### 攻击判定

每帧 step 中：
1. 玩家攻击：若 skill 动作指示释放且距离 <= attack_range 且冷却完毕 -> 对 Bot 造成伤害，重置冷却
2. Bot 攻击：`should_attack()` 返回 True -> 对玩家造成伤害，设置冷却
3. 双方独立维护冷却计数器

### 死亡 & 超时

- 任一方 `hp <= 0` -> `done = True`，存活方获胜
- `step_count >= max_steps` -> `done = True`，按剩余血量判定胜负
- 若双方同时死亡 -> 平局

### 速度惩罚

攻击方在攻击瞬间施加短暂速度衰减（50%），模拟"出招硬直"，防止无限追击连砍的无交互打法。
```

### Cell 6: code — 战斗判定函数

```python
def try_attack(
    attacker_pos: np.ndarray, defender: 'Bot | Player',
    attack_range: float, attack_damage: float,
    cooldown_remaining: int, cooldown_max: int,
    attacker_vel: np.ndarray, speed_penalty: float = 0.5,
) -> tuple[bool, int, float, np.ndarray]:
    """
    尝试攻击。
    返回: (attack_triggered, new_cooldown, actual_damage, new_attacker_vel)
    """
    if cooldown_remaining > 0:
        return False, cooldown_remaining, 0.0, attacker_vel

    dist = np.linalg.norm(defender.center - attacker_pos)
    if dist > attack_range:
        return False, cooldown_remaining, 0.0, attacker_vel

    # 攻击命中
    defender.hp = max(0.0, defender.hp - attack_damage)
    return True, cooldown_max, attack_damage, attacker_vel * speed_penalty
```

### Cell 7: markdown — Gym 环境接口设计

```
## Gym 风格环境接口

### reset()
- 随机化双方出生位置（彼此距离 >= 200px）
- 重置 hp、速度、冷却计数器
- 返回 `obs`（观测向量，24维）

### step(action)
- `action`: `(move_idx, skill0_bool, skill1_bool)` — 来自 RL 模型
- 执行一帧物理更新（玩家 + Bot 分别 update）
- 处理攻击判定、伤害、死亡
- 返回 `(obs, reward, done, info)`

### render()
- 复用现有 pygame 绘制逻辑，新增 Bot 方块(红色)、hp 条、攻击特效

### close()
- pygame.quit()
```

### Cell 8: code — 观测向量构建函数

```python
OBS_DIM = 24  # 最终维度

def build_obs(player: Player, bot: Bot, step_count: int, max_steps: int) -> np.ndarray:
    """
    构建结构化观测向量。

    字段布局（索引从 0 开始）：
    自身状态 (8 维):
      [0:2]   player.pos / FIELD 归一化
      [2:4]   player.vel / max_speed 归一化
      [4]     player.hp / player.max_hp
      [5:7]   player 到自身场地边界的最近距离 (上/下边界,左/右边界) / FIELD 归一化
      [7]     player.attack_cooldown_remaining / player.attack_cooldown

    敌方相对状态 (8 维):
      [8:10]  (bot.pos - player.pos) / FIELD 归一化
      [10:12] (bot.vel - player.vel) / max_speed 归一化
      [12]    bot.hp / bot.max_hp
      [13:15] bot 到自身场地边界的最近距离 归一化
      [15]    bot.attack_cooldown_remaining / bot.attack_cooldown

    距离特征 (2 维):
      [16]    player 与 bot 的距离 / 场地对角线
      [17]    (player 与 bot 的距离 - attack_range) / 场地对角线

    时间特征 (1 维):
      [18]    step_count / max_steps

    剩余 5 维预留 (填 0)
      [19:24] 0
    """
    # 归一化常量
    field_scale     = np.array([FIELD_W, FIELD_H], dtype=np.float32)
    field_diag      = np.sqrt(FIELD_W**2 + FIELD_H**2)
    max_speed_scale = np.array([max(PLAYER_MAX_SPEED, BOT_MAX_SPEED)] * 2, dtype=np.float32)

    # 自身状态
    self_pos_norm       = player.pos / field_scale
    self_vel_norm       = player.vel / max_speed_scale
    self_hp_norm        = np.array([player.hp / player.max_hp], dtype=np.float32)
    self_boundary       = np.array([
        min(player.pos[1], FIELD_H - player.pos[1]) / FIELD_H,
        min(player.pos[0], FIELD_W - player.pos[0]) / FIELD_W,
    ], dtype=np.float32)
    self_cooldown       = np.array([player.attack_cooldown_remaining / max(player.attack_cooldown, 1)],
                                   dtype=np.float32)

    # 敌方相对状态
    enemy_rel_pos_norm  = (bot.pos - player.pos) / field_scale
    enemy_rel_vel_norm  = (bot.vel - player.vel) / max_speed_scale
    enemy_hp_norm       = np.array([bot.hp / bot.max_hp], dtype=np.float32)
    enemy_boundary      = np.array([
        min(bot.pos[1], FIELD_H - bot.pos[1]) / FIELD_H,
        min(bot.pos[0], FIELD_W - bot.pos[0]) / FIELD_W,
    ], dtype=np.float32)
    enemy_cooldown      = np.array([bot.attack_cooldown_remaining / max(bot.attack_cooldown, 1)],
                                   dtype=np.float32)

    # 距离特征
    dist = np.linalg.norm(player.pos - bot.pos)
    dist_norm   = np.array([dist / field_diag], dtype=np.float32)
    dist_range  = np.array([(dist - ATTACK_RANGE) / field_diag], dtype=np.float32)

    # 时间特征
    time_norm = np.array([step_count / max_steps], dtype=np.float32)

    # 拼接
    obs = np.concatenate([
        self_pos_norm, self_vel_norm, self_hp_norm, self_boundary, self_cooldown,
        enemy_rel_pos_norm, enemy_rel_vel_norm, enemy_hp_norm, enemy_boundary, enemy_cooldown,
        dist_norm, dist_range,
        time_norm,
        np.zeros(5, dtype=np.float32),  # 预留
    ])
    return obs.astype(np.float32)
```

### Cell 9: code — ArenaEnv 环境类

```python
# 全局常量
MOVE_ACTIONS      = ["STOP", "W", "A", "S", "D", "WA", "WD", "SA", "SD"]
N_SKILLS          = 2
ATTACK_RANGE      = 45.0   # 攻击判定距离 = 双方 size/2 各 20 + 5px 余量
ATTACK_DAMAGE     = 10.0
ATTACK_COOLDOWN   = 30     # 帧
PLAYER_MAX_SPEED  = 6.0
BOT_MAX_SPEED     = 5.0
MAX_STEPS         = 3600   # 60fps * 60s = 1 分钟时间上限


class ArenaEnv:
    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.step_count = 0

        if render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("RL Arena - 1v1")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 14)

    def reset(self) -> np.ndarray:
        self.step_count = 0

        # 随机出生：距离 >= 200px
        while True:
            p_pos = np.array([
                np.random.uniform(50, FIELD_W - 50),
                np.random.uniform(50, FIELD_H - 50),
            ], dtype=np.float32)
            b_pos = np.array([
                np.random.uniform(50, FIELD_W - 50),
                np.random.uniform(50, FIELD_H - 50),
            ], dtype=np.float32)
            if np.linalg.norm(p_pos - b_pos) >= 200.0:
                break

        self.player = Player(
            pos=p_pos,
            vel=np.array([0.0, 0.0], dtype=np.float32),
            size=20, color=COLOR_PLAYER,
            accel=0.8, friction=0.90, max_speed=PLAYER_MAX_SPEED,
            hp=100.0, max_hp=100.0,
            attack_range=ATTACK_RANGE, attack_damage=ATTACK_DAMAGE,
            attack_cooldown=ATTACK_COOLDOWN, attack_cooldown_remaining=0,
        )
        self.bot = Bot(
            pos=b_pos,
            vel=np.array([0.0, 0.0], dtype=np.float32),
            size=20, color=(255, 80, 80),
            accel=0.6, friction=0.90, max_speed=BOT_MAX_SPEED,
            hp=100.0, max_hp=100.0,
            attack_range=ATTACK_RANGE, attack_damage=ATTACK_DAMAGE,
            attack_cooldown=ATTACK_COOLDOWN, attack_cooldown_remaining=0,
        )

        return build_obs(self.player, self.bot, self.step_count, MAX_STEPS)

    def step(self, action: tuple[int, int, int]) -> tuple[np.ndarray, float, bool, dict]:
        """
        action: (move_idx, skill0_bool, skill1_bool)
        返回: (obs, reward, done, info)
        """
        move_idx, skill0, skill1 = action
        self.step_count += 1

        # === 玩家物理更新 ===
        player_accel = action_to_accel(move_idx, self.player.accel)
        self.player.pos, self.player.vel = physics_update_entity(
            self.player.pos, self.player.vel, player_accel,
            self.player.friction, self.player.max_speed, self.player.size,
        )

        # === Bot 物理更新 ===
        bot_accel = self.bot.get_action(self.player.pos)
        self.bot.pos, self.bot.vel = physics_update_entity(
            self.bot.pos, self.bot.vel, bot_accel,
            self.bot.friction, self.bot.max_speed, self.bot.size,
        )

        # === 攻击处理 ===
        info = {"player_attack": False, "bot_attack": False,
                "player_hp": self.player.hp, "bot_hp": self.bot.hp}

        # 玩家攻击 Bot
        player_attacked = skill0 or skill1  # 任意技能键触发攻击
        if player_attacked:
            attacked, new_cd, dmg, new_vel = try_attack(
                self.player.pos, self.bot,
                self.player.attack_range, self.player.attack_damage,
                self.player.attack_cooldown_remaining, self.player.attack_cooldown,
                self.player.vel,
            )
            if attacked:
                self.player.attack_cooldown_remaining = new_cd
                self.player.vel = new_vel
                info["player_attack"] = True
                info["damage_dealt"] = info.get("damage_dealt", 0.0) + dmg

        # Bot 攻击玩家
        if self.bot.should_attack(self.player.pos):
            attacked, new_cd, dmg, new_vel = try_attack(
                self.bot.pos, self.player,
                self.bot.attack_range, self.bot.attack_damage,
                self.bot.attack_cooldown_remaining, self.bot.attack_cooldown,
                self.bot.vel,
            )
            if attacked:
                self.bot.attack_cooldown_remaining = new_cd
                self.bot.vel = new_vel
                info["bot_attack"] = True
                info["damage_taken"] = info.get("damage_taken", 0.0) + dmg

        # === 冷却递减 ===
        if self.player.attack_cooldown_remaining > 0:
            self.player.attack_cooldown_remaining -= 1
        if self.bot.attack_cooldown_remaining > 0:
            self.bot.attack_cooldown_remaining -= 1

        # === 奖励计算 ===
        reward = compute_reward(info, self.step_count, self.player, self.bot)

        # === 终止判定 ===
        done = False
        if self.player.is_dead():
            done = True
            info["winner"] = "bot"
        elif self.bot.is_dead():
            done = True
            info["winner"] = "player"
        elif self.step_count >= MAX_STEPS:
            done = True
            info["winner"] = "player" if self.player.hp > self.bot.hp else \
                             "bot" if self.bot.hp > self.player.hp else "draw"

        obs = build_obs(self.player, self.bot, self.step_count, MAX_STEPS)
        return obs, reward, done, info

    def render(self):
        if self.render_mode != "human" or self.screen is None:
            return

        self.screen.fill(COLOR_BG)

        # 网格
        for x in range(0, FIELD_W + 1, 50):
            sx = FIELD_OFFSET[0] + x
            pygame.draw.line(self.screen, COLOR_GRID, (sx, FIELD_OFFSET[1]),
                             (sx, FIELD_OFFSET[1] + FIELD_H), 1)
        for y in range(0, FIELD_H + 1, 50):
            sy = FIELD_OFFSET[1] + y
            pygame.draw.line(self.screen, COLOR_GRID, (FIELD_OFFSET[0], sy),
                             (FIELD_OFFSET[0] + FIELD_W, sy), 1)

        # 场地边框
        border_rect = pygame.Rect(FIELD_OFFSET[0], FIELD_OFFSET[1], FIELD_W, FIELD_H)
        pygame.draw.rect(self.screen, COLOR_BORDER, border_rect, 2)

        # 玩家
        pygame.draw.rect(self.screen, self.player.color, self.player.rect)

        # Bot
        pygame.draw.rect(self.screen, self.bot.color, self.bot.rect)

        # 速度指示线
        for entity, line_color in [(self.player, (255, 255, 100)), (self.bot, (255, 255, 100))]:
            cx = int(FIELD_OFFSET[0] + entity.pos[0])
            cy = int(FIELD_OFFSET[1] + entity.pos[1])
            vx = int(entity.vel[0] * 10)
            vy = int(entity.vel[1] * 10)
            pygame.draw.line(self.screen, line_color, (cx, cy), (cx+vx, cy+vy), 2)

        # HP 条
        for entity, is_player in [(self.player, True), (self.bot, False)]:
            bar_w = 40
            bar_h = 6
            bar_x = int(FIELD_OFFSET[0] + entity.pos[0] - bar_w / 2)
            bar_y = int(FIELD_OFFSET[1] + entity.pos[1] - entity.size / 2 - 12)
            hp_ratio = entity.hp / entity.max_hp
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (bar_x, bar_y, bar_w, bar_h))
            hp_color = (80, 220, 80) if is_player else (220, 80, 80)
            pygame.draw.rect(self.screen, hp_color,
                             (bar_x, bar_y, int(bar_w * hp_ratio), bar_h))

        # HUD
        hud = [
            f"Step: {self.step_count}/{MAX_STEPS}",
            f"Player HP: {self.player.hp:.0f}  Bot HP: {self.bot.hp:.0f}",
            f"Player CD: {self.player.attack_cooldown_remaining}",
        ]
        for i, line in enumerate(hud):
            text = self.font.render(line, True, COLOR_HUD)
            self.screen.blit(text, (10, 10 + i * 18))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
```

### Cell 10: markdown — 奖励配置系统

```
## 奖励配置系统

奖励由外部权重向量驱动。环境在 `info` 字典中提供原始事件，`compute_reward()` 根据权重配置计算标量奖励。

### 基础奖励分项

| 分项              | 事件来源         | 默认含义           |
|-------------------|------------------|--------------------|
| `damage_dealt`    | info["damage_dealt"]   | 对 Bot 造成的伤害  |
| `damage_taken`    | info["damage_taken"]   | 自身受到的伤害    |
| `survival_bonus`  | 每帧固定值              | 存活奖励          |
| `distance_penalty`| 玩家与 Bot 的距离      | 距离惩罚/奖励     |
| `kill_bonus`      | done 时 Bot 死亡        | 击杀奖励          |

### 权重配置

通过字典传入，支持正/负/零权重：

```python
reward_weights = {
    "damage_dealt":     2.0,   # 激进型: 高权重
    "damage_taken":    -0.1,   # 激进型: 低惩罚
    "survival_bonus":   0.0,   # 激进型: 不奖励存活
    "distance_penalty": 0.0,
    "kill_bonus":      10.0,
}
```
```

### Cell 11: code — compute_reward 函数

```python
def compute_reward(
    info: dict, step_count: int,
    player: Player, bot: Bot,
    weights: dict | None = None,
) -> float:
    """根据权重配置计算标量奖励。"""
    if weights is None:
        weights = {
            "damage_dealt":     1.0,
            "damage_taken":    -1.0,
            "survival_bonus":   0.01,
            "distance_penalty": 0.0,
            "kill_bonus":       5.0,
        }

    reward = 0.0

    # 造成伤害
    reward += info.get("damage_dealt", 0.0) * weights.get("damage_dealt", 0.0)

    # 受到伤害
    reward += info.get("damage_taken", 0.0) * weights.get("damage_taken", 0.0)

    # 存活奖励（每帧）
    if "survival_bonus" in weights:
        reward += weights["survival_bonus"]

    # 距离惩罚/奖励
    if "distance_penalty" in weights and abs(weights["distance_penalty"]) > 1e-8:
        dist = np.linalg.norm(player.pos - bot.pos)
        field_diag = np.sqrt(FIELD_W**2 + FIELD_H**2)
        dist_norm = dist / field_diag
        reward += weights["distance_penalty"] * dist_norm

    # 击杀奖励
    reward += info.get("kill_bonus", 0.0) * weights.get("kill_bonus", 0.0)

    return reward
```

### Cell 12: markdown — 需要修改的已有代码

```
## 需要修改的已有 Cell

Player 原来只有 pos/vel/size/color/accel/friction/max_speed，现在需要增加：
- hp, max_hp
- attack_range, attack_damage, attack_cooldown, attack_cooldown_remaining
- is_dead() 方法

建议：在原 Player dataclass cell 后面追加一个新 cell 覆盖/扩展 Player 定义。
```

### Cell 13: code — 扩展 Player dataclass

```python
@dataclass
class Player:
    pos: np.ndarray
    vel: np.ndarray
    size: int      = 20
    color: tuple   = COLOR_PLAYER
    accel: float   = 0.8
    friction: float = 0.90
    max_speed: float = 6.0

    hp: float = 100.0
    max_hp: float = 100.0
    attack_range: float = 45.0
    attack_damage: float = 10.0
    attack_cooldown: int = 30
    attack_cooldown_remaining: int = 0

    @property
    def rect(self) -> pygame.Rect:
        screen_x = int(FIELD_OFFSET[0] + self.pos[0] - self.size / 2)
        screen_y = int(FIELD_OFFSET[1] + self.pos[1] - self.size / 2)
        return pygame.Rect(screen_x, screen_y, self.size, self.size)

    @property
    def center(self) -> np.ndarray:
        return self.pos.copy()

    def is_dead(self) -> bool:
        return self.hp <= 0.0

    def take_damage(self, dmg: float):
        self.hp = max(0.0, self.hp - dmg)
```

### Cell 14: markdown — 环境测试

```
## 环境集成测试

用随机动作跑 100 局，验证：
1. reset/step/render 不崩溃
2. done 信号正确触发（死亡/超时）
3. 观测向量维度正确
4. info 字典包含必要字段
```

### Cell 15: code — 集成测试

```python
def test_env_random(render: bool = False, n_episodes: int = 5):
    env = ArenaEnv(render_mode="human" if render else None)

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            move_idx = np.random.randint(0, 9)
            skill0 = np.random.randint(0, 2)
            skill1 = np.random.randint(0, 2)
            obs, reward, done, info = env.step((move_idx, skill0, skill1))
            total_reward += reward
            steps += 1

            if render:
                env.render()

        print(f"Episode {ep+1}: steps={steps}, reward={total_reward:.2f}, "
              f"winner={info.get('winner','?')}, "
              f"player_hp={info.get('player_hp',100):.0f}, "
              f"bot_hp={info.get('bot_hp',100):.0f}")

    env.close()
    print("All tests passed.")

# 无渲染快速测试
test_env_random(render=False, n_episodes=10)

# 有渲染观察一局
# test_env_random(render=True, n_episodes=1)
```

### Cell 16: markdown — 模块 1 完成检查清单

```
## 模块 1 完成检查清单

- [x] 单人 WASD 移动物理（已有）
- [ ] 规则 Bot 追逐+攻击逻辑
- [ ] physics_update 重构为 entity-agnostic
- [ ] Player 扩展战斗属性（hp, attack_*）
- [ ] 战斗攻击判定 + 冷却 + 伤害
- [ ] 死亡/超时终止逻辑
- [ ] 观测向量 build_obs (24 维，含预留)
- [ ] ArenaEnv.reset() / step() / render() / close()
- [ ] 奖励计算 compute_reward（权重驱动）
- [ ] 集成测试通过
- [ ] 可视化渲染正常（含 HP 条）
```

---

## 实施顺序

```
Cell 13: 扩展 Player dataclass        <- 最先，因为 Bot 和 ArenaEnv 都依赖
Cell 2:  实现 Bot 数据类               <- 依赖扩展后的 Player 结构
Cell 4:  重构 physics_update           <- 独立，不依赖上述
Cell 6:  战斗判定 try_attack            <- 独立
Cell 8:  观测向量 build_obs             <- 依赖 Player/Bot 结构
Cell 11: 奖励计算 compute_reward       <- 独立
Cell 9:  ArenaEnv 环境类               <- 整合以上所有
Cell 15: 集成测试                      <- 最后
```
