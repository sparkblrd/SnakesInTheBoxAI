from random import randrange
import pygame
import ctypes
import ctypes.wintypes

CELL_NUMBER = 16
SIZE = 10
RES = CELL_NUMBER * SIZE

GRID_COLOR = (15, 15, 15)


class SnakeGameAI:
    def __init__(self):
        pygame.init()

        self.sc = pygame.display.set_mode([RES, RES], pygame.NOFRAME)
        pygame.display.set_caption("Snake AI")

        self.clock = pygame.time.Clock()

        self.hwnd = pygame.display.get_wm_info()["window"]

        self.reset()

    def spawnApple(self, snake):
        while True:
            apple = (
                randrange(0, RES, SIZE),
                randrange(0, RES, SIZE)
            )
            if apple not in snake:
                return apple

    def reset(self):
        self.direction = "RIGHT"

        self.x = randrange(0, RES, SIZE)
        self.y = randrange(0, RES, SIZE)

        self.snake = [(self.x, self.y)]
        self.length = 1
        self.score = 0

        self.apple = self.spawnApple(self.snake)
        self.game_over = False

        self.frame_iteration = 0

    def handle_window_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    try:
                        WM_NCLBUTTONDOWN = 0x00A1
                        HTCAPTION = 2

                        ctypes.windll.user32.ReleaseCapture()
                        ctypes.windll.user32.SendMessageW(
                            self.hwnd,
                            WM_NCLBUTTONDOWN,
                            HTCAPTION,
                            0
                        )
                    except Exception as e:
                        print("Drag error:", e)

    def playStep(self, action):
        self.handle_window_events()

        self.frame_iteration += 1

        reward = 0

        old_distance = abs(self.x - self.apple[0]) + abs(self.y - self.apple[1])

        self.move(action)

        new_distance = abs(self.x - self.apple[0]) + abs(self.y - self.apple[1])

        if new_distance < old_distance:
            reward = 1
        else:
            reward = -1

        self.snake.append((self.x, self.y))
        self.snake = self.snake[-self.length:]

        if self.isCollision() or self.frame_iteration > 100 * self.length:
            reward = -10
            self.game_over = True
            return reward, self.game_over, self.score

        if self.snake[-1] == self.apple:
            self.length += 1
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self.apple = self.spawnApple(self.snake)

        self.draw()
        self.clock.tick(20)

        return reward, self.game_over, self.score

    def move(self, action):
        directions = ["RIGHT", "DOWN", "LEFT", "UP"]
        index = directions.index(self.direction)

        if action == [1, 0, 0]:
            new_direction = directions[index]
        elif action == [0, 1, 0]:
            new_direction = directions[(index + 1) % 4]
        else:
            new_direction = directions[(index - 1) % 4]

        self.direction = new_direction

        if self.direction == "RIGHT":
            self.x += SIZE
        elif self.direction == "LEFT":
            self.x -= SIZE
        elif self.direction == "DOWN":
            self.y += SIZE
        elif self.direction == "UP":
            self.y -= SIZE

    def isCollision(self, point=None):
        if point is None:
            point = (self.x, self.y)

        x, y = point

        if x < 0 or x > RES - SIZE:
            return True
        if y < 0 or y > RES - SIZE:
            return True
        if point in self.snake[:-1]:
            return True

        return False

    def draw_grid(self):
        for x in range(0, RES, SIZE):
            pygame.draw.line(self.sc, GRID_COLOR, (x, 0), (x, RES), 1)

        for y in range(0, RES, SIZE):
            pygame.draw.line(self.sc, GRID_COLOR, (0, y), (RES, y), 1)

    def draw(self):
        self.sc.fill(pygame.Color("black"))

        self.draw_grid()

        for i, j in self.snake:
            pygame.draw.rect(
                self.sc,
                pygame.Color("green"),
                (i + 1, j + 1, SIZE - 2, SIZE - 2)
            )

        pygame.draw.rect(
            self.sc,
            pygame.Color("red"),
            (self.apple[0] + 1, self.apple[1] + 1, SIZE - 2, SIZE - 2)
        )

        pygame.display.flip()