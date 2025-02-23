import cv2
import numpy as np
import random


class Minesweeper:
    def __init__(self, width=10, height=10, mines=10, cell_size=40):
        self.width = width
        self.height = height
        self.mines = mines
        self.cell_size = cell_size
        self.board = [[' ' for _ in range(width)] for _ in range(height)]
        self.revealed = [[False for _ in range(width)] for _ in range(height)]
        self.mine_locations = set()
        self.game_over = False
        self.setup_mines()
        self.image = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
        self.window_name = 'Minesweeper'
        print("Mine locations:", sorted(list(self.mine_locations)))

    def setup_mines(self):
        mines_placed = 0
        while mines_placed < self.mines:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if (x, y) not in self.mine_locations:
                self.mine_locations.add((x, y))
                mines_placed += 1

    def count_adjacent_mines(self, x, y):
        count = 0
        neighbors_with_mines = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and
                        0 <= new_y < self.height):
                    if (new_x, new_y) in self.mine_locations:
                        count += 1
                        neighbors_with_mines.append((new_x, new_y))
        print(f"Cell ({x}, {y}) has {count} adjacent mines at: {neighbors_with_mines}")
        return count

    def reveal(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height) or self.revealed[y][x]:
            return

        self.revealed[y][x] = True

        if (x, y) in self.mine_locations:
            self.game_over = True
            print(f"Hit mine at ({x}, {y})")
            return

        mines = self.count_adjacent_mines(x, y)
        if mines > 0:
            self.board[y][x] = str(mines)
        else:
            self.board[y][x] = '0'
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    self.reveal(x + dx, y + dy)

    def draw_board(self):
        self.image.fill(0)
        for y in range(self.height):
            for x in range(self.width):
                top_left = (x * self.cell_size, y * self.cell_size)
                bottom_right = ((x + 1) * self.cell_size, (y + 1) * self.cell_size)
                if self.revealed[y][x]:
                    color = (200, 200, 200)  # Светло-серый для раскрытых
                else:
                    color = (100, 100, 100)  # Тёмно-серый для закрытых
                cv2.rectangle(self.image, top_left, bottom_right, color, -1)

                if self.revealed[y][x]:
                    # Мины показываем только после проигрыша
                    if (x, y) in self.mine_locations and self.game_over:
                        center = (x * self.cell_size + self.cell_size // 2,
                                  y * self.cell_size + self.cell_size // 2)
                        cv2.circle(self.image, center, self.cell_size // 3, (0, 0, 255), -1)
                    elif self.board[y][x] != '0' and self.board[y][x] != ' ':
                        text_pos = (x * self.cell_size + self.cell_size // 3,
                                    y * self.cell_size + self.cell_size * 2 // 3)
                        cv2.putText(self.image, self.board[y][x], text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(self.image, top_left, bottom_right, (0, 0, 0), 1)

        status = "Game Over!" if self.game_over else "Playing..."
        cv2.putText(self.image, status, (10, self.height * self.cell_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.game_over:
            grid_x = x // self.cell_size
            grid_y = y // self.cell_size
            self.reveal(grid_x, grid_y)

    def play(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        while True:
            self.draw_board()
            cv2.imshow(self.window_name, self.image)
            unrevealed = sum(row.count(False) for row in self.revealed)
            if unrevealed == self.mines and not self.game_over:
                self.game_over = True
                print("You won!")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def main():
    print("Welcome to Minesweeper! Click cells to play, press 'q' to quit.")
    game = Minesweeper(10, 10, 10, 40)
    game.play()


if __name__ == "__main__":
    main()