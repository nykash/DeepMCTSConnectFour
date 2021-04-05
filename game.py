import numpy as np

class ConnectFour(object):
    def __init__(self, rows=6, cols=7):
        self.board = np.array([[0 for i in range(rows)] for j in range(cols)])
        self.cols = cols
        self.rows = rows

        self.alphabet = {1:"r", -1:"y", 0:"0"}
        self.turn = 1

    def get_legal_moves(self, color):
        return [i for i in range(self.cols) if self.board[i][0] == 0]

    def print_(self, let=True):
        for i in range(self.rows):
            for j in range(self.cols):
                if let:
                    print(self.alphabet[self.board[j][i]], end=" ")
                else:
                    print(self.board[j][i], end=" ")
            print()

    def get_terminal(self, color):
        if(np.all(self.board != 0)):
            return 0.5

        if(self.did_win(color)):
            return 1

        return None

    def did_win(self, c):
        for i in range(self.cols):
            for j in range(self.rows):
                if(self.board[i][j] != c):
                    continue

                down = j + 3 < self.rows
                up = j - 3 >= 0
                right = i + 3 < self.cols
                left = i - 3 >= 0

                if(down and self.board[i][j+1] == c and self.board[i][j+2] == c and self.board[i][j+3] == c):
                    return True

                if(up and self.board[i][j-1] == c and self.board[i][j-2] == c and self.board[i][j-3] == c):
                    return True

                if (right and self.board[i+1][j] == c and self.board[i+2][j] == c and self.board[i+3][j] == c):
                    return True

                if (left and self.board[i-1][j] == c and self.board[i-2][j] == c and self.board[i-3][j] == c):
                    return True

                if (down and left and self.board[i - 1][j + 1] == c and self.board[i - 2][j + 2] == c and
                        self.board[i - 3][j + 3] == c):
                    return True

                if(down and right and self.board[i+1][j+1] == c and self.board[i+2][j+2] == c and self.board[i+3][j+3] == c):
                    return True

                if (up and left and self.board[i - 1][j - 1] == c and self.board[i - 2][j - 2] == c and
                        self.board[i - 3][j - 3] == c):
                    return True

                if (up and right and self.board[i + 1][j - 1] == c and self.board[i + 2][j - 2] == c and
                        self.board[i + 3][j - 3] == c):
                    return True
        return False


    def move(self, move_col, color):
        for i in range(self.rows-1, -1, -1):
            if(self.board[move_col][i] == 0):
                break

        self.board[move_col][i] = color

        return True

    def new_game_with_move(self, move, color):
        b = ConnectFour()
        b.board = self.board.copy()

        b.make_move(move, color)

        return b

    def preprocess(self, who_turn):
        return (self.board * who_turn).reshape((7, 6, 1))

    def clone(self):
        game = ConnectFour()
        board = self.board.copy()
        game.board = board

        return game
