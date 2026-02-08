import tkinter as tk
from tkinter import messagebox
import copy

class Game:
    def __init__(self, top):
        self.top = top
        self.top.title("Tic-Tac-Toe")
        self.board = {i: ' ' for i in range(9)}

        self.play = tk.StringVar(value='computer')
        self.symbol = tk.StringVar(value='X')
        self.first = tk.StringVar(value='yes')

        self.turn = 'me'
        self.start = False
        self.buttons = []

        self.buildControls()
        self.buildBoard()


    def display(self, board):
        for i in range(9):
            self.buttons[i]['text'] = board[i]
            self.buttons[i]['state'] = 'normal' if board[i] == ' ' and self.start else 'disabled'


    def startGame(self):
        self.board = {i: ' ' for i in range(9)}
        self.start = True
        self.turn = 'me' if self.first.get() == 'yes' else self.play.get()
        self.display(self.board)

        if self.turn == 'computer' and self.play.get() == 'computer':
            self.computerMove(self.board, self.otherSymbol(), "Computer")


    def playerMove(self, board, player, sym, pos):
        if not self.start:
            return
        if self.play.get() == 'computer' and self.turn != 'me':
            return

        if board[pos] == ' ':
            board[pos] = sym
            self.display(board)
            state = self.result(board, sym, player)

            if state != 'CONTINUE':
                return

            self.changeTurn()

            if self.start and self.play.get() == 'computer' and self.turn == 'computer':
                self.computerMove(board, self.otherSymbol(), "Computer")


    def computerMove(self, board, sym, player):
        if not self.start or self.turn != 'computer':
            return

        move = self.bestMove(board, sym)

        if move is not None:
            board[move] = sym
            self.display(board)
            state = self.result(board, sym, player)
            if state != 'CONTINUE':
                return

            self.changeTurn()

    def bestMove(self, board, sym):
        opponent = self.otherSymbol() if sym == self.symbol.get() else self.symbol.get()

        for i in range(9):
            if board[i] == ' ':
                board[i] = sym
                if self.Winner(board, sym) == 'WIN':
                    board[i] = ' '
                    return i
                board[i] = ' '

        for i in range(9):
            if board[i] == ' ':
                board[i] = opponent
                if self.Winner(board, opponent) == 'WIN':
                    board[i] = ' '
                    return i
                board[i] = ' '

        if board[4] == ' ':
            return 4

        for i in [0, 2, 6, 8]:
            if board[i] == ' ':
                return i

        for i in range(9):
            if board[i] == ' ':
                return i

        return None


    def Winner(self, board, sym):
        wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6]]

        for w in wins:
            if board[w[0]] == board[w[1]] == board[w[2]] == sym:
                return 'WIN'

        if ' ' not in board.values():
            return 'DRAW'

        return 'CONTINUE'

    def result(self, board, sym, player):

        state = self.Winner(board, sym)

        if state != 'CONTINUE':
            self.display(board)

            if state == 'WIN':
                messagebox.showinfo("Game Over", f"{player} WINS!")

            else:
                messagebox.showinfo("Game Over", "It's a DRAW!")

            again = messagebox.askyesno("Play Again?", "Do you want to play again?")
            if again:
                self.board = {i: ' ' for i in range(9)}
                self.display(self.board)
                self.start = False
            else:
                self.top.destroy()

        return state


    def buildBoard(self):
        for i in range(9):
            btn = tk.Button(
                self.top, text=' ', font=("Arial", 20),
                width=5, height=2,
                bg="white", fg="black",
                bd=1, relief="ridge",
                highlightthickness=0,
                command=lambda i=i: self.click(i)
            )

            btn.grid(row=4 + i // 3, column=i % 3, padx=0, pady=0, sticky="nsew")
            self.buttons.append(btn)

        for r in range(4, 7):
            self.top.grid_rowconfigure(r, weight=1)
        for c in range(3):
            self.top.grid_columnconfigure(c, weight=1)

    def click(self, pos):
        sym = self.symbol.get() if self.turn == 'me' else self.otherSymbol()
        player = 'YOU' if self.turn == 'me' else 'PLAYER 2'
        self.playerMove(self.board, player, sym, pos)

    def otherSymbol(self):
        return 'O' if self.symbol.get() == 'X' else 'X'

    def changeTurn(self):
        if self.play.get() == 'player':
            self.turn = 'player2' if self.turn == 'me' else 'me'
        else:
            self.turn = 'me' if self.turn == 'computer' else 'computer'

    def buildControls(self):
        frame = tk.Frame(self.top, padx=10, pady=10)
        frame.grid(row=0, column=0, columnspan=3, sticky="w")

        tk.Label(frame, text="Play With").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(frame, text="Computer", variable=self.play, value='computer').grid(row=0, column=1)
        tk.Radiobutton(frame, text="Player 2", variable=self.play, value='player').grid(row=0, column=2)

        tk.Label(frame, text="Select").grid(row=1, column=0, sticky="w")
        tk.Radiobutton(frame, text="X", variable=self.symbol, value='X').grid(row=1, column=1)
        tk.Radiobutton(frame, text="O", variable=self.symbol, value='O').grid(row=1, column=2)

        tk.Label(frame, text="Start the game").grid(row=2, column=0, sticky="w")
        tk.Radiobutton(frame, text="Yes", variable=self.first, value='yes').grid(row=2, column=1)
        tk.Radiobutton(frame, text="No", variable=self.first, value='no').grid(row=2, column=2)

        tk.Button(frame, text="start", width=8, command=self.startGame).grid(row=3, column=2, pady=(10, 0), sticky="e")


def main():
    top = tk.Tk()
    game = Game(top)
    top.mainloop()

main()
