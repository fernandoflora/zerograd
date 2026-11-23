import os
import tkinter as tk
import numpy as np

from src.activations import relu, relu_prime
from src.layers import Activation, Dense, Softmax
from src.network import Network

MODEL_PATH = "models/mnist_model.npz"
GRID_SIZE = 28
CELL_SIZE = 10
CANVAS_SIZE = GRID_SIZE * CELL_SIZE


class DigitDrawerApp:
    def __init__(self, root: tk.Tk, net: Network) -> None:
        self.root = root
        self.net = net
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.root.title("zerograd - MNIST Interactive Demo")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        # Main layout
        main_frame = tk.Frame(root, bg="#121212", padx=16, pady=16)
        main_frame.pack()

        # Left: Drawing canvas
        left_frame = tk.Frame(main_frame, bg="#121212")
        left_frame.grid(row=0, column=0, padx=(0, 20))

        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#333333",
            cursor="crosshair",
        )
        self.canvas.pack()

        btn_clear = tk.Button(
            left_frame,
            text="Clear (Space)",
            command=self.clear,
            bg="#222222",
            fg="#ffffff",
            activebackground="#333333",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            font=("monospace", 10),
        )
        btn_clear.pack(fill=tk.X, pady=(10, 0))

        # Right: Prediction & confidence bars
        right_frame = tk.Frame(main_frame, bg="#121212", width=220)
        right_frame.grid(row=0, column=1, sticky="n")

        self.pred_label = tk.Label(
            right_frame,
            text="?",
            bg="#121212",
            fg="#00ff88",
            font=("monospace", 48, "bold"),
        )
        self.pred_label.pack()

        self.conf_label = tk.Label(
            right_frame,
            text="Draw a digit (0-9)",
            bg="#121212",
            fg="#aaaaaa",
            font=("monospace", 10),
        )
        self.conf_label.pack(pady=(0, 12))

        self.bars: list[tuple[tk.Canvas, tk.Label]] = []
        for digit in range(10):
            row = tk.Frame(right_frame, bg="#121212")
            row.pack(fill=tk.X, pady=1)

            lbl_d = tk.Label(
                row, text=f"{digit}", bg="#121212", fg="#cccccc", font=("monospace", 9), width=2
            )
            lbl_d.pack(side=tk.LEFT)

            bar = tk.Canvas(row, width=120, height=12, bg="#1e1e1e", highlightthickness=0)
            bar.pack(side=tk.LEFT, padx=4)

            lbl_p = tk.Label(
                row, text="  0%", bg="#121212", fg="#888888", font=("monospace", 9), width=4
            )
            lbl_p.pack(side=tk.LEFT)

            self.bars.append((bar, lbl_p))

        # Mouse & keyboard bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.root.bind("<space>", lambda _: self.clear())

    def paint(self, event: tk.Event) -> None:
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        # Soft brush: paints center strongly and neighbors softly (like MNIST strokes)
        brush = [
            (0, 0, 1.0),
            (-1, 0, 0.45),
            (1, 0, 0.45),
            (0, -1, 0.45),
            (0, 1, 0.45),
            (-1, -1, 0.2),
            (-1, 1, 0.2),
            (1, -1, 0.2),
            (1, 1, 0.2),
        ]

        for dr, dc, strength in brush:
            r, c = row + dr, col + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                self.grid[r, c] = min(1.0, self.grid[r, c] + strength)
                val = int(self.grid[r, c] * 255)
                color = f"#{val:02x}{val:02x}{val:02x}"
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                self.canvas.create_rectangle(
                    x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE, fill=color, outline=""
                )

        self.update_prediction()

    def update_prediction(self) -> None:
        flat_input = self.grid.reshape(1, 784)
        probs = self.net.predict(flat_input)[0]
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class] * 100.0

        self.pred_label.config(text=str(pred_class))
        self.conf_label.config(text=f"Confidence: {confidence:.1f}%")

        for digit, (bar, lbl_p) in enumerate(self.bars):
            p = probs[digit]
            bar.delete("all")
            fill_color = "#00ff88" if digit == pred_class else "#444444"
            bar.create_rectangle(0, 0, int(p * 120), 12, fill=fill_color, outline="")
            lbl_p.config(
                text=f"{p * 100:3.0f}%",
                fg="#00ff88" if digit == pred_class else "#888888",
            )

    def clear(self) -> None:
        self.grid.fill(0.0)
        self.canvas.delete("all")
        self.pred_label.config(text="?")
        self.conf_label.config(text="Draw a digit (0-9)")
        for bar, lbl_p in self.bars:
            bar.delete("all")
            lbl_p.config(text="  0%", fg="#888888")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python3 -m examples.train_mnist' first."
        )

    net = Network(
        [
            Dense(784, 128),
            Activation(relu, relu_prime),
            Dense(128, 64),
            Activation(relu, relu_prime),
            Dense(64, 10),
            Softmax(),
        ]
    )
    net.load(MODEL_PATH)

    root = tk.Tk()
    DigitDrawerApp(root, net)
    root.mainloop()


if __name__ == "__main__":
    main()