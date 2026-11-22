#Goal: Teach a single neuron the rule y = 2 * x (without tellin it the number 2)

#Training Data: (hours studied, expected grade)
x = 4.0
y_true = 8.0

#1. Start with a random "knob" (weight)
weight = 0.5
learning_rate = 0.01

for epoch in range(1, 21):
    #Forward pass: make a guess
    y_pred = x * weight 

    #Loss: ho far off are we? (Squared Error)
    error = y_pred - y_true
    loss = error ** 2

    #Backpropagation: calculate which way and how much to tuyn the knob
    gradient = 2 * error * x

    #Update: turn the knob in the opposite direction of the error
    weight = weight - learning_rate * gradient
    
    print(f"Epoch {epoch:2d} | Weight: {weight:.4f} | Guess: {y_pred:.4f} | Loss: {loss:.4f}")