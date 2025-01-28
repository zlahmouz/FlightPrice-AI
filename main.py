import numpy as np
import random
import streamlit as st



# Q-learning parameters
alpha = 0.1           # Learning rate
gamma = 0.9           # Discount factor
epsilon = 0.1         # Exploration rate
episodes = 10000      # Number of episodes


# Function to take an action and observe the next state and reward
def step(state, action, prices, probs):
    n, t = state
    if t == 0 or n == 0:
        return state, 0, True  # Terminal state

    price = prices[action]
    prob = probs[action]

    # Determine if a seat is sold
    sold = np.random.rand() < prob
    reward = price * sold
    next_n = n - 1 if sold else n
    next_t = t - 1
    next_state = (next_n, next_t)

    done = next_t == 0 or next_n == 0
    return next_state, reward, done

# Q-learning algorithm
def run_q_learning(prices, probs,max_seats,max_days):
    Q = np.zeros((max_seats + 1, max_days + 1, len(prices)))  # Reinitialize Q-table
    for episode in range(episodes):
        state = (max_seats, max_days)
        while True:
            n, t = state

            # Choose action using epsilon-greedy policy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(prices) - 1)
            else:
                action = np.argmax(Q[n, t, :])

            # Take action and observe reward and next state
            next_state, reward, done = step(state, action, prices, probs)

            # Update Q-value
            next_n, next_t = next_state
            best_next_action = np.argmax(Q[next_n, next_t, :])
            Q[n, t, action] += alpha * (
                reward + gamma * Q[next_n, next_t, best_next_action] - Q[n, t, action]
            )

            # Transition to next state
            state = next_state

            if done:
                break
    return Q

# Streamlit app
def main():
    st.title("Q-learning for Seat Pricing Optimization")

    # User inputs for prices and probabilities
    st.sidebar.header("Set Prices and Probabilities")
    price1 = st.sidebar.number_input("Price 1", value=5, min_value=0)
    price2 = st.sidebar.number_input("Price 2", value=1, min_value=0)
    prob1 = st.sidebar.slider("Probability for Price 1", 0.0, 1.0, 0.1)
    prob2 = st.sidebar.slider("Probability for Price 2", 0.0, 1.0, 0.8)
    max_seats = st.sidebar.number_input("Max Seats", value=40, min_value=0)
    max_days = st.sidebar.number_input("Max days", value=20, min_value=0)

    prices = [price1, price2]
    probs = [prob1, prob2]

    if st.button("Run Q-learning"):
        st.write("Running Q-learning...")
        Q = run_q_learning(prices, probs,max_seats,max_days)
        st.write("Q-learning completed!")

        # Test the learned policy
        total = 0
        state = (max_seats, max_days)
        results = []
        while state[1] > 0 and state[0] > 0:
            n, t = state
            action = np.argmax(Q[n, t, :])
            price = prices[action]
            results.append(f"Day {max_days - t + 1}, Seats remaining: {n}, Price: {price}")
            state, reward, _ = step(state, action, prices, probs)
            total += reward

        st.write("Results:")
        for result in results:
            st.write(result)

        st.write(f"Total revenue: {total}")

if __name__ == "__main__":
    main()
