from __future__ import print_function
import json
import numpy as np
import sys

def graph():
    import numpy as np
    import matplotlib.pyplot as plt

    in_array = [0.001, 0.1, 0.5, 0.99, 1, 1.2, 1.4, 1.6, 1.8, 2]
    out_array = np.log(in_array)

    print("out_array : ", out_array)

    plt.plot(in_array, in_array,
             color='blue', marker="*")

    # red for numpy.log()
    plt.plot(out_array, in_array,
             color='red', marker="o")

    plt.title("numpy.log()")
    plt.xlabel("out_array")
    plt.ylabel("in_array")
    plt.show()

def forward(pi, A, B, O):
    """
    Forward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
    """
    S = len(pi)
    N = len(O)
    alpha = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################
    alpha[:, 0] = np.multiply(pi, B[:, O[0]])
    for i in range(1, N):
        obs = B[:, O[i]]
        temp = np.dot(alpha[:, i-1], A)
        alpha[:, i] = np.multiply(obs, temp)
    # print(alpha)
    return alpha

def backward(pi, A, B, O):
    """
    Backward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    """
    S = len(pi)
    N = len(O)
    beta = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here
    ###################################################
    beta[:, -1] = 1
    for i in range(N-2, -1, -1):
        obs = np.multiply(B[:, O[i+1]], beta[:, i+1])
        matrix = np.dot(A, np.reshape(obs, (S,1)))
        beta[:, i] = np.squeeze(matrix)
    # print(beta)
    return beta

def seqprob_forward(alpha):
    """
    Total probability of observing the whole sequence using the forward messages

    Inputs:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    prob = np.sum(alpha[:, -1])
    return prob

def seqprob_backward(beta, pi, B, O):
    """
    Total probability of observing the whole sequence using the backward messages

    Inputs:
    - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence
        (in terms of the observation index, not the actual symbol)

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here
    ###################################################
    prob = np.sum(B[:, O[0]] * pi * beta[:, 0])
    return prob

def viterbi(pi, A, B, O):
    """
    Viterbi algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - path: A list of the most likely hidden state path (in terms of the state index)
    """
    path = []
    ###################################################
    # Q3.3 Edit here
    ###################################################
    S = len(pi)
    N = len(O)
    gamma = np.zeros([S, N])
    delta = np.zeros([S, N])
    delta[:, 0] = -1
    gamma[:, 0] = np.multiply(pi, B[:, O[0]])
    for i in range(1, N):
        gamma_prev = A * np.reshape(gamma[:, i-1], (S, 1))
        gamma[:, i] = B[:, O[i]] * np.max(gamma_prev, axis=0)
        delta[:, i] = np.argmax(gamma_prev, axis=0)
    prev = np.argmax(gamma[:, N - 1])
    path.append(prev)
    for i in range(N-1, 0, -1):
        new_prev = int(delta[prev, i])
        path.append(new_prev)
        prev = new_prev
    path = path[::-1]
    return path

##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
    model_file = sys.argv[1]
    Osymbols = sys.argv[2]

    #### load data ####
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    #### observation symbols #####
    obs_symbols = data['observations']
    #### state symbols #####
    states_symbols = data['states']

    # import random
    # temp_symbols = list(set(Osymbols))
    # Osymbols = ''
    # for i in range(20):
    #     Osymbols += random.choice(temp_symbols)
    # print(obs_symbols)

    N = len(Osymbols)
    O = [obs_symbols[j] for j in Osymbols]

    alpha = forward(pi, A, B, O)
    beta = backward(pi, A, B, O)

    prob1 = seqprob_forward(alpha)
    prob2 = seqprob_backward(beta, pi, B, O)
    print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

    viterbi_path = viterbi(pi, A, B, O)

    print('Viterbi best path is ')
    for j in viterbi_path:
        print(states_symbols[j], end=' ')

if __name__ == "__main__":
    main()
