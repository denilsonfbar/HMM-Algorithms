import numpy as np
import math  as mt

model = 2

## MODEL 1: dishonest casino
if model == 1:

    n_symbols = 6
    n_states  = 2

    # Indexes of states:
    # fair: 0   loaded: 1
    initial_probabilities = np.array([0.5,0.5])

    # Transition matrix:
    # fair-fair     fair-loaded
    # loaded-fair   loaded-loaded
    transitions_probabilities = np.array([[0.95,0.05], 
                                          [0.1 ,0.9 ]])

    # Emission matrix:
    # fair-1 ... fair-6
    # loaded-1 ... loaded-6
    emission_probabilities = np.array([[1/6,1/6,1/6,1/6,1/6,1/6],
                                       [0.1,0.1,0.1,0.1,0.1,0.5]])

## MODEL 2: Prof. André example
elif model == 2:

    n_symbols = 2
    n_states  = 2

    # Indexes of states:
    # A: 0   B: 1
    initial_probabilities = np.array([0.5,0.5])

    # Transition matrix:
    # AA     AB
    # BA     BB
    transitions_probabilities = np.array([[0.99,0.01], 
                                          [0.3 ,0.7 ]])

    # Emission matrix:
    # A0    A1
    # B0    B1
    emission_probabilities = np.array([[0.5,0.5],
                                       [0.1,0.9]])

## Converting the probabilities to log in base 2:
for i in range(n_states):
    initial_probabilities[i] = mt.log(initial_probabilities[i], 2) 
for i in range(n_states):
    for j in range(n_states):
        transitions_probabilities[i,j] = mt.log(transitions_probabilities[i,j], 2) 
for i in range(n_states):
    for j in range(n_symbols):
        emission_probabilities[i,j] = mt.log(emission_probabilities[i,j], 2) 
# Checking:
print("\nProbabilities in log base 2 values:")
print("Initial probabilities:")
print(initial_probabilities)
print("Transitions probabilities:")
print(transitions_probabilities)
print("Emission probabilities:")
print(emission_probabilities)


## VITERBI DECODING

# Receives a sequence and return the most probable path of states
def viterbi_decoding(sequence):

    n_positions = sequence.size
    viterbi_matrix = np.zeros((n_states,n_positions))
    viterbi_paths = np.zeros((n_states,n_positions))

    for i in range(n_positions):

        emitted_symbol = sequence[i]
        
        for l in range(n_states):
    
            if i == 0:
                viterbi_matrix[l,i] = initial_probabilities[l] + emission_probabilities[l,emitted_symbol]

            else:
                # Probability of most likely path ending in each state:
                previous_position_probs = np.zeros(n_states)
                for k in range(n_states):
                    previous_position_probs[k] = viterbi_matrix[k,i-1] + transitions_probabilities[k,l] 

                viterbi_matrix[l,i] = emission_probabilities[l,emitted_symbol] + np.amax(previous_position_probs) 
                viterbi_paths[l,i-1] = np.argmax(previous_position_probs)
                
    # Finishing the paths:
    for k in range(n_states):
        viterbi_paths[k,n_positions-1] = k

    probability_best_path = np.amax(viterbi_matrix[:,n_positions-1])
    best_path = viterbi_paths[np.argmax(viterbi_matrix[:,n_positions-1])]

    # Checking:
    print("\nViterbi matrix:")
    print (viterbi_matrix)

    print("\nPaths: ")
    print (viterbi_paths)

    print("\nBest path:\tProbability: ", probability_best_path)
    print(best_path)

    return best_path


## FORWARD

# Receives a sequence and the state of the last position
# Returns the sum of the probabilities of all paths ending in the state of the last position informed
def forward(sequence, state_last_position):

    n_positions = sequence.size
    forward_matrix = np.zeros((n_states,n_positions))

    for i in range(n_positions):

        emitted_symbol = sequence[i]
        
        for l in range(n_states):
    
            if i == 0:
                forward_matrix[l,i] = initial_probabilities[l] + emission_probabilities[l,emitted_symbol]

            else:

                for k in range(n_states):
                    aux = forward_matrix[k,i-1] + transitions_probabilities[k,l]
                    if k == 0:
                        previous_sum_probs = aux
                    else:
                        previous_sum_probs = np.logaddexp2(previous_sum_probs, aux) 

                forward_matrix[l,i] = emission_probabilities[l,emitted_symbol] + previous_sum_probs

    total_prob = forward_matrix[state_last_position,n_positions-1]

    # Checking:
    print("\nForward matrix with log base 2 values:")
    print (forward_matrix)

    for i in range(n_positions):
        for l in range(n_states):
            forward_matrix[l,i] = 2 ** forward_matrix[l,i]
    print("\nForward matrix with real values:")
    print (forward_matrix)

    print("\nSum of probabilities of all paths ending in state ", state_last_position, end=': ')
    print(total_prob)

    return total_prob


## TESTING

def experiment(sequence):
    print("\n---------")
    print("\nSequence:")
    print (sequence)
    if model == 1: 
        for s in range(sequence.size):
            sequence[s] -= 1
    viterbi_decoding(sequence)
    forward(sequence, 0)

if model == 1:

    sequence = np.array([6,4,1,2])
    experiment(sequence)
    
    sequence = np.array([6,6,4,1,2])
    experiment(sequence)

    sequence = np.array([6,4,1,2,6])
    experiment(sequence)

    sequence = np.array([1,2,3,4,6,6,6,1,2,3])
    experiment(sequence)  

elif model == 2:

    sequence = np.array([0,1,1])
    experiment(sequence)
