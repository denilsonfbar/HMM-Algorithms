import numpy as np
import math  as mt

## MODEL

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

# Converting the probabilities to log in base 2:
for i in range(n_states):
    initial_probabilities[i] = mt.log(initial_probabilities[i], 2) 
for i in range(n_states):
    for j in range(n_states):
        transitions_probabilities[i,j] = mt.log(transitions_probabilities[i,j], 2) 
for i in range(n_states):
    for j in range(n_symbols):
        emission_probabilities[i,j] = mt.log(emission_probabilities[i,j], 2) 

# Checking:
print("\nProbabilities in log values:")
print(initial_probabilities)
print(transitions_probabilities)
print(emission_probabilities)


## VITERBI DECONDING

# Receives a sequence and presents the best path
def viterbi(sequence):

    n_emissions = sequence.size
    viterbi_matrix = np.zeros((n_states,n_emissions))
    paths = np.zeros((n_states,n_emissions))

    for j in range(n_emissions):

        emitted_symbol = sequence[j]
        
        for i in range(n_states):
    
            if j == 0:
                viterbi_matrix[i,j] = initial_probabilities[i] + emission_probabilities[i,emitted_symbol]
            else:

                # Probability of most likely path ending in each state:
                path_ending_states_probs = np.zeros(n_states)
                for k in range(n_states):
                    path_ending_states_probs[k] = viterbi_matrix[k,j-1] + transitions_probabilities[k,i] + emission_probabilities[i,emitted_symbol]

                viterbi_matrix[i,j] = np.amax(path_ending_states_probs)
                paths[i,j-1] = np.argmax(path_ending_states_probs)
                
    # Finishing the paths:
    for k in range(n_states):
        paths[k,n_emissions-1] = k


    # Checking:
    print("\nViterbi matrix:")
    print (viterbi_matrix)

    print("\nPaths: (0: fair; 1: loaded)")
    print (paths)

    print("\nBest path:")
    print(paths[np.argmax(viterbi_matrix[:,n_emissions-1])])


# Testing:
def experiment(sequence):
    print("\n---------")
    print("\nSequence:")
    print (sequence)
    for s in range(sequence.size):
        sequence[s] -= 1
    viterbi(sequence)

sequence = np.array([6,4,1,2])
experiment(sequence)

sequence = np.array([6,6,4,1,2])
experiment(sequence)

sequence = np.array([6,4,1,2,6])
experiment(sequence)

sequence = np.array([1,2,3,4,6,6,6,1,2,3])
experiment(sequence)
