import numpy as np

verbose = True

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
initial_probabilities = np.log2(initial_probabilities)
transitions_probabilities = np.log2(transitions_probabilities)
emission_probabilities = np.log2(emission_probabilities)


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
                # Probability of most likely path starting in each state:
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

    if verbose:
        print("\nViterbi matrix:")
        print (viterbi_matrix)
        print("\nPaths: ")
        print (viterbi_paths)
        print("\nBest path:\tProbability: ", probability_best_path)
        print(best_path)

    return best_path


# Global variables for reuse of matrices
forward_matrix = []
backward_matrix = []


## FORWARD

# Receives a sequence and the state of the last position
# Returns the sum of the probabilities of all paths finishing in the state of the last position informed
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

    if verbose:
        print("\nForward matrix with log base 2 values:")
        print (forward_matrix)
        print("\nForward matrix with real values:")
        print (np.exp2(forward_matrix))
        print("\nSum of probabilities of all paths finishing in state ", state_last_position, end=': ')
        print(total_prob)

    return total_prob


## BACKWARD

# Receives a sequence and the state of the first position
# Returns the sum of the probabilities of all paths starting in the state of the first position informed
def backward(sequence, state_first_position):

    n_positions = sequence.size
    backward_matrix = np.zeros((n_states,n_positions+1))

    for i in reversed(range(n_positions)):

        emitted_symbol = sequence[i]

        # Initialization:
        if i == n_positions-1:
            for l in range(n_states):
                backward_matrix[l,n_positions] = np.log2(1)
        
        for l in range(n_states):
            # Induction:
            if i != 0:
                for k in range(n_states):
                    aux = backward_matrix[k,i+1] + transitions_probabilities[l,k] + emission_probabilities[k,emitted_symbol]

                    if k == 0:
                        later_sum_probs = aux
                    else:
                        later_sum_probs = np.logaddexp2(later_sum_probs, aux) 

                backward_matrix[l,i] = later_sum_probs
            # Termination:
            else:
                backward_matrix[l,i] = backward_matrix[l,i+1] + emission_probabilities[l,emitted_symbol] + initial_probabilities[l]

    total_prob = backward_matrix[state_first_position,0]

    if verbose:
        print("\nBackward matrix with log base 2 values:")
        print (backward_matrix)
        print("\nBackward matrix with real values:")
        print (np.exp2(backward_matrix))
        print("\nSum of probabilities of all paths starting in state ", state_first_position, end=': ')
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
    backward(sequence, 0)


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
