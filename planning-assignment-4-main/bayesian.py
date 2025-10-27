import numpy as np

class StateGenerator:

    def __init__(self, nrows=8, ncols=7, npieces=10):
        """
        Initialize a generator for sampling valid states from
        an npieces dimensional state space.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.npieces = npieces
        self.rng = np.random.default_rng()

    def sample_state(self):
        """
        Samples a self.npieces length tuple.

        Output:
            Returns a state. A state is as 2-tuple (positions, dimensions), where
             -  Positions is represented as a list of position (c,r) tuples 
             -  Dimensions is a 2-tuple (self.nrows, self.ncols)

            For example, if the dimensions of the board are 2 rows, 3 columns, and the number of pieces
            is 4, then a valid return state would be ([(0, 0) , (1, 0), (2, 0), (1, 1)], (2,3))
        """
        ## Returns positions in decoded format. i.e. list of (c,r) i.e. (x,y)
        ## Without loss of generalization, we assume that positions[1:] are fixes; only
        ## positions[0] will be moved
        positions = self.rng.choice(self.nrows*self.ncols, size=self.npieces, replace=False)
        pos = list(self.decode(p) for p in positions)
        return pos, (self.nrows, self.ncols)

    def decode(self, position):
        r = position // self.ncols
        c = position - self.ncols * r
        return (c, r)

def sample_observation(state):
    """
    Given a state, sample an observation from it. Specifically, the positions[1:] locations are
    all known, while positions[0] should have a noisy observation applied.

    Input:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        A tuple (position, distribution) where:
         - Position is a sampled position which is a 2-tuple (c, r), which represents the sampled observation
         - Distribution is a 2D numpy array representing the observation distribution

    NOTE: the array representing the distribution should have a shape of (nrows, ncols)
    """
    #Extract the true position of piece 0 and the board dimensions
    positions, dimensions = state
    true_position  = positions[0]
    nrows, ncols = dimensions
    
    #Create a 2D numpy array (nrows, ncols) to represent the observation distribution and initialize to zeros
    observation_distribution = np.zeros((nrows, ncols))

    # Set probability = higher value if some adjacent squares are blocked
    adjacent_positions = [(true_position[0]-1, true_position[1]),   #left
                          (true_position[0]+1, true_position[1]),   #right
                          (true_position[0], true_position[1]-1),   #up
                          (true_position[0], true_position[1]+1)]   #down
    
    ### The code below was completed with prompting support from an LLM tool (Copilot)
    
    # For each adjacent position (up, down, left, right):
    # Check if it's within bounds and not occupied
    # If free, assign 10% probability
    # If blocked (edge or other piece), add that 10% to the center probability
    blocked_count = 0
    center_prob = 0.60
    for pos in adjacent_positions:
        if 0 <= pos[0] < ncols and 0 <= pos[1] < nrows:
            if pos in positions[1:]:  # occupied
                blocked_count += 1
            else:  # free
                observation_distribution[pos[1], pos[0]] = 0.10
        else:  # out of bounds
            blocked_count += 1
            
    # Add blocked probabilities to center
    center_prob += blocked_count * 0.10
    observation_distribution[true_position[1], true_position[0]] = center_prob
    
    total = np.sum(observation_distribution)
    if total > 0:
        observation_distribution /= total  # Normalize to sum to 1
    else:
        # If total is 0 (all adjacent blocked), set center to 1
        observation_distribution[true_position[1], true_position[0]] = 1.0
    
    #Sample an observation position based on the observation distribution
    flat_dist = observation_distribution.flatten()
    sampled_idx = np.random.choice(len(flat_dist), p=flat_dist/np.sum(flat_dist))
    sampled_position = (sampled_idx % ncols, sampled_idx // ncols)

    ### End of code completed with LLM tool support
    
    return sampled_position, observation_distribution
    
def sample_transition(state, action):
    """
    Given a state and an action, 
    returns:
         a resulting state, and a probability distribution represented by a 2D numpy array
    If a transition is invalid, returns None for the state, and a zero probability distribution
    NOTE: the array representing the distribution should have a shape of (nrows, ncols)

    Inputs:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        Action: a 2-tuple (dc, dr) representing the difference in positions of position[0] as a result of
                executing this transition.

    Outputs:
        A 2-tuple (new_position, transition_probabilities), where
            - new_position is:
                A 2-tuple (new_column, new_row) if the action is valid.
                None if the action is invalid.
            - transition_probabilities is a 2D numpy array with shape (nrows, ncols) that accurately reflects
                the probability of ending up at a certain position on the board given the action. 
    """ 
    #Extract the true position of piece 0 and the board dimensions
    positions, dimensions = state
    true_position  = positions[0]
    nrows, ncols = dimensions
    dc, dr = action
    new_col = true_position[0] + dc
    new_row = true_position[1] + dr

  ### Code below was completed with prompting support from a LLM tool (Copilot)

    # Check if the new position is valid (within bounds and not occupied)
    if (new_col, new_row) not in positions[1:] and 0 <= new_col < ncols and 0 <= new_row < nrows:
        # Create a zero probability distribution
        transition_probabilities = np.zeros((nrows, ncols))
        # Set the probability of the new position to 1
        transition_probabilities[new_row, new_col] = 1
        return (new_col, new_row), transition_probabilities
    else:
        # Invalid action
        return None, np.zeros((nrows, ncols))
    
    ### End of code prompted with LLM tool support
 
def initialize_belief(initial_state, style="uniform"):
    """
    Create an initial belief, based on the type of belief we want to start with

    Inputs:
        Initial_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        style: an element of the set {"uniform", "dirac"}

    Returns:
        an initial belief, represented by a 2D numpy array with shape (nrows, ncols)

    NOTE:
        The array representing the distribution should have a shape of (nrows, ncols).
        The occupied spaces (if any) should be zeroed out in the belief.
        We define two types of priors: a uniform prior (equal probability over all
        unoccupied spaces), and a dirac prior (which concentrates all the probability
        onto the actual position on the piece).
    
    """
    
    #Extract all pieces positions and board dimensions
    positions, dimensions = initial_state
    nrows, ncols = dimensions
    
    #Create a 2D numpy array (nrows, ncols) to represent the belief and initialize to zeros
    # belief = np.ones((nrows, ncols))
    belief = np.zeros((nrows, ncols))
    
    
    ### The code below was completed with prompting support from an LLM tool (Copilot)
    ## During Debugging, With suggestions from ChatGPT
    # For "uniform" style:
    if style == "uniform":
        # Use precise calculation to avoid floating point errors
        occupied_set = set(positions[1:])  # All pieces are occupied
        free_count = nrows * ncols - len(occupied_set)
        
        if free_count > 0:
            prob = 1.0 / free_count
            for r in range(nrows):
                for c in range(ncols):
                    if (c, r) not in occupied_set:
                        belief[r, c] = prob
                        
    # For "dirac" style:
    elif style == "dirac":
        #Set probability 1.0 at the true position of piece 0
        true_position = positions[0]
        belief[true_position[1], true_position[0]] = 1.0
    
    #Zero out occupied cells
    for pos in positions[1:]:
        belief[pos[1], pos[0]] = 0.0
    
    ### End of code completed with LLM tool support 
    
    return belief
     
### The helper function below was created with LLM Tool support (ChatGPT)
def get_observation_distribution(position, reference_state):
    """Get observation distribution if piece were at given position"""
    positions, dimensions = reference_state
    nrows, ncols = dimensions
    
    dist = np.zeros((nrows, ncols))
    center_prob = 0.60
    
    adjacent_positions = [
        (position[0]-1, position[1]), (position[0]+1, position[1]),
        (position[0], position[1]-1), (position[0], position[1]+1)
    ]
    
    blocked_count = 0
    for pos in adjacent_positions:
        if 0 <= pos[0] < ncols and 0 <= pos[1] < nrows and pos not in positions[1:]:
            dist[pos[1], pos[0]] = 0.10
        else:
            blocked_count += 1
    
    center_prob += blocked_count * 0.10
    dist[position[1], position[0]] = center_prob
    
    return dist
### The helper function above was created with LLM Tool support (ChatGPT)

def belief_update(prior, observation, reference_state):
    """
    Given a prior an observation, compute the posterior belief

    Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        observation: a 2-tuple (col, row) representing the observation of a piece at a position
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    positions, dimensions = reference_state
    nrows, ncols = dimensions
    obs_col, obs_row = observation
    
    ### Code in this function below was completed with prompting support from a LLM tool (Copilot)
    likelihood = np.zeros((nrows, ncols))
    
    for r in range(nrows):
        for c in range(ncols):
            if (c, r) in positions[1:]:  # Skip occupied cells
                continue
                
            # Get observation distribution if piece were at (c, r)
            observation_dist = get_observation_distribution((c, r), reference_state)

            # Likelihood
            likelihood[r, c] = observation_dist[obs_row, obs_col]

    unnormalized_posterior = likelihood * prior

    for pos in positions[1:]:
        unnormalized_posterior[pos[1], pos[0]] = 0.0  # Zero out occupied cells

    total = np.sum(unnormalized_posterior)
    
    if total > 0:
        posterior = unnormalized_posterior / total
    else:
        posterior = np.zeros_like(unnormalized_posterior)

    return posterior


    ### End of code completed with prompting support from a LLM tool (Copilot)

def belief_predict(prior, action, reference_state):
    """
    Given a prior, and an action, compute the posterior belief.

    Actions will be given in terms of dc, dr

   Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        action: a 2-tuple (dc, dr) as defined for action in sample_transition
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    ### Code in this function was completed with prompting support from a LLM tool (Copilot) And Debug with suggestions from ChatGPT
    
    #Create a new belief array initialized to zeros
    nrows, ncols = prior.shape
    posterior = np.zeros((nrows, ncols))
    
    #For each cell in the prior belief:
    # If the prior probability > 0 at that position:
        # Calculate where the piece would move from that position using the action
        # If the move is valid, add the probability to the new position
        # If invalid, the probability is lost (goes to 0)
    for r in range(nrows):
        for c in range(ncols):
            if prior[r, c] > 0:
                current_position = (c, r)
                #Create a temporary state with piece 0 at current_position
                temp_positions = list(reference_state[0])
                temp_positions[0] = current_position
                temp_state = (temp_positions, reference_state[1])
                #Get the resulting position from the action
                new_position, _ = sample_transition(temp_state, action)
                if new_position is not None:
                    posterior[new_position[1], new_position[0]] += prior[r, c]
    
    ### End of code prompted with LLM tool support
    
    for (c,r) in reference_state[0][1:]:
        posterior[r, c] = 0.0  # Zero out occupied cells
    
    total = np.sum(posterior)
    if total > 0:
        posterior /= total  # Normalize to sum to 1
    else:
        posterior = np.zeros_like(posterior)

    return posterior
    
if __name__ == "__main__":
    gen = StateGenerator()
    initial_state = gen.sample_state()
    obs, dist = sample_observation(initial_state)
    print(initial_state)
    print(obs)
    print(dist)
    b = initialize_belief(initial_state, style="uniform")
    print(b)
    print(np.sum(b))
    print(np.count_nonzero(b))
    print(1/np.count_nonzero(b))
    b = belief_update(b, obs, initial_state)
    print(b)
    b = belief_predict(b, (1,0), initial_state)
    print(b)
