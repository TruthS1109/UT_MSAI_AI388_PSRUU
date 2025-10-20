from random import random
from unittest import result
import numpy as np
import queue
from game import BoardState, GameSimulator, Rules
import math
import time 

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.

        TODO: You need to set self.search_alg_fnc here
        """
        # if alg == "bfs":
        #     self.search_alg_fnc = self.bfs_search_algorithm
        # elif alg == "dfs":
        #     self.search_alg_fnc = self.dfs_search_algorithm
        # elif alg == "adversarial":
        #     self.search_alg_fnc = self.adversarial_search_method  # New adversarial search method for assignment 3
        # else:
        #     self.search_alg_fnc = self.adversarial_search_method  # Update Default to adversarial
            
        ## Only adversarial search for assignment 3 
        if alg == "adversarial":
            self.search_alg_fnc = self.adversarial_search_method  # New adversarial search method for assignment 3
        else:
            self.search_alg_fnc = self.adversarial_search_method  # Update Default to adversarial     
        

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        
        #Create new state by updating the moved piece's position
        new_state_list = list(s)
        new_state_list[offset_idx + k] = v
        
        #switch to next player
        next_player = (p + 1) % 2
        
        return (tuple(new_state_list), next_player)
        # return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))
    
    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    
    ##=============================================================================================
    ### Below part for assignment 2: BFS and DFS search algorithms.  To be removed in assignment 3.    
    def bfs_search_algorithm(self):
        """
        Breadth-First Search algorithm to find optimal path from initial state to goal state.

        """
        
        #If initial state is already a goal state   
        if self.is_goal(self.initial_state):
            return [(self.initial_state, None)]
        
        # BFS initialization
        frontier = queue.Queue()
        frontier.put(self.initial_state)  # Each entry is a state
        
        came_from = {self.initial_state: (None, None)}  # state -> (parent_state, action)
        
        ## LLM tool (ChatGPT) suggested these lines of code below
        while not frontier.empty():
            current_state = frontier.get()
            
            actions = self.get_actions(current_state)
            
            for action in actions:
                next_state = self.execute(current_state, action)
                
                #skip if already visited
                if next_state in came_from:
                    continue

                # Record the path
                came_from[next_state] = (current_state, action)

                # Check if we reached a goal state
                if self.is_goal(next_state):
                    # Reconstruct path
                    return self.reconstruct_path(came_from, next_state)
                        # path = []
                        # while next_state is not None:
                        #     parent_state, action = came_from[next_state]
                        #     path.append((next_state, action))
                        #     next_state = parent_state
                        # path.reverse()
                        # return path
                frontier.put(next_state)
                
        ## End of LLM suggested code        
        return []

        
        
    def dfs_search_algorithm(self):
        """
        Depth-First Search algorithm to find a path from initial state to goal state.
        """
        if self.is_goal(self.initial_state):
            return [(self.initial_state, None)]

        stack = [self.initial_state]
        came_from = {self.initial_state: (None, None)}

        ## LLM tool (ChatGPT) suggested the lines of code below
        while stack:
            current_state = stack.pop()

            if self.is_goal(current_state):
                return self.reconstruct_path(came_from, current_state)

            for action in self.get_actions(current_state):
                next_state = self.execute(current_state, action)

                if next_state not in came_from:  # Not visited
                    came_from[next_state] = (current_state, action)
                    stack.append(next_state)
    
        return []
    ## Above part for assignment 2: BFS and DFS search algorithms.  To be removed in assignment 3. 
    ##=============================================================================================
    
    # Define the termination condition for adversarial search
    def is_termination_state(self, state):
        ## This part was completed with prompt assistance of Github Copilot(AI tool)
        s, p = state
        board = BoardState()
        board.state = np.array(s)
        board.decode_state = board.make_state()
        
        return board.is_termination_state()
    
    # Define the evaluation function for adversarial search
    def evaluate_state(self, state, player_idx):
        ## This part was completed with prompt assistance of Github Copilot(AI tool)
        s, current_player = state
        board = BoardState()
        board.state = np.array(s)
        board.decode_state = board.make_state()
        
        # ## The code below was completed with prompt assistance of Github Copilot(AI tool) And also ChatGPT with modifications
        #Termination state evaluation
        if board.is_termination_state():
            winner = board.get_winner()
            if winner == 0:  # WHITE
                return 20000 if player_idx == 0 else -20000
            elif winner == 1:  # BLACK
                return 20000 if player_idx == 1 else -20000
            else:
                return 0  # Draw
            
        ## End of LLM suggested code
        
        #Get actions for both players
        actions_player_0 = len(self.get_actions( (s, player_idx) ))
        actions_player_1 = len(self.get_actions( (s, (player_idx + 1) % 2 ) ))
        
        
        ## Debug with simple heusristic: action advantage + progress towards goal 
        ## The code below was completed with prompt assistance of Github Copilot(AI tool)
        action_advantage = actions_player_0 - actions_player_1
        
        # Progress evaluation based on distance to goal
        if player_idx == 0:   # WHITE (to reach row 7; positions 50-55)
            ball_position = board.decode_single_pos(s[5])  # furthest ball for WHITE
            progress = ball_position[1]  # row number (0-7) ( or ball_position//8)
        else:  # BLACK (to reach row 0; positions 0-5)
            ball_position = board.decode_single_pos(s[11])  # furthest ball for BLACK
            progress = 7 - ball_position[1]  # row number from bottom
            
        # Combine evaluations
        value = action_advantage * 2.0 + progress * 5.0  ##To try with different weights
        
        return value

        
        # #Evaluation based on blocking advantage
        # if player_idx == 0:
        #     opponent_ball_idx = 6
        # else:
        #     opponent_ball_idx = 0
        # opponent_ball_pos = board.state[opponent_ball_idx]
        # # opponent_ball_moves = board.get_valid_knight_moves(opponent_ball_pos)
        # # if len(opponent_ball_moves) == 0:
        # #     return 10000  # High positive value for blocking opponent's ball
        # # else:
        # #     return -100 * len(opponent_ball_moves) + (actions_player_0 - actions_player_1)
        
        # my_blocking_score = sum(abs(board.state[i] - opponent_ball_pos) <= 2 for i in range(player_idx * 6, player_idx * 6 + 6) )
        
        # value = (1.5 * my_blocking_score) + (actions_player_0 - actions_player_1) * 2
        # # value = 2.0 * (actions_player_0 - actions_player_1) + 1.5 * my_blocking_score
        
        # return value

    ## Define order actions by heuristic value for alpha-beta pruning
    def order_actions(self, state, actions, player_idx):
        
        action_values = []
        ## The code below was completed with prompt assistance of Github Copilot(AI tool)
        for action in actions:
            next_state = self.execute(state, action)
            value = self.evaluate_state(next_state, player_idx)
            action_values.append((action, value))
        
        # Sort actions based on their heuristic values
        if player_idx == 0:  # Maximizing player
            action_values.sort(key=lambda x: x[1], reverse=True)
        else:  # Minimizing player
            action_values.sort(key=lambda x: x[1])
        
        ordered_actions = [action for action, value in action_values]
        
        return ordered_actions 

    
    ## Adding adversiarial search algorithm (Minimax) for assignment 3
    def adversarial_search_method(self, state_tup, search_depth=4, val_b=None, val_c=None):
        """
        Adversarial Search algorithm for assignment 3 (minimax).
        """
        
        if state_tup is None:
            state_tup = self.initial_state
            
        initial_state = state_tup
        _, initial_player = initial_state
        
        #Get all actions for current player
        actions = self.get_actions(initial_state)
        
        if not actions:
            return None, self.evaluate_state(initial_state, initial_player)
        
        # The best move using miniMax with Not alpha-beta pruning
        
        # best_action, best_value = self.MiniMaxValue_noAlphaBeta(initial_state, initial_player, search_depth)
        best_action , best_value = self.MiniMaxValue_AlphaBeta(initial_state, initial_player, search_depth)
        
        
        ## the 2 lines below wer added with prompt assistance of Github Copilot(AI tool)
        #If Minimax returns None, use fallback
        if best_action is None and actions :
            #Fallback: select the first valid action 
            best_action = list(actions)[0]
            best_value = self.evaluate_state(initial_state, initial_player)
            
        print(f"[DEBUG] Player {initial_player}, returning {best_action, best_value}")
            
        return best_action, best_value
        
        # # Path format to return
        # next_state = self.execute(initial_state, best_action)
        # return [(initial_state, best_action), (next_state, None)]
        
        # return best_action, best_value
        
        ### Code below for debugging
        # if not actions:
        #     v = self.evaluate_state(initial_state, initial_player)
        #     print(f"[DEBUG] Player {initial_player} has no actions. Returning (None, {v})")
        #     return None, v

        # best_action, best_value = self.MiniMaxValue_noAlphaBeta(initial_state, initial_player, search_depth)

        # if best_action is None:
        #     v = self.evaluate_state(initial_state, initial_player)
        #     print(f"[DEBUG] Player {initial_player} minimax returned None. Fallback to (None, {v})")
        #     return None, v

        # print(f"[DEBUG] Player {initial_player} returning action={best_action}, value={best_value}")
        
        #     # IMPORTANT: If best_action is None, pick a random valid action
        # if best_action is None:
        #     actions = self.get_actions(initial_state)
        # if actions:
        #     best_action = random.choice(list(actions))
        # else:
        #     best_action = None  # edge case: no valid moves
        
        
        # return best_action, best_value
  
    #Define the Minimax value function with with NO alpha-beta pruning following the pseudocode in the lecture slides
    
    # def MiniMaxValue_noAlphaBeta(self, initial_state, initial_player, search_depth):
    #     #MiniMax with no alpha-beta pruning
    #     best_action = None
    #     if search_depth == 0 or self.is_termination_state(initial_state):
    #         return (None, self.evaluate_state(initial_state, initial_player))
        
    #     actions = self.get_actions(initial_state)
    #     if not actions:
    #         return (None, self.evaluate_state(initial_state, initial_player))
        
    #     ## The code below was completed with prompt assistance of Github Copilot(AI tool)
    #     if initial_player == 0:  # Maximizing player
    #         best_value = -math.inf
    #         for action in actions:
    #             next_state = self.execute(initial_state, action)
    #             value = self.MinValue_NoAlphaBeta(next_state, (initial_player + 1) % 2, search_depth - 1)
    #             if value > best_value:
    #                 best_value = value
    #                 best_action = action
    #     else:  # Minimizing player
    #         best_value = math.inf
    #         for action in actions:
    #             next_state = self.execute(initial_state, action)
    #             value = self.MaxValue_NoAlphaBeta(next_state, (initial_player + 1) % 2, search_depth - 1)
    #             if value < best_value:
    #                 best_value = value
    #                 best_action = action
                    
    #     return best_action, best_value
        
        
        
    #Define the Minimax value function with or without alpha-beta pruning following the pseudocode in the lecture slides
    #This function with alpha-beta pruning
    def MiniMaxValue_AlphaBeta(self, initial_state, initial_player, search_depth):
        
        #Intial values 
        alpha = -math.inf
        beta = math.inf
        best_action = None
        #Initialize the best action and value for the initial state
        if initial_player == 0:  # Maximizing player
            best_value = -math.inf
        else:   # Minimizing player
            best_value = math.inf
            
        actions = self.get_actions(initial_state)
        if not actions:
            return (None, self.evaluate_state(initial_state, initial_player))
        
        actions = self.order_actions(initial_state, actions, initial_player) or actions 
        
        ## The code below was completed with prompt assistance of Github Copilot(AI tool)
        for action in actions:
            next_state = self.execute(initial_state, action)
            value = self.MinValue(next_state, (initial_player + 1) % 2, search_depth - 1, alpha, beta)
            
            if initial_player == 0:  # Maximizing player
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
            else:  # Minimizing player
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                
        return best_action, best_value
    
    ## Define the MaxValue function for minimax with alpha-betapruning based on the pseudocode in the lecture slides
    def MaxValue (self, state, max_player, depth, alpha, beta):
        if self.is_termination_state(state) or depth == 0:
            return self.evaluate_state(state, max_player)
        
        value = -math.inf
        actions = self.get_actions(state)
        if not actions:
            return self.evaluate_state(state, max_player)
        
        actions = self.order_actions(state, actions, max_player) or actions
        
        ## The code below was completed with prompt assistance of Github Copilot(AI tool)
        for action in actions:
            next_state = self.execute(state, action)
            value = max(value, self.MinValue(next_state, max_player , depth - 1, alpha, beta))

            if value >= beta:
                return value
            alpha = max(alpha, value)
    
        return value
    
    # ## Define the MinValue function for minimax based on the pseudocode in the lecture slides
    # def MaxValue_noAlphaBeta(self, state, max_player, depth):
    #     if self.is_termination_state(state) or depth == 0:
    #         return self.evaluate_state(state, max_player)
        
    #     value = -math.inf
    #     actions = self.get_actions(state)
    #     if not actions:
    #         return self.evaluate_state(state, max_player)
        
    #     # actions = self.order_actions(state, actions, max_player) or actions
        
    #     ## The code below was completed with prompt assistance of Github Copilot(AI tool)
    #     for action in actions:
    #         next_state = self.execute(state, action)
    #         value = max(value, self.MinValue_NoAlphaBeta(next_state, (max_player + 1) % 2, depth - 1))
        
    #     return value
    
    
    ## Define the MinValue function for minimax with alpha-beta pruning based on the pseudocode in the lecture slides    
    def MinValue(self, state, max_player, depth, alpha, beta):
        if self.is_termination_state(state) or depth == 0:
            return self.evaluate_state(state, max_player)
        
        value = math.inf
        actions = self.get_actions(state)
        if not actions:
            return self.evaluate_state(state, max_player)
        
        # actions = self.order_actions(state, actions, max_player) or actions
        
        ## The code below was completed with prompt assistance of Github Copilot(AI tool)
        for action in actions:
            next_state = self.execute(state, action)
            value = min(value, self.MaxValue(next_state, (max_player + 1) % 2, depth - 1, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        
        return value
    
    # #define the MinValue function for minimax without alpha-beta pruning based on the pseudocode in the lecture slides    
    # def MinValue_NoAlphaBeta(self, state, max_player, depth):
    #     if self.is_termination_state(state) or depth == 0:
    #         return self.evaluate_state(state, max_player)
        
    #     value = math.inf
    #     actions = self.get_actions(state)
    #     if not actions:
    #         return self.evaluate_state(state, max_player)
        
    #     # actions = self.order_actions(state, actions, max_player) or actions
        
    #     ## The code below was completed with prompt assistance of Github Copilot(AI tool)
    #     for action in actions:
    #         next_state = self.execute(state, action)
    #         value = min(value, self.MaxValue_noAlphaBeta(next_state, (max_player + 1) % 2, depth - 1))
        
    #     return value
    
    # #Define the alpha-beta search method as a wrapper for minimax with alpha-beta pruning    
    # def alpha_beta_search_method(self, state_tup, player_idx, search_depth=4):
        
    #     intial_state = state_tup
    #     state = (intial_state, player_idx)

        
    #     best_action = None
    #     best_value = -math.inf
        
    #     best_action, best_value = self.MiniMaxValue_AlphaBeta(state, player_idx, search_depth)
        
    #     return best_action, best_value        

    ## LLM tool (ChatGPT) suggested the implementation for this function below
    def reconstruct_path(self, came_from, goal_state):
        """
        Reconstruct the path from initial state to goal state using the came_from dictionary.
        
        Args:
            came_from: Dictionary mapping state -> (parent_state, action_from_parent)
            goal_state: The goal state we reached
            
        Returns:
            List of (state, action) pairs in order from initial to goal state
        """
        path = []
        current_state = goal_state
        current_action = None
        
        # Work backwards from goal to initial state
        while current_state is not None:
            path.append((current_state, current_action))
            current_state, current_action = came_from.get(current_state, (None, None))
        
        # Reverse the path to get from initial to goal
        path.reverse()
        
        # The first state should have action = None (no action taken yet)
        # Subsequent states should have the action that was taken to reach them
        return path
    
    

    # Alias the search function for easy access
    def search(self):
        """Main search method that uses the selected algorithm."""
        if self.search_alg_fnc is None:
            self.set_search_alg()  # Ensure algorithm is set
        
        return self.search_alg_fnc(self.initial_state)

    """
    Here is an example:

    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
    """

