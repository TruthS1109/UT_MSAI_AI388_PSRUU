import numpy as np

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        TODO: You need to implement this.
        """

        col, row = cr
        return col + row * self.N_COLS

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        row = n // self.N_COLS
        col = n % self.N_COLS
        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """
        if not self.is_valid():
            return False
    
        white_ball_pos = self.decode_single_pos(self.state[5])
        black_ball_pos = self.decode_single_pos(self.state[11])
        
        #check if balls are actually on the correct pieces
        # white_ball_on_piece = self.state[5] in self.state[0:5]
        # black_ball_on_piece = self.state[11] in self.state[6:11]

        if white_ball_pos[1] == 7: #and white_ball_on_piece :
            return True

        if black_ball_pos[1] == 0: #and black_ball_on_piece:
            return True
        
        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        
        TODO: You need to implement this.
        """
        # Check 1: All pieces must be within the board boundaries
        max_pos = self.N_ROWS * self.N_COLS
        for pos in self.state:
            if not isinstance(pos, (int, np.integer)):
                return False
            if pos < 0 or pos >= max_pos:
                return False
            
        # # Check 2: No two pieces can occupy the same position
        # if len(set(self.state)) != 12:
        #     return False
        
        # # Check 3: Each player must have exactly one ball piece
        # if self.state[5] < 50 or self.state[5] > 55:
        #     return False
        # if self.state[11] < 50 or self.state[11] > 55:
        #     return False
        
        # # Check 4: All other pieces must be block pieces
        # for pos in self.state[:5] + self.state[6:11]:
        #     if pos < 0 or pos >= 50:
        #         return False
        
        # Blocks positions: indices 0-4 (white blocks), 6-10 (black blocks)
        white_block_positions = [int(self.state[i]) for i in range(0, 5)]
        black_block_positions = [int(self.state[i]) for i in range(6, 11)]
        
        # Check no duplicate block positions within same player and across both players
        # Combined block set must have length 10
        combined_blocks = white_block_positions + black_block_positions
        if len(set(combined_blocks)) != 10:
            return False

        # # Check 5: All decoded positions must be within the board boundaries
        # # This check is generated with LLM tool assistance
        # for pos in combined_blocks + [white_ball_pos, black_ball_pos]:
        #     col, row = self.decode_single_pos(pos)
        #     if col < 0 or col >= self.N_COLS or row < 0 or row >= self.N_ROWS:
        #         return False

        # Balls must coincide with one of the player's block positions
        white_ball_pos = int(self.state[5])
        black_ball_pos = int(self.state[11])

        if white_ball_pos not in white_block_positions:
            return False
        if black_ball_pos not in black_block_positions:
            return False
        
        # # Check 6: Balls must be on their respective sides of the board(same color as player)
        #  #This check is generated with LLM tool assistance
        #  #White ball (index 5) must be on one of white's pieces (indices 0-4)
        # if self.state[5] not in self.state[0:5]:
        #     return False
        
        # #Black ball (index 11) must be on one of black's pieces (indices 6-10)
        # if self.state[11] not in self.state[6:11]:
        #     return False

        return True

class Rules:
    
    @staticmethod
    def get_knight_moves(pos, board_state):
        """
        Returns the set of possible knight moves for a given position on the board.

        Inputs:
            - pos: the current position of the knight (encoded as a single integer)
            - board_state: the current state of the board (assumed to be a BoardState)

        Output: a set of integers representing the encoded positions the knight can move to
        """
       # Different implementation of single_piece_actions and single_ball_actions
        col, row = board_state.decode_single_pos(pos)
        moves = set()
        
        # Knight move patterns: (±1, ±2) and (±2, ±1)
        patterns = [(1,2), (2,1), (-1,2), (-2,1), 
                   (1,-2), (2,-1), (-1,-2), (-2,-1)]
        
        for dc, dr in patterns:
            new_col, new_row = col + dc, row + dr
            if 0 <= new_col < board_state.N_COLS and 0 <= new_row < board_state.N_ROWS:
                new_pos = board_state.encode_single_pos((new_col, new_row))
                # Check if position is unoccupied
                # if new_pos not in board_state.state:
                moves.add(new_pos)

        return moves

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        TODO: You need to implement this.
        """
        #Get current position of the piece
        current_pos = board_state.state[piece_idx]
        
        #Check if the piece is holding a ball -- if so, it cannot move
        if piece_idx < 6: # White pieces
            if board_state.state[5] == current_pos:
                return set() # No moves possible if holding the ball
        else: # Black pieces
            if board_state.state[11] == current_pos:
                return set() # No moves possible if holding the ball

        # Knight moves that land on unoccupied squares (can't land where any block is)
        candidate_moves = Rules.get_knight_moves(current_pos, board_state)
        # A block may only move to unoccupied spaces on the board (cannot land on any block)
        # Note: balls occupy the same cell as their block; we only check block occupancy (10 block positions)
        block_positions = {int(board_state.state[i]) for i in range(0,5)} | {int(board_state.state[i]) for i in range(6,11)}

        valid_moves = {m for m in candidate_moves if m not in block_positions}
        return valid_moves

        # return set(Rules.get_knight_moves(current_pos, board_state))

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        TODO: You need to implement this.
        """
        if player_idx == 0: # White player
            ball_idx = 5
            piece_indices = list(range(0, 5))  # White pieces are at indices 0-4

        else: # Black player
            ball_idx = 11
            piece_indices = list(range(6, 11))  # Black pieces are at indices 6-10
            
        start_pos = int(board_state.state[ball_idx])

        # Map block positions to piece indices for quick lookup
        # pos_to_piece_idx = {int(board_state.state[i]): i for i in piece_indices}

        # BFS over block positions (nodes = positions of own blocks)
        reachable = set()
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)

        while queue:
            current = queue.pop(0)
            # For every other friendly block, if direct pass is possible, it's a neighbor
            for idx in piece_indices:
                target_pos = int(board_state.state[idx])
                if target_pos == current:
                    continue
                if target_pos in visited:
                    continue
                if Rules.is_valid_ball_path(board_state, current, target_pos):
                    visited.add(target_pos)
                    queue.append(target_pos)
                    reachable.add(target_pos)

        # reachable contains all friendly block positions reachable via any sequence of valid passes
        return reachable

        # current_ball_pos = board_state.state[ball_idx]
        # valid_moves = set()
        
        # #The ball can move to any position occupied by its own pieces
        # for idx in piece_indices:
        #     target_pos = board_state.state[idx]
            
        #     if target_pos == current_ball_pos:
        #         continue  # Skip if it's the same position as the ball
            
            
        #     if Rules.is_valid_ball_path(board_state,current_ball_pos, target_pos):
        #         valid_moves.add(target_pos)
                
        # return valid_moves
        
    @staticmethod
    def is_valid_ball_path(board_state, start_pos, end_pos):
        """
        Checks if moving the ball from current_ball_pos to target_pos is valid according to the game rules.

        Inputs:
            - board_state: the current state of the board (BoardState)
            - current_ball_pos: the current position of the ball
            - target_pos: the position to move the ball to
            - player_idx: the index of the player (0 or 1)

        Output:
            - True if the move is valid, False otherwise
        """
        start_col, start_row = board_state.decode_single_pos(start_pos)
        end_col, end_row = board_state.decode_single_pos(end_pos)
        
        #Check if there is a straight line(horizontal, vertical, diagonal) between start_pos and end_pos
        d_col = end_col - start_col
        d_row = end_row - start_row
        
        #Not a straight line move
        if not (d_col == 0 or d_row == 0 or abs(d_col) == abs(d_row)):
            return False
        
        #Determine the step direction
        col_step = 0 if d_col == 0 else (1 if d_col > 0 else -1)
        row_step = 0 if d_row == 0 else (1 if d_row > 0 else -1)
        
        #Check all positions between start_pos and end_pos for obstructions
        steps = max(abs(d_col), abs(d_row))
        for step in range(1, steps):
            intermediate_col = start_col + step * col_step
            intermediate_row = start_row + step * row_step
            intermediate_pos = board_state.encode_single_pos((intermediate_col, intermediate_row))
            
             # If any piece sits on an intermediate square, path is blocked.
            if intermediate_pos in board_state.state:
                return False  # Obstruction found
            
            
        return True

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        valid_actions = set()
        offset_idx = player_idx * 6 ## Either 0 or 6
        
        # Get valid moves for all pieces
        for rel_idx in range(5):    
            piece_idx = offset_idx + rel_idx
            piece_moves = Rules.single_piece_actions(self.game_state, piece_idx)
            for move in piece_moves:
                valid_actions.add((rel_idx, move))
                
        # Get valid moves for the ball
        ball_moves = Rules.single_ball_actions(self.game_state, player_idx)
        for move in ball_moves:
            valid_actions.add((5, move))
            
        # Remove actions that don't change the player's portion of the state
        current_positions = [int(self.game_state.state[i]) for i in range(offset_idx, offset_idx + 6)]
        valid_actions = {action for action in valid_actions if action[1] != current_positions[action[0]]}


        return valid_actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("Action must be a tuple of (relative_idx, position).")
        
        rel_idx, pos = action
        
        #Check if relative_idx and position are in valid ranges
        if not isinstance(rel_idx, int) or rel_idx < 0 or rel_idx > 5:
            raise ValueError("relative_idx must be an integer in the range [0, 5].")

        # Check position validity
        max_pos = self.game_state.N_ROWS * self.game_state.N_COLS
        if not isinstance(pos, int) or pos < 0 or pos >= max_pos:
            raise ValueError("Position must be an integer in the valid board range [0, 55].")
        
        # #Check is the position is already occupied
        # if pos in self.game_state.state:
        #     #Allow moving the ball to a position occupied by its own pieces
        #     if rel_idx == 5: # Moving the ball
        #         offset_idx = player_idx * 6
        #         friendly_pieces = [self.game_state.state[i] for i in range(offset_idx, offset_idx + 5)]
        #         if pos not in friendly_pieces:
        #             raise ValueError("Ball can only be moved to a position occupied by its own pieces.")
        #     else: # Moving a block piece
        #         raise ValueError("Position is already occupied by another piece.")
                
        #Get all valid actions and check if the action is in the valid actions
        valid_actions = self.generate_valid_actions(player_idx)
        if action not in valid_actions:
            raise ValueError("Action is not valid according to game rules.")
        
        return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        abs_idx = offset_idx + idx
        
        
        # If moving a block (rel_idx in 0..4): update that block position
        # If moving the ball (rel_idx == 5): update the ball index (which is abs_idx)
        self.game_state.update(abs_idx, pos)
