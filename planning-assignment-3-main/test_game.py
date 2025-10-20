import pytest
import numpy as np
from game import BoardState, GameSimulator, AdversarialSearchPlayer, RandomPlayer, PassivePlayer
from search import GameStateProblem

class TestGame:
    
    @pytest.mark.parametrize("p1_class, p2_class, encoded_state_tuple,exp_winner,exp_stat",
        [
            # Test against random player
            (AdversarialSearchPlayer, RandomPlayer, (49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
            (AdversarialSearchPlayer, RandomPlayer, (49, 37, 46,  7, 55,  7, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
            (AdversarialSearchPlayer, RandomPlayer, (49, 37, 46,  0, 55,  0, 50, 51, 52, 53, 54, 52), "WHITE", "No issues"),
            
            # Test against passive player  
            (AdversarialSearchPlayer, PassivePlayer, (44, 37, 46, 34, 40, 34,  1,  2, 52,  4,  5, 52), "WHITE", "No issues"),
            (AdversarialSearchPlayer, PassivePlayer, (44, 37, 46, 28, 40, 28,  1,  2, 52,  4,  5, 52), "WHITE", "No issues"),
            
            # Test as black player
            (RandomPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22,  9, 20, 34, 39, 55, 55), "BLACK", "No issues"),
            (RandomPlayer, AdversarialSearchPlayer, (14, 21, 22, 28, 29, 22, 11, 20, 34, 39, 55, 55), "BLACK", "No issues"),
            
            # Extra credit test case (3 points)
            (AdversarialSearchPlayer, RandomPlayer, (35, 30, 38, 42, 47, 42, 15, 16, 17, 18, 19, 17), "WHITE", "No issues"),
        ])
    def test_adversarial_search(self, p1_class, p2_class, encoded_state_tuple, exp_winner, exp_stat):
        b1 = BoardState()
        b1.state = np.array(encoded_state_tuple)
        b1.decode_state = b1.make_state()
        
        gsp1 = GameStateProblem(b1, b1, 0)
        gsp2 = GameStateProblem(b1, b1, 1)
        
        players = [p1_class(gsp1, 0), p2_class(gsp2, 1)]
        sim = GameSimulator(players)
        sim.game_state = b1
        rounds, winner, status = sim.run()
        
        assert winner == exp_winner, f"Expected {exp_winner} but got {winner}"
        assert status == exp_stat, f"Expected status '{exp_stat}' but got '{status}'"
        

# def test_basic_functionality():
#     """Test if adversarial search returns valid actions"""
#     print("Running basic functionality test...")
    
#     # Create a simple board state
#     b1 = BoardState()
#     b1.state = np.array([49, 37, 46, 41, 55, 41, 50, 51, 52, 53, 54, 52])
#     b1.decode_state = b1.make_state()
    
#     # Create GameStateProblem and player
#     gsp = GameStateProblem(b1, b1, 0)
#     player = AdversarialSearchPlayer(gsp, 0, search_depth=2)
    
#     # Test that policy returns valid (action, value) tuple
#     result = player.policy(b1.make_state())
    
#     print(f"Result: {result}")
    
#     # Basic assertions
#     assert result is not None, "Policy returned None"
#     assert isinstance(result, tuple), "Policy should return tuple"
#     assert len(result) == 2, "Policy should return (action, value)"
    
#     action, value = result
#     assert action is not None, "Action should not be None"
#     assert isinstance(action, tuple), "Action should be a tuple"
#     assert len(action) == 2, "Action should be (relative_idx, position)"
    
#     print("✓ Basic functionality test passed!")
#     print(f"✓ Action: {action}")
#     print(f"✓ Value: {value}")

# if __name__ == "__main__":
#     test_basic_functionality()