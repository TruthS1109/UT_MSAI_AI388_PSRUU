import numpy as np
import pytest
from bayesian import StateGenerator, sample_observation, sample_transition, belief_update, belief_predict, initialize_belief

class TestBayesianInference:

    def test_sample_observation_basic(self):
        """Test that observation sampling returns valid outputs"""
        state = ([(1, 1), (2, 2)], (3, 3))  # Simple 3x3 board
        obs_pos, obs_dist = sample_observation(state)
        
        # Check return types
        assert isinstance(obs_pos, tuple) and len(obs_pos) == 2
        assert isinstance(obs_dist, np.ndarray)
        assert obs_dist.shape == (3, 3)
        
        # Check distribution properties
        assert np.all(obs_dist >= 0)
        assert np.isclose(np.sum(obs_dist), 1.0)
        
        # Check observation is within bounds
        assert 0 <= obs_pos[0] < 3 and 0 <= obs_pos[1] < 3

    def test_sample_observation_blocked_cells(self):
        """Test observation model with blocked adjacent cells"""
        # Piece at (1,1) with blocked right side (piece at (2,1))
        state = ([(1, 1), (2, 1)], (3, 3))
        obs_pos, obs_dist = sample_observation(state)
        
        # Right cell (2,1) should have 0 probability (blocked by piece)
        assert obs_dist[1, 2] == 0
        
        # Check distribution sums to 1
        assert np.isclose(np.sum(obs_dist), 1.0)

    def test_sample_transition_valid(self):
        """Test valid transitions"""
        state = ([(1, 1), (3, 3)], (4, 4))
        action = (1, 0)  # Move right
        new_pos, trans_dist = sample_transition(state, action)
        
        # Should move to (2, 1)
        assert new_pos == (2, 1)
        
        # Transition distribution should be deterministic
        assert trans_dist[1, 2] == 1.0  # row 1, col 2
        assert np.sum(trans_dist) == 1.0

    def test_sample_transition_invalid(self):
        """Test invalid transitions (moving into occupied cell)"""
        state = ([(1, 1), (2, 1)], (4, 4))  # Piece at (1,1), obstacle at (2,1)
        action = (1, 0)  # Move right into occupied cell
        new_pos, trans_dist = sample_transition(state, action)
        
        # Should return None for invalid move
        assert new_pos is None
        # Distribution should be all zeros
        assert np.all(trans_dist == 0)

    def test_initialize_belief_uniform(self):
        """Test uniform belief initialization"""
        state = ([(1, 1), (2, 2)], (3, 3))
        belief = initialize_belief(state, "uniform")
        
        # Should have shape (3, 3)
        assert belief.shape == (3, 3)
        
        # Occupied cells should have 0 probability
        assert belief[1, 1] == 0  # Piece 0 position
        assert belief[2, 2] == 0  # Other piece position
        
        # Free cells should have uniform probability
        free_cells = 3*3 - 2  # 7 free cells
        expected_prob = 1.0 / free_cells
        free_mask = (belief > 0)
        assert np.allclose(belief[free_mask], expected_prob)
        assert np.isclose(np.sum(belief), 1.0)

    def test_initialize_belief_dirac(self):
        """Test dirac belief initialization"""
        state = ([(1, 1), (2, 2)], (3, 3))
        belief = initialize_belief(state, "dirac")
        
        # Should have probability 1 at piece 0 position
        assert belief[1, 1] == 1.0
        # All other cells should be 0
        assert np.sum(belief) == 1.0

    def test_belief_update(self):
        """Test belief update with observation"""
        state = ([(1, 1), (2, 2)], (3, 3))
        prior = initialize_belief(state, "uniform")
        observation = (1, 1)  # Observe at true position
        
        posterior = belief_update(prior, observation, state)
        
        # Posterior should have same shape
        assert posterior.shape == (3, 3)
        # Should still be a valid probability distribution
        assert np.isclose(np.sum(posterior), 1.0)
        assert np.all(posterior >= 0)
        # Posterior should be different from prior
        assert not np.allclose(posterior, prior)

    def test_belief_predict(self):
        """Test belief prediction with action"""
        state = ([(1, 1), (3, 3)], (4, 4))
        # Start with dirac delta at (1,1)
        prior = initialize_belief(state, "dirac")
        action = (1, 0)  # Move right
        
        posterior = belief_predict(prior, action, state)
        
        # After moving right from (1,1), should be at (2,1)
        assert posterior[1, 2] == 1.0  # row 1, col 2
        assert np.sum(posterior) == 1.0

    def test_belief_predict_invalid_move(self):
        """Test belief prediction with invalid move"""
        state = ([(1, 1), (2, 1)], (4, 4))  # Blocked to the right
        prior = initialize_belief(state, "dirac")
        action = (1, 0)  # Move right into blocked cell
        
        posterior = belief_predict(prior, action, state)
        
        # Invalid move should result in zero distribution
        assert np.all(posterior == 0)

    @pytest.mark.parametrize("initial_state,observation_list,prior_style", [
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(3,4),(3,4)], "uniform",
        ),
    ])
    def test_example_observations(self, initial_state, observation_list, prior_style):
        """Test Bayesian update with multiple observations"""
        belief = initialize_belief(initial_state, prior_style)
        
        for observation in observation_list:
            # Store previous belief for comparison
            prev_belief = belief.copy()
            
            # Update belief with observation
            belief = belief_update(belief, observation, initial_state)
            
            # Check belief properties
            assert belief.shape == (8, 7)
            assert np.isclose(np.sum(belief), 1.0)
            assert np.all(belief >= 0)
            
            # Belief should change after update (unless observation is impossible)
            if not np.all(prev_belief == 0):
                assert not np.allclose(belief, prev_belief)

    @pytest.mark.parametrize("initial_state,action_list,prior_style", [
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0)], "uniform",
        ),
        (
            ([(3, 4), (6, 4), (3, 7), (5, 1), (0, 3), (1, 0), (2, 5), (5, 5), (1, 3), (4, 7)], (8, 7)),
            [(0,0),(0,1)], "uniform",
        ),
    ])
    def test_example_actions(self, initial_state, action_list, prior_style):
        """Test Bayesian prediction with multiple actions"""
        belief = initialize_belief(initial_state, prior_style)
        
        for action in action_list:
            # Store previous belief for comparison
            prev_belief = belief.copy()
            
            # Predict new belief after action
            belief = belief_predict(belief, action, initial_state)
            
            # Check belief properties
            assert belief.shape == (8, 7)
            assert np.all(belief >= 0)
            
            # For valid moves, probability mass should be preserved
            if not np.all(belief == 0):
                assert np.isclose(np.sum(belief), 1.0)

    def test_integration_full_bayesian_filter(self):
        """Test complete Bayesian filter cycle: predict -> update"""
        state = ([(1, 1), (3, 3)], (4, 4))
        belief = initialize_belief(state, "dirac")
        
        # Action: move right
        action = (1, 0)
        belief = belief_predict(belief, action, state)
        
        # Should now believe piece is at (2, 1)
        assert belief[1, 2] == 1.0
        
        # Observation: see piece at (2, 1)
        observation = (2, 1)
        belief = belief_update(belief, observation, state)
        
        # Belief should reinforce the correct position
        assert belief[1, 2] > 0.5  # High probability at true position

    def test_edge_case_all_occupied(self):
        """Test edge case where board is nearly full"""
        # Create a state with only one free cell
        state = ([(0, 0), (0, 1), (1, 0)], (2, 2))
        # Only free cell is (1,1)
        
        belief = initialize_belief(state, "uniform")
        
        # Only one free cell, so probability should be 1.0 there
        assert belief[1, 1] == 1.0
        assert np.sum(belief) == 1.0

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-k", "test_bayesian"])