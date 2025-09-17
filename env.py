import numpy as np
from typing import List, Tuple, Literal, Optional, Dict
from dataclasses import dataclass, field
import gymnasium
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

# --- Data Structures (as defined before) ---

GuessMode = Literal['飞', '斋']
DiceFace = Literal[1, 2, 3, 4, 5, 6]

@dataclass(frozen=True, order=True)
class Guess:
    """Defines a guess action, made orderable for easy comparison."""
    # Note: order=True requires careful ordering of fields for comparison logic
    # We will implement a custom comparison function for the complex game logic
    mode: GuessMode
    count: int
    face: DiceFace
        
@dataclass(frozen=True)
class Challenge:
    """Defines a challenge action."""
    pass

Action = Guess | Challenge

@dataclass
class AgentState:
    """Defines the observation an agent receives."""
    my_dice_counts: Tuple[int, ...]
    num_players: int
    total_dice_on_table: int
    player_penalties: Tuple[int, ...]
    current_player_id_idx: int
    is_my_turn: bool
    last_guess: Optional[Guess]
    game_round_history: Tuple[Tuple[int, Guess], ...]

# --- Environment Implementation ---

def env(**kwargs):
    """Initializes the PettingZoo environment."""
    env = LiarDiceEnv(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class LiarDiceEnv(AECEnv):
    metadata = {
        "name": "liar_dice_v1",
        "is_parallelizable": False,
        "render_modes": ["human"],
    }

    def __init__(self, num_players: int = 3, dice_per_player: int = 5, render_mode: Optional[str] = None):
        super().__init__()
        assert num_players >= 2, "Game requires at least 2 players"
        
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.total_dice = num_players * dice_per_player
        self.render_mode = render_mode

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # PettingZoo required attributes
        self._action_spaces = {agent: self._get_action_space() for agent in self.possible_agents}
        self._observation_spaces = {agent: self._get_observation_space() for agent in self.possible_agents}
        
        # Game state attributes
        self.player_hands: Dict[str, np.ndarray] = {}
        self.penalties: Dict[str, int] = {}
        self.round_history: List[Tuple[str, Guess]] = []
        self.last_guess: Optional[Guess] = None
        self.last_guesser: Optional[str] = None
        self._round_loser: Optional[str] = None
        
        self._agent_selector = agent_selector(self.possible_agents)
        
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # --- Core PettingZoo Methods ---

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment for a new match."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.penalties = {agent: 0 for agent in self.agents}
        self._round_loser = None # No loser at the start of the match
        self.agent_selection = None # Initialize agent_selection

        self._start_new_round()
        
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.next()

        # PettingZoo's AEC Env requires reset to return None, and initial obs is fetched via observe()
        # However, it's common for learning loops to expect an initial observation from reset.
        # We will handle this by returning observation and info for the first agent
        # The user should call observe() on the first agent_selection.
        
        # Correction: AecEnv `.reset()` populates the required attributes. The training loop then calls `.observe()` on `agent_selection`.
        return

    def step(self, action: Action):
        """
        Executes a single step for the current agent.
        The step is from the perspective of the agent who just acted.
        """
        if self.terminations[self.agent_selection]:
            # If agent is done, we must call step() on it until all agents are done.
            # aec_env.py requires this. A dummy step is performed.
            self._was_dead_step(action)
            return

        agent_id = self.agent_selection
        
        is_round_over = False
        
        # --- Process Action ---
        if isinstance(action, Challenge):
            is_round_over = True
            if self.last_guess is None:
                # Illegal challenge (cannot challenge on the first turn)
                winner, loser = None, agent_id
            else:
                winner, loser = self._resolve_challenge()
        
        elif isinstance(action, Guess):
            if not self._is_legal(action):
                # Illegal guess, agent who made it loses immediately
                is_round_over = True
                winner, loser = self.last_guesser, agent_id
            else:
                # Legal guess, game continues
                self.round_history.append((agent_id, action))
                self.last_guess = action
                self.last_guesser = agent_id
                
                # Move to the next agent
                self.agent_selection = self._agent_selector.next()
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        # --- Handle Round End ---
        if is_round_over:
            self._round_loser = loser
            self.penalties[loser] += 1
            
            # All agents are "done" for this round.
            # Rewards: -1 for loser, +1 for everyone else
            for ag in self.agents:
                self.rewards[ag] = 1 if ag != loser else -1
            
            self.terminations = {agent: True for agent in self.agents}

        # Accumulate rewards
        self._cumulative_rewards[agent_id] = self.rewards[agent_id]
        
        # If the render mode is human, render the state
        if self.render_mode == "human":
            self.render()

        # If the round is over, we might want to start a new one automatically for the next `observe` call
        # PettingZoo loops will see `dones=True` and should call `reset()`.
        # So we don't start a new round here. The training loop manages episodes.

    def observe(self, agent: str) -> Dict:
        """Returns the observation for a specific agent."""
        dice_counts = np.bincount(self.player_hands[agent], minlength=7)[1:] # Index 0 is unused
        
        state = AgentState(
            my_dice_counts=tuple(dice_counts),
            num_players=self.num_players,
            total_dice_on_table=self.total_dice,
            player_penalties=tuple(self.penalties[a] for a in self.possible_agents),
            current_player_id_idx=self.agent_name_mapping[self.agent_selection],
            is_my_turn=(agent == self.agent_selection),
            last_guess=self.last_guess,
            game_round_history=tuple((self.agent_name_mapping[a], g) for a, g in self.round_history)
        )
        return state.__dict__ # Return as a dictionary for Gym space compatibility

    def render(self, mode: str = "human"):
        """Renders the current state of the game."""
        print("\n" + "="*30)
        print(f"       Liar's Dice State       ")
        print("="*30)
        print(f"Penalties: {[f'{agent}: {self.penalties[agent]}' for agent in self.agents]}")
        print(f"Last Guess: {self.last_guess} by {self.last_guesser}")
        print(f"Round History Length: {len(self.round_history)}")
        
        current_agent_id = self.agent_selection
        print(f"\n---> Turn: {current_agent_id} <---")
        
        # For demonstration, show current agent's hand
        if current_agent_id in self.player_hands:
            print(f"Your Hand ({current_agent_id}): {sorted(self.player_hands[current_agent_id])}")
        print(f"-"*30)

    # --- Helper Methods ---
    
    def _start_new_round(self):
        """Initializes state for a new round of the game."""
        self.round_history = []
        self.last_guess = None
        self.last_guesser = None

        for agent in self.agents:
            self.player_hands[agent] = np.random.randint(1, 7, size=self.dice_per_player)
        
        # Determine starting player
        if self._round_loser is None: # First round of match
            start_agent_idx = np.random.randint(self.num_players)
            start_agent = self.possible_agents[start_agent_idx]
        else: # Loser of previous round starts
            start_agent = self._round_loser
        
        self._agent_selector = agent_selector(self.agents) # Reinitialize with current agent list
        self._agent_selector.reset() # Reset to start with first agent
        
        # Advance to the starting agent
        while self.agent_selection != start_agent:
            self.agent_selection = self._agent_selector.next()
        self.agent_selection = start_agent

    def _resolve_challenge(self) -> Tuple[str, str]:
        """Resolves a challenge and returns (winner, loser)."""
        guess = self.last_guess
        guesser = self.last_guesser
        challenger = self.agent_selection

        actual_count = self._count_dice(guess)

        if actual_count >= guess.count:
            # Guess was true, challenger loses
            return guesser, challenger
        else:
            # Guess was false, guesser loses
            return challenger, guesser

    def _count_dice(self, guess: Guess) -> int:
        """Counts the dice on the table according to the guess mode."""
        total = 0
        target_face = guess.face
        
        for hand in self.player_hands.values():
            if guess.mode == '飞':
                # Wild mode: count target face and 1s
                total += np.sum(hand == target_face)
                total += np.sum(hand == 1)
            else: # '斋' mode
                # Exact mode: only count target face
                total += np.sum(hand == target_face)
        return total

    def _is_legal(self, guess: Guess) -> bool:
        """Checks if a guess is legal given the current game state."""
        # Rule: First guess count must be > num_players
        if self.last_guess is None:
            return guess.count > self.num_players
        
        # Rule: New guess must be greater than the last guess
        return self._is_strictly_greater(new_guess=guess, old_guess=self.last_guess)

    @staticmethod
    def _is_strictly_greater(new_guess: Guess, old_guess: Guess) -> bool:
        """Implements the complex comparison logic for guesses."""
        if new_guess.mode == old_guess.mode:
            if new_guess.count > old_guess.count:
                return True
            if new_guess.count == old_guess.count and new_guess.face > old_guess.face:
                return True
            return False

        if new_guess.mode == '斋' and old_guess.mode == '飞':
            # A '斋' guess's value is roughly double
            if (new_guess.count * 2) > old_guess.count:
                return True
            if (new_guess.count * 2) == old_guess.count and new_guess.face > old_guess.face:
                return True # This is a common house rule, assuming it applies
            return False

        if new_guess.mode == '飞' and old_guess.mode == '斋':
            # To go from '斋' to '飞', you need more than half the '斋' count
            required_count = (old_guess.count * 2) + 1
            if new_guess.count > required_count:
                return True
            if new_guess.count == required_count and new_guess.face > old_guess.face:
                return True
            return False
            
        return False
    
    def _get_observation_space(self) -> gymnasium.spaces.Space:
        """Defines the observation space for an agent."""
        # This is a simplified representation. A real NN would need more sophisticated encoding.
        return gymnasium.spaces.Dict({
            "my_dice_counts": gymnasium.spaces.MultiDiscrete([self.dice_per_player + 1] * 6),
            "num_players": gymnasium.spaces.Discrete(self.num_players + 1, start=2),
            "total_dice_on_table": gymnasium.spaces.Discrete(self.num_players * self.dice_per_player + 1),
            "player_penalties": gymnasium.spaces.MultiDiscrete([10] * self.num_players), # Assuming max 10 penalties
            "current_player_id_idx": gymnasium.spaces.Discrete(self.num_players),
            "is_my_turn": gymnasium.spaces.Discrete(2),
            # last_guess and history are complex; they would be tokenized/embedded in a real model
            # For simplicity, we omit them from the formal space definition but they are in the obs dict
            "last_guess": gymnasium.spaces.Dict({
                "mode": gymnasium.spaces.Discrete(2), # 0:飞, 1:斋
                "count": gymnasium.spaces.Discrete(self.total_dice + 1),
                "face": gymnasium.spaces.Discrete(7, start=1)
            }),
            "game_round_history": gymnasium.spaces.Sequence(gymnasium.spaces.Tuple((
                gymnasium.spaces.Discrete(self.num_players), 
                gymnasium.spaces.Box(low=0, high=self.total_dice, shape=(3,)) # (mode, count, face)
                )))
        })

    def _get_action_space(self) -> gymnasium.spaces.Space:
        """
        Defines the action space. This is very complex.
        A common approach is to have one large discrete space and use an action mask.
        Total actions = 1 (challenge) + 2 (modes) * total_dice * 6 (faces)
        We will use a simpler space and rely on the agent to generate valid objects.
        """
        # A proper implementation for a library like SB3 would require flattening this
        # into a single Discrete space with an action mask.
        return gymnasium.spaces.Discrete(1 + (2 * self.total_dice * 6))

# --- Example Usage ---
