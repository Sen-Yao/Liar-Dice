from typing import List, Tuple, Dict, Optional
from env import LiarDiceEnv, Guess, Challenge, Action

def is_strictly_greater(new_guess, old_guess) -> bool:
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

def get_legal_actions(observation: Dict, num_players: int) -> List[Action]:
    """Get all legal actions for current observation"""
    legal_actions = []
    total_dice = 5 * num_players
    last_guess = observation['last_guess']

    # Always add challenge if there's a previous guess
    if last_guess is not None:
        legal_actions.append(Challenge())

    # Add legal guesses
    for mode in ['飞', '斋']:
        for count in range(1, total_dice + 1):
            for face in range(1, 7):
                if mode == '飞' and face == 1:
                    continue  # Flying mode can't guess face 1

                guess = Guess(mode=mode, count=count, face=face)

                if last_guess is None:
                    if count >= num_players:  # First guess must be at least num_players
                        legal_actions.append(guess)
                elif is_strictly_greater(guess, last_guess):
                    legal_actions.append(guess)

    return legal_actions