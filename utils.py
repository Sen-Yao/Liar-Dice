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