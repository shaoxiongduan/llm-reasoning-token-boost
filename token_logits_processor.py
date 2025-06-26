import torch
from transformers import AutoTokenizer


class TokenLogitsProcessor:
    def __init__(self, token_id_boosts=None):
        """
        Initialize logits processor with token ID boosts.
        
        Args:
            token_id_boosts: Dict[int, float] - mapping of token_id -> boost_factor
        """
        self.token_id_boosts = token_id_boosts or {}
        print(f"Token ID boost mappings: {self.token_id_boosts}")
    
    def __call__(self, past_token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Apply token ID boosts to logits.
        
        Args:
            past_token_ids: List of previously generated token IDs
            logits: Logits tensor of shape [vocab_size]
            
        Returns:
            Modified logits tensor
        """
        # Apply boosts to specific token IDs
        for token_id, boost_factor in self.token_id_boosts.items():
            if 0 <= token_id < logits.shape[-1]:  # Safety check for valid token ID
                logits[token_id] *= boost_factor
        
        return logits


def find_token_ids(tokenizer, strings):
    """
    Helper function to find token IDs for given strings.
    Tries multiple encoding variations to capture different contexts.
    
    Args:
        tokenizer: The tokenizer to use
        strings: List of strings to find token IDs for
        
    Returns:
        Dict[str, List[int]] - mapping of string -> list of possible token IDs
    """
    result = {}
    
    # Common generic tokens to exclude (spaces, punctuation, etc.)
    generic_tokens = set()
    for generic_str in [' ', '\t', '\n', '.', ',', '!', '?', ':', ';']:
        generic_tokens.update(tokenizer.encode(generic_str, add_special_tokens=False))
    
    for string in strings:
        token_ids = set()
        
        # Try different variations
        variations = [
            string,           # Direct
            " " + string,     # With leading space
            string + " ",     # With trailing space
            " " + string + " ", # With both spaces
            string + ",",     # With trailing comma
            string + ", ",    # With trailing comma and space
            " " + string + ",", # With leading space and trailing comma
            " " + string + ", ", # With leading space, trailing comma and space
            string + ".",     # With trailing period
            string + ". ",    # With trailing period and space
            " " + string + ".", # With leading space and trailing period
            " " + string + ". " # With leading space, trailing period and space
        ]
        
        for variation in variations:
            encoded = tokenizer.encode(variation, add_special_tokens=False)
            token_ids.update(encoded)
        
        # Filter out generic tokens like spaces
        filtered_token_ids = [tid for tid in token_ids if tid not in generic_tokens]
        
        result[string] = sorted(filtered_token_ids)
        # Convert token IDs back to strings for display
        token_strings = [repr(tokenizer.decode([token_id])) for token_id in result[string]]
        print(f"'{string}' -> token IDs: {result[string]} -> tokens: {token_strings}")
    
    return result


def create_reasoning_token_logits_processor(tokenizer, boost_factor=5.0, penalty_factor=-5.0):
    """
    Create a logits processor that boosts reasoning tokens and penalizes certain unwanted tokens.
    
    Args:
        tokenizer: The tokenizer to use for finding token IDs
        boost_factor: Boost factor for reasoning tokens
        penalty_factor: Penalty factor for unwanted tokens
        
    Returns:
        TokenLogitsProcessor: Configured logits processor
    """
    # Find token IDs for reasoning-related strings
    reasoning_strings = ["think", "reason", "analyze", "consider", "however", "but", "instead", "Think", "Reason", "Analyze", "Consider", "However", "But", "Instead"]
    unwanted_strings = []
    
    reasoning_mappings = find_token_ids(tokenizer, reasoning_strings)
    unwanted_mappings = find_token_ids(tokenizer, unwanted_strings)
    
    token_id_boosts = {}
    
    # Add boosts for reasoning tokens
    for token_ids in reasoning_mappings.values():
        for token_id in token_ids:
            token_id_boosts[token_id] = boost_factor
    
    # Add penalties for unwanted tokens
    for token_ids in unwanted_mappings.values():
        for token_id in token_ids:
            token_id_boosts[token_id] = penalty_factor
    
    print(f"\nFinal token_id_boosts: {token_id_boosts}")
    
    return TokenLogitsProcessor(token_id_boosts=token_id_boosts) 