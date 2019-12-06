
from e2cnn.nn import FieldType
from typing import List, Dict, Tuple
from collections import defaultdict


__all__ = ["check_consecutive_numbers", "indexes_from_labels"]


def check_consecutive_numbers(list: List[int]) -> bool:
    
    m = M = list[0]
    s = 0
    
    for l in list:
        assert l >= 0
        m = min(m, l)
        M = max(M, l)
        s += l
    
    S = M*(M+1)/2 - (m-1)*m/2
    
    return S == s
    
    
def indexes_from_labels(in_type: FieldType, labels: List[str]) -> Dict[str, Tuple[bool, List[int], List[int]]]:
    r"""
    
    
    Args:
        in_type (FieldType): the input field type
        labels (list): a list of strings long as the list :attr:'~e2cnn.nn.FieldType.representations`
                of the input :attr:`in_type`

    Returns:

    """
    assert len(labels) == len(in_type)
    
    indeces = defaultdict(lambda: [])
    fields = defaultdict(lambda: [])
    
    current_position = 0
    for c, (l, r) in enumerate(zip(labels, in_type.representations)):
        # append the indeces of the current field to the list corresponding to this label
        indeces[l] += list(range(current_position, current_position + r.size))
        fields[l].append(c)
        current_position += r.size
    
    groups = {}
    
    for l in labels:
        contiguous = check_consecutive_numbers(indeces[l])
        groups[l] = contiguous, fields[l], indeces[l]
    
    return groups

