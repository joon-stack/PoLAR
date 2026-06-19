"""Dataset mixtures used by the PoLAR public training/evaluation path."""

from typing import Dict, List, Tuple


OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    "bridge": [
        ("bridge_dataset", 1.0),
    ],
    "simpler": [
        ("carrot", 1.0),
        ("eggplant", 1.0),
        ("spoon", 1.0),
        ("stack", 1.0),
    ],
}
