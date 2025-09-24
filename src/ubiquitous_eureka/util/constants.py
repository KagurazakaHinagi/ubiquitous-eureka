"""Constants used throughout the codebase."""

from typing import Final
from types import MappingProxyType

from toolz import keymap

# Constants for Basic Chemistry

UNKNOWN_ELEMENT: Final[str] = "X"
UNKNOWN_ATOMIC_NUMBER: Final[int] = 0

# fmt: off

ELEMENT_NAME_TO_ATOMIC_NUMBER: Final[MappingProxyType[str, int]] = MappingProxyType(keymap(str.upper, {
    "H": 1,    "He": 2,   "Li": 3,   "Be": 4,   "B": 5,   "C": 6,   "N": 7,    "O": 8,    "F": 9,   "Ne": 10,
    "Na": 11,  "Mg": 12,  "Al": 13,  "Si": 14,  "P": 15,  "S": 16,  "Cl": 17,  "Ar": 18,  "K": 19,  "Ca": 20,
    "Sc": 21,  "Ti": 22,  "V": 23,   "Cr": 24,  "Mn": 25, "Fe": 26, "Co": 27,  "Ni": 28,  "Cu": 29, "Zn": 30,
    "Ga": 31,  "Ge": 32,  "As": 33,  "Se": 34,  "Br": 35, "Kr": 36, "Rb": 37,  "Sr": 38,  "Y": 39,  "Zr": 40,
    "Nb": 41,  "Mo": 42,  "Tc": 43,  "Ru": 44,  "Rh": 45, "Pd": 46, "Ag": 47,  "Cd": 48,  "In": 49, "Sn": 50,
    "Sb": 51,  "Te": 52,  "I": 53,   "Xe": 54,  "Cs": 55, "Ba": 56, "La": 57,  "Ce": 58,  "Pr": 59, "Nd": 60,
    "Pm": 61,  "Sm": 62,  "Eu": 63,  "Gd": 64,  "Tb": 65, "Dy": 66, "Ho": 67,  "Er": 68,  "Tm": 69, "Yb": 70,
    "Lu": 71,  "Hf": 72,  "Ta": 73,  "W": 74,   "Re": 75, "Os": 76, "Ir": 77,  "Pt": 78,  "Au": 79, "Hg": 80,
    "Tl": 81,  "Pb": 82,  "Bi": 83,  "Po": 84,  "At": 85, "Rn": 86, "Fr": 87,  "Ra": 88,  "Ac": 89, "Th": 90,
    "Pa": 91,  "U": 92,   "Np": 93,  "Pu": 94,  "Am": 95, "Cm": 96, "Bk": 97,  "Cf": 98,  "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,"Sg": 106, "Bh": 107,"Hs": 108, "Mt": 109,"Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
    UNKNOWN_ELEMENT: UNKNOWN_ATOMIC_NUMBER
}))

ATOMIC_NUMBER_TO_ELEMENT: Final[MappingProxyType[int | str, str]] = MappingProxyType(
    {v: k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()} |
    {str(v): k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()}
)

METAL_ELEMENTS: Final[frozenset[str]] = frozenset(map(str.upper, [
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
]))

# fmt: on

# Constants for Protein
BACKBONE_ATOMS: Final[tuple[str, ...]] = ("N", "CA", "C", "O")

STANDARD_AA: Final[tuple[str, ...]] = tuple(
    sorted(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
    )
)

DICT_THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
    " * ": "*",
}

# Constants for Biotite AtomArray
ATOM_ARRAY_MANDATORY_ANNOTATIONS: Final[tuple[str, ...]] = (
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
)

# TEMPORARY
ATOM_TYPES: Final[tuple[str, ...]] = (
    "C", "N", "O", "S", "P",           # Common biological atoms
    "Mg", "Ca", "Zn", "Fe", "Mn",      # Metal ions
    "Cl", "Br", "I", "F",              # Halogens
    "Na", "K", "Se",                   # Others
    "SOLVENT", "UNKNOWN"               # Special cases
)


# Physical constraint constants for common atoms
COMMON_BOND_LENGTHS: Final[dict[tuple[str, str], tuple[float, float]]] = {
    ('C', 'C'): (1.40, 1.65),   ('C', 'N'): (1.25, 1.50),   ('C', 'O'): (1.20, 1.50),
    ('C', 'S'): (1.70, 1.90),   ('N', 'N'): (1.10, 1.50),   ('N', 'O'): (1.15, 1.45),
    ('O', 'O'): (1.20, 1.50),   ('S', 'S'): (2.00, 2.10),   ('C', 'P'): (1.75, 1.90),
    ('N', 'P'): (1.65, 1.80),   ('O', 'P'): (1.45, 1.70),
    # Metal coordination bonds
    ('Mg', 'O'): (1.90, 2.30),  ('Ca', 'O'): (2.20, 2.60),  ('Zn', 'N'): (1.90, 2.20),
    ('Zn', 'O'): (1.90, 2.20),  ('Fe', 'N'): (1.90, 2.20),  ('Fe', 'O'): (1.90, 2.20),
}

COMMON_BOND_ANGLES: Final[dict[tuple[str, str, str], tuple[float, float]]] = {
    ('C', 'C', 'C'): (100.0, 130.0),  ('C', 'N', 'C'): (100.0, 125.0),
    ('C', 'O', 'C'): (100.0, 125.0),  ('N', 'C', 'O'): (110.0, 130.0),
    ('O', 'P', 'O'): (100.0, 120.0),
} # in degrees

COMMON_COORDINATION_NUMBERS: Final[dict[str, int]] = {
    'C': 4, 'N': 3, 'O': 2, 'S': 2, 'P': 4,
    'Mg': 6, 'Ca': 8, 'Zn': 4, 'Fe': 6, 'Mn': 6
}

COMMON_VAN_DER_WAALS_RADII: Final[dict[str, float]] = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'Mg': 1.73, 'Ca': 2.31, 'Zn': 1.39, 'Fe': 1.32, 'Mn': 1.35,
    'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'F': 1.47,
}