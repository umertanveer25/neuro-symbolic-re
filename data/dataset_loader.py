"""
Dataset loader and standardizer for Requirements Engineering datasets:
- Promise_Exp.csv (Training and primary benchmark)
- FNFC.csv (External generalization dataset 1)
- nfr.csv (External generalization dataset 2)
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Tuple, Dict

# Unified ISO/IEC 25010 class labels (12 classes standard)
VALID_CLASSES = ['F', 'SE', 'US', 'O', 'PE', 'LF', 'A', 'MN', 'SC', 'FT', 'L', 'PO']

# Deterministic ISO/IEC 25010 taxonomy alignment mapping for external FNFC corpus (14 classes -> 12 standard classes)
# Mapped strictly according to ISO/IEC 25010:2011 Clause 4.2:
# - AU (Auditability)        -> SE (ISO 25010 Section 4.2.6: Security, Authenticity & Non-repudiation)
# - P (Performance)          -> PE (ISO 25010 Section 4.2.2: Performance Efficiency)
# - R (Reliability)          -> A  (ISO 25010 Section 4.2.5: Reliability & Availability)
# - LL (Legal)               -> L  (ISO 25010 Section 4.2.7: Legal, Regulatory & Compliance)
# - M (Maintainability)      -> MN (ISO 25010 Section 4.2.7: Maintainability, Modularity & Analysability)
FNFC_CLASS_MAPPING = {
    'F': 'F',    # Functional Suitability
    'SE': 'SE',  # Security
    'AU': 'SE',  # Auditability -> Security (ISO 25010 Clause 4.2.6)
    'P': 'PE',   # Performance -> Performance Efficiency (ISO 25010 Clause 4.2.2)
    'PE': 'PE',  # Performance Efficiency
    'A': 'A',    # Availability (ISO 25010 Clause 4.2.5)
    'R': 'A',    # Reliability -> Availability & Reliability (ISO 25010 Clause 4.2.5)
    'US': 'US',  # Usability
    'O': 'O',    # Operational & Compatibility
    'LL': 'L',   # Legal & Regulatory (ISO 25010 Clause 4.2.7)
    'L': 'L',
    'M': 'MN',   # Maintainability -> Maintainability & Modularity (ISO 25010 Clause 4.2.7)
    'MN': 'MN',
    'LF': 'LF',  # Look and Feel
    'FT': 'FT',  # Fault Tolerance
    'SC': 'SC',  # Scalability
}


def get_taxonomy_mapping_table() -> pd.DataFrame:
    """Exports formal taxonomy alignment table between FNFC and PROMISE/ISO 25010."""
    mapping_data = [
        ('F', 'Functional', 'F', 'Functional Suitability (ISO 25010 §4.2.1)'),
        ('SE', 'Security', 'SE', 'Security (ISO 25010 §4.2.6)'),
        ('AU', 'Auditability', 'SE', 'Security & Non-repudiation (ISO 25010 §4.2.6)'),
        ('P', 'Performance', 'PE', 'Performance Efficiency (ISO 25010 §4.2.2)'),
        ('PE', 'Performance Efficiency', 'PE', 'Performance Efficiency (ISO 25010 §4.2.2)'),
        ('A', 'Availability', 'A', 'Availability & Reliability (ISO 25010 §4.2.5)'),
        ('R', 'Reliability', 'A', 'Availability & Reliability (ISO 25010 §4.2.5)'),
        ('US', 'Usability', 'US', 'Usability (ISO 25010 §4.2.4)'),
        ('O', 'Operational', 'O', 'Compatibility & Operational (ISO 25010 §4.2.3)'),
        ('LL', 'Legal', 'L', 'Legal & Regulatory Compliance (ISO 25010 §4.2.7)'),
        ('L', 'Legal', 'L', 'Legal & Regulatory Compliance (ISO 25010 §4.2.7)'),
        ('M', 'Maintainability', 'MN', 'Maintainability & Modularity (ISO 25010 §4.2.7)'),
        ('MN', 'Maintainability', 'MN', 'Maintainability & Modularity (ISO 25010 §4.2.7)'),
        ('LF', 'Look and Feel', 'LF', 'Look and Feel / Aesthetics'),
        ('FT', 'Fault Tolerance', 'FT', 'Fault Tolerance (ISO 25010 §4.2.5)'),
        ('SC', 'Scalability', 'SC', 'Scalability & Capacity (ISO 25010 §4.2.2)')
    ]
    return pd.DataFrame(mapping_data, columns=['FNFC_Label', 'FNFC_Concept', 'PROMISE_Label', 'ISO_25010_Clause'])



def clean_text(text: str) -> str:
    """Standardizes requirement text cleaning."""
    if not isinstance(text, str):
        return ""
    # Strip leading/trailing whitespace and surrounding quotes
    text = text.strip().strip("'").strip('"')
    # Replace escaped quotes or weird formatting
    text = text.replace('\\%', '%').replace('\\"', '"').replace("''", "'")
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_promise_exp(file_path: str = r"B:\Dataset\Promise_Exp.csv") -> pd.DataFrame:
    """Loads and standardizes Promise_Exp.csv."""
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.rename(columns={'Requirement': 'text', 'Type': 'label'})
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[df['label'].isin(VALID_CLASSES) & (df['text'].str.len() > 10)].copy()
    df = df.reset_index(drop=True)
    return df[['text', 'label']]


def load_fnfc(file_path: str = r"B:\Dataset\FNFC.csv") -> pd.DataFrame:
    """Loads and standardizes FNFC.csv with ontology mapping."""
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.rename(columns={'text': 'text', 'class': 'raw_label'})
    df['text'] = df['text'].apply(clean_text)
    df['raw_label'] = df['raw_label'].astype(str).str.strip().str.upper()
    df['label'] = df['raw_label'].map(FNFC_CLASS_MAPPING)
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin(VALID_CLASSES) & (df['text'].str.len() > 10)].copy()
    df = df.reset_index(drop=True)
    return df[['text', 'label']]


def load_nfr(file_path: str = r"B:\Dataset\nfr.csv") -> pd.DataFrame:
    """Loads and standardizes nfr.csv."""
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df.rename(columns={'RequirementText': 'text', 'class': 'label'})
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].astype(str).str.strip().str.upper()
    df = df[df['label'].isin(VALID_CLASSES) & (df['text'].str.len() > 10)].copy()
    df = df.reset_index(drop=True)
    return df[['text', 'label']]


if __name__ == '__main__':
    p_df = load_promise_exp()
    print(f"Loaded Promise_Exp: {len(p_df)} samples, {p_df['label'].nunique()} classes.")
    fnfc_df = load_fnfc()
    print(f"Loaded FNFC: {len(fnfc_df)} samples, {fnfc_df['label'].nunique()} classes.")
    nfr_df = load_nfr()
    print(f"Loaded nfr: {len(nfr_df)} samples, {nfr_df['label'].nunique()} classes.")
