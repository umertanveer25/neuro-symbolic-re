"""
EARS (Easy Approach to Requirements Syntax) Rule Engine.
Validates requirements conformity to standardized aerospace/software engineering patterns:
- Ubiquitous (The <system> shall ...)
- Event-driven (When <trigger>, the <system> shall ...)
- State-driven (While <in state>, the <system> shall ...)
- Optional Feature (Where <feature>, the <system> shall ...)
- Unwanted Behavior (If <error condition>, then the <system> shall ...)
"""

import re
from typing import Dict, Any, Optional

EARS_PATTERNS = {
    'ubiquitous': re.compile(r'^(the\s+[\w\s\-]+?\s+(shall|must|will)\s+.+)$', re.IGNORECASE),
    'event_driven': re.compile(r'^(when\s+[\w\s\-]+?,\s*(the\s+[\w\s\-]+?\s+(shall|must|will)\s+.+))$', re.IGNORECASE),
    'state_driven': re.compile(r'^(while\s+[\w\s\-]+?,\s*(the\s+[\w\s\-]+?\s+(shall|must|will)\s+.+))$', re.IGNORECASE),
    'optional_feature': re.compile(r'^(where\s+[\w\s\-]+?,\s*(the\s+[\w\s\-]+?\s+(shall|must|will)\s+.+))$', re.IGNORECASE),
    'unwanted_behavior': re.compile(r'^(if\s+[\w\s\-]+?,\s*(then\s+)?(the\s+[\w\s\-]+?\s+(shall|must|will)\s+.+))$', re.IGNORECASE)
}


def analyze_ears_pattern(requirement_text: str) -> Dict[str, Any]:
    """
    Checks if requirement conforms to an EARS syntactic pattern.
    Returns:
        {
            'is_ears_compliant': bool,
            'pattern': str, # 'ubiquitous', 'event_driven', etc.
            'confidence': float
        }
    """
    cleaned = requirement_text.strip().strip('"').strip("'")
    
    for pattern_name, regex in EARS_PATTERNS.items():
        if regex.match(cleaned):
            return {
                'is_ears_compliant': True,
                'pattern': pattern_name,
                'confidence': 1.0
            }
            
    # Soft match: Contains '<system> shall <verb>' anywhere
    soft_regex = re.compile(r'(the\s+[\w\s\-]+?\s+(shall|must|will)\s+\w+)', re.IGNORECASE)
    if soft_regex.search(cleaned):
        return {
            'is_ears_compliant': True,
            'pattern': 'quasi_ubiquitous',
            'confidence': 0.8
        }
        
    return {
        'is_ears_compliant': False,
        'pattern': 'non_standard',
        'confidence': 0.0
    }
