"""
ISO/IEC 25010 Software Quality Requirements & Rules Engine.
Validates requirement text against syntactic structures, domain keywords,
and identifies quality flaws (ambiguity, weak words, missing constraints).
"""

import re
from typing import Dict, List, Tuple

# ISO/IEC 25010 Quality Characteristics and their diagnostic semantic anchors
ISO_25010_TAXONOMY = {
    'F': {
        'name': 'Functional Suitability',
        'keywords': ['allow', 'calculate', 'provide', 'generate', 'process', 'enable', 'display', 'create', 'update', 'delete', 'store', 'search', 'transmit', 'receive'],
        'subcharacteristics': ['Functional Completeness', 'Functional Correctness', 'Functional Appropriateness']
    },
    'PE': {
        'name': 'Performance Efficiency',
        'keywords': ['seconds', 'milliseconds', 'response time', 'throughput', 'latency', 'refresh', 'bandwidth', 'load', 'capacity', 'transactions per second', 'concurrent', 'memory', 'cpu'],
        'subcharacteristics': ['Time Behavior', 'Resource Utilization', 'Capacity']
    },
    'SE': {
        'name': 'Security',
        'keywords': ['encrypt', 'authenticate', 'authorization', 'credential', 'password', 'role', 'access control', 'confidential', 'privilege', 'tamper', 'audit log', 'firewall', 'token'],
        'subcharacteristics': ['Confidentiality', 'Integrity', 'Non-repudiation', 'Accountability', 'Authenticity']
    },
    'US': {
        'name': 'Usability',
        'keywords': ['user', 'intuitive', 'interface', 'readable', 'font', 'click', 'navigation', 'screen', 'view', 'help', 'feedback', 'guidance', 'accessible', 'error message', 'display'],
        'subcharacteristics': ['Learnability', 'Operability', 'User Error Protection', 'User Interface Aesthetics', 'Accessibility']
    },
    'LF': {
        'name': 'Look and Feel',
        'keywords': ['color', 'font', 'theme', 'appearance', 'branding', 'logo', 'layout', 'styling', 'visual', 'schema', 'palette', 'icon'],
        'subcharacteristics': ['User Interface Aesthetics']
    },
    'A': {
        'name': 'Availability & Reliability',
        'keywords': ['available', 'uptime', '24/7', 'downtime', 'operational hours', '99.9%', 'continuous', 'accessible', 'mtbf', 'scheduled maintenance'],
        'subcharacteristics': ['Availability', 'Maturity', 'Fault Tolerance']
    },
    'FT': {
        'name': 'Fault Tolerance',
        'keywords': ['failover', 'backup', 'recover', 'redundant', 'crash', 'exception', 'fault', 'graceful degradation', 'power failure', 'disconnect', 'fallback'],
        'subcharacteristics': ['Fault Tolerance', 'Recoverability']
    },
    'MN': {
        'name': 'Maintainability',
        'keywords': ['modular', 'maintain', 'configurable', 'upgrade', 'log', 'diagnose', 'extensible', 'reusable', 'code', 'refactor', 'testable', 'plug-in'],
        'subcharacteristics': ['Modularity', 'Reusability', 'Analyzability', 'Modifiability', 'Testability']
    },
    'SC': {
        'name': 'Scalability',
        'keywords': ['scale', 'concurrent users', 'grow', 'expand', 'volume', 'workload', 'elastic', 'horizontal', 'peak load', 'throughput increase'],
        'subcharacteristics': ['Capacity', 'Elasticity']
    },
    'PO': {
        'name': 'Portability',
        'keywords': ['install', 'platform', 'os', 'windows', 'linux', 'ios', 'android', 'browser', 'migrate', 'cross-platform', 'compatibility', 'export', 'import'],
        'subcharacteristics': ['Adaptability', 'Installability', 'Replaceability']
    },
    'L': {
        'name': 'Legal & Regulatory',
        'keywords': ['compliance', 'regulation', 'gdpr', 'hipaa', 'copyright', 'license', 'statutory', 'law', 'standard', 'privacy policy', 'terms', 'fbi', 'dhs'],
        'subcharacteristics': ['Compliance']
    },
    'O': {
        'name': 'Operational & Compatibility',
        'keywords': ['integrate', 'legacy system', 'database', 'protocol', 'api', 'server', 'third-party', 'hardware', 'environment', 'interface', 'interoperate'],
        'subcharacteristics': ['Co-existence', 'Interoperability']
    }
}

# Ambiguous / weak phrases forbidden by ISO/IEC 25010 & IEEE 830 quality guidelines
FORBIDDEN_WEAK_PHRASES = [
    r'\betc\b', r'\band so on\b', r'\bas fast as possible\b', r'\buser-friendly\b',
    r'\bif possible\b', r'\bnormal business\b', r'\bapproximate\b', r'\badequate\b',
    r'\brobust\b', r'\beasy to use\b', r'\bseamless\b', r'\bsomehow\b', r'\bvarious\b'
]

# Strict modal verbs required for formal specification (IEEE 830 / ISO 29148)
VALID_MODAL_VERBS = ['shall', 'must', 'will', 'should', 'required to', 'is able to']


def validate_iso_quality_rules(text: str, target_class: str) -> Tuple[bool, List[str]]:
    """
    Validates whether a requirement text satisfies ISO/IEC 25010 quality standards
    and conforms to the target requirement class.
    
    Returns:
        (is_valid, list_of_reasons)
    """
    violations = []
    text_lower = text.lower().strip()
    
    # 1. Structural Length Check
    words = text_lower.split()
    if len(words) < 5:
        violations.append("Too short (< 5 words), lacks actionable specification")
    if len(words) > 85:
        violations.append("Too verbose (> 85 words), likely compound requirement violating atomicity")
        
    # 2. IEEE 830 / ISO 29148 Modal Verb Check
    has_modal = any(re.search(r'\b' + re.escape(mv) + r'\b', text_lower) for mv in VALID_MODAL_VERBS)
    if not has_modal:
        violations.append("Lacks normative modal verb ('shall', 'must', 'will', 'should')")
        
    # 3. Flaw Detection: Vagueness & Ambiguity Filter
    for weak_regex in FORBIDDEN_WEAK_PHRASES:
        if re.search(weak_regex, text_lower):
            violations.append(f"Contains non-deterministic/vague quality flaw matching: {weak_regex}")
            
    # 4. Semantic Alignment Check against ISO 25010 Taxonomy
    target_class_norm = target_class.upper().strip()
    if target_class_norm in ISO_25010_TAXONOMY:
        expected_keywords = ISO_25010_TAXONOMY[target_class_norm]['keywords']
        matches = [kw for kw in expected_keywords if kw in text_lower]
        # For non-functional requirements (NFRs), require at least one domain characteristic marker
        if target_class_norm != 'F' and len(matches) == 0:
            violations.append(f"Lacks ISO 25010 characteristic markers for class '{target_class_norm}'")
            
    is_valid = len(violations) == 0
    return is_valid, violations
