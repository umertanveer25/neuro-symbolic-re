"""
Enhanced Neuro-Symbolic Requirement Generator.
Massively expanded lexical slots and combinatorial EARS syntactic patterns
aligned with ISO/IEC 25010 to provide high-entropy, diverse vocabularies
for linear models (TF-IDF, Logistic Regression, SVM) and tree models alike.
"""

import random
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
from symbolic.iso25010_rules import validate_iso_quality_rules, ISO_25010_TAXONOMY
from symbolic.ears_syntax import analyze_ears_pattern

SYSTEM_COMPONENTS = [
    "The system", "The web application", "The client interface", "The API gateway",
    "The microservice architecture", "The database engine", "The security module",
    "The authentication service", "The backend pipeline", "The reporting service",
    "The transaction processor", "The distributed server cluster", "The user dashboard",
    "The background scheduler", "The communication layer", "The data persistence subsystem",
    "The cryptographic key manager", "The telemetry ingestion pipeline", "The microkernel subsystem",
    "The access management gateway", "The container orchestrator", "The distributed message broker",
    "The identity provider", "The event processing engine", "The session management service",
    "The distributed cache cluster", "The analytics calculation engine", "The disaster recovery node"
]

MODAL_OPERATORS = ["shall", "must"]

OPERATIONAL_MODES = [
    "under peak concurrency", "during sustained peak load", "under maximum operational stress",
    "during steady-state operations", "under simulated network degradation", "during failover conditions",
    "under 95th percentile workload", "during batch synchronization windows"
]

# Rich combinatorial templates per ISO 25010 quality characteristic
ADVANCED_TEMPLATES = {
    'PE': [
        "{subj} {modal} process incoming user transactions within {v_ms} milliseconds {op_mode}.",
        "{subj} {modal} maintain an end-to-end response latency of less than {v_s} seconds {op_mode}.",
        "{subj} {modal} refresh dynamic visual telemetry every {v_s} seconds to preserve interface accuracy.",
        "When handling batch data uploads, {subj} {modal} execute records at a minimum sustained throughput of {v_t} records per second.",
        "While under ninety percent server workload, {subj} {modal} maintain memory utilization below {v_pct} percent.",
        "{subj} {modal} complete complex analytical search queries within {v_s} seconds across ten million records.",
        "When multiple concurrent requests are dispatched, {subj} {modal} respond to ninety-five percent of queries in under {v_ms} milliseconds.",
        "{subj} {modal} optimize network bandwidth consumption to not exceed {v_bw} megabytes per second during video streaming.",
        "While processing payment authorizations, {subj} {modal} enforce strict computational turnaround within {v_ms} milliseconds."
    ],
    'SE': [
        "{subj} {modal} encrypt all persisted database records and sensitive credentials using AES-256 cryptographic standards.",
        "{subj} {modal} enforce multi-factor authentication protocols prior to granting administrative access privileges.",
        "{subj} {modal} enforce granular role-based access control policies to restrict unauthorized schema modification.",
        "When an invalid credential entry occurs {v_num} consecutive times, {subj} {modal} lock the account and generate an audit log.",
        "{subj} {modal} transmit all network communications across public domains exclusively via secure TLS 1.3 encrypted tunnels.",
        "{subj} {modal} mask confidential payment card information and personally identifiable data in accordance with PCI-DSS.",
        "{subj} {modal} maintain immutable audit trails of all user authorization attempts and administrative privilege escalations.",
        "When session inactivity exceeds {v_min} minutes, {subj} {modal} automatically terminate active sessions to prevent session hijacking.",
        "{subj} {modal} sanitize all input parameters to prevent SQL injection and cross-site scripting vulnerabilities."
    ],
    'US': [
        "{subj} {modal} display contextual diagnostic guidance within {v_s} seconds following an invalid input submission.",
        "{subj} {modal} enable primary workflow execution within no more than {v_num} intuitive navigation clicks from the homepage.",
        "{subj} {modal} render all quantitative data views using high-contrast color palettes and scalable typography for accessibility.",
        "When a form validation error occurs, {subj} {modal} visually highlight the exact interactive field requiring user correction.",
        "{subj} {modal} provide clear visual progress indicators during operations requiring more than {v_s} seconds to complete.",
        "{subj} {modal} support keyboard-only navigation across all dialogue windows in compliance with WCAG 2.1 accessibility guidelines.",
        "{subj} {modal} prompt the user with a confirmation dialogue prior to irreversible record deletion.",
        "While navigating complex configuration workflows, {subj} {modal} present persistent step-by-step breadcrumb feedback."
    ],
    'A': [
        "{subj} {modal} maintain an operational service availability of at least 99.{v_pct}% during scheduled core operational hours.",
        "{subj} {modal} guarantee continuous service uptime, limiting unplanned downtime to less than {v_min} minutes per calendar month.",
        "When unscheduled server hardware faults occur, {subj} {modal} restore full user service within {v_min} minutes.",
        "{subj} {modal} be available for transactional processing twenty-four hours a day, seven days a week.",
        "While undergoing automated background database backup, {subj} {modal} remain accessible in uninterrupted read-only mode.",
        "{subj} {modal} achieve a mean time between failures (MTBF) exceeding {v_hrs} continuous operational hours."
    ],
    'FT': [
        "If primary network connectivity fails, then {subj} {modal} automatically switch to redundant backup connections without data loss.",
        "When an unhandled runtime exception arises, {subj} {modal} execute graceful degradation and record a detailed diagnostic stack trace.",
        "If a server node crashes unexpectedly, then {subj} {modal} initiate automated failover to standby replicas within {v_s} seconds.",
        "{subj} {modal} recover transactional integrity from write-ahead database logs upon system restart following power interruption.",
        "If an external web service becomes unresponsive, then {subj} {modal} isolate the fault and provide cached responses.",
        "When storage disk corruption is detected, {subj} {modal} seamlessly redirect I/O operations to mirror redundant arrays."
    ],
    'MN': [
        "{subj} {modal} be architected into decoupled modular components to facilitate independent unit testing and automated continuous deployment.",
        "{subj} {modal} externalize runtime configuration properties to permit threshold modifications without recompilation of source code.",
        "{subj} {modal} produce standardized diagnostic logging files with millisecond timestamps to accelerate defect triage and maintenance.",
        "{subj} {modal} provide extensible plug-in hooks through documented RESTful application programming interfaces.",
        "{subj} {modal} maintain clean class separation to ensure automated integration test coverage exceeds eighty percent.",
        "{subj} {modal} support database schema migrations using automated version-controlled script execution."
    ],
    'SC': [
        "{subj} {modal} scale horizontally across cloud nodes to support an increase from {v_u1} to {v_u2} active concurrent sessions.",
        "{subj} {modal} sustain an annual historical database growth rate of at least {v_pct} percent without query degradation.",
        "When incoming transaction volume spikes by {v_pct} percent, {subj} {modal} automatically spin up dynamic container instances.",
        "{subj} {modal} maintain linear throughput scaling across distributed compute clusters as additional worker nodes are attached.",
        "{subj} {modal} support simultaneous search queries from over {v_u1} active concurrent client sessions."
    ],
    'PO': [
        "{subj} {modal} execute seamlessly across Windows, Linux, and macOS environments without modification to underlying source code.",
        "{subj} {modal} render responsive web user interfaces across Google Chrome, Mozilla Firefox, Apple Safari, and Microsoft Edge browsers.",
        "{subj} {modal} permit seamless relational database migration between PostgreSQL and Oracle platforms.",
        "{subj} {modal} be packaged as standardized OCI container images deployable on any Kubernetes certified cluster.",
        "{subj} {modal} export and import project datasets using platform-agnostic JSON and XML standardized schemas."
    ],
    'L': [
        "{subj} {modal} comply with statutory global data privacy mandates, including the General Data Protection Regulation and HIPAA.",
        "{subj} {modal} record verifiable audit logs of user consent revocations to ensure statutory legal compliance.",
        "{subj} {modal} display mandatory copyright attribution and terms of service licensing agreements before registration.",
        "{subj} {modal} enforce automated data retention rules to securely purge archived customer records after the legal statutory period.",
        "{subj} {modal} restrict physical data residency and storage to jurisdictionally compliant regional data centers."
    ],
    'LF': [
        "{subj} {modal} adhere to the corporate design schema and branding aesthetics specified by the corporate design authority.",
        "{subj} {modal} apply consistent corporate color palettes, typography, and icon themes across all operational dialogs.",
        "{subj} {modal} maintain uniform layout margins, component spacing, and button alignments across all application screens.",
        "{subj} {modal} embed the official enterprise logo on all generated PDF reports and printable export documents.",
        "{subj} {modal} preserve consistent visual styling between desktop client dashboards and mobile web layouts."
    ],
    'O': [
        "{subj} {modal} interoperate with existing enterprise Active Directory and LDAP authentication servers.",
        "{subj} {modal} interface with external legacy relational database systems via standard JDBC and ODBC database drivers.",
        "{subj} {modal} transmit asynchronous event notifications to third-party message brokers using the AMQP protocol.",
        "{subj} {modal} integrate with commercial payment processing gateways via authenticated HTTPS REST APIs.",
        "{subj} {modal} operate within existing corporate virtual private network infrastructure without requiring firewall modifications."
    ]
}


class EnhancedNeuroSymbolicAugmenter:
    """
    Generates high-entropy, diverse, domain-rich synthetic requirements
    with full lexical and syntactic validation and runtime diversity guards.
    """
    def __init__(self, random_seed: int = 42, min_jaccard_distance: float = 0.25):
        self.rng = random.Random(random_seed)
        self.min_jaccard_distance = min_jaccard_distance
        
    def generate_candidate(self, target_class: str) -> str:
        templates = ADVANCED_TEMPLATES.get(target_class, [])
        if not templates:
            return ""
        template = self.rng.choice(templates)
        subj = self.rng.choice(SYSTEM_COMPONENTS)
        modal = self.rng.choice(MODAL_OPERATORS)
        op_mode = self.rng.choice(OPERATIONAL_MODES)
        
        # Randomized realistic technical values
        v_ms = self.rng.choice([50, 75, 100, 150, 200, 250, 300, 500])
        v_s = self.rng.choice([1, 2, 3, 5, 8, 10, 15, 30])
        v_t = self.rng.choice([500, 1000, 2500, 5000, 10000])
        v_pct = self.rng.choice([80, 85, 90, 95, 99])
        v_bw = self.rng.choice([10, 25, 50, 100])
        v_num = self.rng.choice([3, 4, 5])
        v_min = self.rng.choice([5, 10, 15, 30])
        v_hrs = self.rng.choice([500, 1000, 2000, 5000])
        v_u1 = self.rng.choice([1000, 2000, 5000])
        v_u2 = v_u1 * self.rng.choice([2, 5, 10])
        
        text = template.format(
            subj=subj, modal=modal, op_mode=op_mode,
            v_ms=v_ms, v_s=v_s, v_t=v_t, v_pct=v_pct,
            v_bw=v_bw, v_num=v_num, v_min=v_min, v_hrs=v_hrs,
            v_u1=v_u1, v_u2=v_u2
        )
        return text

    def _jaccard_distance(self, s1: str, s2: str) -> float:
        set1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s1.lower()))
        set2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', s2.lower()))
        if not set1 or not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 1.0 - (intersection / union)

    def generate_verified_batch(self, target_class: str, n_needed: int) -> Tuple[List[str], Dict[str, int]]:
        verified = []
        stats = {
            'attempted': 0,
            'passed': 0,
            'rejected_iso': 0,
            'rejected_ears': 0,
            'rejected_diversity': 0
        }
        
        max_attempts = n_needed * 15
        while len(verified) < n_needed and stats['attempted'] < max_attempts:
            stats['attempted'] += 1
            cand = self.generate_candidate(target_class)
            
            # Step 1: ISO 25010 Quality & Characteristic Validation
            is_valid, _ = validate_iso_quality_rules(cand, target_class)
            if not is_valid:
                stats['rejected_iso'] += 1
                continue
                
            # Step 2: EARS Syntactic Pattern Validation
            ears_res = analyze_ears_pattern(cand)
            if not ears_res['is_ears_compliant']:
                stats['rejected_ears'] += 1
                continue
                
            # Step 3: Semantic Diversity Guard (Prevent duplicate / near-duplicate generation)
            if verified and self.min_jaccard_distance > 0:
                min_dist = min(self._jaccard_distance(cand, prev) for prev in verified)
                if min_dist < self.min_jaccard_distance:
                    stats['rejected_diversity'] += 1
                    continue
                
            stats['passed'] += 1
            verified.append(cand)
            
        return verified, stats

    @staticmethod
    def compute_corpus_diversity(corpus: List[str]) -> Dict[str, float]:
        """
        Computes formal linguistic and vocabulary diversity metrics on a requirements corpus:
          - Type-Token Ratio (TTR)
          - Shannon Vocabulary Entropy (bits)
          - Total running tokens & vocabulary size
          - Mean pairwise Jaccard distance across random sampled subset
        """
        if not corpus:
            return {'ttr': 0.0, 'entropy': 0.0, 'vocab_size': 0, 'total_tokens': 0, 'mean_jaccard_dist': 0.0}
            
        all_tokens = []
        token_sets = []
        for text in corpus:
            toks = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_tokens.extend(toks)
            token_sets.append(set(toks))
            
        vocab = set(all_tokens)
        total_tokens = len(all_tokens)
        ttr = len(vocab) / total_tokens if total_tokens > 0 else 0.0
        
        # Shannon vocabulary entropy
        import math
        counts = Counter(all_tokens)
        entropy = -sum((c / total_tokens) * math.log2(c / total_tokens) for c in counts.values()) if total_tokens > 0 else 0.0
        
        # Sample pairwise Jaccard distance (up to 200 pairs)
        rng = random.Random(42)
        n_samples = min(len(token_sets), 50)
        sample_indices = rng.sample(range(len(token_sets)), n_samples)
        jaccard_dists = []
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                s1 = token_sets[sample_indices[i]]
                s2 = token_sets[sample_indices[j]]
                if s1 and s2:
                    dist = 1.0 - (len(s1 & s2) / len(s1 | s2))
                    jaccard_dists.append(dist)
                    
        mean_jaccard = float(np.mean(jaccard_dists)) if jaccard_dists else 0.0
        
        return {
            'ttr': round(ttr, 4),
            'shannon_entropy': round(entropy, 4),
            'vocab_size': len(vocab),
            'total_tokens': total_tokens,
            'mean_jaccard_distance': round(mean_jaccard, 4)
        }

