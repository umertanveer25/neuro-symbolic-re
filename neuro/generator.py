"""
Enhanced Neuro-Symbolic Requirement Generator.
Massively expanded lexical slots and combinatorial EARS syntactic patterns
aligned with ISO/IEC 25010 to provide high-entropy, diverse vocabularies
for linear models (TF-IDF, Logistic Regression, SVM) and tree models alike.
"""

import random
import re
from typing import List, Dict, Tuple
from symbolic.iso25010_rules import validate_iso_quality_rules, ISO_25010_TAXONOMY
from symbolic.ears_syntax import analyze_ears_pattern

SYSTEM_COMPONENTS = [
    "The system", "The web application", "The client interface", "The API gateway",
    "The microservice architecture", "The database engine", "The security module",
    "The authentication service", "The backend pipeline", "The reporting service",
    "The transaction processor", "The distributed server cluster", "The user dashboard",
    "The background scheduler", "The communication layer", "The data persistence subsystem"
]

# Rich combinatorial templates per ISO 25010 quality characteristic
ADVANCED_TEMPLATES = {
    'PE': [
        "{subj} shall process incoming user transactions within {v_ms} milliseconds under maximum peak concurrency.",
        "{subj} shall maintain an end-to-end response latency of less than {v_s} seconds during steady-state operations.",
        "{subj} shall refresh dynamic visual telemetry every {v_s} seconds to preserve interface accuracy.",
        "When handling batch data uploads, {subj} shall execute records at a minimum sustained throughput of {v_t} records per second.",
        "While under ninety percent server workload, {subj} shall maintain memory utilization below {v_pct} percent.",
        "{subj} shall complete complex analytical search queries within {v_s} seconds across ten million records.",
        "When multiple concurrent requests are dispatched, {subj} shall respond to ninety-five percent of queries in under {v_ms} milliseconds.",
        "{subj} shall optimize network bandwidth consumption to not exceed {v_bw} megabytes per second during video streaming.",
        "While processing payment authorizations, {subj} shall enforce strict computational turnaround within {v_ms} milliseconds."
    ],
    'SE': [
        "{subj} shall encrypt all persisted database records and sensitive credentials using AES-256 cryptographic standards.",
        "{subj} shall enforce multi-factor authentication protocols prior to granting administrative access privileges.",
        "{subj} shall enforce granular role-based access control policies to restrict unauthorized schema modification.",
        "When an invalid credential entry occurs {v_num} consecutive times, {subj} shall lock the account and generate an audit log.",
        "{subj} shall transmit all network communications across public domains exclusively via secure TLS 1.3 encrypted tunnels.",
        "{subj} shall mask confidential payment card information and personally identifiable data in accordance with PCI-DSS.",
        "{subj} shall maintain immutable audit trails of all user authorization attempts and administrative privilege escalations.",
        "When session inactivity exceeds {v_min} minutes, {subj} shall automatically terminate active sessions to prevent session hijacking.",
        "{subj} shall sanitize all input parameters to prevent SQL injection and cross-site scripting vulnerabilities."
    ],
    'US': [
        "{subj} shall display contextual diagnostic guidance within {v_s} seconds following an invalid input submission.",
        "{subj} shall enable primary workflow execution within no more than {v_num} intuitive navigation clicks from the homepage.",
        "{subj} shall render all quantitative data views using high-contrast color palettes and scalable typography for accessibility.",
        "When a form validation error occurs, {subj} shall visually highlight the exact interactive field requiring user correction.",
        "{subj} shall provide clear visual progress indicators during operations requiring more than {v_s} seconds to complete.",
        "{subj} shall support keyboard-only navigation across all dialogue windows in compliance with WCAG 2.1 accessibility guidelines.",
        "{subj} shall prompt the user with a confirmation dialogue prior to irreversible record deletion.",
        "While navigating complex configuration workflows, {subj} shall present persistent step-by-step breadcrumb feedback."
    ],
    'A': [
        "{subj} shall maintain an operational service availability of at least 99.{v_pct}% during scheduled core operational hours.",
        "{subj} shall guarantee continuous service uptime, limiting unplanned downtime to less than {v_min} minutes per calendar month.",
        "When unscheduled server hardware faults occur, {subj} shall restore full user service within {v_min} minutes.",
        "{subj} shall be available for transactional processing twenty-four hours a day, seven days a week.",
        "While undergoing automated background database backup, {subj} shall remain accessible in uninterrupted read-only mode.",
        "{subj} shall achieve a mean time between failures (MTBF) exceeding {v_hrs} continuous operational hours."
    ],
    'FT': [
        "If primary network connectivity fails, then {subj} shall automatically switch to redundant backup connections without data loss.",
        "When an unhandled runtime exception arises, {subj} shall execute graceful degradation and record a detailed diagnostic stack trace.",
        "If a server node crashes unexpectedly, then {subj} shall initiate automated failover to standby replicas within {v_s} seconds.",
        "{subj} shall recover transactional integrity from write-ahead database logs upon system restart following power interruption.",
        "If an external web service becomes unresponsive, then {subj} shall isolate the fault and provide cached responses.",
        "When storage disk corruption is detected, {subj} shall seamlessly redirect I/O operations to mirror redundant arrays."
    ],
    'MN': [
        "{subj} shall be architected into decoupled modular components to facilitate independent unit testing and automated continuous deployment.",
        "{subj} shall externalize runtime configuration properties to permit threshold modifications without recompilation of source code.",
        "{subj} shall produce standardized diagnostic logging files with millisecond timestamps to accelerate defect triage and maintenance.",
        "{subj} shall provide extensible plug-in hooks through documented RESTful application programming interfaces.",
        "{subj} shall maintain clean class separation to ensure automated integration test coverage exceeds eighty percent.",
        "{subj} shall support database schema migrations using automated version-controlled script execution."
    ],
    'SC': [
        "{subj} shall scale horizontally across cloud nodes to support an increase from {v_u1} to {v_u2} active concurrent sessions.",
        "{subj} shall sustain an annual historical database growth rate of at least {v_pct} percent without query degradation.",
        "When incoming transaction volume spikes by {v_pct} percent, {subj} shall automatically spin up dynamic container instances.",
        "{subj} shall maintain linear throughput scaling across distributed compute clusters as additional worker nodes are attached.",
        "{subj} shall support simultaneous search queries from over {v_u1} active concurrent client sessions."
    ],
    'PO': [
        "{subj} shall execute seamlessly across Windows, Linux, and macOS environments without modification to underlying source code.",
        "{subj} shall render responsive web user interfaces across Google Chrome, Mozilla Firefox, Apple Safari, and Microsoft Edge browsers.",
        "{subj} shall permit seamless relational database migration between PostgreSQL and Oracle platforms.",
        "{subj} shall be packaged as standardized OCI container images deployable on any Kubernetes certified cluster.",
        "{subj} shall export and import project datasets using platform-agnostic JSON and XML standardized schemas."
    ],
    'L': [
        "{subj} shall comply with statutory global data privacy mandates, including the General Data Protection Regulation and HIPAA.",
        "{subj} shall record verifiable audit logs of user consent revocations to ensure statutory legal compliance.",
        "{subj} shall display mandatory copyright attribution and terms of service licensing agreements before registration.",
        "{subj} shall enforce automated data retention rules to securely purge archived customer records after the legal statutory period.",
        "{subj} shall restrict physical data residency and storage to jurisdictionally compliant regional data centers."
    ],
    'LF': [
        "{subj} shall adhere to the corporate design schema and branding aesthetics specified by the corporate design authority.",
        "{subj} shall apply consistent corporate color palettes, typography, and icon themes across all operational dialogs.",
        "{subj} shall maintain uniform layout margins, component spacing, and button alignments across all application screens.",
        "{subj} shall embed the official enterprise logo on all generated PDF reports and printable export documents.",
        "{subj} shall preserve consistent visual styling between desktop client dashboards and mobile web layouts."
    ],
    'O': [
        "{subj} shall interoperate with existing enterprise Active Directory and LDAP authentication servers.",
        "{subj} shall interface with external legacy relational database systems via standard JDBC and ODBC database drivers.",
        "{subj} shall transmit asynchronous event notifications to third-party message brokers using the AMQP protocol.",
        "{subj} shall integrate with commercial payment processing gateways via authenticated HTTPS REST APIs.",
        "{subj} shall operate within existing corporate virtual private network infrastructure without requiring firewall modifications."
    ]
}


class EnhancedNeuroSymbolicAugmenter:
    """
    Generates high-entropy, diverse, domain-rich synthetic requirements
    with full lexical and syntactic validation.
    """
    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)
        
    def generate_candidate(self, target_class: str) -> str:
        templates = ADVANCED_TEMPLATES.get(target_class, [])
        if not templates:
            return ""
        template = self.rng.choice(templates)
        subj = self.rng.choice(SYSTEM_COMPONENTS)
        
        # Randomized realistic technical values
        v_ms = self.rng.choice([50, 100, 150, 200, 250, 300, 500])
        v_s = self.rng.choice([1, 2, 3, 5, 10, 15, 30])
        v_t = self.rng.choice([500, 1000, 2500, 5000, 10000])
        v_pct = self.rng.choice([80, 85, 90, 95, 99])
        v_bw = self.rng.choice([10, 25, 50, 100])
        v_num = self.rng.choice([3, 4, 5])
        v_min = self.rng.choice([5, 10, 15, 30])
        v_hrs = self.rng.choice([500, 1000, 2000, 5000])
        v_u1 = self.rng.choice([1000, 2000, 5000])
        v_u2 = v_u1 * self.rng.choice([2, 5, 10])
        
        text = template.format(
            subj=subj, v_ms=v_ms, v_s=v_s, v_t=v_t, v_pct=v_pct,
            v_bw=v_bw, v_num=v_num, v_min=v_min, v_hrs=v_hrs,
            v_u1=v_u1, v_u2=v_u2
        )
        return text

    def generate_verified_batch(self, target_class: str, n_needed: int) -> Tuple[List[str], Dict[str, int]]:
        verified = []
        stats = {'attempted': 0, 'passed': 0, 'rejected_iso': 0, 'rejected_ears': 0}
        
        max_attempts = n_needed * 12
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
                
            stats['passed'] += 1
            verified.append(cand)
            
        return verified, stats
