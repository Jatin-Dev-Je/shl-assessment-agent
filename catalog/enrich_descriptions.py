"""
catalog/enrich_descriptions.py
────────────────────────────────
Generates rule-based descriptions for catalog items that have empty description fields.
No API calls needed — uses name patterns + test type + job levels.
Run BEFORE rebuild_index so vectors are built on rich text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"

TYPE_DESCRIPTIONS: dict[str, str] = {
    "P": "personality and behavior assessment measuring workplace behavioural style, motivations, and interpersonal tendencies",
    "A": "ability and aptitude assessment measuring cognitive reasoning including numerical, verbal, and inductive reasoning",
    "K": "knowledge and skills assessment measuring specific technical or domain knowledge",
    "B": "biodata and situational judgment assessment measuring real-world workplace decision-making",
    "S": "simulation assessment replicating real job tasks and workplace situations",
    "C": "competency assessment evaluating job-relevant skills against a competency framework",
    "D": "development and 360 assessment providing feedback for professional growth and coaching",
    "E": "assessment exercise such as in-tray, role-play, or case study for assessment centres",
}

NAME_HINTS: list[tuple[str, str]] = [
    (r"OPQ32r|Occupational Personality Questionnaire", "Measures 32 workplace behaviour dimensions including leadership style, influencing, and thinking style."),
    (r"Verify.*G\+|Verify Interactive G\+|SHL Verify Interactive G", "Adaptive reasoning battery covering inductive, numerical, and deductive reasoning in a single administration."),
    (r"Verify.*Numerical|Numerical Reason", "Measures ability to interpret numerical data, charts, and statistical information."),
    (r"Verify.*Verbal|Verbal Reason", "Measures ability to evaluate verbal arguments and draw logical conclusions from written information."),
    (r"Verify.*Inductive|Inductive Reason", "Measures ability to identify patterns and rules from abstract visual information."),
    (r"Verify.*Deductive|Deductive Reason", "Measures ability to apply rules and draw logical deductions."),
    (r"SVAR.*Spoken English|Spoken English.*SVAR", "Automated spoken English screen evaluating accent clarity, fluency, and comprehension for contact centre roles."),
    (r"Contact Center.*Simulation|Call Simulation", "Simulates inbound contact centre calls testing customer service skills and call-handling behaviour."),
    (r"Customer Service.*Simulation|Customer Service.*Phone", "Simulates customer service phone interactions assessing responsiveness, empathy, and problem resolution."),
    (r"OPQ.*Leadership|Leadership Report", "Interprets OPQ32r results through a leadership lens, benchmarking candidates against leadership competencies."),
    (r"Universal Competency Report|OPQ UCR", "Translates OPQ32r personality data into a Universal Competency Framework report for selection and development."),
    (r"OPQ.*Sales|MQ Sales Report", "Interprets OPQ32r results in a sales-specific frame highlighting behaviours linked to sales effectiveness."),
    (r"Sales Transformation", "Measures personality predictors of success in modern, digital-first sales environments for individual contributors and managers."),
    (r"Global Skills Assessment|GSA\b", "Self-reported skills inventory assessing proficiency across a broad range of workplace competencies for talent audit and reskilling."),
    (r"Global Skills Development", "Development report derived from the Global Skills Assessment identifying skill gaps and recommending learning actions."),
    (r"Dependability and Safety Instrument|DSI\b", "Measures personality predictors of dependability, safety-conscious behaviour, integrity, and rule compliance across industries."),
    (r"Safety.*Dependability.*8\.0|Safety & Dependability", "Industry-normed personality battery for manufacturing and industrial settings measuring safety behaviour predictors with sector-specific norms."),
    (r"Workplace.*Health.*Safety", "Knowledge test measuring understanding of health, safety, and workplace regulatory requirements."),
    (r"Graduate Scenarios", "Situational judgment test designed specifically for graduate-level candidates assessing work-context decision-making and professional judgement."),
    (r"HIPAA", "Knowledge test measuring understanding of HIPAA privacy and security regulations for healthcare staff handling patient data."),
    (r"Medical Terminology", "Knowledge test assessing familiarity with clinical and medical vocabulary required in healthcare administration roles."),
    (r"Financial Accounting", "Knowledge test measuring understanding of financial accounting principles, practices, and reporting standards."),
    (r"Basic Statistics", "Knowledge test assessing understanding of foundational statistical concepts, data interpretation, and methods."),
    (r"Smart Interview.*Coding|Live Coding", "Adaptive live-coding interview platform enabling custom technical exercises in multiple programming languages including Java, Python, and SQL."),
    (r"Linux Programming", "Knowledge assessment measuring proficiency in Linux system programming, shell scripting, and command-line operations."),
    (r"Networking.*Implementation", "Knowledge assessment measuring understanding of networking protocols, TCP/IP, architecture, and infrastructure implementation."),
    (r"Microsoft Word", "Simulation-based assessment of Microsoft Word skills including document formatting, styles, mail merge, and editing."),
    (r"Microsoft Excel", "Simulation-based assessment of Microsoft Excel skills including formulas, pivot tables, data analysis, and spreadsheet management."),
    (r"Microsoft PowerPoint", "Simulation-based assessment of Microsoft PowerPoint skills including presentation design and slide management."),
    (r"Microsoft Outlook", "Simulation-based assessment of Microsoft Outlook skills including email management, calendar scheduling, and task organisation."),
    (r"Entry Level.*Customer Serv", "Behavioural and personality assessment for entry-level customer service roles in retail and contact centre environments."),
    (r"Motivation Questionnaire|MQ\b", "Measures what motivates and engages an individual at work across 18 motivational dimensions tied to sales and performance behaviours."),
    (r"Enterprise Leadership", "Interprets OPQ32r or assessment data through an enterprise leadership lens for CXO and senior executive evaluation."),
    (r"Apprentice", "Job-focused assessment battery calibrated for apprentice and entry-level candidates entering trades, technical, or operational roles."),
    (r"Managerial|Management", "Assessment measuring managerial potential and effectiveness including people leadership, delegation, and decision-making."),
    (r"Java\b", "Knowledge assessment measuring Java programming proficiency including OOP, data structures, and modern Java APIs."),
    (r"Python\b", "Knowledge assessment measuring Python programming proficiency including syntax, libraries, and scripting."),
    (r"SQL\b", "Knowledge assessment measuring SQL query writing, database design, and data manipulation proficiency."),
    (r"JavaScript|JS\b", "Knowledge assessment measuring JavaScript programming proficiency for web development roles."),
    (r"C\+\+|C Programming", "Knowledge assessment measuring C or C++ programming proficiency for systems and embedded development."),
    (r"Data Science|Machine Learning|Data Analysis", "Knowledge assessment measuring data science and analytics skills including statistical methods and ML concepts."),
    (r"Retail|Sales Associate", "Behavioural and situational assessment for retail and sales associate roles measuring customer orientation and service skills."),
    (r"Call Centre|Call Center", "Assessment battery designed for call centre roles measuring communication, resilience, and customer service skills."),
    (r"Administrative|Admin", "Assessment measuring administrative skills including organisation, attention to detail, and office software proficiency."),
    (r"Accounting|Bookkeeping", "Knowledge assessment measuring accounting and bookkeeping skills including ledger management and financial reporting."),
]


def build_description(name: str, test_type: str, job_levels: list[str]) -> str:
    types = [t.strip() for t in test_type.split(",") if t.strip()]
    type_descs = [TYPE_DESCRIPTIONS.get(t, "assessment") for t in types]
    base = f"{name} is a {' and '.join(type_descs)}."

    for pattern, hint in NAME_HINTS:
        if re.search(pattern, name, re.IGNORECASE):
            base += " " + hint
            break

    if job_levels:
        levels_str = ", ".join(job_levels[:5])
        base += f" Target audience: {levels_str}."

    return base.strip()


def enrich_catalog(catalog_path: Path = CATALOG_PATH) -> None:
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    enriched = 0
    for item in catalog:
        if not item.get("description", "").strip():
            jl = item.get("job_levels", [])
            if isinstance(jl, str):
                try:
                    jl = json.loads(jl)
                except Exception:
                    jl = []
            item["description"] = build_description(
                item.get("name", ""),
                item.get("test_type", ""),
                jl,
            )
            enriched += 1

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Enriched {enriched} / {len(catalog)} items.")
    print("\nSamples:")
    for item in catalog[:3]:
        print(f"  [{item['name']}]")
        print(f"  {item['description'][:120]}")
        print()


if __name__ == "__main__":
    enrich_catalog()
