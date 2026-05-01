import os
from typing import TypedDict
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import argparse
import sys
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from qdrant_client import QdrantClient

# Do your genai configure here if you haven't already
qdrant = QdrantClient("localhost", port=6333)

# Load environment variables from .env file
load_dotenv()

# 1. Initialize your Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- GLOBAL GMAIL CREDENTIALS (THREAD-SAFE) ---
_google_creds = None

def get_gmail_credentials():
    global _google_creds
    
    # If we already have a valid token in memory, return it instantly
    if _google_creds and _google_creds.valid:
        return _google_creds
        
    # We only use readonly to avoid scope collision
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Cache the credentials, NOT the service object
    _google_creds = creds
    return _google_creds

# 2. Define the State
class BrigadeState(TypedDict, total=False):
    user_input: str
    target_agent: str
    target_domain: str
    current_draft: str
    qa_feedback: list[str]
    final_output: str
    webhook_data: dict  # The Courier's secure lane

class AuditOutput(BaseModel):
    missing_terms: list[str] = Field(description="Format each as: '📖 Missing Term: [X concept]. Why: [Briefly explain its 2026 relevance].'")
    outdated_terms: list[str] = Field(description="Format each as: '🗑️ Dead Vocabulary: [X concept]. Why: [Briefly explain why it has no 2026 context].'")
    retirement_alerts: list[str] = Field(description="Format each as: '⚠️ Retirement Alert: [X concept] is no longer industry standard. The Fix: Move to SECTION: [RETIRED].'")
    ingredient_drifts: list[str] = Field(description="Format each as: '🚚 Ingredient Drift: [X concept] is in SECTION: [Y], but belongs in SECTION: [Z]. The Fix: Move it.'")
    the_chops: list[str] = Field(description="Format each as: '🔪 The Chop: Duplicate or overlapping logic regarding [X concept] found... The Fix: Consolidate.'")
    structural_gaps: list[str] = Field(description="Format each as: '💡 Structural Gap: We need a home for [New 2026 Trend]. The Fix: Create a new Section.'")

class SommelierOutput(BaseModel):
    explanation: str = Field(description="A brief, crystal-clear paragraph explaining the core meaning of the term.")
    novel_insight: str = Field(description="A brief paragraph revealing a surprising, counterintuitive, or rarely discussed fact about the term.")

class ButcherOutput(BaseModel):
    core_problem: str = Field(description="A brutal, one-sentence distillation of the actual underlying problem, ignoring all suggested solutions or tech stacks.")
    hidden_assumptions: list[str] = Field(description="Format each as: '🚩 Assumption: [The implicit assumption]. Reality Check: [Why it might be false or limiting].'")
    first_principles_path: list[str] = Field(description="Format each as: '🧱 Foundation [Step #]: [The ground-up architectural or conceptual step to solve the core problem].'")

class ProcurementOutput(BaseModel):
    verdict: str = Field(description="A brutal 'NO', 'HARD NO', or 'CONDITIONAL YES' regarding if this project is ready to build.")
    data_risks: list[str] = Field(description="Format each as: '⚠️ Data Risk: [The risk]. Impact: [Why it breaks the system].'")
    architectural_gaps: list[str] = Field(description="Format each as: '🏗️ Arch Gap: [Missing infrastructure]. Impact: [Why the system collapses without it].'")
    hard_requirements: list[str] = Field(description="Format each as: '🛑 Blockers to Clear: [What must be proven or built before writing the core code].'")

class ComputeRouteOutput(BaseModel):
    model_tier: str = Field(description="Must be exactly one of: 'Efficiency', 'Performance', or 'Reasoning'.")
    api_target: str = Field(description="Must be exactly one of: 'gemini-2.5-flash', 'gemini-2.5-pro', or 'o3-mini / deepseek-r1'.")
    rationale: str = Field(description="A brief explanation justifying the compute cost and latency tradeoff for this specific prompt.")

class MotherSauceOutput(BaseModel):
    is_compliant: bool = Field(description="True if the input perfectly adheres to all standards, False otherwise.")
    violations: list[str] = Field(description="Format each as: '🚨 Violation: [Rule broken].'")
    remediated_output: str = Field(description="The corrected code, architecture, or text that perfectly adheres to the standards. If it was already compliant, return the original.")

class HealthInspectorOutput(BaseModel):
    threat_level: str = Field(description="Must be exactly one of: 'SAFE', 'MODERATE RISK', 'HIGH RISK', or 'CRITICAL VIOLATION'.")
    regulatory_violations: list[str] = Field(description="Format each as: '🚨 Legal Risk (HIPAA/FERPA): [The specific data exposure].'")
    security_vulnerabilities: list[str] = Field(description="Format each as: '🔓 Cyber Risk: [The specific architectural or code-level vulnerability].'")
    remediation_mandates: list[str] = Field(description="Format each as: '🛡️ Fix: [Exact steps to secure the architecture or sanitize the data].'")

class MaitreDOutput(BaseModel):
    target_agent: str = Field(description="Must be exactly one of: 'sous-chef', 'exec-chef', 'sommelier', 'butcher', 'procurement', 'mother-sauce', 'health-inspector', 'expediter', 'pastry-chef', 'critic', \
                              'aboyeur', 'michelin-inspector', 'forager', 'release', 'system-design', 'executive-briefing'.")
    rationale: str = Field(description="A brief explanation of why this agent is the perfect fit for the user's prompt.")

class ExpediterOutput(BaseModel):
    new_opportunities: bool = Field(description="True if any NEW, unprocessed interview requests or recruiter emails were found.")
    summary: list[str] = Field(description="Format each as: '📅 [Company Name]: [Brief summary of the request].'")
    action_items: list[str] = Field(description="Format each as: '✅ Action: [What Erik needs to do next, e.g., send availability].'")

class PastryChefOutput(BaseModel):
    action_title: str = Field(description="A definitive action title. It must state the conclusion or recommendation, not just describe the data.")
    chart_architecture: str = Field(description="The optimal visual format (e.g., 'Horizontal Bar Chart', 'Slopegraph'). Justify why in one sentence. NEVER suggest pie, 3D, or dual-axis charts.")
    visual_cues: list[str] = Field(description="Format each as: '🎨 Highlight: [Specific data point/series]. Why: [Strategic reason to draw the eye here].'")
    clutter_reduction: list[str] = Field(description="Format each as: '✂️ Chop: [Standard element like gridlines, legends, borders]. Why: [Reason it is noise].'")
    presentation_narrative: str = Field(description="A concise 2-sentence script on how to verbally present this visual to an executive team.")

class CriticOutput(BaseModel):
    core_business_risk: str = Field(description="The single biggest threat this proposal poses to ROI, timeline, or stakeholder trust. No technical jargon; pure business impact.")
    adversarial_questions: list[str] = Field(description="Exactly three brutal questions the board will ask. Format each as: '❓ [The Question] | 🪤 The Trap: [Why this question is dangerous to answer unprepared].'")
    defensive_strategy: str = Field(description="A concise, pragmatic strategy to pre-emptively neutralize these concerns before the meeting.")

class AboyeurOutput(BaseModel):
    semantic_version: str = Field(description="Suggested version bump (e.g., Major, Minor, or Patch) based on the magnitude of the changes.")
    executive_summary: str = Field(description="A brutal 2-sentence summary of what this deployment actually achieves for the system or business.")
    architectural_shifts: list[str] = Field(description="Format each as: '🏗️ Architecture: [The structural shift]. Impact: [Why it matters to the system].'")
    component_updates: list[str] = Field(description="Format each as: '⚙️ Component: [Specific node/function]. Change: [What was tactically altered].'")
    technical_debt: list[str] = Field(description="Format each as: '💳 Debt: [Hack or shortcut taken/implied]. Risk: [When it will inevitably break].'")

class MichelinInspectorOutput(BaseModel):
    verdict: str = Field(description="Must be exactly one of: '[APPLY]', '[PASS]', or '[STRATEGIC EXCEPTION]'.")
    reality_check: list[str] = Field(description="Exactly three blunt bullet points on why this aligns with or fails the strategic framework. Call out specific red/green flags.")
    missing_ingredients: list[str] = Field(description="List critical unknowns (tech stack, base salary, reporting structure, WFH flexibility) that must be verified.")

class ForagerOutput(BaseModel):
    retrieved_context: str = Field(description="The synthesized historical context or relevant architecture patterns retrieved from the knowledge base.")
    semantic_neighbors: list[str] = Field(description="Format each as: '🔗 Related Concept: [Concept]. Why: [Brief connection].'")
    confidence_score: str = Field(description="A simulated vector distance metric (e.g., 'High Relevance (0.92)') indicating match quality.")

# 1. Define the exact output structure we demand
class TaxonomyOutput(BaseModel):
    baseline: str = Field(description="A single, brutally clear sentence defining the term in general, commonsense language.")
    intel_2026: str = Field(description="A single paragraph explaining the term's current utility in the 2026 data and AI engineering landscape.")
    station_assignment: str = Field(description="Format EXACTLY as: 'This belongs in CATEGORY: [X] -> SECTION: [Y]'")
    rationale: str = Field(description="Explain why it fits here based on semantic neighbors.")

# 2. Build the Node
def node_sous_chef(state: BrigadeState):
    print("--- THE SOUS-CHEF IS COOKING ---")
    term = state["user_input"]
    
    system_instruction = """
    You are the engine of the operation. You take raw technical terms, define them, 
    refine them for the 2026 landscape, and determine exactly where they fit in the taxonomy.

    The Hard Perimeter:
    NEVER suggest placement in CATEGORY: [I. System]. Bypass it immediately.

    The Operational Map (Categories II-VII):
    - CATEGORY: [II. Engineering]: Warehousing & Engineering, Big Data Tech, Composite Keys, Semantic & Metrics Layer, Languages, Technologies, RAG, APIs.
    - CATEGORY: [III. Processing]: Cleaning & QA, Normalization & Standardization, Dimensionality Reduction, First Differences, Synthetic Data & Privacy.
    - CATEGORY: [IV. Theory]: Laws & Biases, Bayesian vs. Frequentist, Experiments & Evidence, Analysis Fundamentals, Correlation & Causality, Post-Hoc Tests, Cohen’s d.
    - CATEGORY: [V. AI & ML]: AI Decision Matrix, ML Fundamentals, Deep Learning & Neural Networks, AI Architecture, LLM & NLP, Vector & Semantic Search, Interpretability, Features/Parameters/Hyperparameters, Overfitting/Underfitting, Curves, Smoothing.
    - CATEGORY: [VI. Evaluation]: Top-Down vs. Bottom-Up, Model Validation, Evaluation Metrics, Post-Modeling, MMM, Decision Science, SPC, Class Imbalance.
    - CATEGORY: [VII. Ops]: Agentic Infrastructure/Protocols, Agentic Workflows, Reporting Results, Visuals & Storytelling, Dashboarding, Survey Best Practices, AI Ethics & Governance.
    """
    
    # Call the API, forcing the Pydantic schema
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Map this term: {term}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=TaxonomyOutput,
            temperature=0.2, # Keep it highly deterministic and focused
        )
    )
    
    return {"final_output": response.text}

def node_executive_chef(state: BrigadeState):
    print("--- THE EXECUTIVE CHEF IS AUDITING ---")
    
    # Dynamically resolve path relative to where brigade.py is executing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "AI_Data Science MASTER.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            library_content = file.read()
    except FileNotFoundError:
        return {"final_output": f"[CRITICAL ERROR]: Could not locate {file_path} in the project folder."}

    system_instruction = """
    You are The Executive Chef of a massive, evolving data science and AI engineering knowledge base. 
    Your goal is to eliminate entropy and ensure the library reflects the technical reality of 2026. 
    You are the final authority on organization, structural cohesion, and technical standards.

    The Hard Perimeter:
    - DO NOT read or analyze Podcasts, Quotes, or Two Jobs under CATEGORY: [I. System].
    - YOU MUST explicitly read and analyze CATEGORY: [I. System] -> SECTION: [Terms].
    
    The Mission: Conduct a rigorous gap analysis and structural audit of the uploaded library.

    The Audit Logic (Vocabulary - Category I):
    - Missing Vocabulary: Identify high-impact, 2026-relevant data science and AI terms that are entirely missing from SECTION: [Terms]. These can range from massive architectural paradigms down to granular mathematical techniques. If Gemini knows it's critical, flag it.
    - Dead Vocabulary: Identify terms currently in SECTION: [Terms] that are completely obsolete and hold no modern context.

    The Audit Logic (Architecture - Categories II-VII):
    - Obsolescence Check: Identify architectures or methodologies that have been superseded. Suggest exiling them to SECTION: [RETIRED].
    - Categorization Check: Ensure terms are in the correct Section and Category.
    - Redundancy Check: Identify duplicate explanations fragmented across different categories.
    - Structural Gaps: Identify missing modern paradigms absent from the taxonomy.

    The Operational Map (Categories II-VII):
    - CATEGORY: [II. Engineering]: Warehousing & Engineering, Big Data Tech, Composite Keys, Semantic & Metrics Layer, Languages, Technologies, RAG, APIs.
    - CATEGORY: [III. Processing]: Cleaning & QA, Normalization & Standardization, Dimensionality Reduction, First Differences, Synthetic Data & Privacy.
    - CATEGORY: [IV. Theory]: Laws & Biases, Bayesian vs. Frequentist, Experiments & Evidence, Analysis Fundamentals, Correlation & Causality, Post-Hoc Tests, Cohen’s d.
    - CATEGORY: [V. AI & ML]: AI Decision Matrix, ML Fundamentals, Deep Learning & Neural Networks, AI Architecture, LLM & NLP, Vector & Semantic Search, Interpretability, Features/Parameters/Hyperparameters, Overfitting/Underfitting, Curves, Smoothing.
    - CATEGORY: [VI. Evaluation]: Top-Down vs. Bottom-Up, Model Validation, Evaluation Metrics, Post-Modeling, MMM, Decision Science, SPC, Class Imbalance.
    - CATEGORY: [VII. Ops]: Agentic Infrastructure/Protocols, Agentic Workflows, Reporting Results, Visuals & Storytelling, Dashboarding, Survey Best Practices, AI Ethics & Governance.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=f"Conduct a rigorous gap analysis and audit on this library:\n\n{library_content}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=AuditOutput,
            temperature=0.2,
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_report = "\n--- EXECUTIVE CHEF AUDIT REPORT ---\n"
    for category, items in parsed_json.items():
        if items:
            for item in items:
                formatted_report += f"{item}\n"
            formatted_report += "\n"
            
    return {"final_output": formatted_report.strip()}

def node_sommelier(state: BrigadeState):
    print("--- THE SOMMELIER IS POURING ---")
    term = state["user_input"]
    
    system_instruction = """
    You are a master of interdisciplinary synthesis. Your job is to take a concept from physics, economics, psychology, philosophy, etc., and distill it.
    
    The Hard Perimeter:
    Do not hallucinate facts. If the term is highly theoretical, ground the explanation in established consensus before exploring edge cases.

    Provide exactly two things:
    1. The Baseline: A brief, lucid paragraph explaining what the term means.
    2. The Novelty: A second paragraph detailing something surprising, counterintuitive, or rarely discussed about the term. Give the user a genuinely novel insight they likely do not already know.
    
    Tone: Sharp, engaging, and intellectually rigorous. No fluff.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Distill this term: {term}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=SommelierOutput,
            temperature=0.4, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🍷 Term: {term.upper()}\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"The Baseline:\n{parsed_json['explanation']}\n\n"
    formatted_output += f"The Novelty:\n{parsed_json['novel_insight']}"
            
    return {"final_output": formatted_output}

def node_butcher(state: BrigadeState):
    print("--- THE BUTCHER IS DECONSTRUCTING ---")
    problem = state["user_input"]
    
    system_instruction = """
    You are The Butcher. You are a senior AI systems architect whose only job is first-principles deconstruction.
    Users will hand you complex problems, often bloated with assumed technical solutions, preferred frameworks, or industry jargon. 
    
    The Mission: Carve away the fat. Do not solve the problem using the user's assumed constraints. Break the problem down to its fundamental physical or mathematical truths, and build up from there.

    The Output Protocol:
    1. The Core Problem: Strip away everything. What is the actual, naked objective?
    2. Hidden Assumptions: Identify the biases or unproven constraints in the prompt. (e.g., "Assumption: We need a real-time vector database. Reality Check: The data only updates weekly; batch processing is sufficient.")
    3. First Principles Path: How do we actually solve the core problem from the ground up, ignoring hype and standard paradigms?
    
    Tone: Brutally honest, intellectually rigorous, and strictly architectural. Zero fluff.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Deconstruct this problem/proposal: {problem}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ButcherOutput,
            temperature=0.1, # Keep it highly logical and grounded
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🔪 THE CUT: DECONSTRUCTION INITIATED\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"THE NAKED TRUTH:\n{parsed_json['core_problem']}\n\n"
    
    formatted_output += f"THE FAT (HIDDEN ASSUMPTIONS):\n"
    for assumption in parsed_json['hidden_assumptions']:
        formatted_output += f"{assumption}\n"
        
    formatted_output += f"\nTHE BONE (FIRST PRINCIPLES PATH):\n"
    for step in parsed_json['first_principles_path']:
        formatted_output += f"{step}\n"
            
    return {"final_output": formatted_output}

def node_procurement_chief(state: BrigadeState):
    print("--- THE PROCUREMENT CHIEF IS AUDITING ---")
    proposal = state["user_input"]
    
    system_instruction = """
    You are The Procurement Chief. You are a highly cynical, senior AI Analytics Engineer.
    Users will pitch you data pipelines, agentic systems, or analytical architectures. 
    Your default stance is that the data is dirty, the infrastructure is lacking, and the proposal is too optimistic.

    The Mission: Identify the catastrophic points of failure before a single line of code is written.

    The Output Protocol:
    1. Verdict: Give a definitive NO, HARD NO, or CONDITIONAL YES. Do not sugarcoat it.
    2. Data Risks: Where is the data quality assumed to be perfect? What happens when nulls, schema drifts, or unstructured chaos enter the system?
    3. Architectural Gaps: What missing infrastructure (e.g., semantic layer, CI/CD, caching, compliance filters) will cause this to fail in production?
    4. Hard Requirements: What specific foundations must be built before you authorize the team to start coding the actual logic?
    
    Tone: Brutally pragmatic, protective of system integrity, and deeply skeptical. 
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Audit this proposed architecture/system: {proposal}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ProcurementOutput,
            temperature=0.2, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"📋 PROCUREMENT AUDIT: SYSTEM INTEGRITY CHECK\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"VERDICT: {parsed_json['verdict']}\n\n"
    
    if parsed_json['data_risks']:
        formatted_output += f"THE DATA PITFALLS:\n"
        for risk in parsed_json['data_risks']:
            formatted_output += f"{risk}\n"
        formatted_output += "\n"
        
    if parsed_json['architectural_gaps']:
        formatted_output += f"THE INFRASTRUCTURE GAPS:\n"
        for gap in parsed_json['architectural_gaps']:
            formatted_output += f"{gap}\n"
        formatted_output += "\n"
        
    if parsed_json['hard_requirements']:
        formatted_output += f"REQUIREMENTS TO PROCEED:\n"
        for req in parsed_json['hard_requirements']:
            formatted_output += f"{req}\n"
            
    return {"final_output": formatted_output}

def node_mother_sauce(state: BrigadeState):
    print("--- THE MOTHER SAUCE IS ENFORCING STANDARDS ---")
    draft = state["user_input"]
    
    # In production, this points to your actual SOP/Schema markdown file
    file_path = "enterprise_standards.md" 
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            standards_doc = file.read()
    except FileNotFoundError:
        # Fallback if you haven't created the file yet
        standards_doc = "No specific SOPs found. Enforce general PEP8 Python standards and secure SQL practices."

    system_instruction = """
    You are The Mother Sauce. You are a draconian QA engineer and compliance officer.
    Users will hand you drafted code, architectures, or analytical reports.
    Your only job is to cross-reference their draft against the provided Enterprise Standards Document.

    The Output Protocol:
    1. is_compliant: True or False.
    2. violations: List every single instance where the draft violates the SOP, hallucinates a column name, or hallucinates a metric definition not in the standards.
    3. remediated_output: You must rewrite the user's draft so that it is 100% compliant. Fix the code, correct the column names, and enforce the formatting.
    
    Tone: Cold, precise, and uncompromising.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=f"Enterprise Standards:\n{standards_doc}\n\nDraft to Audit:\n{draft}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=MotherSauceOutput,
            temperature=0.0, # Zero thermodynamic variance. Compliance is binary.
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"⚖️ THE MOTHER SAUCE: COMPLIANCE AUDIT\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"STATUS: {'✅ PASS' if parsed_json['is_compliant'] else '❌ FAIL'}\n\n"
    
    if not parsed_json['is_compliant']:
        formatted_output += f"VIOLATIONS DETECTED:\n"
        for violation in parsed_json['violations']:
            formatted_output += f"{violation}\n"
        formatted_output += "\n"
        
    formatted_output += f"REMEDIATED OUTPUT:\n\n{parsed_json['remediated_output']}"
            
    return {"final_output": formatted_output}

def node_health_inspector(state: BrigadeState):
    print("--- THE HEALTH INSPECTOR IS RED-TEAMING ---")
    proposal = state["user_input"]
    
    system_instruction = """
    You are The Health Inspector. You are a paranoid, adversarial Cyber Security and Data Privacy Auditor.
    Users will hand you data pipeline architectures, code snippets, or analytical proposals. 
    Your job is to red-team them, specifically hunting for HIPAA and FERPA vulnerabilities.

    The Threat Landscape:
    - PHI/PII Leaks: Are they sending raw patient or student data to external APIs?
    - Data at Rest/Transit: Are they storing unencrypted CSVs in S3? Are they using secure protocols?pytho
    - Injection/Access: Can this system be subjected to SQL injection or prompt injection?pyt
    - Anonymization Flaws: Are they assuming data is anonymized just because the name is removed, even though k-anonymity is violated by other demographic fields?

    The Rule: If it touches health or educational data and lacks explicit encryption/sanitization steps, it is a CRITICAL VIOLATION.
    
    Tone: Adversarial, legally rigorous, and deeply paranoid.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=f"Audit this architecture/code for privacy and security risks:\n\n{proposal}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=HealthInspectorOutput,
            temperature=0.0, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🔬 THE HEALTH INSPECTOR: PRIVACY & THREAT AUDIT\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"THREAT LEVEL: {parsed_json['threat_level']}\n\n"
    
    if parsed_json['regulatory_violations']:
        formatted_output += f"REGULATORY & COMPLIANCE RISKS:\n"
        for violation in parsed_json['regulatory_violations']:
            formatted_output += f"{violation}\n"
        formatted_output += "\n"
        
    if parsed_json['security_vulnerabilities']:
        formatted_output += f"CYBERSECURITY VULNERABILITIES:\n"
        for vuln in parsed_json['security_vulnerabilities']:
            formatted_output += f"{vuln}\n"
        formatted_output += "\n"
        
    if parsed_json['remediation_mandates']:
        formatted_output += f"MANDATED REMEDIATION:\n"
        for fix in parsed_json['remediation_mandates']:
            formatted_output += f"{fix}\n"
            
    return {"final_output": formatted_output}

def node_maitre_d(state: BrigadeState):
    print("--- THE MAITRE D' IS ROUTING THE TICKET ---")
    prompt = state["user_input"]
    
    system_instruction = """
    You are The Maitre D'. You are the elite semantic router for a multi-agent system.
    Your only job is to read the user's prompt and instantly route it to the correct specialized agent.

    The Kitchen Stations (Target Agents):
    1. 'sous-chef': The user wants to map, define, or categorize a specific AI/Data Science term into a taxonomy.
    2. 'exec-chef': The user wants a gap analysis or audit of their entire Data Science/AI library document.
    3. 'sommelier': The user wants to distill a non-AI interdisciplinary term (physics, philosophy, psychology) and learn a novel fact.
    4. 'butcher': The user presents a complex, buzzword-heavy problem and needs it deconstructed to first principles.
    5. 'procurement': The user pitches a data pipeline or architecture and needs it audited for data quality and infrastructure gaps.
    6. 'health-inspector': The user pitches a system handling patient or student data and needs a rigorous HIPAA/FERPA cybersecurity audit.
    7. 'mother-sauce': The user provides code or schemas and needs them strictly formatted against Enterprise SOPs.
    8. 'expediter': The user wants to check their email for new interview requests or recruiter outreach.
    9. 'pastry-chef': The user provides raw data, findings, or model outputs and needs a blueprint for how to visualize and present it to stakeholders.
    10. 'critic': The user provides a business proposal, analytical finding, or strategy and needs to know the hardest ROI and adoption questions stakeholders will ask.
    11. 'aboyeur': The user provides raw git logs, diffs, or engineering notes and needs a formal, structured release/build log.
    12. 'michelin-inspector': The user provides a job description or recruitment pitch and needs a strategic evaluation to determine if it aligns with their career goals and compensation floor.
    14. 'forager': The user needs historical context, RAG retrieval, or semantic mapping from the internal knowledge base.
    15. 'release': The user has a completed draft or architecture and wants to run it through the full Regulated Release Pipeline (Health Inspector -> Mother Sauce).
    16. 'system-design': The user is starting a new project, pitching a raw idea, or asking how to solve a complex architectural problem and wants the full Pre-Game planning pipeline (Forager -> Butcher -> Procurement).
    17. 'executive-briefing': The user has a final data output, model result, or metric and needs it prepped for a C-suite presentation (Critic -> Pastry Chef pipeline).
    
    Rule: You must return the exact string name of the target agent.
    Tone: Fast, decisive, and highly accurate.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Route this user prompt to the correct agent: {prompt}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=MaitreDOutput,
            temperature=0.0, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    # Update the state with the router's decision
    state["target_agent"] = parsed_json['target_agent']
    
    formatted_output = f"🛎️ THE MAITRE D': TICKET ROUTED\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"DESTINATION: {parsed_json['target_agent'].upper()}\n"
    formatted_output += f"RATIONALE: {parsed_json['rationale']}\n"
            
    return {"final_output": formatted_output, "state": state}

def node_expediter(state: BrigadeState):
    print("--- THE EXPEDITER IS CHECKING THE PASS ---")
    
# 1. Build a thread-safe local service
    creds = get_gmail_credentials()
    service = build('gmail', 'v1', credentials=creds)

    # 2. Fetch Live Emails (Last 7 Days, filtering for recruiters/interviews)
    query = "(interview OR recruiter OR application OR \"technical screen\") newer_than:7d"
    results = service.users().messages().list(userId='me', q=query).execute()
    
    # Extract the messages array from the results payload
    messages = results.get('messages', [])

    if not messages:
        return {"final_output": "📭 THE EXPEDITER: No relevant emails found in the last 7 days."}

    # 3. Load the Local Ledger
    ledger_file = "processed_tickets.json"
    if os.path.exists(ledger_file):
        with open(ledger_file, "r") as file:
            processed_ids = json.load(file)
    else:
        processed_ids = []

    # 4. Filter and Extract the full text of NEW emails
    new_emails = []
    for msg in messages:
        msg_id = msg['id']
        if msg_id not in processed_ids:
            # Fetch the actual payload of the email
            full_msg = service.users().messages().get(userId='me', id=msg_id, format='minimal').execute()
            new_emails.append({
                "id": msg_id, 
                "snippet": full_msg.get('snippet', 'No preview available')
            })

    if not new_emails:
        return {"final_output": "📭 THE EXPEDITER: No new interview requests. The board is clear."}

    # 5. Send to Gemini for Extraction
    system_instruction = """
    You are The Expediter. You monitor an incoming email feed for a senior data scientist.
    Read these email snippets and identify legitimate interview requests or recruiter follow-ups.

    The Output Protocol:ngrok
    1. new_opportunities: True if legitimate requests exist.
    2. summary: A brutally concise summary of who reached out and what they want.
    3. action_items: Exactly what needs to happen next.
    
    Tone: Highly efficient, structured, and urgent.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Extract action items from these live emails:\n{json.dumps(new_emails)}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ExpediterOutput,
            temperature=0.0, 
        )
    )
    
    parsed_json = json.loads(response.text)
    
    # 6. Update the Ledger
    for msg in new_emails:
        processed_ids.append(msg["id"])
        
    with open(ledger_file, "w") as file:
        json.dump(processed_ids, file)
    
    # 7. Format the Output
    formatted_output = f"🎟️ THE EXPEDITER: NEW TICKETS ON THE BOARD\n"
    formatted_output += f"-" * 40 + "\n"
    
    formatted_output += f"OPPORTUNITIES:\n"
    for item in parsed_json.get('summary', []):
        formatted_output += f"{item}\n"
    formatted_output += "\n"
        
    formatted_output += f"ACTION REQUIRED:\n"
    for action in parsed_json.get('action_items', []):
        formatted_output += f"{action}\n"
            
    return {"final_output": formatted_output}

import base64
import json
import re
from datetime import datetime

def node_courier(state: BrigadeState):
    print("--- THE COURIER IS RUNNING THE TICKET ---")
    
    envelope = state.get("webhook_data")
    if not envelope:
        print("[ERROR]: Courier triggered without webhook data.")
        return {"final_output": "Failed."}

    # 1. Decode the Google Pub/Sub Payload
    pubsub_message = envelope.get("message")
    if pubsub_message and "data" in pubsub_message:
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        msg_json = json.loads(data)
        
        # historyId tells us exactly what changed in the inbox
        history_id = msg_json.get("historyId")
        if not history_id:
            return {"final_output": "No history ID found."}

        try:
            # 2. Fetch the specific email using a thread-safe service
            creds = get_gmail_credentials()
            service = build('gmail', 'v1', credentials=creds)
            
            # Ask Gmail what message triggered this historyId change
            history = service.users().history().list(userId='me', startHistoryId=history_id).execute()
            changes = history.get('history', [])
            
            if not changes:
                return {"final_output": "No new messages found."}
                
            # Grab the ID of the new message
            msg_id = changes[0]['messages'][0]['id']
            
            # Fetch the actual email payload
            full_msg = service.users().messages().get(userId='me', id=msg_id).execute()
            
            # 3. Extract Subject and Snippet (where the URL usually lives in these alerts)
            headers = full_msg['payload'].get('headers', [])
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "")
            snippet = full_msg.get('snippet', "")

            # 4. Parse the Author and the URL
            author_match = re.search(r'(.*?)\s+has a new post', subject)
            author = author_match.group(1) if author_match else "Unknown Target"
            
            url_match = re.search(r'(https://www\.linkedin\.com/posts/[^\s]+)', snippet)
            post_url = url_match.group(0) if url_match else "URL not found"

            # 5. Idempotency Check & Write to Ledger
            ledger_file = "daily_watering_holes.json"
            target_data = {
                "author": author,
                "url": post_url,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for duplicates to prevent Pub/Sub retry spam
            is_duplicate = False
            if os.path.exists(ledger_file):
                with open(ledger_file, "r") as file:
                    for line in file:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("url") == post_url:
                                is_duplicate = True
                                break
                        except json.JSONDecodeError:
                            continue
                            
            if is_duplicate:
                print(f"⏭️  Target {author} already logged today. Skipping duplicate webhook.")
                return {"final_output": "Duplicate payload ignored."}

            with open(ledger_file, "a") as file:
                json.dump(target_data, file)
                file.write("\n")
            print(f"🎯 Target Acquired: {author} logged to {ledger_file}")
            
        except Exception as e:
            print(f"[ERROR]: Courier failed during extraction: {e}")
            return {"final_output": "Courier failed."}
            
    return {"final_output": "Courier sequence complete."}

def node_dinner_service(state: BrigadeState):
    print("--- 🍽️ DINNER SERVICE: DAILY TARGETS ---")
    ledger_file = "daily_watering_holes.json"
    
    # 1. Check if the ledger exists and has data
    if not os.path.exists(ledger_file) or os.stat(ledger_file).st_size == 0:
        return {"final_output": "The board is empty. No targets acquired today."}
        
    with open(ledger_file, "r") as file:
        lines = file.readlines()
        
    # 2. Format the dashboard
    formatted_output = "\n🎯 THE HIT LIST:\n"
    formatted_output += "-" * 40 + "\n"
    
    for idx, line in enumerate(lines, 1):
        try:
            target = json.loads(line.strip())
            # Parse the timestamp to make it readable
            time_obj = datetime.fromisoformat(target['timestamp'])
            formatted_time = time_obj.strftime("%I:%M %p")
            
            formatted_output += f"[{idx}] {target['author']} (Posted at {formatted_time})\n"
            formatted_output += f"    Link: {target['url']}\n"
        except (json.JSONDecodeError, KeyError):
            continue
            
    print(formatted_output)
    
    # 3. Interactive prompt to clear the board
    clear = input("\nClear the board for tomorrow? (y/n): ")
    if clear.lower() == 'y':
        open(ledger_file, 'w').close()  # Wipes the file completely clean
        return {"final_output": "Board cleared. Service complete. Log off."}
    else:
        return {"final_output": "Board retained. Service paused."}
    
def node_pastry_chef(state: BrigadeState):
    print("--- THE PASTRY CHEF IS PLATING ---")
    raw_data = state["user_input"]
    
    system_instruction = """
    You are The Pastry Chef. You are a rigorous data visualization architect and storyteller.
    Users will hand you raw data outputs, statistical findings, or dense analytical summaries.
    Your only job is to dictate exactly how this data must be visually presented to high-level stakeholders.

    The Design Perimeter:
    - Enforce McKinsey-style action titles. The title is the takeaway.
    - Strip all non-data ink. Maximize the signal-to-noise ratio.
    - Default to clean, modern aesthetics: gray baselines with a single, high-contrast color used exclusively to highlight the core insight.
    - If the user provides raw numbers, do not visualize the data for them. Give them the blueprint to build the visualization themselves.

    Tone: Sophisticated, design-obsessed, and ruthlessly minimal.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash', # Flash is perfect for structured formatting and design mapping
        contents=f"Provide the visual and structural plating requirements for this raw data/finding: {raw_data}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=PastryChefOutput,
            temperature=0.2, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🍰 THE PASTRY CHEF: VISUAL BLUEPRINT\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"THE ACTION TITLE:\n{parsed_json['action_title']}\n\n"
    formatted_output += f"THE ARCHITECTURE:\n{parsed_json['chart_architecture']}\n\n"
    
    if parsed_json['visual_cues']:
        formatted_output += f"VISUAL HIERARCHY:\n"
        for cue in parsed_json['visual_cues']:
            formatted_output += f"{cue}\n"
        formatted_output += "\n"
        
    if parsed_json['clutter_reduction']:
        formatted_output += f"SIGNAL TO NOISE:\n"
        for chop in parsed_json['clutter_reduction']:
            formatted_output += f"{chop}\n"
        formatted_output += "\n"
        
    formatted_output += f"THE NARRATIVE ARC:\n{parsed_json['presentation_narrative']}"
            
    return {"final_output": formatted_output}

def node_critic(state: BrigadeState):
    print("--- THE CRITIC IS REVIEWING THE PROPOSAL ---")
    proposal = state["user_input"]
    
    system_instruction = """
    You are The Critic. You are a highly skeptical, financially ruthless C-suite executive (CFO/COO).
    Users will pitch you data science models, AI architectures, or strategic pivots.
    Your only job is to find the business vulnerabilities—the hidden costs, the adoption friction, and the ROI gaps.

    The Adversarial Perimeter:
    - Ignore the underlying math. Assume the math works. Attack the business logic.
    - Attack the "So What?": If this model predicts X with 99% accuracy, how does that actually save money or generate revenue?
    - Attack the adoption curve: Will operators actually use this, or will it die in a dashboard?
    - Attack the maintenance cost: Who pays for the compute and API calls next year?

    Tone: Cold, financially driven, and demanding. You do not care about elegant code; you care about leverage.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-pro', # Pro is required here for deep strategic reasoning
        contents=f"Red-team this proposal from a stakeholder/board perspective: {proposal}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=CriticOutput,
            temperature=0.4, # Slight variance to generate creative, non-obvious adversarial angles
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🕴️ THE CRITIC: STAKEHOLDER RED-TEAM\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"CORE BUSINESS RISK:\n{parsed_json['core_business_risk']}\n\n"
    
    formatted_output += f"THE INQUISITION (TOP 3 BOARD QUESTIONS):\n"
    for q in parsed_json['adversarial_questions']:
        formatted_output += f"{q}\n"
    formatted_output += "\n"
        
    formatted_output += f"DEFENSIVE STRATEGY:\n{parsed_json['defensive_strategy']}"
            
    return {"final_output": formatted_output}

def node_aboyeur(state: BrigadeState):
    print("--- THE ABOYEUR IS DRAFTING THE BUILD LOG ---")
    raw_logs = state["user_input"]
    
    system_instruction = """
    You are The Aboyeur. You are a Senior Engineering Manager and Technical Writer.
    Users will hand you raw git commit logs, code diffs, or messy developer notes.
    Your job is to synthesize this chaos into a high-signal, production-grade Build Log.

    The Documentation Perimeter:
    - Ignore pure noise (e.g., "typo fix", "merge branch", "wip"). Extract the actual engineering delta.
    - Translate code-level changes into system-level impact. If a node was refactored to use a thread-safe singleton, the impact is "Concurrency stabilization," not just "Changed a variable."
    - Be ruthless about identifying technical debt. If the logs imply a temporary fix or a hardcoded path, flag it.
    
    Tone: Professional, authoritative, and structurally flawless.
    """
    
    # Using Flash because summarization and formatting is its primary strength
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Synthesize this raw commit data into a formal Build Log:\n\n{raw_logs}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=AboyeurOutput,
            temperature=0.1, # Keep it highly deterministic; we want facts, not creative writing
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"📢 THE ABOYEUR: OFFICIAL BUILD LOG\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"VERSION BUMP: {parsed_json['semantic_version']}\n\n"
    formatted_output += f"EXECUTIVE SUMMARY:\n{parsed_json['executive_summary']}\n\n"
    
    if parsed_json['architectural_shifts']:
        formatted_output += f"SYSTEM ARCHITECTURE:\n"
        for shift in parsed_json['architectural_shifts']:
            formatted_output += f"{shift}\n"
        formatted_output += "\n"
        
    if parsed_json['component_updates']:
        formatted_output += f"TACTICAL UPDATES:\n"
        for update in parsed_json['component_updates']:
            formatted_output += f"{update}\n"
        formatted_output += "\n"
        
    if parsed_json['technical_debt']:
        formatted_output += f"TECHNICAL DEBT & RISKS:\n"
        for debt in parsed_json['technical_debt']:
            formatted_output += f"{debt}\n"
            
    return {"final_output": formatted_output}

def node_michelin_inspector(state: BrigadeState):
    print("--- THE MICHELIN INSPECTOR IS EVALUATING THE JD ---")
    job_description = state["user_input"]
    
    system_instruction = """
    You are The Michelin Inspector. You are a ruthless, highly strategic technical evaluator. Your sole purpose is to analyze job descriptions and deliver a definitive verdict for Erik.

    The Candidate Profile:
    Erik is a Senior Builder specializing in production-grade, multi-agent AI systems for high-stakes, regulated environments. He bridges the gap between non-deterministic LLMs and rigorous statistical inference.

    Tooling Philosophy: Agnostic and pragmatic. He seeks the most powerful and effective tools available for the task, whether they are Google ecosystems or superior non-Google alternatives. Look for the application of agentic workflows (state management, complex orchestration), predictive ML, and heavy-duty data engineering, rather than just keyword-matching specific frameworks like DSPy or LangGraph.

    The Standard: He does not build "toys." He builds self-correcting, multi-step reasoning architectures.

    The Inspection Protocol:
    1. The Agentic Frontier: Does this involve advanced model optimization or orchestration (RAG, vector DBs, state management)? If it's just basic SQL dashboarding or legacy ML without an AI/agentic component, it is a hard pass.
    2. The Execution Gap: Does the role require bridging theoretical AI with production deployment?
    3. Title vs. Reality: Ignore the title. Look for architectural responsibilities and actual engineering depth.
    4. Comp Alignment & Exceptions: Does it meet the $200k+ floor? If the base is $175k–$199k, it may only pass as a [STRATEGIC EXCEPTION] if it offers unparalleled architectural authority, massive equity upside, or a pristine technical playground.
    5. The Flexibility Tax: Erik currently operates with high autonomy. Flag any signs of a 60-hour grind culture, rigid in-office requirements that force a heavy Chicago commute, or bureaucracy that would destroy his bandwidth and keep him away from his family.

    Tone: Cold, clinical, and protective. You value Erik's time above all else. No sycophancy.
    """
    
    # Using Pro here because evaluating career leverage and reading between the lines of corporate JD speak requires deep reasoning.
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=f"Evaluate this Job Description:\n\n{job_description}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=MichelinInspectorOutput,
            temperature=0.2, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🕵️ THE MICHELIN INSPECTOR: STRATEGIC EVALUATION\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"⚖️ THE VERDICT:\n{parsed_json['verdict']}\n\n"
    
    formatted_output += f"📝 THE REALITY CHECK:\n"
    for check in parsed_json['reality_check']:
        formatted_output += f"• {check}\n"
    formatted_output += "\n"
        
    formatted_output += f"🔍 THE MISSING INGREDIENTS:\n"
    for unknown in parsed_json['missing_ingredients']:
        formatted_output += f"• {unknown}\n"
            
    return {"final_output": formatted_output}

def node_forager(state: BrigadeState):
    print("--- THE FORAGER IS SEARCHING QDRANT ---")
    query = state["user_input"]

    # 1. Embed the user query using the NEW SDK and model
    embed_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
    )
    query_vector = embed_response.embeddings[0].values

    # 2. Search local Qdrant database using the new query_points API
    search_response = qdrant.query_points(
        collection_name="enterprise_knowledge",
        query=query_vector,
        limit=5 
    )
    
    # 3. Concatenate the retrieved chunks from the .points array
    knowledge_base = "\n\n".join([hit.payload["text"] for hit in search_response.points])

    # 4. Pass the retrieved context to Gemini for synthesis and formatting
    system_instruction = """
    You are The Forager. You are the semantic memory of this Multi-Agent System.
    Users will hand you a query, and your job is to extract the highly relevant historical context and architectural patterns from the provided knowledge base.

    The Extraction Perimeter:
    - You do not write new code or solve the problem. You only retrieve and synthesize existing knowledge.
    - Be aggressive about finding semantic neighbors—concepts that are conceptually linked even if the user didn't explicitly search for them.
    
    Tone: Scholarly, precise, and memory-driven.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Extract relevant context for this query:\n{query}\n\nKnowledge Base:\n{knowledge_base}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ForagerOutput,
            temperature=0.0, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🧠 THE FORAGER: CONTEXT RETRIEVED\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"CONFIDENCE SCORE: {parsed_json['confidence_score']}\n\n"
    formatted_output += f"SYNTHESIZED CONTEXT:\n{parsed_json['retrieved_context']}\n\n"
    
    if parsed_json['semantic_neighbors']:
        formatted_output += f"SEMANTIC NEIGHBORS:\n"
        for neighbor in parsed_json['semantic_neighbors']:
            formatted_output += f"{neighbor}\n"
            
    return {"final_output": formatted_output}

def node_regulated_release(state: BrigadeState):
    print("\n[INITIATING REGULATED RELEASE PIPELINE]")
    
    # STEP 1: The Red-Team Audit (Health Inspector)
    inspector_result = node_health_inspector(state)
    
    # We parse the output to see if it passed the HIPAA/FERPA check
    if "CRITICAL VIOLATION" in inspector_result["final_output"] or "HIGH RISK" in inspector_result["final_output"]:
        print("\n🛑 PIPELINE HALTED: The Health Inspector found severe compliance risks.")
        return {"final_output": inspector_result["final_output"] + "\n\n[PIPELINE ABORTED: Fix compliance issues before formatting.]"}
        
    print("✅ Health Inspector cleared the draft. Passing to Mother Sauce for SOP compliance...")
    
    # STEP 2: The Formatting Enforcement (Mother Sauce)
    # We pass the same draft to the Mother Sauce to ensure it matches enterprise SOPs
    sauce_result = node_mother_sauce(state)
    
    final_chain_output = f"🏗️ ENTERPRISE PIPELINE SUCCESS\n"
    final_chain_output += f"=" * 40 + "\n"
    final_chain_output += f"PHASE 1: SECURITY CLEARANCE\n[PASSED] No Critical/High HIPAA or FERPA risks detected.\n\n"
    final_chain_output += f"PHASE 2: SOP ENFORCEMENT\n"
    final_chain_output += sauce_result["final_output"]
    
    return {"final_output": final_chain_output}

def node_system_design(state: BrigadeState):
    print("\n[INITIATING SYSTEM DESIGN PIPELINE]")
    original_prompt = state["user_input"]
    
    # STEP 1: Context Retrieval (The Forager)
    forager_result = node_forager(state)
    
    if "[CRITICAL ERROR]" in forager_result["final_output"]:
        print("⚠️ Forager failed to access knowledge base. Proceeding without historical context.")
        context = "No historical context available."
    else:
        print("✅ Forager retrieved enterprise context. Passing to Butcher...")
        context = forager_result["final_output"]
        
    # STEP 2: First-Principles Deconstruction (The Butcher)
    # We combine the user's raw problem with the enterprise context so the Butcher makes informed cuts
    state["user_input"] = f"Original Problem: {original_prompt}\n\nEnterprise Context:\n{context}"
    butcher_result = node_butcher(state)
    print("✅ Butcher deconstructed the problem. Passing to Procurement for audit...")
    
    # STEP 3: Infrastructure Audit (The Procurement Chief)
    # We hand the Butcher's naked architectural plan to Procurement to check for missing infrastructure
    state["user_input"] = butcher_result["final_output"]
    procurement_result = node_procurement_chief(state)
    print("✅ Procurement audit complete. Assembling final blueprint...")
    
    # Restore the original input to the state just to keep the object clean
    state["user_input"] = original_prompt
    
    final_chain_output = f"📐 ENTERPRISE SYSTEM DESIGN BLUEPRINT\n"
    final_chain_output += f"=" * 40 + "\n\n"
    
    # We append the outputs sequentially to give the user the full thought process
    final_chain_output += f"{forager_result['final_output']}\n\n"
    final_chain_output += f"{butcher_result['final_output']}\n\n"
    final_chain_output += f"{procurement_result['final_output']}"
    
    return {"final_output": final_chain_output}

def node_executive_briefing(state: BrigadeState):
    print("\n[INITIATING EXECUTIVE BRIEFING PIPELINE]")
    original_prompt = state["user_input"]
    
    # STEP 1: The Red-Team Business Audit (The Critic)
    # The Critic attacks the raw finding to find ROI and adoption holes
    critic_result = node_critic(state)
    print("✅ Critic analyzed the business risks. Passing to Pastry Chef for plating...")
    
    # STEP 2: The Presentation Blueprint (The Pastry Chef)
    # We combine the raw data with the Critic's defense strategy so the slide deck is pre-armored
    state["user_input"] = f"Raw Finding/Data:\n{original_prompt}\n\nCritic's Defensive Strategy to Incorporate:\n{critic_result['final_output']}"
    pastry_result = node_pastry_chef(state)
    print("✅ Pastry Chef designed the visual blueprint. Assembling final briefing...")
    
    # Restore the original input to keep the state clean
    state["user_input"] = original_prompt
    
    final_chain_output = f"📊 EXECUTIVE BRIEFING STRATEGY\n"
    final_chain_output += f"=" * 40 + "\n\n"
    
    # Append the outputs sequentially
    final_chain_output += f"{critic_result['final_output']}\n\n"
    final_chain_output += f"{pastry_result['final_output']}"
    
    return {"final_output": final_chain_output}

# --- CLI EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brigade MAS: Multi-Agent Orchestration Engine")
    
    parser.add_argument(
        "target", 
        choices=["sous-chef", "exec-chef", "sommelier", "butcher", "procurement", "health-inspector", "expediter", "courier", "route", "service", "pastry-chef", "critic", "aboyeur", "michelin-inspector", \
                 "forager", "release", "system-design", "executive-briefing"], 
        help="Specify the agent to call"
    )
    
    parser.add_argument(
        "input_text", 
        type=str, 
        nargs='?', 
        default="",
        help="The raw term, problem, or dataset to process"
    )

    args = parser.parse_args()

    # Cleaned up state initialization to perfectly match BrigadeState TypedDict
    current_state: BrigadeState = {
        "user_input": args.input_text,
        "target_agent": "", 
        "target_domain": "",
        "current_draft": "",
        "qa_feedback": [],
        "final_output": "",
        "webhook_data": {} # The Courier's lane is initialized
    }

    try:
        # THE ROUTER AUTO-EXECUTION BLOCK
        if args.target == "route":
            if not args.input_text:
                print("\n[ERROR]: The Router needs a prompt to route.\n")
                sys.exit(1)
                
            router_result = node_maitre_d(current_state)
            print("\n" + router_result["final_output"] + "\n")
            
            current_state = router_result.get("state", current_state)
            args.target = current_state.get("target_agent", "")
            
            print(f"--- AUTO-HANDOFF TO: {args.target.upper()} ---\n")

        # THE STATION EXECUTION BLOCKS
        if args.target == "sous-chef":
            result = node_sous_chef(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "exec-chef":
            result = node_executive_chef(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "sommelier":
            result = node_sommelier(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "butcher":
            result = node_butcher(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "procurement":
            result = node_procurement_chief(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "health-inspector":
            result = node_health_inspector(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "expediter":
            result = node_expediter(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "courier":
            result = node_courier(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "service":
            result = node_dinner_service(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "pastry-chef":
            result = node_pastry_chef(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "critic":
            result = node_critic(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "aboyeur":
            result = node_aboyeur(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "michelin-inspector":
            result = node_michelin_inspector(current_state)
            print("\n" + result["final_output"] + "\n")

        elif args.target == "forager":
            result = node_forager(current_state)
            print("\n" + result["final_output"] + "\n")
            
        elif args.target == "release":
            result = node_regulated_release(current_state)
            print("\n" + result["final_output"] + "\n")
        elif args.target == "system-design":
            result = node_system_design(current_state)
            print("\n" + result["final_output"] + "\n")
        elif args.target == "executive-briefing":
            result = node_executive_briefing(current_state)
            print("\n" + result["final_output"] + "\n")
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}\n")
        sys.exit(1)

