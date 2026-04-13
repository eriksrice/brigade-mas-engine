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

# Load environment variables from .env file
load_dotenv()

# 1. Initialize your Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 2. Define the State
class BrigadeState(TypedDict):
    user_input: str
    target_domain: str
    current_draft: str
    qa_feedback: list[str]
    final_output: str

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
    target_agent: str = Field(description="Must be exactly one of: 'sous-chef', 'exec-chef', 'sommelier', 'butcher', 'procurement', 'mother-sauce', 'health-inspector', or 'expediter'.")
    rationale: str = Field(description="A brief explanation of why this agent is the perfect fit for the user's prompt.")

class ExpediterOutput(BaseModel):
    new_opportunities: bool = Field(description="True if any NEW, unprocessed interview requests or recruiter emails were found.")
    summary: list[str] = Field(description="Format each as: '📅 [Company Name]: [Brief summary of the request].'")
    action_items: list[str] = Field(description="Format each as: '✅ Action: [What Erik needs to do next, e.g., send availability].'")

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
    
    file_path = r"C:\Users\erice\Downloads\Brigade_Project\AI_Data Science MASTER.txt" 
    
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

def node_the_range(state: BrigadeState):
    print("--- THE RANGE IS CALIBRATING COMPUTE ---")
    task = state["user_input"]
    
    system_instruction = """
    You are The Range. You are a senior DevOps and AI infrastructure router.
    Your only job is to look at a user's prompt and determine the cheapest, fastest model capable of executing it flawlessly.
    
    The Compute Tiers:
    1. Efficiency (gemini-2.5-flash): Use for text summarization, formatting, basic classification, RAG retrieval, and simple entity extraction. High speed, near-zero cost.
    2. Performance (gemini-2.5-pro): Use for complex coding, architectural gap analysis, nuance-heavy writing, and multi-step logic. Medium speed, medium cost.
    3. Reasoning (o3-mini / deepseek-r1): Use ONLY for extreme mathematical proofs, hyper-complex logic puzzles, or zero-shot novel algorithm design. Very slow, very expensive.

    The Rule: Always route to the lowest possible tier that will succeed. Do not waste expensive compute on simple tasks.
    
    Tone: Purely architectural and financially ruthless.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Analyze this task and route the compute: {task}",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ComputeRouteOutput,
            temperature=0.1, 
        )
    )
    
    import json
    parsed_json = json.loads(response.text)
    
    formatted_output = f"🎛️ THE RANGE: COMPUTE ALLOCATED\n"
    formatted_output += f"-" * 40 + "\n"
    formatted_output += f"TIER: {parsed_json['model_tier'].upper()}\n"
    formatted_output += f"TARGET MODEL: {parsed_json['api_target']}\n\n"
    formatted_output += f"JUSTIFICATION:\n{parsed_json['rationale']}"
            
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
    - Data at Rest/Transit: Are they storing unencrypted CSVs in S3? Are they using secure protocols?
    - Injection/Access: Can this system be subjected to SQL injection or prompt injection?
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
    
    # 1. Google OAuth 2.0 Authentication Flow
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    
    # Check if we already logged in previously
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    # If not valid, trigger the browser login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials so you don't have to log in every time
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Build the Gmail Service
    service = build('gmail', 'v1', credentials=creds)

    # 2. Fetch Live Emails (Last 7 Days, filtering for recruiters/interviews)
    query = "(interview OR recruiter OR application OR \"technical screen\") newer_than:7d"
    results = service.users().messages().list(userId='me', q=query).execute()
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

    The Output Protocol:
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

# --- CLI EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brigade MAS: Multi-Agent Orchestration Engine")
    
    parser.add_argument(
        "target", 
        choices=["sous-chef", "exec-chef", "sommelier", "butcher", "procurement", "the-range", "health-inspector", "expediter", "route"], 
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

    # Added target_agent to the state dictionary
    current_state: BrigadeState = {
        "user_input": args.input_text,
        "target_agent": "", 
        "target_domain": "",
        "compute_route": "",
        "current_draft": "",
        "compliance_flags": [],
        "final_output": ""
    }

    try:
        # THE ROUTER AUTO-EXECUTION BLOCK
        if args.target == "route":
            if not args.input_text:
                print("\n[ERROR]: The Router needs a prompt to route.\n")
                sys.exit(1)
                
            # Run the Maitre D'
            router_result = node_maitre_d(current_state)
            print("\n" + router_result["final_output"] + "\n")
            
            # Update current state based on Maitre D' decision
            current_state = router_result["state"]
            
            # Reassign args.target to the agent the Maitre D' picked
            args.target = current_state["target_agent"]
            
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
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}\n")
        sys.exit(1)