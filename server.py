# ═══════════════════════════════════════════════════════════════
#  ResumeAI — Python Flask Backend
#  Flask + SQLite + NumPy + Groq SDK
# ═══════════════════════════════════════════════════════════════

import os, json, re, time, threading
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from groq import Groq                          # ← Groq SDK
from dotenv import load_dotenv
from database import init_db, save_analysis, get_all_analyses, get_stats_summary

# ─── Load env ────────────────────────────────────────────────
load_dotenv()

# ─── Init ────────────────────────────────────────────────────
app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute"],
    storage_uri="memory://"
)

# ── Groq client ───────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))           # ← GROQ_API_KEY
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # override in .env if needed
PORT       = int(os.getenv("PORT", 5000))

UPLOAD_DIR  = "uploads"
MAX_MB      = int(os.getenv("MAX_FILE_SIZE_MB", 10))
ALLOWED_EXT = {".pdf", ".doc", ".docx", ".txt"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Init DB ─────────────────────────────────────────────────
init_db()

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

def schedule_delete(path, delay=300):
    def _del():
        time.sleep(delay)
        try: os.remove(path)
        except: pass
    threading.Thread(target=_del, daemon=True).start()

def extract_text(filepath, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        import pdfplumber
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    if ext in (".doc", ".docx"):
        import mammoth
        with open(filepath, "rb") as f:
            return mammoth.extract_raw_text(f).value
    raise ValueError("Unsupported format")

def call_groq(messages, system="", max_tokens=1000):
    """Drop-in replacement for call_claude() — uses Groq's chat completions."""
    groq_messages = []
    if system:
        groq_messages.append({"role": "system", "content": system})
    groq_messages.extend(messages)

    resp = client.chat.completions.create(
        model      = GROQ_MODEL,
        max_tokens = max_tokens,
        messages   = groq_messages
    )
    return resp.choices[0].message.content or ""

def parse_json(raw):
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)

# NumPy helper — compute a weighted average score for dimensions
def compute_weighted_score(dimensions: list[dict]) -> float:
    weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
    scores  = np.array([d["score"] for d in dimensions], dtype=float)
    if len(scores) != len(weights):
        return float(np.mean(scores))
    return round(float(np.dot(weights, scores)), 1)

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("public", "index.html")

# ── Health ───────────────────────────────────────────────────
@app.get("/api/health")
def health():
    stats = get_stats_summary()
    return jsonify({
        "status" : "ok",
        "version": "2.0.0",
        "model"  : GROQ_MODEL,
        "db"     : stats,
        "time"   : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    })

# ── Upload ───────────────────────────────────────────────────
@app.post("/api/upload")
@limiter.limit("20 per minute")
def upload():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["resume"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PDF, DOC, DOCX, or TXT."}), 415

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{filename}")
    file.save(filepath)
    schedule_delete(filepath)

    if os.path.getsize(filepath) > MAX_MB * 1024 * 1024:
        os.remove(filepath)
        return jsonify({"error": f"File too large. Max {MAX_MB}MB."}), 413

    try:
        text = extract_text(filepath, filename)
        if len(text.strip()) < 30:
            return jsonify({"error": "Could not extract enough text. Try a different format."}), 422
        return jsonify({"success": True, "filename": filename, "textLength": len(text), "text": text[:8000]})
    except Exception as e:
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500

# ── Analyze ──────────────────────────────────────────────────
@app.post("/api/analyze")
@limiter.limit("20 per minute")
def analyze():
    body        = request.get_json(force=True)
    resume_text = body.get("resumeText", "").strip()
    target_role = body.get("targetRole", "")

    if len(resume_text) < 30:
        return jsonify({"error": "Resume text too short or missing."}), 400

    system = "You are a world-class resume analyst. Return ONLY valid JSON — no markdown, no code fences."
    prompt = f"""Analyze this resume{f' for the role "{target_role}"' if target_role else ''} and return ONLY valid JSON:

{{
  "score": <0-100>,
  "scoreLabel": "<Poor|Fair|Good|Strong|Excellent>",
  "scoreDesc": "<2 sentences>",
  "stats": {{"skills":<int>,"experience":"<str>","education":"<str>","projects":<int>}},
  "skillDimensions": [
    {{"name":"Content Quality","score":<0-100>}},
    {{"name":"ATS Compatibility","score":<0-100>}},
    {{"name":"Formatting","score":<0-100>}},
    {{"name":"Impact & Metrics","score":<0-100>}},
    {{"name":"Skills Relevance","score":<0-100>}}
  ],
  "strengths"      : ["<s1>","<s2>","<s3>"],
  "improvements"   : ["<i1>","<i2>","<i3>","<i4>"],
  "topSkills"      : ["<sk1>","<sk2>","<sk3>","<sk4>","<sk5>","<sk6>"],
  "missingSkills"  : ["<m1>","<m2>","<m3>","<m4>"],
  "atsKeywords"    : ["<k1>","<k2>","<k3>","<k4>","<k5>"],
  "summary"        : "<2-paragraph overview>",
  "careerLevel"    : "<Student|Entry Level|Mid Level|Senior|Executive>",
  "industryFit"    : ["<ind1>","<ind2>","<ind3>"],
  "jobMatches"     : [
    {{"title":"<t>","company":"<c>","emoji":"<e>","location":"<l>","salary":"<s>","type":"<Full-time|Internship|Contract>","match":<70-99>,"desc":"<2 sentences>","skills":["<sk1>","<sk2>","<sk3>"]}}
  ]
}}

Resume:
{resume_text[:4000]}"""

    try:
        raw      = call_groq([{"role": "user", "content": prompt}], system, max_tokens=1800)
        analysis = parse_json(raw)

        # NumPy: recompute weighted score from dimensions
        if "skillDimensions" in analysis:
            analysis["weightedScore"] = compute_weighted_score(analysis["skillDimensions"])

        save_analysis(resume_text[:500], target_role, analysis.get("score", 0), json.dumps(analysis))
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ── Chat ─────────────────────────────────────────────────────
@app.post("/api/chat")
@limiter.limit("20 per minute")
def chat():
    body        = request.get_json(force=True)
    message     = body.get("message", "").strip()
    resume_text = body.get("resumeText", "")
    history     = body.get("history", [])

    if not message:
        return jsonify({"error": "Message is required."}), 400

    system  = "You are an expert AI career advisor. Give practical, concise, encouraging advice. Plain text only."
    context = f"\n\nResume context:\n{resume_text[:1500]}" if resume_text else ""
    msgs    = history[-6:] + [{"role": "user", "content": message + context}]

    try:
        reply = call_groq(msgs, system, max_tokens=600)
        return jsonify({"success": True, "reply": reply})
    except Exception as e:
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500

# ── Build Resume ─────────────────────────────────────────────
@app.post("/api/build-resume")
@limiter.limit("20 per minute")
def build_resume():
    body = request.get_json(force=True)
    name, role = body.get("name","").strip(), body.get("role","").strip()
    if not name or not role:
        return jsonify({"error": "Name and target role are required."}), 400

    system = "You are a professional resume writer. Create ATS-optimized resumes. Plain text only."
    prompt = f"""Create a complete, professional, ATS-optimized resume:

Name: {name}
Target Role: {role}
Contact: {body.get('contact','Not provided')}
Education: {body.get('education','Not provided')}
Skills: {body.get('skills','Not provided')}
Experience/Projects: {body.get('experience','Not provided')}
Style: {body.get('template','Modern Professional')}

Instructions:
1. Compelling 2-3 sentence professional summary
2. Strong bullet points with action verbs (Led, Built, Developed, Optimized)
3. Group skills by category
4. Add 2-3 recommended certifications
5. Key Achievements section with 2-3 impactful bullets
Return clean plain text resume only."""

    try:
        resume = call_groq([{"role": "user", "content": prompt}], system, max_tokens=1500)
        return jsonify({"success": True, "resume": resume})
    except Exception as e:
        return jsonify({"error": f"Resume generation failed: {str(e)}"}), 500

# ── Cover Letter ─────────────────────────────────────────────
@app.post("/api/cover-letter")
@limiter.limit("20 per minute")
def cover_letter():
    body        = request.get_json(force=True)
    resume_text = body.get("resumeText","").strip()
    job_title   = body.get("jobTitle","").strip()
    if not resume_text or not job_title:
        return jsonify({"error": "Resume text and job title are required."}), 400

    prompt = f"""Write a professional cover letter for:
Job Title: {job_title}
Company: {body.get('companyName','the company')}
Tone: {body.get('tone','Professional and enthusiastic')}

Resume:
{resume_text[:2000]}

Write a 3-paragraph cover letter (Opening, Why I'm a fit, Closing). Personalized, not generic. Plain text only."""

    try:
        letter = call_groq([{"role": "user", "content": prompt}],
                           "You are an expert cover letter writer.", max_tokens=800)
        return jsonify({"success": True, "letter": letter})
    except Exception as e:
        return jsonify({"error": f"Cover letter failed: {str(e)}"}), 500

# ── Interview Prep ───────────────────────────────────────────
@app.post("/api/interview-prep")
@limiter.limit("20 per minute")
def interview_prep():
    body      = request.get_json(force=True)
    job_title = body.get("jobTitle","").strip()
    if not job_title:
        return jsonify({"error": "Job title is required."}), 400

    qtype  = body.get("questionType","mixed")
    resume = body.get("resumeText","")
    prompt = f"""Generate 8 {qtype} interview questions for "{job_title}".
{f'Resume context: {resume[:1000]}' if resume else ''}

Return ONLY valid JSON array:
[{{"question":"<q>","type":"<Behavioral|Technical|Situational|HR>","tip":"<1-sentence tip>","sample":"<2-sentence sample answer>"}}]"""

    try:
        raw       = call_groq([{"role": "user", "content": prompt}], max_tokens=1500)
        questions = parse_json(raw)
        return jsonify({"success": True, "questions": questions})
    except Exception as e:
        return jsonify({"error": f"Interview prep failed: {str(e)}"}), 500

# ── Skills Gap ───────────────────────────────────────────────
@app.post("/api/skills-gap")
@limiter.limit("20 per minute")
def skills_gap():
    body           = request.get_json(force=True)
    current_skills = body.get("currentSkills","").strip()
    target_role    = body.get("targetRole","").strip()
    if not current_skills or not target_role:
        return jsonify({"error": "currentSkills and targetRole are required."}), 400

    prompt = f"""Skills gap analysis:
Current Skills: {current_skills}
Target Role: {target_role}

Return ONLY valid JSON:
{{"matchingSkills":["<s1>"],"missingSkills":[{{"skill":"<n>","priority":"<High|Medium|Low>","learnIn":"<timeframe>","resource":"<resource>"}}],"roadmap":["<step1>","<step2>","<step3>","<step4>","<step5>"],"timeToReady":"<timeframe>","overallGap":"<Small|Moderate|Large>","encouragement":"<motivating 2 sentences>"}}"""

    try:
        raw = call_groq([{"role": "user", "content": prompt}], max_tokens=1200)
        gap = parse_json(raw)
        return jsonify({"success": True, "gap": gap})
    except Exception as e:
        return jsonify({"error": f"Skills gap failed: {str(e)}"}), 500

# ── Job Recommendations ──────────────────────────────────────
@app.post("/api/jobs")
@limiter.limit("20 per minute")
def jobs():
    body = request.get_json(force=True)
    if not body.get("skills") and not body.get("targetRole"):
        return jsonify({"error": "Provide skills or targetRole."}), 400

    prompt = f"""Suggest 6 ideal job matches:
Skills: {body.get('skills','Not specified')}
Experience: {body.get('experience','Fresher')}
Education: {body.get('education','Not specified')}
Target Role: {body.get('targetRole','Open')}

Return ONLY valid JSON array:
[{{"title":"<t>","company":"<c>","emoji":"<e>","location":"<l>","salary":"<s>","type":"<Full-time|Internship|Contract>","match":<70-99>,"desc":"<2-sentence desc>","tags":["<t1>","<t2>","<t3>"]}}]"""

    try:
        raw  = call_groq([{"role": "user", "content": prompt}], max_tokens=1200)
        jobs = parse_json(raw)
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        return jsonify({"error": f"Job recommendations failed: {str(e)}"}), 500

# ── Analytics ────────────────────────────────────────────────
@app.get("/api/analytics")
def analytics():
    """Returns aggregate stats computed with NumPy from the SQLite history."""
    rows = get_all_analyses()
    if not rows:
        return jsonify({"message": "No data yet.", "count": 0})

    scores = np.array([r["score"] for r in rows], dtype=float)
    return jsonify({
        "totalAnalyses" : len(rows),
        "avgScore"      : round(float(np.mean(scores)), 1),
        "medianScore"   : round(float(np.median(scores)), 1),
        "maxScore"      : int(np.max(scores)),
        "minScore"      : int(np.min(scores)),
        "stdDev"        : round(float(np.std(scores)), 1),
        "percentile75"  : round(float(np.percentile(scores, 75)), 1),
        "histogram"     : np.histogram(scores, bins=[0,40,60,75,90,101])[0].tolist(),
        "bucketLabels"  : ["Poor(0-40)","Fair(40-60)","Good(60-75)","Strong(75-90)","Excellent(90+)"]
    })

# ═══════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": f"Route not found: {request.method} {request.path}"}), 404

@app.errorhandler(429)
def rate_limit_hit(e):
    return jsonify({"error": "Too many requests. Please wait a moment."}), 429

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error. Please try again."}), 500

# ═══════════════════════════════════════════════════════════════
#  START
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n╔═══════════════════════════════════════╗")
    print(f"║   🚀 ResumeAI (Python) running        ║")
    print(f"║   URL  : http://localhost:{PORT}          ║")
    print(f"║   Model: {GROQ_MODEL:<29}║")
    print( "╚═══════════════════════════════════════╝\n")

    key = os.getenv("GROQ_API_KEY","")
    if not key or "xxx" in key:
        print("⚠️  WARNING: GROQ_API_KEY not set in .env!\n")

    app.run(host="0.0.0.0", port=PORT, debug=os.getenv("FLASK_DEBUG","false").lower()=="true")

    if __name__ == "__main__":
     app.run(host="0.0.0.0", port=5000)