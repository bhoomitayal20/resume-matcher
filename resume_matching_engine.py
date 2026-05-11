import math

# ============================================================
# SKILL_ALIASES — exactly as provided in problem sheet
# ============================================================
SKILL_ALIASES = {
    # Languages
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    # Web — Frontend
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android", "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# ============================================================
# RESUME DATASET — 10 candidates
# ============================================================
RESUMES = {
    "Arjun Sharma":    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",
    "Priya Nair":      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",
    "Rahul Gupta":     "Java, Spring Boot, MySql, Microservices, Docker, kubernates",
    "Sneha Patel":     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",
    "Vikram Singh":    "C++, Algoritms, Data Structure, competitive programming, python",
    "Ananya Krishnan": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",
    "Karan Mehta":     "Python, Sklearn, XGboost, feature engineering, SQL, tableau",
    "Deepika Rao":     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",
    "Aditya Kumar":    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",
    "Meera Iyer":      "python, R, statistics, ML, regression, clustering, Power-BI",
}

# ============================================================
# JOB DESCRIPTIONS — 3 JDs
# ============================================================
JD_RAW = {
    "JD1": {
        "label": "Kakao (ML Engineer)",
        "skills": ["python", "machine learning", "deep learning", "tensorflow",
                   "pytorch", "sql", "data visualization", "nlp", "bert",
                   "feature engineering", "statistics"]
    },
    "JD2": {
        "label": "Naver (Backend Engineer)",
        "skills": ["java", "spring boot", "mysql", "postgresql", "microservices",
                   "docker", "kubernetes", "rest api", "ci/cd", "redis"]
    },
    "JD3": {
        "label": "Line (Frontend Engineer)",
        "skills": ["javascript", "react", "vue", "typescript", "rest api",
                   "html/css", "node.js", "graphql", "redux", "jest", "aws"]
    },
}

# ============================================================
# STEP 1: Normalize skills — multi-word first, then single token
# ============================================================
MULTI_WORD_KEYS = sorted(
    [k for k in SKILL_ALIASES if " " in k or "-" in k],
    key=lambda x: -len(x)
)

def normalize_skills(raw_string):
    tokens = [t.strip().lower() for t in raw_string.split(",")]
    result = []
    seen = set()
    for token in tokens:
        matched = None
        # Try multi-word / hyphenated phrases first
        for phrase in MULTI_WORD_KEYS:
            if token == phrase:
                matched = SKILL_ALIASES[phrase]
                break
        # Fallback: single token lookup
        if matched is None:
            matched = SKILL_ALIASES.get(token)
        # Step 2: Deduplicate — only add if not seen before
        if matched and matched not in seen:
            result.append(matched)
            seen.add(matched)
    return result

# ============================================================
# STEP 3: Build shared vocabulary (alphabetically sorted)
# ============================================================
def build_vocabulary(normalized_resumes):
    all_skills = set()
    for skills in normalized_resumes.values():
        all_skills.update(skills)
    return sorted(all_skills)

# ============================================================
# STEP 4: Compute TF-IDF vectors for resumes
# ============================================================
def compute_tfidf(normalized_resumes, vocab):
    vocab_index = {skill: i for i, skill in enumerate(vocab)}
    N_DOCS = len(normalized_resumes)

    # Document frequency: how many resumes contain each skill
    df = {skill: 0 for skill in vocab}
    for skills in normalized_resumes.values():
        for s in set(skills):
            df[s] += 1

    # IDF = ln(N / df)  — no smoothing
    idf = {skill: math.log(N_DOCS / df[skill]) for skill in vocab}

    # TF-IDF vector per resume
    vectors = {}
    for name, skills in normalized_resumes.items():
        N = len(skills)          # unique skills after dedup
        vec = [0.0] * len(vocab)
        for skill in skills:
            tf = 1.0 / N         # TF = 1/N after deduplication
            vec[vocab_index[skill]] = tf * idf[skill]
        vectors[name] = vec

    return vectors, idf, df

# ============================================================
# STEP 5: Build binary JD vectors over same vocabulary
# ============================================================
def build_jd_vector(skill_list, vocab_index):
    vec = [0] * len(vocab_index)
    for raw in skill_list:
        r = raw.lower()
        matched = None
        for phrase in MULTI_WORD_KEYS:
            if r == phrase:
                matched = SKILL_ALIASES[phrase]
                break
        if matched is None:
            matched = SKILL_ALIASES.get(r)
        if matched and matched in vocab_index:
            vec[vocab_index[matched]] = 1
    return vec

# ============================================================
# STEP 6: Cosine similarity
# ============================================================
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ============================================================
# MAIN — run the full pipeline
# ============================================================
def main():
    print("=" * 60)
    print("   REDROB HACKATHON — RESUME MATCHING ENGINE")
    print("=" * 60)

    # Step 1 & 2: Normalize + Deduplicate
    print("\n[STEP 1-2] Normalized & Deduplicated Skills")
    print("-" * 60)
    normalized = {}
    for name, raw in RESUMES.items():
        skills = normalize_skills(raw)
        normalized[name] = skills
        print(f"  {name}: {skills}")

    # Step 3: Vocabulary
    vocab = build_vocabulary(normalized)
    vocab_index = {s: i for i, s in enumerate(vocab)}
    print(f"\n[STEP 3] Vocabulary — {len(vocab)} terms (alphabetical)")
    print(f"  {vocab}")

    # Step 4: TF-IDF
    tfidf_vectors, idf, df = compute_tfidf(normalized, vocab)
    print(f"\n[STEP 4] TF-IDF Vectors (non-zero only)")
    print("-" * 60)
    for name, vec in tfidf_vectors.items():
        nz = [(vocab[i], round(v, 4)) for i, v in enumerate(vec) if v > 0]
        print(f"  {name}: {nz}")

    # Step 5: JD Vectors
    print(f"\n[STEP 5] JD Binary Vectors")
    print("-" * 60)
    jd_vectors = {}
    for jd_id, jd_data in JD_RAW.items():
        vec = build_jd_vector(jd_data["skills"], vocab_index)
        jd_vectors[jd_id] = vec
        present = [vocab[i] for i, v in enumerate(vec) if v == 1]
        print(f"  {jd_id} — {jd_data['label']}: {present}")

    # Step 6: Cosine similarity + Ranking
    print(f"\n[STEP 6] Cosine Similarity Scores")
    print("-" * 60)
    print("\n" + "=" * 60)
    print("   FINAL RESULTS")
    print("=" * 60)

    for jd_id, jd_vec in jd_vectors.items():
        scores = []
        for name, resume_vec in tfidf_vectors.items():
            sim = cosine_similarity(resume_vec, jd_vec)
            scores.append((name, sim))

        # Sort: descending score, alphabetical name for ties
        scores.sort(key=lambda x: (-x[1], x[0]))

        label = JD_RAW[jd_id]["label"]
        jd_display = jd_id[:2] + "-" + jd_id[2:]
        print(f"\n{jd_display} — {label}")
        top3 = scores[:3]
        output = ", ".join(f"{n}({s:.2f})" for n, s in top3)
        print(output)

        # Show all scores for transparency
        print("  [All scores]")
        for name, score in scores:
            print(f"    {name}: {score:.6f}")

if __name__ == "__main__":
    main()