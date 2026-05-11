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

# multi-word / hyphenated keys sorted longest first
MULTI_WORD_KEYS = sorted(
    [k for k in SKILL_ALIASES if " " in k or "-" in k],
    key=lambda x: -len(x)
)

# ============================================================
# PROMPT 1 — normalize_skills (matches Redrob AI output)
# ============================================================
def normalize_skills(skill_string, alias_dict):
    """
    Normalize a string of skills by splitting on commas, lowercasing each token,
    checking multi-word phrases first, and then mapping to canonical skills using
    the provided alias dictionary. Unknown tokens are discarded.
    Args:
        skill_string (str): The input string of skills.
        alias_dict (dict): A dictionary mapping skill aliases to their canonical forms.
    Returns:
        list: A list of normalized skills.
    """
    # Split the input string into individual skills
    skills = [skill.strip() for skill in skill_string.split(',')]

    # Initialize an empty list to store the normalized skills
    normalized_skills = []
    seen = set()

    # Iterate over each skill
    for skill in skills:
        # Lowercase the skill
        skill = skill.lower()
        matched = None

        # Check if the skill is a multi-word phrase (longest first)
        for phrase in MULTI_WORD_KEYS:
            if skill == phrase.lower():
                matched = alias_dict[phrase]
                break

        # If not multi-word, check if the skill is a single-word alias
        if matched is None:
            matched = alias_dict.get(skill)

        # Deduplicate — only add if not seen before
        if matched and matched not in seen:
            normalized_skills.append(matched)
            seen.add(matched)

    # Return the list of normalized skills
    return normalized_skills


# ============================================================
# PROMPT 2 — build_vocabulary, compute_document_frequency,
#             compute_tf_idf (matches Redrob AI output)
# ============================================================
def build_vocabulary(resumes):
    """
    Build a shared vocabulary from all resumes.
    Args:
        resumes (list): A list of lists, where each inner list contains the skills in a resume.
    Returns:
        list: A sorted list of unique skills.
    """
    vocabulary = set()
    for resume in resumes:
        for skill in resume:
            vocabulary.add(skill)
    return sorted(list(vocabulary))


def compute_document_frequency(resumes, vocabulary):
    """
    Compute the document frequency for each skill.
    Args:
        resumes (list): A list of lists, where each inner list contains the skills in a resume.
        vocabulary (list): A sorted list of unique skills.
    Returns:
        dict: A dictionary where the keys are the skills and the values are their document frequencies.
    """
    document_frequency = {}
    for skill in vocabulary:
        document_frequency[skill] = sum(1 for resume in resumes if skill in resume)
    return document_frequency


def compute_tf_idf(resumes, vocabulary, document_frequency):
    """
    Compute the TF-IDF vectors for each resume.
    Args:
        resumes (list): A list of lists, where each inner list contains the skills in a resume.
        vocabulary (list): A sorted list of unique skills.
        document_frequency (dict): A dictionary where the keys are the skills and values are document frequencies.
    Returns:
        list: A list of dictionaries, where each dictionary represents the TF-IDF vector for a resume.
    """
    tf_idf_vectors = []
    for resume in resumes:
        unique_skills = len(set(resume))
        tf_idf_vector = {}
        for skill in vocabulary:
            tf = 1 / unique_skills if skill in resume else 0
            idf = math.log(10 / document_frequency[skill]) \
                  if document_frequency[skill] > 0 else 0
            tf_idf_vector[skill] = tf * idf
        tf_idf_vectors.append(tf_idf_vector)
    return tf_idf_vectors


# ============================================================
# PROMPT 3 — build_binary_vectors, compute_cosine_similarity,
#             get_top_candidates (matches Redrob AI output)
# ============================================================
def build_binary_vectors(job_descriptions, vocabulary):
    """
    Build binary vectors for the job descriptions.
    Args:
        job_descriptions (list): A list of lists, where each inner list contains the skills in a job description.
        vocabulary (list): A sorted list of unique skills.
    Returns:
        list: A list of binary vectors, where each binary vector represents a job description.
    """
    binary_vectors = []
    for job_description in job_descriptions:
        binary_vector = [1 if skill in job_description else 0 for skill in vocabulary]
        binary_vectors.append(binary_vector)
    return binary_vectors


def compute_cosine_similarity(tf_idf_vector, binary_vector):
    """
    Compute the cosine similarity between a TF-IDF vector and a binary vector.
    Args:
        tf_idf_vector (dict): A dictionary where the keys are the skills and the values are their TF-IDF scores.
        binary_vector (list): A binary vector where presence of a skill is 1 and absence is 0.
    Returns:
        float: The cosine similarity between the TF-IDF vector and the binary vector.
    """
    tf_idf_scores = [tf_idf_vector[skill] for skill in tf_idf_vector]
    dot_product = sum(a * b for a, b in zip(tf_idf_scores, binary_vector))
    magnitude_tfidf = math.sqrt(sum(a ** 2 for a in tf_idf_scores))
    magnitude_binary = math.sqrt(sum(a ** 2 for a in binary_vector))
    if magnitude_tfidf == 0 or magnitude_binary == 0:
        return 0.0
    return dot_product / (magnitude_tfidf * magnitude_binary)


def get_top_candidates(candidate_names, tf_idf_vectors, binary_vectors):
    """
    Get the top 3 candidates per job description.
    Args:
        candidate_names (list): List of candidate names.
        tf_idf_vectors (list): A list of TF-IDF vector dicts for each resume.
        binary_vectors (list): A list of binary vectors for each job description.
    Returns:
        list: A list of lists, where each inner list contains the top 3 candidates for a job description.
    """
    top_candidates = []
    for binary_vector in binary_vectors:
        candidates = []
        for j, tf_idf_vector in enumerate(tf_idf_vectors):
            similarity = compute_cosine_similarity(tf_idf_vector, binary_vector)
            candidates.append((j, similarity))
        # Sort by descending similarity, alphabetical name for ties
        candidates.sort(key=lambda x: (-x[1], candidate_names[x[0]]))
        top_candidates.append(
            [(candidate_names[c[0]], round(c[1], 2)) for c in candidates[:3]]
        )
    return top_candidates


# ============================================================
# MAIN — run the full pipeline
# ============================================================
def main():
    print("=" * 60)
    print("   REDROB HACKATHON — RESUME MATCHING ENGINE")
    print("=" * 60)

    # Step 1 & 2: Normalize + Deduplicate all resumes
    print("\n[STEP 1-2] Normalized & Deduplicated Skills")
    print("-" * 60)
    candidate_names = list(RESUMES.keys())
    normalized_resumes = []
    for name, raw in RESUMES.items():
        skills = normalize_skills(raw, SKILL_ALIASES)
        normalized_resumes.append(skills)
        print(f"  {name}: {skills}")

    # Step 3: Build shared vocabulary
    vocabulary = build_vocabulary(normalized_resumes)
    print(f"\n[STEP 3] Vocabulary — {len(vocabulary)} terms (alphabetical)")
    print(f"  {vocabulary}")

    # Step 4: Compute document frequency and TF-IDF vectors
    document_frequency = compute_document_frequency(normalized_resumes, vocabulary)
    tf_idf_vectors = compute_tf_idf(normalized_resumes, vocabulary, document_frequency)
    print(f"\n[STEP 4] TF-IDF Vectors (non-zero only)")
    print("-" * 60)
    for i, tf_idf_vector in enumerate(tf_idf_vectors):
        nz = [(s, round(v, 4)) for s, v in tf_idf_vector.items() if v > 0]
        print(f"  {candidate_names[i]}: {nz}")

    # Step 5: Build JD binary vectors
    print(f"\n[STEP 5] JD Binary Vectors")
    print("-" * 60)
    jd_skill_lists = []
    for jd_id, jd_data in JD_RAW.items():
        # Normalize JD skills using same alias map
        normalized_jd = []
        for raw in jd_data["skills"]:
            r = raw.lower()
            matched = None
            for phrase in MULTI_WORD_KEYS:
                if r == phrase:
                    matched = SKILL_ALIASES[phrase]
                    break
            if matched is None:
                matched = SKILL_ALIASES.get(r)
            if matched and matched in vocabulary:
                normalized_jd.append(matched)
        jd_skill_lists.append(normalized_jd)
        print(f"  {jd_id} — {jd_data['label']}: {normalized_jd}")

    binary_vectors = build_binary_vectors(jd_skill_lists, vocabulary)

    # Step 6: Cosine similarity and ranking
    print(f"\n[STEP 6] Cosine Similarity & Top Candidates")
    print("-" * 60)
    top_candidates = get_top_candidates(candidate_names, tf_idf_vectors, binary_vectors)

    print("\n" + "=" * 60)
    print("   FINAL RESULTS")
    print("=" * 60)
    jd_labels = list(JD_RAW.items())
    for i, candidates in enumerate(top_candidates):
        jd_id = jd_labels[i][0]
        label = jd_labels[i][1]["label"]
        jd_display = jd_id[:2] + "-" + jd_id[2:]
        print(f"\n{jd_display} — {label}")
        print(", ".join(f"{name}({score:.2f})" for name, score in candidates))

        # Show all scores
        print("  [All scores]")
        all_scores = []
        binary_vector = binary_vectors[i]
        for j, tf_idf_vector in enumerate(tf_idf_vectors):
            sim = compute_cosine_similarity(tf_idf_vector, binary_vector)
            all_scores.append((candidate_names[j], sim))
        all_scores.sort(key=lambda x: (-x[1], x[0]))
        for name, score in all_scores:
            print(f"    {name}: {score:.6f}")


if __name__ == "__main__":
    main()
