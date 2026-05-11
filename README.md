Run this in the VS Code terminal:
bashcat > README.md << 'EOF'
# Resume Matching Engine — Redrob AI Campus Hackathon

## Problem Statement
Build a Resume Matching Engine that matches 10 student resumes against 3 Job Descriptions from Korean tech companies using TF-IDF and Cosine Similarity.

## Approach
1. **Normalize Skills** — Split on commas, lowercase, apply alias mapping, discard unknown tokens
2. **Deduplicate** — Each canonical skill appears only once per resume
3. **Build Vocabulary** — Shared alphabetically sorted vocabulary from all resumes
4. **Compute TF-IDF** — TF = 1/N, IDF = ln(10/df), no external libraries
5. **Build JD Vectors** — Binary vectors over same vocabulary
6. **Cosine Similarity** — Rank top 3 candidates per JD

## Results
| JD | Company | Top 3 Candidates |
|---|---|---|
| JD-1 | Kakao (ML Engineer) | Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40) |
| JD-2 | Naver (Backend Engineer) | Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19) |
| JD-3 | Line (Frontend Engineer) | Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35) |

## How to Run
```bash
python3 resume_matching_engine.py
```

## Libraries Used
- `math` (standard library only — as per hackathon rules)

## Author
Bhoomi Tayal
EOF
Then add and push it:
bashgit add README.md
git commit -m "Add README"
git push
