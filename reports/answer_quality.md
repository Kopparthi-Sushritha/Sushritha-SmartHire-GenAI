# SmartHire Answer Quality Evaluation

## 1. Retrieval Relevance

**Hit rate:** 100.00%

A retrieval test is counted as a hit when at least one of the top-N retrieved jobs contains a relevant job title, skill, or description keyword.

| Test Profile | Hit | Top Job | Score |
|---|---|---|---:|
| Machine Learning Candidate | PASS | Data Scientist Machine Learning | 0.7163 |
| Data Science Candidate | PASS | Data Scientist Machine Learning | 0.7186 |
| Software Engineer Candidate | PASS | Software Engineer | 0.7325 |

---

## 2. Mentor Answer Quality

The mentor answers are checked for basic correctness, grounding, and helpfulness indicators.

| Question | Correctness | Grounding | Helpfulness |
|---|---|---|---|
| What skills should I learn for a career in data science? | PASS | PASS | PASS |
| How can I improve my career based on the available career notes? | PASS | PASS | PASS |

### Mentor Answers

#### Question 1

**Question:** What skills should I learn for a career in data science?

**Answer:**

I don't know based on the provided career notes.

**Retrieved sources:**
- data_analyst_roadmap.md (score: 0.6143)
- data_analyst_roadmap.md (score: 0.6031)
- resume_writing_tips.md (score: 0.3571)

#### Question 2

**Question:** How can I improve my career based on the available career notes?

**Answer:**

Based on the provided career notes, you can improve your career development and job search in the following ways:

* **Optimize your resume:** 
  * Keep it to one page if you are a student or early-career candidate, making it easy to scan in under a minute (Source 1).
  * Structure it with contact details, a short summary, skills, experience/projects, and education, placing the most relevant section highest (Source 1).
  * Write strong bullet points starting with action verbs, include results or numbers where possible, and avoid vague lines (Source 1).
  * Tailor your resume by reading the job description, noting repeated skills, and including those exact words (if true for you) to help pass automated filters (Source 2). Avoid common mistakes like long paragraph objectives, listing irrelevant courses/skills, and spelling or formatting errors (Source 2).
  * Group your skills (into categories like Languages, Tools, Databases) and only list specific tools and languages asked for by the job if you actually know them (Source 1 & 2).

* **Transition into a data analyst role (if switching from a non-data job):**
  * Follow the fastest path: 
    1. Learn SQL first and practise on real datasets (Source 3).
    2. Rebuild hand-made reports into a dashboard (Source 3).
    3. Complete two or three portfolio projects analyzing public datasets (Source 3).
    4. Apply to "junior data analyst" or "business analyst" roles (Source 3).
  * Focus on entry-level expectations such as being comfortable writing SQL queries without help, building a clean dashboard and explaining it, and turning vague business questions into data questions (Source 3). Note that you do not need machine learning, deep learning, or a statistics degree at the start (Source 3).

**Retrieved sources:**
- resume_writing_tips.md (score: 0.4387)
- resume_writing_tips.md (score: 0.4386)
- data_analyst_roadmap.md (score: 0.3460)

---

## 3. Hallucination Check

The mentor should refuse to answer when the requested information is not supported by the career notes.

**Hallucination test: PASS**

**Question:** What is the salary of a software engineer at Google?

**Mentor response:**

I don't know based on the provided career notes.

---

## 4. Summary

- Retrieval hit rate: 100.00%
- Mentor tests passing all basic quality checks: 2/2
- Hallucination refusal: PASS

Note: These are small automated evaluation checks. Correctness and helpfulness should also be reviewed manually for a reliable assessment.