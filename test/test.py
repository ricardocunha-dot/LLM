import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI

# Configuração do Azure

client = AzureOpenAI(
    api_key="API_KEY",
    azure_endpoint="https://deliveryaihub7596701848.services.ai.azure.com/",
    api_version="2024-03-01-preview",
)


# Ler dados

jobs = pd.read_csv("jobs.csv")
candidates = pd.read_csv("candidates.csv")


# Função DeepSeek

def evaluate_candidate_deepseek(candidate, job):
    prompt = f"""
    Evaluate the candidate's suitability for the job below.
    Return in the format:
    SCORE: <score 0-100>
    REASON: <short justification>

    Job:
    Title: {job['title']}
    Requirements: {job['requirements']}
    Minimum experience: {job['min_experience']} years

    Candidate:
    Name: {candidate['name']}
    Current role: {candidate['role']}
    Skills: {candidate['skills']}
    Experience: {candidate['experience_years']} years
    """
    
    response = client.chat.completions.create(
        model="DeepSeek-V3-0324",
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        reasoning = response.choices[0].message.reasoning_content.strip()
    except AttributeError:
        reasoning = str(response.choices[0])

    score = 0
    reason = ""
    try:
        lines = reasoning.splitlines()
        for line in lines:
            if line.lower().startswith("score:"):
                score = int(line.split(":", 1)[1].strip())
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()
    except:
        score = 0
        reason = reasoning

    return score, reason


# Preparar TF-IDF

def text_for_job(job):
    return f"{job['title']} {job['requirements']} {job['min_experience']} anos"

def text_for_candidate(candidate):
    return f"{candidate['role']} {candidate['skills']} {candidate['experience_years']} anos"

jobs['text'] = jobs.apply(text_for_job, axis=1)
candidates['text'] = candidates.apply(text_for_candidate, axis=1)

vectorizer = TfidfVectorizer()
all_texts = pd.concat([jobs['text'], candidates['text']])
tfidf_matrix = vectorizer.fit_transform(all_texts)
job_matrix = tfidf_matrix[:len(jobs)]
candidate_matrix = tfidf_matrix[len(jobs):]


# Função para top candidatos

def top_candidates(job_id, top_n=3, prefilter_n=5):
    job_idx = jobs[jobs['id'] == job_id].index[0]
    job_vec = job_matrix[job_idx]

    similarities = cosine_similarity(job_vec, candidate_matrix)[0]

    df_candidates = candidates.copy()
    df_candidates['cosine_similarity'] = similarities

    top_prefilter = df_candidates.sort_values(by='cosine_similarity', ascending=False).head(prefilter_n)

    candidate_scores = []
    for _, candidate in top_prefilter.iterrows():
        score, reason = evaluate_candidate_deepseek(candidate, jobs.iloc[job_idx])
        candidate_scores.append({
            "name": candidate['name'],
            "cosine_similarity": candidate['cosine_similarity'],
            "score": score,
            "reason": reason
        })

    sorted_candidates = sorted(candidate_scores, key=lambda x: x['score'], reverse=True)[:top_n]
    return top_prefilter, sorted_candidates


# Pipeline

print("Available jobs:")
for _, job in jobs.iterrows():
    print(f"{job['id']}: {job['title']}")

job_id = input("Select a job by ID: ").strip()
prefilter_df, top3 = top_candidates(job_id)

print(f"\nCandidatos pré-filtrados pela similaridade de cosseno:")
print(prefilter_df[['name', 'cosine_similarity']].to_string(index=False))

print(f"\nTop {len(top3)} candidatos segundo DeepSeek:")
for idx, cand in enumerate(top3, start=1):
    print(f"{idx}. Name: {cand['name']}, Score: {cand['score']}, Cosine Similarity: {cand['cosine_similarity']:.3f}")
    print(f"   Justification: {cand['reason']}\n")