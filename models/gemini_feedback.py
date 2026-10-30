import os
import json
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_educational_fallback(
    topic: str,
    similarity_score: float,
    speech_metrics: Dict[str, Any]
) -> str:
    """
    Generates structured, high-quality rule-based feedback when Gemini API is unavailable.
    Provides detailed summary, strengths, weaknesses, missing concepts, suggestions, and tips.
    """
    # Parse metrics
    wpm = speech_metrics.get("speaking_rate", 0.0)
    pause_ratio = speech_metrics.get("pause_ratio", 0.0)
    fillers = speech_metrics.get("total_fillers", 0)
    filler_pct = speech_metrics.get("filler_percentage", 0.0)
    
    # 1. Summary
    summary = f"The student explained the concept of **{topic}**. "
    if similarity_score >= 80.0:
        summary += "Their explanation shows an excellent, precise grasp of the core concepts, aligning closely with standard academic definitions."
    elif similarity_score >= 60.0:
        summary += "Their explanation indicates a good general understanding of the topic, though some minor definitions or technical nuances were omitted."
    elif similarity_score >= 40.0:
        summary += "Their explanation suggests an average understanding. The student mentioned several terms but did not connect them with complete accuracy or depth."
    else:
        summary += "The explanation shows a poor grasp of the concept. The student spoke very briefly or missed the fundamental principles of the topic."

    # 2. Strengths
    strengths = []
    if similarity_score >= 60.0:
        strengths.append(f"Strong conceptual match ({similarity_score}%) demonstrating correct understanding of {topic} terminology.")
    if wpm >= 110.0 and wpm <= 150.0:
        strengths.append(f"Excellent speaking pace of {wpm} WPM, which is within the ideal professional speaking range (110-150 WPM).")
    elif wpm > 0:
        strengths.append(f"Speech structure was continuous with a measured rate of {wpm} WPM.")
    if filler_pct <= 2.0 and fillers < 5:
        strengths.append("High speech clarity with very minimal filler words used (e.g. 'um', 'uh', 'like').")
    if pause_ratio <= 0.15:
        strengths.append("Fluent transition between sentences with a solid, confident pause ratio.")
    
    if not strengths:
        strengths.append("The student successfully recorded their voice and attempted to elaborate on the selected technical concept.")

    # 3. Weaknesses
    weaknesses = []
    if similarity_score < 60.0:
        weaknesses.append(f"Conceptual alignment score of {similarity_score}% is low, indicating gaps in defining or explaining {topic} correctly.")
    if wpm < 100.0 and wpm > 0:
        weaknesses.append(f"The speaking rate was too slow ({wpm} WPM), which might lead to disengagement. Ideal is 110-150 WPM.")
    elif wpm > 160.0:
        weaknesses.append(f"The speaking rate was too fast ({wpm} WPM), which can make it difficult for listeners to follow.")
    if filler_pct > 3.0:
        weaknesses.append(f"Frequent use of filler words ({fillers} detected, {filler_pct}% of total speech), reducing speech professionalism.")
    if pause_ratio > 0.22:
        weaknesses.append(f"Relatively high pause ratio ({int(pause_ratio * 100)}%), which suggests hesitation or difficulty recalling the concepts.")

    if not weaknesses:
        weaknesses.append("No major speech delivery weaknesses detected. Excellent performance!")

    # 4. Missing Concepts (Domain knowledge mapping)
    missing_concepts = []
    if topic == "Machine Learning":
        missing_concepts = ["Supervised vs. Unsupervised learning", "Training vs. Testing datasets", "Generalization and Overfitting"]
    elif topic == "Artificial Intelligence":
        missing_concepts = ["Reasoning/Problem solving", "Natural Language Processing (NLP)", "Computer Vision", "Expert Systems"]
    elif topic == "Cloud Computing":
        missing_concepts = ["IaaS, PaaS, SaaS delivery models", "Public, Private, Hybrid deployment models", "On-demand scalability"]
    elif topic == "Deep Learning":
        missing_concepts = ["Neural Networks & Multi-layer Perceptrons", "Backpropagation/Loss functions", "Architectures like CNNs/Transformers"]
    elif topic == "Python":
        missing_concepts = ["Interpreted and Dynamic typing", "Standard Libraries/Packages", "Object-Oriented/Functional paradigms"]
    elif topic == "Operating System":
        missing_concepts = ["Process/CPU Scheduling", "Virtual Memory/Paging", "File Systems & I/O management"]
    elif topic == "DBMS":
        missing_concepts = ["Data integrity and ACID properties", "Relational vs. NoSQL databases", "Structured Query Language (SQL)"]
    elif topic == "Computer Networks":
        missing_concepts = ["TCP/IP protocol stack", "Routing and Switching", "IP Addressing (IPv4/IPv6)", "Client-Server model"]
    elif topic == "Cyber Security":
        missing_concepts = ["Confidentiality, Integrity, Availability (CIA Triad)", "Cryptography (Symmetric/Asymmetric)", "Firewalls & Threat vectors"]
    elif topic == "Data Science":
        missing_concepts = ["Interdisciplinary integration (math, stats, CS)", "Data cleaning and preprocessing", "Data Visualization & insights"]
    else:
        missing_concepts = ["Core structural components", "Practical industry applications", "Associated terminology"]

    if similarity_score >= 85.0:
        # Filter down if score is excellent
        missing_concepts = [f"Advanced applications or sub-areas of {topic}"]

    # 5. Suggestions
    suggestions = [
        f"Review the academic textbook definition of {topic} to ensure all foundational components are fully integrated into your summary.",
        "Practice speaking with a structural outline (e.g., Introduction, Core Definition, Examples, Applications) to minimize hesitation."
    ]
    if similarity_score < 70.0:
        suggestions.append(f"Focus specifically on learning the relationships between key technical terms under {topic}.")

    # 6. Communication Tips
    comm_tips = []
    if filler_pct > 2.0:
        comm_tips.append("Reduce filler words by pausing silently for a split second instead of saying 'um' or 'like'.")
    if wpm < 100.0:
        comm_tips.append("Try to increase your speaking pace by practicing pre-written technical scripts to build oral fluency.")
    elif wpm > 150.0:
        comm_tips.append("Slow down! Articulate technical terms clearly and insert natural pauses after key sentences.")
    if pause_ratio > 0.20:
        comm_tips.append("Organize your thoughts before pressing record; this will minimize long gaps in speech delivery.")
    
    if not comm_tips:
        comm_tips.append("Maintain your current vocal cadence, pitch modulation, and confidence level.")

    # Format output as a nice Markdown report
    feedback_md = f"""### Summary
{summary}

### Strengths
{chr(10).join([f"- {s}" for s in strengths])}

### Weaknesses
{chr(10).join([f"- {w}" for w in weaknesses])}

### Missing Concepts
{chr(10).join([f"- {mc}" for mc in missing_concepts])}

### Suggestions
{chr(10).join([f"- {s}" for s in suggestions])}

### Communication Tips
{chr(10).join([f"- {t}" for t in comm_tips])}

### Overall Feedback
Your overall understanding of **{topic}** is rated as **{ "Excellent" if similarity_score >= 80.0 else "Good" if similarity_score >= 60.0 else "Average" if similarity_score >= 40.0 else "Poor" }**. By working on the suggestions and speech tips listed above, you will be well-prepared to explain this concept in job interviews, university vivas, and technical presentations. Keep practicing!"""
    return feedback_md

def generate_ai_feedback(
    topic: str,
    transcript: str,
    similarity_score: float,
    speech_metrics: Dict[str, Any]
) -> str:
    """
    Generates educational and encouraging feedback using the Google Gemini API.
    Gracefully falls back to a rule-based generator if API key is missing or calls fail.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini API Key missing. Generating educational fallback feedback...")
        return generate_educational_fallback(topic, similarity_score, speech_metrics)

    # Prompt constructing
    prompt = f"""
    You are an expert academic evaluator, university professor, and communication coach.
    Provide constructive, encouraging, and detailed feedback for a student explaining a technical topic.

    Evaluation Context:
    - Topic: {topic}
    - Student Transcript: "{transcript}"
    - Concept Similarity Score: {similarity_score}% (how closely it aligns with the academic standard)
    - Speech Metrics:
      * Speaking Rate: {speech_metrics.get('speaking_rate', 0.0)} words per minute
      * Pause Ratio: {speech_metrics.get('pause_ratio', 0.0)} (fraction of silence)
      * Filler Word Count: {speech_metrics.get('total_fillers', 0)} ({speech_metrics.get('filler_percentage', 0.0)}% of speech)
      * Confidence Score: {speech_metrics.get('confidence_score', 0.0)}%

    Please generate a feedback assessment containing EXACTLY these sections, formatted in Markdown:
    
    ### Summary
    [A brief paragraph summarizing what the student explained and how well they understood the concept.]
    
    ### Strengths
    [Bullet points highlighting conceptual or communication-related strengths (e.g. good vocabulary, correct flow, perfect speed, low fillers).]
    
    ### Weaknesses
    [Bullet points detailing conceptual gaps or speech flow delivery issues.]
    
    ### Missing Concepts
    [Bullet points listing 2-3 specific technical sub-topics/aspects of "{topic}" that were missing but should have been discussed.]
    
    ### Suggestions
    [Bullet points detailing clear action steps to improve technical grasp of {topic}.]
    
    ### Communication Tips
    [Bullet points showing how to improve speech delivery, e.g., speed, fillers, or pauses.]
    
    ### Overall Feedback
    [An encouraging wrap-up summarizing the performance category (Excellent/Good/Average/Poor) and motivating the student to practice.]
    
    Be supportive, educational, professional, and clear. Do not include extra headings, intro, or outro text outside the requested sections.
    """

    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # We use gemini-1.5-flash as the standard, efficient model for text generation
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        print("Sending request to Google Gemini API...")
        response = model.generate_content(prompt)
        feedback_text = response.text.strip()
        
        if not feedback_text:
            raise ValueError("Empty response received from Gemini API.")
            
        return feedback_text
        
    except Exception as e:
        print(f"Gemini API execution error: {e}. Falling back to rule-based feedback...")
        return generate_educational_fallback(topic, similarity_score, speech_metrics)
