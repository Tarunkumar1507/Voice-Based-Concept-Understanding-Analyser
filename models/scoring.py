from typing import Dict, Any

def calculate_fluency_score(
    pause_ratio: float,
    filler_percentage: float,
    speaking_rate: float
) -> float:
    """
    Calculates the Speech Fluency Score (0 - 100) based on pause ratio,
    filler word density, and estimated speaking rate.
    
    Penalties:
    - Pause Ratio: Ideal <= 0.15. Penalty applied for higher ratio.
    - Filler Percentage: Ideal <= 2.0%. Penalty applied for higher density.
    - Speaking Rate: Ideal is 110 to 150 words per minute (WPM).
    """
    if speaking_rate == 0.0:
        return 0.0

    score = 100.0

    # 1. Pause Ratio Penalty (Max 35 points)
    # Deduct points if pause ratio is greater than 0.15
    if pause_ratio > 0.15:
        pause_penalty = (pause_ratio - 0.15) * 80.0
        score -= min(35.0, pause_penalty)

    # 2. Filler Word Penalty (Max 35 points)
    # Deduct points if filler words exceed 2.0% of total speech
    if filler_percentage > 2.0:
        filler_penalty = (filler_percentage - 2.0) * 8.0
        score -= min(35.0, filler_penalty)

    # 3. Speaking Rate Penalty (Max 30 points)
    # Ideal range is 110 - 150 WPM. Too fast or too slow gets penalized.
    if speaking_rate < 110.0:
        rate_penalty = (110.0 - speaking_rate) * 0.4
        score -= min(30.0, rate_penalty)
    elif speaking_rate > 150.0:
        rate_penalty = (speaking_rate - 150.0) * 0.4
        score -= min(30.0, rate_penalty)

    return round(max(0.0, min(100.0, score)), 2)

def calculate_confidence_score(
    rms_energy: float,
    pause_ratio: float,
    mean_pitch: float
) -> float:
    """
    Calculates the Confidence Score (0 - 100) based on volume level (RMS Energy),
    pauses, and vocal modulation (Pitch presence).
    
    Penalties:
    - RMS Energy: Extremely low energy indicates mumbling/whispering (ideal >= 0.01).
    - Pause Ratio: High pause ratio indicates hesitation (ideal <= 0.20).
    - Pitch: No pitch detected (value 0.0) indicates monotone/robotic/silent speech.
    """
    if rms_energy == 0.0 and mean_pitch == 0.0:
        return 0.0

    score = 100.0

    # 1. Volume (RMS Energy) Penalty (Max 40 points)
    # Ideal rms_energy is >= 0.015. Soft speaking (< 0.015) is penalized.
    if rms_energy < 0.015:
        energy_penalty = (0.015 - rms_energy) * 2000.0
        score -= min(40.0, energy_penalty)

    # 2. Hesitation (Pause Ratio) Penalty (Max 40 points)
    # Deduct points for pauses over 0.20
    if pause_ratio > 0.20:
        hesitation_penalty = (pause_ratio - 0.20) * 100.0
        score -= min(40.0, hesitation_penalty)

    # 3. Monotone/Silence Penalty (Max 20 points)
    # If no pitch is detected (0.0), it suggests monotone reading or silent gaps
    if mean_pitch == 0.0:
        score -= 20.0

    return round(max(0.0, min(100.0, score)), 2)

def generate_report_scores(
    semantic_score: float,
    fluency_score: float,
    confidence_score: float
) -> Dict[str, Any]:
    """
    Aggregates semantic, fluency, and confidence scores using specific weights:
    - Semantic Similarity (70%)
    - Speech Fluency (20%)
    - Confidence (10%)
    
    Returns overall score, letter grade, and performance category.
    """
    # Calculate weighted average
    overall_score = (semantic_score * 0.70) + (fluency_score * 0.20) + (confidence_score * 0.10)
    overall_score = round(max(0.0, min(100.0, overall_score)), 2)

    # Letter Grade
    if overall_score >= 90.0:
        letter_grade = "A"
    elif overall_score >= 80.0:
        letter_grade = "B"
    elif overall_score >= 70.0:
        letter_grade = "C"
    elif overall_score >= 60.0:
        letter_grade = "D"
    else:
        letter_grade = "F"

    # Performance Category
    if overall_score >= 80.0:
        performance_category = "Excellent"
    elif overall_score >= 60.0:
        performance_category = "Good"
    elif overall_score >= 40.0:
        performance_category = "Average"
    else:
        performance_category = "Poor"

    return {
        "semantic_score": semantic_score,
        "fluency_score": fluency_score,
        "confidence_score": confidence_score,
        "overall_score": overall_score,
        "letter_grade": letter_grade,
        "performance_category": performance_category
    }
