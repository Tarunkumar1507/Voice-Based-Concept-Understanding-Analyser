import os
import sys
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from database.database import init_db, save_report, get_history, delete_report, clear_history
from utils.helper import load_concepts, save_uploaded_file, plot_waveform, cleanup_files
from models.whisper_model import transcribe_audio
from models.semantic_model import evaluate_understanding
from models.filler_analysis import analyze_fillers
from models.audio_features import extract_audio_features
from models.scoring import calculate_fluency_score, calculate_confidence_score, generate_report_scores
from models.gemini_feedback import generate_ai_feedback
from reports.pdf_report import generate_pdf_report

# Load environment variables
load_dotenv()

# Define directories
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Ensure required directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize SQLite database
init_db()

# Streamlit Page Config
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional CSS for visual excellence (slate/emerald/cyan palette)
st.markdown("""
<style>
    /* Primary font & background styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Title Banner */
    .title-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .title-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(14,165,233,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .title-banner h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .title-banner p {
        font-size: 1.1rem;
        color: #93c5fd;
        font-weight: 300;
        margin: 0;
    }
    
    /* Modern Section Cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        border-color: #cbd5e1;
    }
    
    /* Progress bar coloring customization */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    }
    
    /* Status indicators */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-excellent { background-color: #d1fae5; color: #065f46; }
    .status-good { background-color: #dbeafe; color: #1e40af; }
    .status-average { background-color: #fef3c7; color: #92400e; }
    .status-poor { background-color: #fee2e2; color: #991b1b; }
    
    /* Sidebar header styling */
    .sidebar-header {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.5rem;
        font-size: 1.2rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">🎙️ VBCUA Panel</div>', unsafe_allow_html=True)
    
    # Nav selection
    menu_option = st.radio(
        "Navigation Menu",
        ["Upload & Analyze", "Assessment History", "Concepts Guide"]
    )
    
    st.markdown("---")
    st.markdown("**AI Service Integration**")
    
    # Check Gemini API Key and prioritize user settings/input over dotenv if user supplies it
    api_key_status = "Connected" if os.getenv("GEMINI_API_KEY") else "Not Configured (Fallback Active)"
    
    user_api_key = st.text_input(
        "Override Gemini API Key",
        type="password",
        help="Input a Google Gemini API Key. If left empty, the application falls back to a rules-based evaluation engine."
    )
    
    if user_api_key:
        os.environ["GEMINI_API_KEY"] = user_api_key
        api_key_status = "Connected (User Key Provided)"
        
    if "Connected" in api_key_status:
        st.success(f"Gemini API: {api_key_status}")
    else:
        st.warning(f"Gemini API: {api_key_status}")
        
    st.markdown("---")
    st.caption("Developed with OpenAI Whisper, Sentence-BERT & Librosa.")
    st.caption("Version 1.0.0 (Stable)")

# Load Reference Concepts
concepts_dict = load_concepts()

# Initialize session states to persist analysis outputs
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "last_analyzed_file" not in st.session_state:
    st.session_state.last_analyzed_file = None

# ----------------- MENU: UPLOAD & ANALYZE -----------------
if menu_option == "Upload & Analyze":
    # Elegant Title
    st.markdown("""
    <div class="title-banner">
        <h1>Voice-Based Concept Understanding Analyser</h1>
        <p>Record or upload technical concept explanations to assess oral fluency, acoustic confidence, and semantic domain knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Inputs Section
    st.subheader("1. Submission Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        student_name = st.text_input(
            "Student Name",
            placeholder="e.g. Alexander Fleming",
            help="Enter the student's full name for assessment tracking and report generation."
        )
    with col2:
        topic_options = list(concepts_dict.keys())
        selected_topic = st.selectbox(
            "Selected Assessment Topic",
            options=topic_options,
            help="Select the concept the student is explaining."
        )
        
    st.markdown("---")
    st.subheader("2. Audio Source File")
    
    uploaded_file = st.file_uploader(
        "Upload audio recording (.wav or .mp3)",
        type=["wav", "mp3"],
        help="Select a clean voice recording. Maximum duration should ideally be under 5 minutes."
    )
    
    if uploaded_file:
        # File details summary
        st.audio(uploaded_file, format=uploaded_file.type)
        
        # Enable action button
        is_profile_valid = len(student_name.strip()) > 0
        
        if not is_profile_valid:
            st.warning("Please enter the student name in the Profile section before running the evaluation.")
            
        analyze_clicked = st.button(
            "Analyze Concept Understanding",
            disabled=not is_profile_valid,
            type="primary"
        )
        
        # Run Evaluation Pipeline
        if analyze_clicked:
            # Check if this file was already analyzed to prevent re-running on redraws
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # STEP 1: Save file locally
                status_text.text("Saving uploaded audio file...")
                progress_bar.progress(10)
                temp_audio_path = save_uploaded_file(uploaded_file, UPLOAD_DIR)
                st.session_state.last_analyzed_file = temp_audio_path
                
                # STEP 2: Whisper speech-to-text
                model_label = os.getenv("WHISPER_MODEL", "tiny").capitalize()
                status_text.text(f"Transcribing speech with OpenAI Whisper ({model_label})...")
                progress_bar.progress(30)
                transcript = transcribe_audio(temp_audio_path)
                
                if not transcript.strip():
                    raise ValueError("No speech or words could be transcribed from this audio. Please verify the audio signal.")
                
                # STEP 3: Filler word analysis
                status_text.text("Analyzing filler words and phrase usage...")
                progress_bar.progress(45)
                filler_stats = analyze_fillers(transcript)
                
                # STEP 4: Audio feature extraction using Librosa
                status_text.text("Extracting acoustic and temporal features (Librosa)...")
                progress_bar.progress(60)
                audio_features = extract_audio_features(temp_audio_path, total_words=filler_stats["total_words"])
                
                # Update fillers per minute and fillers per 100 words with actual duration
                filler_stats = analyze_fillers(transcript, duration_seconds=audio_features["duration"])
                
                # STEP 5: Load reference concept and execute Sentence-BERT similarity
                status_text.text("Computing Sentence-BERT semantic similarity embeddings...")
                progress_bar.progress(75)
                ref_text = concepts_dict.get(selected_topic, "")
                semantic_results = evaluate_understanding(transcript, ref_text)
                
                # STEP 6: Execute scoring logic
                status_text.text("Calculating composite academic scores...")
                progress_bar.progress(85)
                
                fluency_score = calculate_fluency_score(
                    pause_ratio=audio_features["pause_ratio"],
                    filler_percentage=filler_stats["filler_percentage"],
                    speaking_rate=audio_features["speaking_rate"]
                )
                
                confidence_score = calculate_confidence_score(
                    rms_energy=audio_features["rms_energy"],
                    pause_ratio=audio_features["pause_ratio"],
                    mean_pitch=audio_features["mean_pitch"]
                )
                
                composite_scores = generate_report_scores(
                    semantic_score=semantic_results["similarity_score"],
                    fluency_score=fluency_score,
                    confidence_score=confidence_score
                )
                
                # STEP 7: Generate Waveform visualization image
                status_text.text("Generating audio waveform plot...")
                waveform_filename = f"wf_{int(datetime.now().timestamp())}.png"
                temp_waveform_path = os.path.join(OUTPUT_DIR, waveform_filename)
                plot_waveform(temp_audio_path, temp_waveform_path)
                
                # STEP 8: Generate Google Gemini feedback report
                status_text.text("Querying AI feedback services...")
                progress_bar.progress(95)
                ai_feedback = generate_ai_feedback(
                    topic=selected_topic,
                    transcript=transcript,
                    similarity_score=composite_scores["semantic_score"],
                    speech_metrics={
                        "speaking_rate": audio_features["speaking_rate"],
                        "pause_ratio": audio_features["pause_ratio"],
                        "total_fillers": filler_stats["total_fillers"],
                        "filler_percentage": filler_stats["filler_percentage"],
                        "confidence_score": confidence_score
                    }
                )
                
                # STEP 9: Generate Report PDF
                pdf_filename = f"report_{int(datetime.now().timestamp())}.pdf"
                pdf_report_path = os.path.join(OUTPUT_DIR, pdf_filename)
                generate_pdf_report(
                    student_name=student_name,
                    topic=selected_topic,
                    transcript=transcript,
                    scores=composite_scores,
                    speech_metrics=audio_features,
                    feedback=ai_feedback,
                    waveform_image_path=temp_waveform_path,
                    output_path=pdf_report_path
                )
                
                # STEP 10: Save to SQLite database
                status_text.text("Persisting results to Database...")
                save_report(
                    student_name=student_name,
                    topic=selected_topic,
                    transcript=transcript,
                    semantic_score=composite_scores["semantic_score"],
                    speech_score=composite_scores["fluency_score"],
                    overall_score=composite_scores["overall_score"],
                    feedback=ai_feedback
                )
                
                # Store in session state
                st.session_state.analysis_results = {
                    "student_name": student_name,
                    "topic": selected_topic,
                    "transcript": transcript,
                    "scores": composite_scores,
                    "audio_features": audio_features,
                    "filler_stats": filler_stats,
                    "feedback": ai_feedback,
                    "waveform_image_path": temp_waveform_path,
                    "pdf_report_path": pdf_report_path
                }
                
                progress_bar.progress(100)
                status_text.empty()
                st.success("Analysis Complete!")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Analysis Pipeline Failure: {e}")
                st.session_state.analysis_results = None

    # 2. Results Dashboard
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        
        st.markdown("---")
        st.header("Assessment Results Dashboard")
        
        # Row 1: Metrics display
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        
        with col_m1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Overall Score</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #047857;">{}%</div>
            </div>
            """.format(res["scores"]["overall_score"]), unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Semantic Match</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #0284c7;">{}%</div>
            </div>
            """.format(res["scores"]["semantic_score"]), unsafe_allow_html=True)
            
        with col_m3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Speech Fluency</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #1e3a8a;">{}%</div>
            </div>
            """.format(res["scores"]["fluency_score"]), unsafe_allow_html=True)
            
        with col_m4:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Acoustic Confidence</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #0ea5e9;">{}%</div>
            </div>
            """.format(res["scores"]["confidence_score"]), unsafe_allow_html=True)
            
        with col_m5:
            # Map category class
            cat = res["scores"]["performance_category"]
            badge_class = "status-poor"
            if cat == "Excellent": badge_class = "status-excellent"
            elif cat == "Good": badge_class = "status-good"
            elif cat == "Average": badge_class = "status-average"
            
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Classification</div>
                <div style="font-size: 1.6rem; font-weight: 700; margin-top: 0.4rem;"><span class="status-badge {}">{}</span></div>
                <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.2rem;">Grade {}</div>
            </div>
            """.format(badge_class, cat, res["scores"]["letter_grade"]), unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Audio Waveform plot displaying
        st.subheader("Speech Audio Waveform Representation")
        if os.path.exists(res["waveform_image_path"]):
            st.image(res["waveform_image_path"], use_container_width=True)
            
        # Analysis breakdown tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📖 Concept Alignment",
            "💬 Fluency & Fillers",
            "🔊 Acoustic Profile",
            "🤖 AI Feedback",
            "📥 Export Report"
        ])
        
        with tab1:
            st.subheader("Domain Semantic Matching")
            st.write(f"**Selected Topic**: {res['topic']}")
            
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.info("**Student Transcribed Speech**")
                st.write(f"\"{res['transcript']}\"")
            with c_s2:
                st.success("**Reference Concept Definition**")
                st.write(concepts_dict[res['topic']])
                
            st.markdown("---")
            st.metric("Sentence-BERT Cosine Similarity", f"{res['scores']['semantic_score']}%")
            st.markdown(f"Understanding Level Assessment: **{res['scores']['performance_category']}**")

        with tab2:
            st.subheader("Fluency Analysis Breakdown")
            c_f1, c_f2 = st.columns(2)
            
            with c_f1:
                st.metric("Pause Ratio", f"{int(res['audio_features']['pause_ratio'] * 100)}%", 
                          help="Ratio of silent periods detected in the audio file relative to overall speaking duration.")
                st.write(f"- **Total Duration**: {res['audio_features']['duration']} seconds")
                st.write(f"- **Active Speaking Time**: {res['audio_features']['speaking_time']} seconds")
                st.write(f"- **Total Silent gaps**: {res['audio_features']['silent_time']} seconds")
            
            with c_f2:
                st.metric("Estimated Speaking Rate", f"{res['audio_features']['speaking_rate']} WPM",
                          help="Estimated words per minute. Professional talking rate typically ranges between 110 and 150 WPM.")
                st.write(f"- **Word count**: {res['filler_stats']['total_words']} words")
                
            st.markdown("---")
            st.subheader("Filler Word Usage Statistics")
            st.metric("Total Filler Words Count", f"{res['filler_stats']['total_fillers']} ({res['filler_stats']['filler_percentage']}%)")
            
            if res['filler_stats']['filler_counts']:
                df_fillers = pd.DataFrame({
                    "Filler Word / Phrase": list(res['filler_stats']['filler_counts'].keys()),
                    "Count": list(res['filler_stats']['filler_counts'].values())
                })
                st.bar_chart(df_fillers.set_index("Filler Word / Phrase"))
            else:
                st.success("Outstanding speech delivery! No standard fillers (um, uh, you know) detected.")

        with tab3:
            st.subheader("Vocal Acoustic Characteristics")
            c_a1, c_a2, c_a3 = st.columns(3)
            
            with c_a1:
                st.metric("RMS Energy", res['audio_features']['rms_energy'],
                          help="Average amplitude energy. Low values (< 0.015) can represent soft whispering or high distances from the mic.")
            with c_a2:
                st.metric("Zero Crossing Rate", res['audio_features']['zcr'],
                          help="Rate of sign-changes in the signal. Helpful to differentiate voiced sound, unvoiced speech and background hiss.")
            with c_a3:
                pitch_val = res['audio_features']['mean_pitch']
                st.metric("Mean Pitch (F0)", f"{pitch_val} Hz" if pitch_val > 0 else "Not Detected",
                          help="Average fundamental frequency. Normal conversational limits range from 80-250 Hz.")
                
            st.caption("Acoustic parameters are calculated directly from raw speech tensors using the Librosa DSP library.")

        with tab4:
            st.subheader("AI Generative Assessment Feedback")
            st.markdown(res["feedback"])

        with tab5:
            st.subheader("Export Assessment Report")
            st.write("Click the button below to download the official PDF report compiling assessment grades, acoustic waveforms, transcriptions, and AI recommendations.")
            
            if os.path.exists(res["pdf_report_path"]):
                with open(res["pdf_report_path"], "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Evaluation Report",
                        data=pdf_file,
                        file_name=f"VBCUA_Report_{res['student_name'].replace(' ', '_')}_{res['topic'].replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

# ----------------- MENU: ASSESSMENT HISTORY -----------------
elif menu_option == "Assessment History":
    st.markdown("""
    <div class="title-banner">
        <h1>Evaluation History Log</h1>
        <p>Review and manage historical assessment reports saved locally in the SQLite Database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    history_records = get_history()
    
    if not history_records:
        st.info("No assessments have been recorded yet. Select 'Upload & Analyze' to run your first technical evaluation.")
    else:
        # Action controls for full deletion
        col_h1, col_h2 = st.columns([6, 1])
        with col_h2:
            if st.button("Clear All History", type="secondary"):
                if clear_history():
                    st.success("History database successfully wiped.")
                    st.rerun()
                    
        # Process history data to table dataframe
        history_df = pd.DataFrame(history_records)
        history_df.rename(columns={
            "student_name": "Student Name",
            "topic": "Topic Name",
            "semantic_score": "Semantic Similarity %",
            "speech_score": "Speech Fluency %",
            "overall_score": "Overall Rating %",
            "created_at": "Evaluation Timestamp"
        }, inplace=True)
        
        # Display main grid view
        st.dataframe(
            history_df[["id", "Student Name", "Topic Name", "Semantic Similarity %", "Speech Fluency %", "Overall Rating %", "Evaluation Timestamp"]],
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown("---")
        st.subheader("Detailed Report Inspection")
        
        selected_record_id = st.selectbox(
            "Select Evaluation ID to review detailed analysis",
            options=[r["id"] for r in history_records],
            format_func=lambda x: f"ID {x} - {next(item['student_name'] for item in history_records if item['id'] == x)} ({next(item['topic'] for item in history_records if item['id'] == x)})"
        )
        
        if selected_record_id:
            # Find the specific record
            rec = next(item for item in history_records if item["id"] == selected_record_id)
            
            c_det1, c_det2 = st.columns([1, 1])
            with c_det1:
                st.markdown(f"**Student**: {rec['student_name']}")
                st.markdown(f"**Topic**: {rec['topic']}")
                st.markdown(f"**Date Evaluated**: {rec['created_at']}")
            with c_det2:
                st.markdown(f"**Semantic Similarity**: {rec['semantic_score']}%")
                st.markdown(f"**Speech Fluency**: {rec['speech_score']}%")
                st.markdown(f"**Overall Weighted Score**: {rec['overall_score']}%")
                
            st.markdown("**Speech Transcript**:")
            st.info(f"\"{rec['transcript']}\"")
            
            st.markdown("**AI Assessment Feedback**:")
            st.markdown(rec["feedback"])
            
            # Allow individual deletion
            if st.button(f"Delete Evaluation ID {selected_record_id}", type="secondary"):
                if delete_report(selected_record_id):
                    st.success(f"Report ID {selected_record_id} successfully deleted.")
                    st.rerun()

# ----------------- MENU: CONCEPTS GUIDE -----------------
elif menu_option == "Concepts Guide":
    st.markdown("""
    <div class="title-banner">
        <h1>Reference Concepts Guide</h1>
        <p>Review pre-configured academic definitions for semantic similarity comparisons, and extend the conceptual repository.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Pre-configured Reference Core Concepts")
    
    # Display concepts in expandable boxes
    for concept, explanation in concepts_dict.items():
        with st.expander(concept):
            st.write(explanation)
            
    st.markdown("---")
    st.subheader("💡 Add/Extend Conceptual Database")
    st.write("Easily extend the system with new conceptual topics and reference academic descriptions.")
    
    with st.form("extend_concepts_form"):
        new_concept_name = st.text_input("New Concept Topic Name", placeholder="e.g. Data Structure")
        new_concept_description = st.text_area(
            "Reference Academic Explanation",
            placeholder="Introduce the technical definition, structural properties, and main elements of the concept..."
        )
        
        submit_btn = st.form_submit_button("Add Topic to Repository")
        
        if submit_btn:
            if not new_concept_name.strip() or not new_concept_description.strip():
                st.error("Please enter both the concept name and explanation description.")
            elif new_concept_name in concepts_dict:
                st.error("This topic is already configured in the concepts database.")
            else:
                concepts_dict[new_concept_name] = new_concept_description
                
                # Write back to JSON file
                concepts_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "reference_concepts",
                    "concepts.json"
                )
                try:
                    with open(concepts_path, "w", encoding="utf-8") as f:
                        json.dump(concepts_dict, f, indent=2)
                    st.success(f"Topic '{new_concept_name}' successfully added to concepts database!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to update concepts database JSON: {ex}")
