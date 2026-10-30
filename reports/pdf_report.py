import os
import re
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# NumberedCanvas helper for footer page numbering: "Page X of Y"
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header banner (top of every page except first)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Voice-Based Concept Understanding Analyser (VBCUA) - Assessment Report")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (on all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        # Generation Date footer
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.drawString(54, 40, f"Generated automatically on {date_str} | VBCUA System")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()

def clean_xml_string(text: str) -> str:
    """
    Escapes characters that violate XML parsers used by ReportLab Paragraphs.
    """
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Restore allowed tags
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;u&gt;", "<u>").replace("&lt;/u&gt;", "</u>")
    text = text.replace("&lt;br/&gt;", "<br/>").replace("&lt;br &gt;", "<br/>")
    return text

def parse_markdown_to_flowables(md_text: str, styles: Any) -> List[Any]:
    """
    Parses a markdown feedback string into structured ReportLab flowables (Paragraphs).
    """
    flowables = []
    lines = md_text.split("\n")
    
    current_paragraph = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_paragraph:
                # Output previous text block
                p_text = clean_xml_string(" ".join(current_paragraph))
                flowables.append(Paragraph(p_text, styles["BodyText"]))
                flowables.append(Spacer(1, 6))
                current_paragraph = []
            continue
            
        # Parse headings
        if stripped.startswith("### "):
            if current_paragraph:
                p_text = clean_xml_string(" ".join(current_paragraph))
                flowables.append(Paragraph(p_text, styles["BodyText"]))
                flowables.append(Spacer(1, 6))
                current_paragraph = []
            heading_text = clean_xml_string(stripped[4:])
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(heading_text, styles["FeedbackHeading"]))
            flowables.append(Spacer(1, 4))
            
        # Parse bullet points
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if current_paragraph:
                p_text = clean_xml_string(" ".join(current_paragraph))
                flowables.append(Paragraph(p_text, styles["BodyText"]))
                flowables.append(Spacer(1, 6))
                current_paragraph = []
            # Convert **text** in bullets to bold tags
            bullet_content = stripped[2:]
            bullet_content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", bullet_content)
            bullet_content = clean_xml_string(bullet_content)
            flowables.append(Paragraph(bullet_content, styles["FeedbackBullet"]))
            
        else:
            # Regular text line - accumulate paragraphs
            processed_line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", stripped)
            current_paragraph.append(processed_line)
            
    # Output remaining text block
    if current_paragraph:
        p_text = clean_xml_string(" ".join(current_paragraph))
        flowables.append(Paragraph(p_text, styles["BodyText"]))
        
    return flowables

def generate_pdf_report(
    student_name: str,
    topic: str,
    transcript: str,
    scores: Dict[str, Any],
    speech_metrics: Dict[str, Any],
    feedback: str,
    waveform_image_path: str,
    output_path: str
) -> str:
    """
    Generates a beautifully styled assessment PDF report using ReportLab.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Page setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54, # 0.75 inch
        rightMargin=54,
        topMargin=54,
        bottomMargin=64
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Define custom colors
    c_primary = colors.HexColor("#0f172a")    # Slate 900
    c_secondary = colors.HexColor("#1e3a8a")  # Blue 900
    c_accent = colors.HexColor("#0284c7")     # Sky 600
    c_text = colors.HexColor("#334155")       # Slate 700
    c_bg_light = colors.HexColor("#f8fafc")   # Slate 50
    c_border = colors.HexColor("#e2e8f0")     # Slate 200
    
    # Modifying default styles
    styles['Normal'].textColor = c_text
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 13
    
    styles['BodyText'].textColor = c_text
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 14

    # Custom styles
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=6
    ))
    
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='TableLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b")
    ))

    styles.add(ParagraphStyle(
        name='TableValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=c_text
    ))

    styles.add(ParagraphStyle(
        name='ScoreMetric',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_accent,
        alignment=1 # Center aligned
    ))
    
    styles.add(ParagraphStyle(
        name='ScoreLabel',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1 # Center aligned
    ))

    styles.add(ParagraphStyle(
        name='FeedbackHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='FeedbackBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    ))

    story = []
    
    # 1. Header Banner
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", styles["MainTitle"]))
    story.append(Paragraph("Acoustic Fluency & Semantic Knowledge Assessment Report", styles["Subtitle"]))
    
    # 2. Metadata Section (Student Details)
    metadata_data = [
        [
            Paragraph("Student Name:", styles["TableLabel"]),
            Paragraph(clean_xml_string(student_name), styles["TableValue"]),
            Paragraph("Assessment Date:", styles["TableLabel"]),
            Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["TableValue"])
        ],
        [
            Paragraph("Evaluated Topic:", styles["TableLabel"]),
            Paragraph(clean_xml_string(topic), styles["TableValue"]),
            Paragraph("Overall Grade:", styles["TableLabel"]),
            Paragraph(f"<b>{scores.get('letter_grade', 'N/A')}</b> ({scores.get('performance_category', 'N/A')})", styles["TableValue"])
        ]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[1.2*inch, 2.3*inch, 1.3*inch, 2.2*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_bg_light),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.25, c_border),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 15))
    
    # 3. Score Summary Banner
    score_cards = [
        [
            Paragraph("Semantic Score", styles["ScoreLabel"]),
            Paragraph("Fluency Score", styles["ScoreLabel"]),
            Paragraph("Confidence Score", styles["ScoreLabel"]),
            Paragraph("Overall Performance Score", styles["ScoreLabel"])
        ],
        [
            Paragraph(f"{scores.get('semantic_score', 0.0)}%", styles["ScoreMetric"]),
            Paragraph(f"{scores.get('fluency_score', 0.0)}%", styles["ScoreMetric"]),
            Paragraph(f"{scores.get('confidence_score', 0.0)}%", styles["ScoreMetric"]),
            Paragraph(f"<font color='#047857'><b>{scores.get('overall_score', 0.0)}%</b></font>", styles["ScoreMetric"])
        ]
    ]
    
    score_table = Table(score_cards, colWidths=[1.75*inch, 1.75*inch, 1.75*inch, 1.75*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")), # Mint green bg
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1.0, colors.HexColor("#bbf7d0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bbf7d0")),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 15))
    
    # 4. Speech Delivery Metrics Table
    story.append(Paragraph("Acoustic & Speech Delivery Metrics", styles["SectionHeader"]))
    
    metrics_data = [
        [
            Paragraph("Speaking Duration", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('duration', 0.0)} seconds", styles["TableValue"]),
            Paragraph("Estimated Speech Rate", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('speaking_rate', 0.0)} WPM", styles["TableValue"])
        ],
        [
            Paragraph("Active Speaking Time", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('speaking_time', 0.0)} seconds", styles["TableValue"]),
            Paragraph("Mean Speech Energy (RMS)", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('rms_energy', 0.0000)}", styles["TableValue"])
        ],
        [
            Paragraph("Silent/Pause Time", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('silent_time', 0.0)} seconds", styles["TableValue"]),
            Paragraph("Mean Pitch (F0)", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('mean_pitch', 0.0)} Hz" if speech_metrics.get('mean_pitch', 0.0) > 0 else "N/A", styles["TableValue"])
        ],
        [
            Paragraph("Pause-to-Speech Ratio", styles["TableLabel"]),
            Paragraph(f"{int(speech_metrics.get('pause_ratio', 0.0) * 100)}%", styles["TableValue"]),
            Paragraph("Filler Words Count", styles["TableLabel"]),
            Paragraph(f"{speech_metrics.get('total_fillers', 0)} ({speech_metrics.get('filler_percentage', 0.0)}%)", styles["TableValue"])
        ]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 2.0*inch, 1.6*inch, 1.9*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.25, c_border),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # 5. Audio Waveform
    if waveform_image_path and os.path.exists(waveform_image_path):
        waveform_elements = []
        waveform_elements.append(Paragraph("Speech Signal Waveform Visualization", styles["SectionHeader"]))
        try:
            # We scale the image down to fit the report neatly (approx 6.5 inches wide, 1.8 inches tall)
            img = Image(waveform_image_path, width=6.5*inch, height=1.8*inch)
            waveform_elements.append(img)
            waveform_elements.append(Spacer(1, 15))
            story.append(KeepTogether(waveform_elements))
        except Exception as e:
            print(f"Failed to embed waveform image in PDF: {e}")

    # 6. Transcript
    transcript_elements = []
    transcript_elements.append(Paragraph("Transcribed Speech Response", styles["SectionHeader"]))
    escaped_transcript = clean_xml_string(transcript if transcript else "[No speech transcribed]")
    transcript_elements.append(Paragraph(f"<i>\"{escaped_transcript}\"</i>", styles["BodyText"]))
    transcript_elements.append(Spacer(1, 15))
    story.append(KeepTogether(transcript_elements))
    
    # 7. AI Educational Feedback (Using KeepTogether or letting it break naturally)
    feedback_elements = [Paragraph("AI Educational Evaluation & Feedback", styles["SectionHeader"])]
    flowables = parse_markdown_to_flowables(feedback, styles)
    feedback_elements.extend(flowables)
    story.append(KeepTogether(feedback_elements))
    
    # Build Document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
