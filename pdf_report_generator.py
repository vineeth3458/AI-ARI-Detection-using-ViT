from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime
from PIL import Image
import os

class MedicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3b82f6'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='FindingText',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
    
    def generate_report(self, analysis_data, image=None, patient_info=None):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        story.append(Paragraph("AI-Powered ARI Detection Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        report_info = [
            ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['Image Filename:', analysis_data.get('filename', 'N/A')],
            ['Analysis ID:', str(analysis_data.get('id', 'N/A'))],
            ['AI Model:', analysis_data.get('model', 'N/A')]
        ]
        
        if patient_info:
            for key, value in patient_info.items():
                report_info.append([f'{key}:', value])
        
        info_table = Table(report_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8'))
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        if image:
            story.append(Paragraph("Medical Image", self.styles['SectionHeader']))
            try:
                img_buffer = BytesIO()
                img_resized = image.copy()
                img_resized.thumbnail((400, 400), Image.Resampling.LANCZOS)
                img_resized.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                rl_image = RLImage(img_buffer, width=4*inch, height=4*inch)
                story.append(rl_image)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"Image could not be embedded: {str(e)}", 
                                     self.styles['BodyText']))
        
        story.append(Paragraph("Diagnostic Results", self.styles['SectionHeader']))
        
        diagnosis = analysis_data.get('prediction', 'N/A')
        confidence = analysis_data.get('confidence', 0) * 100
        
        diagnosis_color = colors.green if diagnosis == 'Normal' else colors.red
        
        diagnosis_data = [
            ['Diagnosis:', diagnosis],
            ['Confidence:', f'{confidence:.2f}%'],
        ]
        
        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 12),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 12),
            ('FONT', (1, 0), (1, 0), 'Helvetica-Bold', 14),
            ('TEXTCOLOR', (1, 0), (1, 0), diagnosis_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9fafb'))
        ]))
        
        story.append(diagnosis_table)
        story.append(Spacer(1, 0.2*inch))
        
        probabilities = analysis_data.get('probabilities', {})
        if probabilities:
            story.append(Paragraph("Classification Probabilities", self.styles['SectionHeader']))
            
            prob_data = [['Condition', 'Probability']]
            for condition, prob in sorted(probabilities.items(), 
                                        key=lambda x: x[1], reverse=True):
                prob_data.append([condition, f'{prob*100:.2f}%'])
            
            prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
            prob_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
                ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
                 [colors.white, colors.HexColor('#f0f4f8')])
            ]))
            
            story.append(prob_table)
            story.append(Spacer(1, 0.2*inch))
        
        findings = analysis_data.get('findings')
        if findings:
            story.append(Paragraph("Radiological Findings", self.styles['SectionHeader']))
            story.append(Paragraph(findings, self.styles['FindingText']))
            story.append(Spacer(1, 0.1*inch))
        
        key_features = analysis_data.get('key_features', [])
        if key_features:
            story.append(Paragraph("Key Features Identified", self.styles['SectionHeader']))
            for idx, feature in enumerate(key_features, 1):
                story.append(Paragraph(f"{idx}. {feature}", self.styles['BodyText']))
            story.append(Spacer(1, 0.1*inch))
        
        recommendations = analysis_data.get('recommendations')
        if recommendations:
            story.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))
            story.append(Paragraph(recommendations, self.styles['FindingText']))
        
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Important Disclaimer", self.styles['SectionHeader']))
        disclaimer_text = """This report is generated by an AI-powered diagnostic assistance system 
        and is intended for research and educational purposes only. It should not be used as the sole 
        basis for clinical decision-making. All diagnoses must be confirmed by qualified medical 
        professionals, including radiologists and physicians. The AI system's predictions are 
        probabilistic and may contain errors. Always consult with appropriate medical experts 
        before making any treatment decisions."""
        
        story.append(Paragraph(disclaimer_text, self.styles['FindingText']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_batch_report(self, analyses_list, images_dict=None):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        story.append(Paragraph("Batch Analysis Report", self.styles['CustomTitle']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             self.styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        summary_data = [['Total Images Analyzed:', str(len(analyses_list))]]
        
        predictions_count = {}
        for analysis in analyses_list:
            pred = analysis.get('prediction', 'Unknown')
            predictions_count[pred] = predictions_count.get(pred, 0) + 1
        
        for pred, count in predictions_count.items():
            summary_data.append([f'{pred} Cases:', str(count)])
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica-Bold', 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f4f8'))
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        for idx, analysis in enumerate(analyses_list, 1):
            story.append(Paragraph(f"Analysis {idx}: {analysis.get('filename', 'N/A')}", 
                                 self.styles['SectionHeader']))
            
            analysis_data = [
                ['Diagnosis:', analysis.get('prediction', 'N/A')],
                ['Confidence:', f"{analysis.get('confidence', 0)*100:.2f}%"],
                ['Timestamp:', analysis.get('timestamp', 'N/A')]
            ]
            
            analysis_table = Table(analysis_data, colWidths=[2*inch, 4*inch])
            analysis_table.setStyle(TableStyle([
                ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
                ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            
            story.append(analysis_table)
            story.append(Spacer(1, 0.2*inch))
            
            if idx < len(analyses_list):
                story.append(PageBreak())
        
        doc.build(story)
        buffer.seek(0)
        return buffer
