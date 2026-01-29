# -*- coding: utf-8 -*-
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
import datetime
import textwrap

class PDFReportEngine:
    def __init__(self, buffer, patient_data, predict_result, nlg_report):
        self.buffer = buffer
        self.data = patient_data
        self.res = predict_result
        self.text = nlg_report
        
        self.c = canvas.Canvas(self.buffer, pagesize=A4)
        self.width, self.height = A4
        self.margin = 20 * mm
        self.y = self.height - self.margin
        
        self.hospital_name = "Yichang Central People's Hospital"
        self.system_name = "ATBAD 3-Year Mortality Predictor (SVM)"

    def _draw_header(self):
        self.c.setFillColor(colors.HexColor("#003366"))
        self.c.rect(0, self.height - 30*mm, self.width, 30*mm, fill=1, stroke=0)
        
        self.c.setFillColor(colors.white)
        self.c.setFont("Times-Bold", 18)
        self.c.drawString(self.margin, self.height - 15*mm, "ATBAD Clinical Risk Assessment")
        
        self.c.setFont("Times-Roman", 10)
        self.c.drawString(self.margin, self.height - 22*mm, f"{self.hospital_name} | {self.system_name}")
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.c.drawRightString(self.width - self.margin, self.height - 15*mm, f"Generated: {now}")
        
        self.y -= 40 * mm

    def _draw_patient_table(self):
        self.c.setFillColor(colors.black)
        self.c.setFont("Times-Bold", 12)
        self.c.drawString(self.margin, self.y, "1. Patient Clinical Profile")
        self.y -= 5 * mm
        
        col_width = (self.width - 2*self.margin) / 2
        row_height = 7 * mm
        start_y = self.y
        
        # --- ATBAD 7个特征 ---
        # 格式化处理：Yes/No转换
        chd_str = "Yes" if self.data.get('coronary heart disease') == 1 else "No"
        renal_str = "Yes" if self.data.get('renal dysfunction') == 1 else "No"
        
        rows = [
            ("Age (years)", f"{self.data.get('age', 0)}"),
            ("Heart Rate (bpm)", f"{self.data.get('HR', 0)}"),
            ("BUN (mmol/L)", f"{self.data.get('BUN', 0)}"),
            ("Hemoglobin (g/L)", f"{self.data.get('HGB', 0)}"),
            ("Hospitalization (days)", f"{self.data.get('hospitalization', 0)}"),
            ("Coronary Heart Disease", chd_str),
            ("Renal Dysfunction", renal_str)
        ]
        
        self.c.setFont("Times-Roman", 10)
        self.c.setLineWidth(0.5)
        self.c.setStrokeColor(colors.grey)
        
        for i, (label, val) in enumerate(rows):
            current_y = start_y - (i * row_height)
            if i % 2 == 0:
                self.c.setFillColor(colors.whitesmoke)
                self.c.rect(self.margin, current_y - row_height + 2*mm, self.width - 2*self.margin, row_height, fill=1, stroke=0)
            
            self.c.setFillColor(colors.black)
            self.c.setFont("Times-Bold", 10)
            self.c.drawString(self.margin + 5*mm, current_y - 4*mm, label)
            
            self.c.setFont("Times-Roman", 10)
            self.c.drawString(self.margin + col_width, current_y - 4*mm, str(val))
            
            self.c.line(self.margin, current_y - 7*mm, self.width - self.margin, current_y - 7*mm)
            
        self.y = start_y - (len(rows) * row_height) - 10*mm

    def _draw_risk_gauge(self):
        self.c.setFont("Times-Bold", 12)
        self.c.drawString(self.margin, self.y, "2. Mortality Risk Stratification")
        self.y -= 10 * mm
        
        bar_width = self.width - 2*self.margin
        bar_height = 10 * mm
        prob = self.res['prob']
        threshold = self.res['threshold']
        
        self.c.setFillColor(colors.lightgrey)
        self.c.roundRect(self.margin, self.y, bar_width, bar_height, 2*mm, fill=1, stroke=0)
        
        # 红色代表高危
        fill_color = colors.red if prob >= threshold else colors.green
        # 计算填充长度 (归一化到 Bar 宽度)
        fill_width = min(bar_width, bar_width * prob) # 假设最大 100%
        
        self.c.setFillColor(fill_color)
        self.c.roundRect(self.margin, self.y, fill_width, bar_height, 2*mm, fill=1, stroke=0)
        
        # 画阈值线
        thresh_x = self.margin + (bar_width * threshold)
        self.c.setStrokeColor(colors.black)
        self.c.setLineWidth(2)
        self.c.line(thresh_x, self.y - 2*mm, thresh_x, self.y + bar_height + 2*mm)
        self.c.drawString(thresh_x - 5*mm, self.y + bar_height + 3*mm, f"Cut-off: {threshold}")
        
        self.y -= 8 * mm
        self.c.setFont("Times-Bold", 14)
        self.c.setFillColor(fill_color)
        self.c.drawString(self.margin, self.y, f"Predicted Risk: {prob:.1%}")
        self.c.drawRightString(self.width - self.margin, self.y, f"Status: {self.res['risk_label']}")
        
        self.y -= 15 * mm

    def _draw_narrative(self):
        self.c.setFillColor(colors.black)
        self.c.setFont("Times-Bold", 12)
        self.c.drawString(self.margin, self.y, "3. AI Clinical Analysis")
        self.y -= 8 * mm
        
        self.c.setFont("Times-Roman", 10)
        line_height = 5 * mm
        max_width = 90
        
        clean_text = self.text.replace("### ", "").replace("**", "").replace("#### ", "")
        
        lines = []
        for paragraph in clean_text.split('\n'):
            wrapped = textwrap.wrap(paragraph, width=max_width)
            if not wrapped:
                lines.append("")
            else:
                lines.extend(wrapped)
        
        for line in lines:
            if self.y < self.margin:
                self.c.showPage()
                self._draw_header()
                self.y = self.height - 50*mm
                self.c.setFont("Times-Roman", 10)
            
            self.c.drawString(self.margin, self.y, line)
            self.y -= line_height

    def _draw_footer(self):
        self.c.saveState()
        self.c.setFont("Times-Italic", 8)
        self.c.setFillColor(colors.grey)
        footer_text = "Generated by ATBAD-SVM Predictor. (C) 2026 Yichang Central People's Hospital."
        self.c.drawCentredString(self.width / 2, 10*mm, footer_text)
        self.c.restoreState()

    def generate(self):
        self._draw_header()
        self._draw_patient_table()
        self._draw_risk_gauge()
        self._draw_narrative()
        self._draw_footer()
        self.c.save()
        self.buffer.seek(0)
        return self.buffer
