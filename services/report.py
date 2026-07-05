import os
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


class ReportGenerator:

    def __init__(self, filename="reports/report.pdf"):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.filename = filename
        self.styles = getSampleStyleSheet()

    def generate(self, df_results, best_model, target, task_type):

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        doc = SimpleDocTemplate(self.filename)
        elements = []

        # Title
        elements.append(
            Paragraph("🤖 AutoML Studio Report", self.styles["Title"])
        )
        elements.append(Spacer(1, 12))

        # Info
        info_text = f"""
        <b>Date:</b> {datetime.datetime.now()}<br/>
        <b>Target:</b> {target}<br/>
        <b>Task:</b> {task_type}<br/>
        <b>Best Model:</b> {best_model}
        """

        elements.append(Paragraph(info_text, self.styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Table
        data = [["Model"] + list(df_results.columns)]

        for idx, row in df_results.iterrows():
            data.append([idx] + list(row.values))

        table = Table(data)

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ]))

        elements.append(table)

        # Build PDF
        doc.build(elements)

        return self.filename