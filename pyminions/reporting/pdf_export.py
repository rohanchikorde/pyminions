import pdfkit
import os

def export_html_to_pdf(html_path, pdf_path=None):
    if pdf_path is None:
        pdf_path = os.path.splitext(html_path)[0] + '.pdf'
    pdfkit.from_file(html_path, pdf_path)
    return pdf_path
