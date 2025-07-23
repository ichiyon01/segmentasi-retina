import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from PIL import Image
import numpy as np

class PDFGenerator:
    def __init__(self):
        self.margin = 40
        self.header_space = 100
        self.img_width = 150
        self.img_height = 120
        self.spacing = 25

    def draw_header(self, c, width, height):
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2, height - self.margin, "Perbandingan Segmentasi Retina")
        c.setFont("Helvetica", 10)
        c.drawCentredString(width / 2, height - self.margin - 15, "Model U-Net dengan CLAHE vs Tanpa CLAHE")
        c.line(self.margin, height - self.margin - 25, width - self.margin, height - self.margin - 25)

        # Tambahkan legenda warna
        legend_y = height - self.margin - 45
        legend_x = self.margin

        c.setFont("Helvetica", 8)
        for name, color in zip([
            "above ILM", "ILM-IPL/INL", "IPL/INL-RPE", "RPE-BM", "under BM", "PED"
        ], [
            (0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255), (0, 0, 255), (0, 255, 255)
        ]):
            c.setFillColorRGB(color[0] / 255, color[1] / 255, color[2] / 255)
            c.rect(legend_x, legend_y, 10, 10, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.drawString(legend_x + 15, legend_y, name)
            legend_x += 100

    def generate(self, images):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        page_number = 1

        self.draw_header(c, width, height)
        y_offset = height - self.margin - self.header_space - 100

        for idx, (ori, clahe_img, no_clahe_img, name) in enumerate(images):
            if y_offset - self.img_height < 100:
                c.setFont("Helvetica", 8)
                c.drawRightString(width - self.margin, 20, f"Halaman {page_number}")
                c.showPage()
                page_number += 1
                self.draw_header(c, width, height)
                y_offset = height - self.margin - self.header_space - 100

            x_positions = [
                self.margin,
                self.margin + self.img_width + self.spacing,
                self.margin + 2 * (self.img_width + self.spacing)
            ]

            c.setFont("Helvetica-Bold", 9)
            c.drawString(x_positions[0], y_offset + self.img_height + 12, "Gambar Asli")
            c.drawString(x_positions[1], y_offset + self.img_height + 12, "U-Net + CLAHE")
            c.drawString(x_positions[2], y_offset + self.img_height + 12, "U-Net Tanpa CLAHE")

            for i, img in enumerate([ori, clahe_img, no_clahe_img]):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                c.drawImage(ImageReader(img_byte_arr),
                            x_positions[i], y_offset,
                            width=self.img_width, height=self.img_height)

            c.setFont("Helvetica", 8)
            c.drawString(self.margin, y_offset - 12, f"Nama file: {name}")
            c.setStrokeColor(colors.grey)
            c.setLineWidth(0.3)
            c.line(self.margin, y_offset - 20, width - self.margin, y_offset - 20)

            y_offset -= (self.img_height + self.spacing + 30)

        c.setFont("Helvetica", 8)
        c.drawRightString(width - self.margin, 20, f"Halaman {page_number}")
        c.save()
        buffer.seek(0)
        return buffer
