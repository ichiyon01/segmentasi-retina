import numpy as np
import cv2
from PIL import Image

class_names = ["above ILM", "ILM-IPL/INL", "IPL/INL-RPE", "RPE-BM", "under BM", "PED"]
class_colors = [
    (0, 0, 0),       # 0: hitam - Above ILM
    (255, 0, 0),     # 1: merah - ILM-IPL/INL
    (255, 255, 0),   # 2: kuning - IPL/INL-RPE
    (255, 255, 255), # 3: putih - RPE-BM
    (0, 0, 255),     # 4: biru - under BM
    (0, 255, 255)    # 5: cyan - PED
]

class ImageProcessor:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size

    def CalHistogram(self, channel):
        hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        return hist

    def ClipHistogram(self, hist, clip_limit):
        excess = hist - clip_limit
        excess[excess < 0] = 0
        total_excess = np.sum(excess)

        hist = np.minimum(hist, clip_limit)
        redist = total_excess // 256
        hist += redist

        remainder = total_excess % 256
        for i in range(256):
            if remainder <= 0:
                break
            hist[i] += 1
            remainder -= 1

        return hist

    def CreateMapping(self, hist, block_size):
        cdf = np.cumsum(hist)
        cdf_min = cdf[np.nonzero(cdf)][0]
        mapping = np.round((cdf - cdf_min) / float(block_size - cdf_min) * 255).astype(np.uint8)
        return mapping

    def ComputeMappings(self, image, n_rows, n_cols, cell_h, cell_w, clip_limit):
        mappings = []
        for i in range(n_rows):
            row_maps = []
            for j in range(n_cols):
                y0, y1 = i * cell_h, (i + 1) * cell_h
                x0, x1 = j * cell_w, (j + 1) * cell_w
                cell = image[y0:y1, x0:x1]

                hist = self.CalHistogram(cell)
                clipped_hist = self.ClipHistogram(hist, clip_limit)
                mapping = self.CreateMapping(clipped_hist, cell.size)
                row_maps.append(mapping)
            mappings.append(row_maps)
        return mappings

    def ApplyInterpolation(self, image, mappings, n_rows, n_cols, cell_h, cell_w):
        height, width = image.shape
        result = np.zeros_like(image, dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                row = min(i // cell_h, n_rows - 2)
                col = min(j // cell_w, n_cols - 2)

                y_ratio = (i - row * cell_h) / cell_h
                x_ratio = (j - col * cell_w) / cell_w

                mappings_ij = mappings[row][col]
                mappings_i1j = mappings[row + 1][col]
                mappings_ij1 = mappings[row][col + 1]
                mappings_i1j1 = mappings[row + 1][col + 1]

                pixel_val = image[i, j]

                top = (1 - x_ratio) * mappings_ij[pixel_val] + x_ratio * mappings_ij1[pixel_val]
                bottom = (1 - x_ratio) * mappings_i1j[pixel_val] + x_ratio * mappings_i1j1[pixel_val]
                interpolated_val = (1 - y_ratio) * top + y_ratio * bottom

                result[i, j] = np.clip(interpolated_val, 0, 255)

        return result

    def HistogramEqualizationClaheGrayscale(self, image, clip_limit=10, grid_size=(8, 8)):
        height, width = image.shape
        n_rows, n_cols = grid_size
        cell_h, cell_w = height // n_rows, width // n_cols

        mappings = self.ComputeMappings(image, n_rows, n_cols, cell_h, cell_w, clip_limit)
        equalized_image = self.ApplyInterpolation(image, mappings, n_rows, n_cols, cell_h, cell_w)
        return equalized_image

    def preprocess(self, pil_img, use_clahe=False):
        img_gray = np.array(pil_img.convert("L").resize(self.img_size))

        if use_clahe:
            clip_limit = 4.0
            grid_size = (4, 4)
            img_gray = self.HistogramEqualizationClaheGrayscale(
                img_gray, clip_limit=clip_limit, grid_size=grid_size
            )

        # Terapkan denoising menggunakan OpenCV
        img_denoised = cv2.fastNlMeansDenoising(img_gray, None, h=40, templateWindowSize=7, searchWindowSize=21)
        
        if use_clahe:
            clip_limit = 4.0
            grid_size = (4, 4)
            img_denoised = self.HistogramEqualizationClaheGrayscale(
                img_denoised, clip_limit=clip_limit, grid_size=grid_size
            )

        # Normalisasi ke [0, 1]
        img_norm = img_denoised / 255.0

        # Ubah dimensi jadi (1, H, W, 1)
        return np.expand_dims(img_norm, axis=(0, -1)), img_denoised

    def predict(self, image_tensor, model):
        pred = model.predict(image_tensor)[0]
        pred_class = np.argmax(pred, axis=-1)
        mask_rgb = np.zeros((*pred_class.shape, 3), dtype=np.uint8)
        for i, color in enumerate(class_colors):
            mask_rgb[pred_class == i] = color
        return Image.fromarray(mask_rgb)
