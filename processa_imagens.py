import os
import cv2
import numpy as np

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao ler {img_path}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                blur = cv2.GaussianBlur(eq, (3, 3), 0)
                edges = cv2.Canny(blur, 120, 250)
                img_out = img.copy()
                img_out[edges > 0] = [0, 0, 255]  # destaca contornos em vermelho
                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, img_out)
                print(f"Processado: {out_path}")

if __name__ == "__main__":
    input_dir = input("Digite o diretório de entrada das imagens: ")
    output_dir = input("Digite o diretório de saída para imagens processadas: ")
    process_images(input_dir, output_dir)