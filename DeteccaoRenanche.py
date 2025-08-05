import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QRect
import cv2

from ImageAI import NeuralAI  # Certifique-se de que ImageAI.py está no mesmo diretório ou no PYTHONPATH

class PredictFromCamera(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predição por Áreas da Foto")
        self.setGeometry(100, 100, 400, 300)
        self.model_input_size = (64, 64)  # valor padrão
        self.model_path = ""
        self.class_txt_path = ""
        self.precisao_path = ""
        self.precisao_val_path = ""
        self.json_path = ""
        self.camera_index = 0
        self.areas = []
        self.neural_ai = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_btn = QPushButton("Selecionar Modelo (.hdf5)")
        self.model_btn.clicked.connect(self.select_model)
        layout.addWidget(self.model_btn)

        self.class_btn = QPushButton("Selecionar Arquivo de Classes (.txt)")
        self.class_btn.clicked.connect(self.select_class_txt)
        layout.addWidget(self.class_btn)

        self.precisao_btn = QPushButton("Selecionar Precisão (.npy)")
        self.precisao_btn.clicked.connect(self.select_precisao)
        layout.addWidget(self.precisao_btn)

        self.precisao_val_btn = QPushButton("Selecionar Precisão Val (.npy)")
        self.precisao_val_btn.clicked.connect(self.select_precisao_val)
        layout.addWidget(self.precisao_val_btn)

        self.json_btn = QPushButton("Selecionar Áreas (JSON)")
        self.json_btn.clicked.connect(self.select_json)
        layout.addWidget(self.json_btn)

        self.camera_btn = QPushButton("Selecionar Câmera")
        self.camera_btn.clicked.connect(self.select_camera)
        layout.addWidget(self.camera_btn)

        self.image_btn = QPushButton("Selecionar Imagem de Arquivo")
        self.image_btn.clicked.connect(self.select_image_file)
        layout.addWidget(self.image_btn)

        self.predict_file_btn = QPushButton("Predizer Áreas na Imagem de Arquivo")
        self.predict_file_btn.clicked.connect(self.predict_on_image_file)
        layout.addWidget(self.predict_file_btn)

        self.predict_camera_btn = QPushButton("Predizer Áreas Usando Câmera")
        self.predict_camera_btn.clicked.connect(self.predict_on_camera)
        layout.addWidget(self.predict_camera_btn)

        self.image_view = QLabel("Imagem não carregada")
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("border: 2px solid gray; min-height: 200px;")
        layout.addWidget(self.image_view)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #0055aa; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        from PyQt5.QtWidgets import QTextEdit
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("border: 1px solid #aaa; background: #f8f8f8; min-height: 160px; padding: 6px;")
        self.output_text.setMinimumHeight(160)
        layout.addWidget(self.output_text)

        self.image_path = None
        self.captured_img = None
        self.setLayout(layout)

    def draw_areas_on_pixmap(self, pixmap):
        if not pixmap or not self.areas:
            return pixmap
        painter = QPainter(pixmap)
        painter.setPen(Qt.red)
        for area in self.areas:
            x, y, w, h = area["x"], area["y"], area["width"], area["height"]
            painter.drawRect(x, y, w, h)
        painter.end()
        return pixmap

    def select_image_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Imagens (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path:
            self.image_path = path
            self.captured_img = cv2.imread(path)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = self.draw_areas_on_pixmap(pixmap)
                self.image_view.setPixmap(pixmap.scaled(self.image_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image_view.setText("")
            else:
                self.image_view.setText("Erro ao carregar imagem.")
            QMessageBox.information(self, "Imagem Selecionada", f"Imagem carregada: {path}")
            self.status_label.setText(f"Imagem selecionada: {os.path.basename(path)}")

    def predict_on_image_file(self):
        if not self.areas or not self.model_path:
            QMessageBox.warning(self, "Erro", "Selecione o modelo e o arquivo JSON de áreas antes.")
            return
        self.load_neural_ai()
        if self.captured_img is None:
            QMessageBox.warning(self, "Erro", "Nenhuma imagem de arquivo carregada.")
            return
        img = self.captured_img.copy()
        nenhuma_conf_aceitavel = True
        output_lines = []
        for area in self.areas:
            x, y, w, h = area["x"], area["y"], area["width"], area["height"]
            crop = img[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, self.model_input_size)
            # Realce de contornos metálicos
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            edges = cv2.Canny(eq, 120, 250)
            crop_resized[edges > 0] = [0, 0, 255]  # borda vermelha
            temp_path = "temp_crop.png"
            cv2.imwrite(temp_path, crop_resized)
            if "threshold" in area:
                self.neural_ai.threshold = float(area["threshold"])
            else:
                self.neural_ai.threshold = 0.5
            # Checa se a predição retorna 'Nenhuma classe com confiança aceitável.'
            pred_result = self.neural_ai.predict_image(temp_path, info=True, classe_=[area.get("class", "")])
            if isinstance(pred_result, str) and "Nenhuma classe com confiança aceitável" in pred_result:
                output_lines.append(f"Área {area.get('class','')} (Ponto: {area.get('point','')}, Threshold: {area.get('threshold',0.5)}): Nenhuma classe com confiança aceitável.")
            else:
                nenhuma_conf_aceitavel = False
                output_lines.append(f"Área {area.get('class','')} (Ponto: {area.get('point','')}, Threshold: {area.get('threshold',0.5)}): {pred_result}")
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")
        if nenhuma_conf_aceitavel:
            output_lines.append("\nPeça passou: Nenhuma classe com confiança aceitável em todas as áreas.")
        else:
            output_lines.append("\nPeça não passou: Alguma área teve classe com confiança aceitável.")
        self.output_text.setPlainText("\n".join(output_lines))

    def predict_on_camera(self):
        if not self.areas or not self.model_path:
            QMessageBox.warning(self, "Erro", "Selecione o modelo e o arquivo JSON de áreas antes.")
            return
        self.load_neural_ai()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            QMessageBox.warning(self, "Erro", f"Não foi possível abrir a câmera de índice {self.camera_index}.")
            return
        QMessageBox.information(self, "Captura", "Pressione 's' para capturar a foto.")
        img = None
        while True:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, "Erro", "Não foi possível capturar imagem da câmera.")
                cap.release()
                return
            cv2.imshow("Captura de Foto", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                img = frame.copy()
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(qimg)
                pixmap = self.draw_areas_on_pixmap(pixmap)
                self.image_view.setPixmap(pixmap.scaled(self.image_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.image_view.setText("")
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyAllWindows()
        if img is None:
            QMessageBox.warning(self, "Erro", "Nenhuma imagem disponível para predição.")
            return
        self.captured_img = img.copy()
        encontrou_passa = False
        encontrou_nao_passa = False
        output_lines = []
        for area in self.areas:
            x, y, w, h = area["x"], area["y"], area["width"], area["height"]
            crop = img[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, self.model_input_size)
            temp_path = "temp_crop.png"
            cv2.imwrite(temp_path, crop_resized)
            if "threshold" in area:
                self.neural_ai.threshold = float(area["threshold"])
            else:
                self.neural_ai.threshold = 0.5
            if "classe_passa" in area:
                pred_passa = self.neural_ai.predict_image(temp_path, info=True, classe_=[area["classe_passa"]])
                if pred_passa:
                    encontrou_passa = True
                    output_lines.append(f"Área encontrada: Classe PASSA={area['classe_passa']}, Ponto={area.get('point','')}, Threshold={area.get('threshold',0.5)}")
            if "classe_nao_passa" in area:
                pred_nao_passa = self.neural_ai.predict_image(temp_path, info=True, classe_=[area["classe_nao_passa"]])
                if pred_nao_passa:
                    encontrou_nao_passa = True
                    output_lines.append(f"Área encontrada: Classe NÃO PASSA={area['classe_nao_passa']}, Ponto={area.get('point','')}, Threshold={area.get('threshold',0.5)}")
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")
        if encontrou_passa and not encontrou_nao_passa:
            output_lines.append("\nPeça passou: encontrou classe_passa e não encontrou classe_nao_passa.")
        else:
            output_lines.append("\nPeça não passou: encontrou classe_nao_passa ou não encontrou classe_passa.")
        self.output_text.setPlainText("\n".join(output_lines))

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Modelo", "", "Modelos Keras (*.keras)")
        if path:
            self.model_path = path
    def clear_image(self):
        self.image_view.clear()
        self.image_view.setText("Imagem não carregada")
        self.image_path = None
        self.captured_img = None
        self.status_label.setText("")
        self.result_label.setText("")
        self.output_text.setText("")

    def select_class_txt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Classes", "", "Arquivos TXT (*.txt)")
        if path:
            self.class_txt_path = path
            self.status_label.setText(f"Arquivo de classes selecionado: {os.path.basename(path)}")

    def select_precisao(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Precisão", "", "Arquivos NPY (*.npy)")
        if path:
            self.precisao_path = path
            self.status_label.setText(f"Precisão selecionada: {os.path.basename(path)}")

    def select_precisao_val(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Precisão Val", "", "Arquivos NPY (*.npy)")
        if path:
            self.precisao_val_path = path
            self.status_label.setText(f"Precisão Val selecionada: {os.path.basename(path)}")

    def select_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Áreas JSON", "", "Arquivos JSON (*.json)")
        if path:
            self.json_path = path
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.areas = json.load(f)
            self.status_label.setText(f"Áreas JSON selecionadas: {os.path.basename(path)}")

    def select_camera(self):
        idx, ok = QInputDialog.getInt(self, "Índice da Câmera", "Digite o índice da câmera:", 0, 0, 10, 1)
        if ok:
            self.camera_index = idx
            self.status_label.setText(f"Câmera selecionada: índice {idx}")

    def load_neural_ai(self):
        self.neural_ai = NeuralAI(threshold=0.5)
        if self.model_path:
            self.neural_ai.model, _ = self.neural_ai.load_model_by_name(self.model_path)
            # Detecta tamanho de entrada do modelo
            try:
                input_shape = self.neural_ai.model.input_shape
                print(f"[INFO] input_shape do modelo carregado: {input_shape}")
                if len(input_shape) == 4:
                    self.model_input_size = (input_shape[1], input_shape[2])
                else:
                    self.model_input_size = (40, 40)  # Tamanho padrão
            except Exception:
                self.model_input_size = (64, 64)
        # Carrega modelo e arquivos auxiliares
        if self.model_path:
            self.neural_ai.model, _ = self.neural_ai.load_model_by_name(self.model_path)
        if self.class_txt_path:
            try:
                img_list = np.loadtxt(self.class_txt_path, dtype=str)
                # Garante lista 1D mesmo para um único valor
                if isinstance(img_list, str):
                    img_list = [img_list]
                elif img_list is None or (hasattr(img_list, 'size') and img_list.size == 0):
                    QMessageBox.critical(self, "Erro", "Arquivo de classes está vazio ou mal formatado.")
                    img_list = []
                elif isinstance(img_list, np.ndarray):
                    img_list = img_list.flatten().tolist()
                if not img_list:
                    QMessageBox.critical(self, "Erro", "Arquivo de classes está vazio ou mal formatado.")
                else:
                    self.neural_ai.img_list = img_list
                    self.neural_ai.labelencoder.fit(self.neural_ai.img_list)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar arquivo de classes: {e}")
        if self.precisao_path:
            self.neural_ai.precisao = np.load(self.precisao_path)
        if self.precisao_val_path:
            self.neural_ai.precisao_val = np.load(self.precisao_val_path)

    def capture_and_predict(self):
        if not self.areas or not self.model_path:
            QMessageBox.warning(self, "Erro", "Selecione o modelo e o arquivo JSON de áreas antes.")
            return
        self.load_neural_ai()
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            QMessageBox.warning(self, "Erro", f"Não foi possível abrir a câmera de índice {self.camera_index}.")
            return
        QMessageBox.information(self, "Captura", "Pressione 's' para capturar a foto.")
        while True:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, "Erro", "Não foi possível capturar imagem da câmera.")
                cap.release()
                return
            cv2.imshow("Captura de Foto", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                img = frame.copy()
                break
            elif key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyAllWindows()

        results = []
        for area in self.areas:
            x, y, w, h = area["x"], area["y"], area["width"], area["height"]
            crop = img[y:y+h, x:x+w]
            crop_resized = cv2.resize(crop, self.model_input_size)
            temp_path = "temp_crop.png"
            cv2.imwrite(temp_path, crop_resized)
            # Aplica o threshold da área
            if "threshold" in area:
                self.neural_ai.threshold = float(area["threshold"])
            else:
                self.neural_ai.threshold = 0.5
            # Predição
            pred = self.neural_ai.predict_image(temp_path, info=True, classe_=[area["class"]])
            results.append(f'Área {area["class"]} (Ponto: {area.get("point","")}, Threshold: {area.get("threshold",0.5)}): veja terminal para detalhes')
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictFromCamera()
    window.show()
    sys.exit(app.exec_())