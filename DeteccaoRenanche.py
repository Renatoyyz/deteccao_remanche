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
        self.config_file = "config_predicao.json"
        self.init_ui()
        self.load_config()
        self.auto_load_files_from_config()
    def auto_load_files_from_config(self):
        # Carrega arquivos automaticamente se os caminhos estiverem definidos
        import os
        if self.model_path and os.path.isfile(self.model_path):
            self.neural_ai = NeuralAI(threshold=0.5)
            self.neural_ai.model, _ = self.neural_ai.load_model_by_name(self.model_path)
        if self.class_txt_path and os.path.isfile(self.class_txt_path):
            try:
                img_list = np.loadtxt(self.class_txt_path, dtype=str)
                if isinstance(img_list, str):
                    img_list = [img_list]
                elif img_list is None or (hasattr(img_list, 'size') and img_list.size == 0):
                    img_list = []
                elif isinstance(img_list, np.ndarray):
                    img_list = img_list.flatten().tolist()
                if img_list:
                    if not self.neural_ai:
                        self.neural_ai = NeuralAI(threshold=0.5)
                    self.neural_ai.img_list = img_list
                    self.neural_ai.labelencoder.fit(self.neural_ai.img_list)
            except Exception:
                pass
        if self.precisao_path and os.path.isfile(self.precisao_path):
            if not self.neural_ai:
                self.neural_ai = NeuralAI(threshold=0.5)
            self.neural_ai.precisao = np.load(self.precisao_path)
        if self.precisao_val_path and os.path.isfile(self.precisao_val_path):
            if not self.neural_ai:
                self.neural_ai = NeuralAI(threshold=0.5)
            self.neural_ai.precisao_val = np.load(self.precisao_val_path)
        if self.json_path and os.path.isfile(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.areas = json.load(f)
            except Exception:
                self.areas = []

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
        # Botão para salvar configurações
        self.save_config_btn = QPushButton("Salvar Configurações")
        self.save_config_btn.clicked.connect(self.save_config)
        layout.addWidget(self.save_config_btn)
        self.setLayout(layout)
    def save_config(self):
        config = {
            "model_path": self.model_path,
            "class_txt_path": self.class_txt_path,
            "precisao_path": self.precisao_path,
            "precisao_val_path": self.precisao_val_path,
            "json_path": self.json_path,
            "camera_index": self.camera_index
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Configuração", self.config_file, "JSON (*.json)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Sucesso", f"Configuração salva em:\n{file_path}")
        else:
            QMessageBox.warning(self, "Aviso", "Nenhum nome de arquivo fornecido.")

    def load_config(self):
        import os
        if os.path.isfile(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.model_path = config.get("model_path", "")
                self.class_txt_path = config.get("class_txt_path", "")
                self.precisao_path = config.get("precisao_path", "")
                self.precisao_val_path = config.get("precisao_val_path", "")
                self.json_path = config.get("json_path", "")
                self.camera_index = config.get("camera_index", 0)
                # Atualiza status na interface
                if self.model_path:
                    self.status_label.setText(f"Modelo carregado: {os.path.basename(self.model_path)}")
                if self.class_txt_path:
                    self.status_label.setText(f"Arquivo de classes selecionado: {os.path.basename(self.class_txt_path)}")
                if self.precisao_path:
                    self.status_label.setText(f"Precisão selecionada: {os.path.basename(self.precisao_path)}")
                if self.precisao_val_path:
                    self.status_label.setText(f"Precisão Val selecionada: {os.path.basename(self.precisao_val_path)}")
                if self.json_path:
                    self.status_label.setText(f"Áreas JSON selecionadas: {os.path.basename(self.json_path)}")
                self.camera_btn.setText(f"Selecionar Câmera (Atual: {self.camera_index})")
            except Exception as e:
                QMessageBox.warning(self, "Aviso", f"Falha ao carregar configuração: {e}")

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
        pontos_nao_passou = []
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
            # Verifica classe_nao_passa
            if "classe_nao_passa" in area:
                pred_nao_passa = self.neural_ai.predict_image(temp_path, info=True, classe_=[area["classe_nao_passa"]])
                if pred_nao_passa:
                    pontos_nao_passou.append(str(area.get('point','')))
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")
        if not pontos_nao_passou:
            self.output_text.setPlainText("Peça aprovada")
        else:
            texto = "\n".join([f"Ponto = {p} não passou" for p in pontos_nao_passou])
            self.output_text.setPlainText(f"{texto}\nPeça não passou")

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
        pontos_nao_passou = []
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
            # Verifica classe_nao_passa
            if "classe_nao_passa" in area:
                pred_nao_passa = self.neural_ai.predict_image(temp_path, info=True, classe_=[area["classe_nao_passa"]])
                if pred_nao_passa:
                    pontos_nao_passou.append(str(area.get('point','')))
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")
        if not pontos_nao_passou:
            self.output_text.setPlainText("Peça aprovada")
        else:
            texto = "\n".join([f"Ponto = {p} não passou" for p in pontos_nao_passou])
            self.output_text.setPlainText(f"{texto}\nPeça não passou")

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

        nenhuma_conf_aceitavel = True
        pontos_nao_passou = []
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
            pred_result = self.neural_ai.predict_image(temp_path, info=True, classe_=[area.get("class", "")])
            if not (isinstance(pred_result, str) and "Nenhuma classe com confiança aceitável" in pred_result):
                nenhuma_conf_aceitavel = False
                pontos_nao_passou.append(str(area.get('point','')))
        self.result_label.setText("Predição realizada. Veja terminal para detalhes.")
        if nenhuma_conf_aceitavel:
            self.output_text.setPlainText("Peça passou: Nenhuma classe com confiança aceitável em todas as áreas.")
        else:
            self.output_text.setPlainText("\n".join([f"Ponto = {p} não passou" for p in pontos_nao_passou]))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictFromCamera()
    window.show()
    sys.exit(app.exec_())