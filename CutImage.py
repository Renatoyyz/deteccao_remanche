import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QCheckBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt, QRect

class CropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.crop_rect = QRect()
        self.show_rect = False
        self.areas_rects = []  # lista de áreas para desenhar

    def set_crop_rect(self, rect):
        self.crop_rect = rect
        self.show_rect = not rect.isNull()
        self.update()

    def set_areas_rects(self, rects):
        self.areas_rects = rects
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # Desenha áreas já selecionadas
        painter.setPen(Qt.blue)
        for r in self.areas_rects:
            painter.drawRect(r)
        # Desenha área atual
        if self.show_rect:
            painter.setPen(Qt.red)
            painter.drawRect(self.crop_rect)

class ImageCropper(QWidget):
    def capture_image(self):
        try:
            import cv2
        except ImportError:
            QMessageBox.warning(self, "Erro", "OpenCV (cv2) não está instalado. Instale com 'pip install opencv-python'.")
            return
        cam_index, ok = QInputDialog.getInt(self, "Escolher Câmera", "Digite o índice da câmera (normalmente 0, 1...):", 0, 0, 10, 1)
        if not ok:
            return
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            QMessageBox.warning(self, "Erro", f"Não foi possível abrir a câmera de índice {cam_index}.")
            return
        QMessageBox.information(self, "Captura", "Pressione 's' para capturar a foto e fechar a janela.")
        temp_path = None
        while True:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, "Erro", "Não foi possível capturar imagem da câmera.")
                cap.release()
                return
            cv2.imshow("Captura de Foto", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                temp_path = "captured_image_temp.png"
                cv2.imwrite(temp_path, frame)
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyAllWindows()
        if temp_path:
            self.image_path = temp_path
            self.original_pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.save_button.setEnabled(True)
            self.image_label.setText("")
            self.crop_rect = QRect()
            self.image_label.set_crop_rect(QRect())
            self.update_areas_rects()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cropper de Imagem")
        self.setGeometry(100, 100, 900, 700)
        self.image_path = None
        self.original_pixmap = None
        self.cropping = False
        self.start_point = None
        self.end_point = None
        self.crop_rect = QRect()
        self.areas = []  # Armazena as áreas selecionadas
        self.selection_mode = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.image_label = CropLabel(self)
        self.image_label.setText("Nenhuma imagem carregada")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        layout.addWidget(self.image_label)

        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.capture_button = QPushButton("Tirar Foto")
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        self.selection_checkbox = QCheckBox("Modo Seleção de Áreas")
        self.selection_checkbox.stateChanged.connect(self.toggle_selection_mode)
        layout.addWidget(self.selection_checkbox)

        self.save_button = QPushButton("Salvar Área Selecionada")
        self.save_button.clicked.connect(self.save_cropped_image)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.save_json_button = QPushButton("Salvar Áreas em JSON")
        self.save_json_button.clicked.connect(self.save_areas_json)
        layout.addWidget(self.save_json_button)

        self.clear_button = QPushButton("Limpar Áreas Selecionadas")
        self.clear_button.clicked.connect(self.clear_areas)
        layout.addWidget(self.clear_button)

        self.setLayout(layout)
    def init_ui(self):
        layout = QVBoxLayout()
        self.image_label = CropLabel(self)
        self.image_label.setText("Nenhuma imagem carregada")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        layout.addWidget(self.image_label)

        self.load_button = QPushButton("Carregar Imagem")
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.capture_button = QPushButton("Tirar Foto")
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        self.selection_checkbox = QCheckBox("Modo Seleção de Áreas")
        self.selection_checkbox.stateChanged.connect(self.toggle_selection_mode)
        layout.addWidget(self.selection_checkbox)

        self.select_folder_button = QPushButton("Selecionar Pasta de Destino")
        self.select_folder_button.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_button)

        self.save_button = QPushButton("Salvar Área Selecionada")
        self.save_button.clicked.connect(self.save_cropped_image)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.save_json_button = QPushButton("Salvar Áreas em JSON")
        self.save_json_button.clicked.connect(self.save_areas_json)
        layout.addWidget(self.save_json_button)

        self.load_areas_json_btn = QPushButton("Carregar Áreas de JSON")
        self.load_areas_json_btn.clicked.connect(self.load_areas_from_json)
        layout.addWidget(self.load_areas_json_btn)

        self.save_all_areas_btn = QPushButton("Salvar Todas Áreas do JSON como Imagens")
        self.save_all_areas_btn.clicked.connect(self.save_all_areas_from_json)
        layout.addWidget(self.save_all_areas_btn)

        self.clear_button = QPushButton("Limpar Áreas Selecionadas")
        self.clear_button.clicked.connect(self.clear_areas)
        layout.addWidget(self.clear_button)

        self.setLayout(layout)

        self.selected_folder = None
        self.loaded_areas_json = None
    def load_areas_from_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Carregar Áreas de JSON", "areas.json", "JSON (*.json)")
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.loaded_areas_json = json.load(f)
                QMessageBox.information(self, "Sucesso", f"Áreas carregadas de:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Erro", f"Falha ao carregar JSON: {e}")
        else:
            self.loaded_areas_json = None

    def save_all_areas_from_json(self):
        if not self.loaded_areas_json:
            QMessageBox.warning(self, "Aviso", "Nenhum arquivo de áreas JSON carregado.")
            return
        if not self.original_pixmap:
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem carregada.")
            return
        folder = self.selected_folder
        if not folder:
            QMessageBox.warning(self, "Aviso", "Nenhuma pasta de destino definida. Selecione uma pasta antes de salvar.")
            return
        pixmap_size = self.original_pixmap.size()
        import os
        existing_files = [f for f in os.listdir(folder) if f.endswith('.png')]
        next_num = len(existing_files) + 1
        base_name = os.path.basename(folder)
        for idx, area in enumerate(self.loaded_areas_json):
            crop_rect = QRect(area["x"], area["y"], area["width"], area["height"])
            crop_rect = crop_rect.intersected(QRect(0, 0, pixmap_size.width(), pixmap_size.height()))
            cropped = self.original_pixmap.copy(crop_rect)
            # Realce de contornos metálicos
            from PyQt5.QtGui import QImage
            import numpy as np
            import cv2
            cropped_img = cropped.toImage()
            width = cropped_img.width()
            height = cropped_img.height()
            ptr = cropped_img.bits()
            ptr.setsize(cropped_img.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2GRAY)
            eq = cv2.equalizeHist(gray)
            blur = cv2.GaussianBlur(eq, (3, 3), 0)
            edges = cv2.Canny(blur, 120, 250)
            arr_rgb[edges > 0] = [0, 0, 255]
            out_img = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2RGB)
            out_qimg = QImage(out_img.data, width, height, 3*width, QImage.Format_RGB888)
            out_pixmap = QPixmap.fromImage(out_qimg)
            file_name = f"{base_name}_area_{next_num}.png"
            file_path = os.path.join(folder, file_name)
            out_pixmap.save(file_path)
            next_num += 1
        QMessageBox.information(self, "Sucesso", f"Todas as áreas do JSON foram salvas como imagens na pasta:\n{folder}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Escolher Pasta para Salvar", "")
        if folder:
            self.selected_folder = folder
            QMessageBox.information(self, "Pasta Selecionada", f"Pasta de destino definida:\n{folder}")
        else:
            self.selected_folder = None
            QMessageBox.warning(self, "Aviso", "Nenhuma pasta selecionada.")
    def update_areas_rects(self):
        # Atualiza a visualização das áreas já selecionadas
        rects = []
        label_size = self.image_label.size()
        pixmap_size = self.original_pixmap.size() if self.original_pixmap else None
        if not pixmap_size:
            self.image_label.set_areas_rects([])
            return
        scaled_pixmap = self.original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        display_size = scaled_pixmap.size()
        offset_x = (label_size.width() - display_size.width()) // 2
        offset_y = (label_size.height() - display_size.height()) // 2
        scale_x = display_size.width() / pixmap_size.width()
        scale_y = display_size.height() / pixmap_size.height()
        for area in self.areas:
            x = int(area["x"] * scale_x) + offset_x
            y = int(area["y"] * scale_y) + offset_y
            w = int(area["width"] * scale_x)
            h = int(area["height"] * scale_y)
            rects.append(QRect(x, y, w, h))
        self.image_label.set_areas_rects(rects)

    def toggle_selection_mode(self, state):
        self.selection_mode = state == 2

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Escolher Imagem", "",
                                                   "Imagens (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_name:
            self.image_path = file_name
            self.original_pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.save_button.setEnabled(True)
            self.image_label.setText("")
            self.crop_rect = QRect()
            self.image_label.set_crop_rect(QRect())
            self.update_areas_rects()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_pixmap:
            rel_pos = event.pos() - self.image_label.pos()
            if self.image_label.rect().contains(rel_pos):
                self.cropping = True
                self.start_point = rel_pos
                self.end_point = rel_pos
                self.crop_rect = QRect()
                self.image_label.set_crop_rect(QRect())

    def mouseMoveEvent(self, event):
        if self.cropping and self.original_pixmap:
            rel_pos = event.pos() - self.image_label.pos()
            if self.image_label.rect().contains(rel_pos):
                self.end_point = rel_pos
                self.crop_rect = QRect(self.start_point, self.end_point).normalized()
                self.image_label.set_crop_rect(self.crop_rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.cropping:
            rel_pos = event.pos() - self.image_label.pos()
            if self.image_label.rect().contains(rel_pos):
                self.cropping = False
                self.end_point = rel_pos
                self.crop_rect = QRect(self.start_point, self.end_point).normalized()
                self.image_label.set_crop_rect(self.crop_rect)
                # Se modo seleção estiver ativo, pede nome da classe e salva área
                if self.selection_mode and not self.crop_rect.isNull():
                    classe_passa, ok1 = QInputDialog.getText(self, "Classe que PASSA", "Digite o nome da classe que indica que a peça PASSA nesta área:")
                    if not ok1 or not classe_passa:
                        return
                    classe_nao_passa, ok2 = QInputDialog.getText(self, "Classe que NÃO PASSA", "Digite o nome da classe que indica que a peça NÃO PASSA nesta área:")
                    if not ok2 or not classe_nao_passa:
                        return
                    point_name, ok3 = QInputDialog.getText(self, "Nome do Ponto", "Digite o nome do ponto para esta área:")
                    if not ok3 or not point_name:
                        return
                    threshold, ok4 = QInputDialog.getDouble(self, "Threshold de Acurácia", "Digite o threshold para detecção (0.0 a 1.0):", 0.5, 0.0, 1.0, 2)
                    if not ok4:
                        return
                    label_size = self.image_label.size()
                    pixmap_size = self.original_pixmap.size()
                    scaled_pixmap = self.original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    display_size = scaled_pixmap.size()
                    offset_x = (label_size.width() - display_size.width()) // 2
                    offset_y = (label_size.height() - display_size.height()) // 2
                    crop_x = self.crop_rect.x() - offset_x
                    crop_y = self.crop_rect.y() - offset_y
                    crop_x = max(0, crop_x)
                    crop_y = max(0, crop_y)
                    scale_x = pixmap_size.width() / display_size.width()
                    scale_y = pixmap_size.height() / display_size.height()
                    area_coords = {
                        "classe_passa": classe_passa,
                        "classe_nao_passa": classe_nao_passa,
                        "point": point_name,
                        "threshold": float(threshold),
                        "x": int(crop_x * scale_x),
                        "y": int(crop_y * scale_y),
                        "width": int(self.crop_rect.width() * scale_x),
                        "height": int(self.crop_rect.height() * scale_y)
                    }
                    self.areas.append(area_coords)
                    self.update_areas_rects()
                    QMessageBox.information(self, "Área Adicionada", f"Classe PASSA: {classe_passa}\nClasse NÃO PASSA: {classe_nao_passa}\nPonto: {point_name}\nThreshold: {threshold}\nCoordenadas: {area_coords}")
    def clear_areas(self):
        self.areas = []
        self.update_areas_rects()

    def save_cropped_image(self):
        if self.original_pixmap and not self.crop_rect.isNull():
            label_size = self.image_label.size()
            pixmap_size = self.original_pixmap.size()
            scaled_pixmap = self.original_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            display_size = scaled_pixmap.size()
            offset_x = (label_size.width() - display_size.width()) // 2
            offset_y = (label_size.height() - display_size.height()) // 2
            crop_x = self.crop_rect.x() - offset_x
            crop_y = self.crop_rect.y() - offset_y
            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)
            scale_x = pixmap_size.width() / display_size.width()
            scale_y = pixmap_size.height() / display_size.height()
            crop = QRect(
                int(crop_x * scale_x),
                int(crop_y * scale_y),
                int(self.crop_rect.width() * scale_x),
                int(self.crop_rect.height() * scale_y)
            )
            crop = crop.intersected(QRect(0, 0, pixmap_size.width(), pixmap_size.height()))
            cropped = self.original_pixmap.copy(crop)

            # Realce de contornos metálicos
            from PyQt5.QtGui import QImage
            import numpy as np
            import cv2
            cropped_img = cropped.toImage()
            width = cropped_img.width()
            height = cropped_img.height()
            ptr = cropped_img.bits()
            ptr.setsize(cropped_img.byteCount())
            arr = np.array(ptr).reshape(height, width, 4)
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2GRAY)
            # Equalização para destacar metal
            eq = cv2.equalizeHist(gray)
            # Canny para contornos
            edges = cv2.Canny(eq, 120, 250)
            # Mescla bordas destacadas com imagem original
            arr_rgb[edges > 0] = [0, 0, 255]  # borda vermelha
            out_img = cv2.cvtColor(arr_rgb, cv2.COLOR_BGR2RGB)
            out_qimg = QImage(out_img.data, width, height, 3*width, QImage.Format_RGB888)
            out_pixmap = QPixmap.fromImage(out_qimg)

            # Usar pasta previamente selecionada
            folder = self.selected_folder
            if not folder:
                QMessageBox.warning(self, "Aviso", "Nenhuma pasta de destino definida. Selecione uma pasta antes de salvar.")
                return

            base_name = os.path.basename(folder)
            existing = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith(('.png', '.jpg', '.jpeg'))]
            next_num = len(existing) + 1
            file_name = f"{base_name}_{next_num}.png"
            file_path = os.path.join(folder, file_name)

            out_pixmap.save(file_path)
            QMessageBox.information(self, "Sucesso", f"Área salva em:\n{file_path}\n(Contornos metálicos destacados)")
        else:
            QMessageBox.warning(self, "Aviso", "Nenhuma área selecionada.")

    def save_areas_json(self):
        if not self.areas:
            QMessageBox.warning(self, "Aviso", "Nenhuma área selecionada para salvar.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Áreas em JSON", "areas.json", "JSON (*.json)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.areas, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "Sucesso", f"Áreas salvas em:\n{file_path}")
        else:
            QMessageBox.warning(self, "Aviso", "Nenhum nome de arquivo fornecido.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageCropper()
    window.show()
    sys.exit(app.exec_())