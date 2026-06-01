"""PySide6 UI layout for FourierLab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ImageView(QLabel):
    """Image display label with centered, aspect-ratio-preserving content."""

    def __init__(self, title: str, min_height: int = 240) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #f8fafc; border: 1px solid #d8dee8; "
            "border-radius: 8px; color: #4b5563; font-size: 14px; padding: 8px; }"
        )
        self.setScaledContents(False)


class SliderRow(QWidget):
    """A slider with a fixed-width numeric value label."""

    def __init__(self, minimum: int, maximum: int, value: int) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(max(1, (maximum - minimum) // 8))
        self.value_label = QLabel(str(value))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(42)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(lambda val: self.value_label.setText(str(val)))


class Ui_MainWindow:
    """Build all widgets used by MainWindow."""

    def setup_ui(self, window: QMainWindow) -> None:
        window.setWindowTitle("FourierLab —— 二维傅里叶图像实验平台")
        window.resize(1800, 1050)
        window.setMinimumSize(1280, 820)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(False)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #d8dee8; background: #ffffff; }"
            "QTabBar::tab { min-width: 190px; min-height: 34px; padding: 8px 18px; "
            "font-size: 15px; font-weight: 600; color: #374151; background: #eef2f7; "
            "border: 1px solid #d8dee8; border-bottom: none; margin-right: 4px; "
            "border-top-left-radius: 7px; border-top-right-radius: 7px; }"
            "QTabBar::tab:selected { background: #ffffff; color: #175cd3; }"
            "QWidget { background: #ffffff; color: #1f2937; }"
            "QGroupBox { font-weight: 600; border: 1px solid #d8dee8; border-radius: 8px; "
            "margin-top: 10px; padding: 14px 10px 10px 10px; background: #fbfcfe; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
            "QPushButton { min-height: 32px; padding: 6px 12px; border: 1px solid #cbd5e1; "
            "border-radius: 6px; background: #f8fafc; }"
            "QPushButton:hover { background: #eef6ff; border-color: #93c5fd; }"
            "QPushButton:pressed { background: #dbeafe; }"
            "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { min-height: 28px; "
            "border: 1px solid #cbd5e1; border-radius: 5px; padding: 2px 6px; background: #ffffff; }"
            "QTextEdit { border: 1px solid #d8dee8; border-radius: 8px; background: #fbfcfe; padding: 8px; }"
        )
        self.tabs.addTab(self._build_filter_tab(), "基础傅里叶实验")
        self.tabs.addTab(self._build_watermark_tab(), "频域数字水印")
        self.tabs.addTab(self._build_encrypt_tab(), "频域加密解密")
        window.setCentralWidget(self.tabs)

    def _build_watermark_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 3)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        controls.setMinimumWidth(310)
        controls.setMaximumWidth(380)
        group = QGroupBox("水印参数")
        form = QFormLayout(group)
        self.watermark_open_button = QPushButton("导入原图")
        self.watermark_text_input = QLineEdit("FourierLab2026")
        self.watermark_seed_spin = QSpinBox()
        self.watermark_seed_spin.setRange(0, 99999999)
        self.watermark_seed_spin.setValue(2026)
        self.watermark_delta_spin = QDoubleSpinBox()
        self.watermark_delta_spin.setRange(1.0, 200.0)
        self.watermark_delta_spin.setValue(50.0)
        self.watermark_r1_spin = QSpinBox()
        self.watermark_r1_spin.setRange(1, 512)
        self.watermark_r1_spin.setValue(32)
        self.watermark_r2_spin = QSpinBox()
        self.watermark_r2_spin.setRange(2, 512)
        self.watermark_r2_spin.setValue(96)
        self.watermark_embed_button = QPushButton("嵌入水印")
        self.watermark_extract_button = QPushButton("提取水印")
        self.watermark_selfcheck_button = QPushButton("水印自检")
        self.watermark_open_marked_button = QPushButton("导入含水印图像")
        self.watermark_save_image_button = QPushButton("保存含水印图像")
        self.watermark_save_params_button = QPushButton("保存水印参数 JSON")
        self.watermark_load_params_button = QPushButton("读取水印参数 JSON")
        for row in [
            (self.watermark_open_button,),
            ("水印文字", self.watermark_text_input),
            ("seed 密钥", self.watermark_seed_spin),
            ("嵌入强度 Δ", self.watermark_delta_spin),
            ("中频内半径 r1", self.watermark_r1_spin),
            ("中频外半径 r2", self.watermark_r2_spin),
            (self.watermark_embed_button,),
            (self.watermark_selfcheck_button,),
            (self.watermark_extract_button,),
            (self.watermark_open_marked_button,),
            (self.watermark_save_image_button,),
            (self.watermark_save_params_button,),
            (self.watermark_load_params_button,),
        ]:
            form.addRow(*row)
        controls_layout.addWidget(group)
        self.watermark_clear_log_button = QPushButton("清空日志")
        self.watermark_export_log_button = QPushButton("导出日志 TXT")
        controls_layout.addWidget(self.watermark_clear_log_button)
        controls_layout.addWidget(self.watermark_export_log_button)
        controls_layout.addStretch(1)
        top.addWidget(controls)

        views = QWidget()
        grid = QGridLayout(views)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.watermark_original_view = ImageView("原图")
        self.watermark_result_view = ImageView("含水印图像")
        self.watermark_spectrum_view = ImageView("原图频谱")
        self.watermark_mask_view = ImageView("水印嵌入位置图")
        self.watermark_diff_view = ImageView("差异图 abs(watermarked-original)")
        self.watermark_extract_view = QTextEdit()
        self.watermark_extract_view.setReadOnly(True)
        self.watermark_extract_view.setPlaceholderText("提取出的水印文字")
        grid.addWidget(self.watermark_original_view, 0, 0)
        grid.addWidget(self.watermark_result_view, 0, 1)
        grid.addWidget(self.watermark_spectrum_view, 0, 2)
        grid.addWidget(self.watermark_mask_view, 1, 0)
        grid.addWidget(self.watermark_diff_view, 1, 1)
        grid.addWidget(self.watermark_extract_view, 1, 2)
        top.addWidget(views)
        top.setSizes([340, 1260])
        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.watermark_log_text = QTextEdit()
        self.watermark_log_text.setReadOnly(True)
        self.watermark_log_text.setMinimumHeight(260)
        self.watermark_log_text.setMaximumHeight(380)
        self.watermark_log_text.setStyleSheet("QTextEdit { font-family: 'Microsoft YaHei'; font-size: 14px; line-height: 1.35; }")
        self.watermark_info_text = QTextEdit()
        self.watermark_info_text.setReadOnly(True)
        self.watermark_info_text.setMinimumHeight(260)
        self.watermark_info_text.setMaximumHeight(380)
        self.watermark_info_text.setStyleSheet("QTextEdit { font-family: 'Microsoft YaHei'; font-size: 14px; line-height: 1.35; }")
        self.watermark_info_text.setText("数字水印是将标识信息嵌入图像的一种方法。低频区域影响图像主体，修改后容易被察觉；高频区域容易被压缩和去噪破坏；中频区域兼顾隐蔽性与稳定性，因此本实验选择在傅里叶频谱的中频环带嵌入文字水印。")
        bottom.addWidget(self.watermark_log_text)
        bottom.addWidget(self.watermark_info_text)
        bottom.setSizes([920, 680])
        layout.addWidget(bottom, 1)
        return tab

    def _build_filter_tab(self) -> QWidget:
        tab = QWidget()
        root_layout = QVBoxLayout(tab)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(main_splitter, 3)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setWidget(self._build_filter_control_panel())
        control_scroll.setMinimumWidth(290)
        control_scroll.setMaximumWidth(350)
        main_splitter.addWidget(control_scroll)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(10, 10, 10, 10)
        image_layout.setSpacing(12)
        self.original_view = ImageView("原始图像", min_height=260)
        self.result_view = ImageView("重建图像", min_height=260)
        image_layout.addWidget(self.original_view, 1)
        image_layout.addWidget(self.result_view, 1)
        main_splitter.addWidget(image_panel)

        freq_panel = QWidget()
        freq_layout = QGridLayout(freq_panel)
        freq_layout.setContentsMargins(10, 10, 10, 10)
        freq_layout.setHorizontalSpacing(12)
        freq_layout.setVerticalSpacing(12)
        self.spectrum_view = ImageView("原始幅度谱", min_height=240)
        self.mask_view = ImageView("滤波器模板", min_height=240)
        self.filtered_spectrum_view = ImageView("滤波后频谱", min_height=240)
        self.phase_view = ImageView("相位谱", min_height=240)
        freq_layout.addWidget(self.spectrum_view, 0, 0)
        freq_layout.addWidget(self.mask_view, 0, 1)
        freq_layout.addWidget(self.filtered_spectrum_view, 1, 0)
        freq_layout.addWidget(self.phase_view, 1, 1)
        main_splitter.addWidget(freq_panel)
        main_splitter.setSizes([320, 620, 760])

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumHeight(260)
        self.stats_text.setMaximumHeight(380)
        self.stats_text.setStyleSheet("QTextEdit { font-family: Consolas, 'Microsoft YaHei'; font-size: 13px; line-height: 1.35; }")
        self.principle_text = QTextEdit()
        self.principle_text.setReadOnly(True)
        self.principle_text.setMinimumHeight(260)
        self.principle_text.setMaximumHeight(380)
        self.principle_text.setStyleSheet("QTextEdit { font-family: 'Microsoft YaHei'; font-size: 14px; line-height: 1.35; }")
        self.formula_text = QTextEdit()
        self.formula_text.setReadOnly(True)
        self.formula_text.setMinimumHeight(260)
        self.formula_text.setMaximumHeight(380)
        self.formula_text.setStyleSheet("QTextEdit { font-family: Consolas, 'Microsoft YaHei'; font-size: 13px; line-height: 1.35; }")
        bottom.addWidget(self.stats_text)
        bottom.addWidget(self.principle_text)
        bottom.addWidget(self.formula_text)
        bottom.setSizes([500, 760, 520])
        root_layout.addWidget(bottom, 1)
        return tab

    def _build_filter_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        file_group = QGroupBox("文件")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        self.open_button = QPushButton("打开图片")
        self.color_check = QCheckBox("彩色 RGB 处理")
        self.save_button = QPushButton("保存重建结果")
        self.reset_button = QPushButton("重置")
        file_layout.addWidget(self.open_button)
        file_layout.addWidget(self.color_check)
        file_layout.addWidget(self.save_button)
        file_layout.addWidget(self.reset_button)
        layout.addWidget(file_group)

        filter_group = QGroupBox("频域滤波")
        filter_layout = QFormLayout(filter_group)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["不滤波", "理想低通", "理想高通", "理想带通", "理想带阻", "高斯低通", "高斯高通", "巴特沃斯低通", "巴特沃斯高通"])
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["80%", "90%", "95%", "99%"])
        filter_layout.addRow("滤波器", self.filter_combo)
        filter_layout.addRow("自动低通能量", self.energy_combo)
        layout.addWidget(filter_group)

        param_group = QGroupBox("参数")
        param_layout = QFormLayout(param_group)
        self.radius_row = SliderRow(1, 512, 60)
        self.r1_row = SliderRow(1, 512, 30)
        self.r2_row = SliderRow(1, 512, 120)
        self.sigma_row = SliderRow(1, 512, 60)
        self.radius_slider = self.radius_row.slider
        self.r1_slider = self.r1_row.slider
        self.r2_slider = self.r2_row.slider
        self.sigma_slider = self.sigma_row.slider
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 8)
        self.order_spin.setValue(2)
        self.live_check = QCheckBox("参数改变后自动应用")
        param_layout.addRow("截止半径 r", self.radius_row)
        param_layout.addRow("带通/带阻内半径 r1", self.r1_row)
        param_layout.addRow("带通/带阻外半径 r2", self.r2_row)
        param_layout.addRow("高斯 sigma", self.sigma_row)
        param_layout.addRow("巴特沃斯阶数 n", self.order_spin)
        param_layout.addRow("", self.live_check)
        layout.addWidget(param_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(8)
        self.fft_button = QPushButton("重新计算 FFT")
        self.apply_button = QPushButton("应用滤波并重建")
        self.demo_button = QPushButton("一键演示完整流程")
        self.auto_button = QPushButton("自动低通调参")
        self.ifft_button = QPushButton("仅逆变换重建")
        self.spectrum3d_button = QPushButton("查看 3D 频谱")
        self.export_grid_button = QPushButton("一键导出实验图")
        for button in [self.fft_button, self.apply_button, self.demo_button, self.auto_button, self.ifft_button, self.spectrum3d_button, self.export_grid_button]:
            action_layout.addWidget(button)
        layout.addWidget(action_group)
        layout.addStretch(1)
        return panel

    def _build_noise_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(360)
        file_group = QGroupBox("周期噪声实验")
        file_layout = QFormLayout(file_group)
        self.noise_open_button = QPushButton("导入图片")
        self.noise_direction_combo = QComboBox()
        self.noise_direction_combo.addItems(["横向", "纵向", "斜向"])
        self.noise_freq_spin = QSpinBox()
        self.noise_freq_spin.setRange(1, 120)
        self.noise_freq_spin.setValue(18)
        self.noise_amp_spin = QDoubleSpinBox()
        self.noise_amp_spin.setRange(1.0, 120.0)
        self.noise_amp_spin.setValue(35.0)
        self.noise_dx_spin = QSpinBox()
        self.noise_dx_spin.setRange(-512, 512)
        self.noise_dx_spin.setValue(0)
        self.noise_dy_spin = QSpinBox()
        self.noise_dy_spin.setRange(-512, 512)
        self.noise_dy_spin.setValue(18)
        self.noise_notch_radius_spin = QSpinBox()
        self.noise_notch_radius_spin.setRange(1, 80)
        self.noise_notch_radius_spin.setValue(6)
        self.noise_add_button = QPushButton("添加周期噪声")
        self.noise_auto_button = QPushButton("自动估计陷波位置")
        self.noise_denoise_button = QPushButton("陷波去噪")
        file_layout.addRow(self.noise_open_button)
        file_layout.addRow("噪声方向", self.noise_direction_combo)
        file_layout.addRow("噪声频率 freq", self.noise_freq_spin)
        file_layout.addRow("噪声强度 amplitude", self.noise_amp_spin)
        file_layout.addRow("陷波 dx", self.noise_dx_spin)
        file_layout.addRow("陷波 dy", self.noise_dy_spin)
        file_layout.addRow("陷波半径", self.noise_notch_radius_spin)
        file_layout.addRow(self.noise_add_button)
        file_layout.addRow(self.noise_auto_button)
        file_layout.addRow(self.noise_denoise_button)
        controls_layout.addWidget(file_group)
        controls_layout.addStretch(1)
        top.addWidget(controls)
        views = QWidget()
        grid = QGridLayout(views)
        self.noise_original_view = ImageView("原图")
        self.noise_noisy_view = ImageView("加噪图")
        self.noise_spectrum_view = ImageView("加噪图频谱")
        self.noise_mask_view = ImageView("陷波滤波模板")
        self.noise_result_view = ImageView("去噪重建图")
        grid.addWidget(self.noise_original_view, 0, 0)
        grid.addWidget(self.noise_noisy_view, 0, 1)
        grid.addWidget(self.noise_spectrum_view, 0, 2)
        grid.addWidget(self.noise_mask_view, 1, 0)
        grid.addWidget(self.noise_result_view, 1, 1)
        top.addWidget(views)
        top.setSizes([320, 1150])
        self.noise_info_text = QTextEdit()
        self.noise_info_text.setReadOnly(True)
        self.noise_info_text.setMaximumHeight(130)
        self.noise_info_text.setText("周期噪声在空间域表现为规则条纹，在频域中通常表现为远离中心的对称亮点。通过陷波滤波去除这些异常频率峰值，可以减弱条纹噪声。")
        layout.addWidget(self.noise_info_text)
        return tab

    def _build_compression_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(360)
        group = QGroupBox("压缩参数")
        form = QFormLayout(group)
        self.compress_open_button = QPushButton("导入图片")
        self.compress_ratio_combo = QComboBox()
        self.compress_ratio_combo.addItems(["80%", "90%", "95%", "99%"])
        self.compress_run_button = QPushButton("按能量保留率压缩重建")
        form.addRow(self.compress_open_button)
        form.addRow("能量保留率", self.compress_ratio_combo)
        form.addRow(self.compress_run_button)
        controls_layout.addWidget(group)
        controls_layout.addStretch(1)
        top.addWidget(controls)
        views = QWidget()
        grid = QGridLayout(views)
        self.compress_original_view = ImageView("原图", 260)
        self.compress_spectrum_view = ImageView("频谱图", 260)
        self.compress_mask_view = ImageView("保留频率模板", 260)
        self.compress_result_view = ImageView("压缩重建图", 260)
        grid.addWidget(self.compress_original_view, 0, 0)
        grid.addWidget(self.compress_spectrum_view, 0, 1)
        grid.addWidget(self.compress_mask_view, 1, 0)
        grid.addWidget(self.compress_result_view, 1, 1)
        top.addWidget(views)
        top.setSizes([320, 1150])
        self.compress_info_text = QTextEdit()
        self.compress_info_text.setReadOnly(True)
        self.compress_info_text.setMaximumHeight(130)
        self.compress_info_text.setText("自然图像的主要能量通常集中在低频区域。保留少量主要频率成分即可恢复图像整体内容，这体现了频域压缩的基本思想。")
        layout.addWidget(self.compress_info_text)
        return tab

    def _build_phase_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(360)
        group = QGroupBox("幅度/相位交换")
        box = QVBoxLayout(group)
        self.phase_open_a_button = QPushButton("导入图像 A")
        self.phase_open_b_button = QPushButton("导入图像 B")
        self.phase_run_button = QPushButton("执行幅度谱/相位谱交换")
        box.addWidget(self.phase_open_a_button)
        box.addWidget(self.phase_open_b_button)
        box.addWidget(self.phase_run_button)
        controls_layout.addWidget(group)
        controls_layout.addStretch(1)
        top.addWidget(controls)
        views = QWidget()
        grid = QGridLayout(views)
        self.phase_a_view = ImageView("图像 A")
        self.phase_b_view = ImageView("图像 B")
        self.phase_ab_view = ImageView("A幅度 + B相位")
        self.phase_ba_view = ImageView("B幅度 + A相位")
        self.phase_a_mag_view = ImageView("A幅度谱")
        self.phase_a_phase_view = ImageView("A相位谱")
        self.phase_b_mag_view = ImageView("B幅度谱")
        self.phase_b_phase_view = ImageView("B相位谱")
        for i, view in enumerate([self.phase_a_view, self.phase_b_view, self.phase_ab_view, self.phase_ba_view, self.phase_a_mag_view, self.phase_a_phase_view, self.phase_b_mag_view, self.phase_b_phase_view]):
            grid.addWidget(view, i // 4, i % 4)
        top.addWidget(views)
        top.setSizes([320, 1150])
        self.phase_info_text = QTextEdit()
        self.phase_info_text.setReadOnly(True)
        self.phase_info_text.setMaximumHeight(130)
        self.phase_info_text.setText("幅度谱表示各频率成分强弱，相位谱决定频率成分在空间中的排列关系。实验中，重建图像通常更接近提供相位谱的图像，说明相位对图像结构具有重要作用。")
        layout.addWidget(self.phase_info_text)
        return tab

    def _build_encrypt_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 3)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        controls.setMinimumWidth(290)
        controls.setMaximumWidth(350)
        group = QGroupBox("图像与密钥")
        form = QFormLayout(group)
        self.encrypt_open_button = QPushButton("导入图片")
        self.encrypt_seed_spin = QSpinBox()
        self.encrypt_seed_spin.setRange(0, 99999999)
        self.encrypt_seed_spin.setValue(1236)
        self.encrypt_strength_spin = QDoubleSpinBox()
        self.encrypt_strength_spin.setRange(0.0, 3.0)
        self.encrypt_strength_spin.setSingleStep(0.1)
        self.encrypt_strength_spin.setValue(1.0)
        self.encrypt_run_button = QPushButton("加密")
        self.decrypt_run_button = QPushButton("解密")
        self.encrypt_save_button = QPushButton("保存加密预览图 PNG")
        self.encrypt_save_package_button = QPushButton("保存可解密加密包 NPZ")
        self.encrypt_load_package_button = QPushButton("导入加密包并解密")
        self.decrypt_save_button = QPushButton("保存解密图")
        for row in [(self.encrypt_open_button,), ("seed 密钥", self.encrypt_seed_spin), ("相位扰动强度", self.encrypt_strength_spin), (self.encrypt_run_button,), (self.decrypt_run_button,), (self.encrypt_save_button,), (self.encrypt_save_package_button,), (self.encrypt_load_package_button,), (self.decrypt_save_button,)]:
            form.addRow(*row)
        controls_layout.addWidget(group)
        controls_layout.addStretch(1)
        top.addWidget(controls)
        views = QWidget()
        grid = QGridLayout(views)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.encrypt_original_view = ImageView("原图", 240)
        self.encrypt_magnitude_view = ImageView("幅度谱", 240)
        self.encrypt_phase_view = ImageView("相位谱", 240)
        self.encrypt_result_view = ImageView("加密图", 240)
        self.decrypt_result_view = ImageView("解密图", 240)
        grid.addWidget(self.encrypt_original_view, 0, 0)
        grid.addWidget(self.encrypt_magnitude_view, 0, 1)
        grid.addWidget(self.encrypt_phase_view, 0, 2)
        grid.addWidget(self.encrypt_result_view, 1, 0)
        grid.addWidget(self.decrypt_result_view, 1, 1)
        top.addWidget(views)
        top.setSizes([320, 1280])
        self.encrypt_explain_text = QTextEdit()
        self.encrypt_explain_text.setReadOnly(True)
        self.encrypt_explain_text.setMinimumHeight(170)
        self.encrypt_explain_text.setMaximumHeight(260)
        self.encrypt_explain_text.setStyleSheet("QTextEdit { font-family: 'Microsoft YaHei'; font-size: 14px; line-height: 1.35; }")
        self.encrypt_explain_text.setText("这是教学型相位扰动实验，不是严格密码学加密。普通图像文件无法保存完整复数频谱和相位扰动状态，因此保存成 png/jpg 后再重新打开，不能保证完整解密。")
        self.encrypt_metrics_text = QTextEdit()
        self.encrypt_metrics_text.setReadOnly(True)
        self.encrypt_metrics_text.setMinimumHeight(180)
        self.encrypt_metrics_text.setMaximumHeight(280)
        self.encrypt_metrics_text.setStyleSheet("QTextEdit { font-family: Consolas, 'Microsoft YaHei'; font-size: 13px; line-height: 1.35; }")
        layout.addWidget(self.encrypt_explain_text, 1)
        layout.addWidget(self.encrypt_metrics_text, 1)
        return tab

    def _build_hybrid_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(top, 1)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(360)
        file_group = QGroupBox("图像导入")
        file_layout = QVBoxLayout(file_group)
        self.hybrid_open_a_button = QPushButton("导入图像 A（低频轮廓）")
        self.hybrid_open_b_button = QPushButton("导入图像 B / 文字层（高频细节）")
        self.hybrid_save_button = QPushButton("保存融合结果")
        for b in [self.hybrid_open_a_button, self.hybrid_open_b_button, self.hybrid_save_button]:
            file_layout.addWidget(b)
        controls_layout.addWidget(file_group)

        text_group = QGroupBox("文字高频层（扩展）")
        text_layout = QFormLayout(text_group)
        self.hybrid_text_input = QLineEdit("FourierLab")
        self.hybrid_font_size = QSpinBox()
        self.hybrid_font_size.setRange(12, 220)
        self.hybrid_font_size.setValue(72)
        self.hybrid_bold_check = QCheckBox("加粗")
        self.hybrid_bold_check.setChecked(True)
        self.hybrid_text_color = QComboBox()
        self.hybrid_text_color.addItems(["白色", "黑色", "红色", "蓝色", "绿色"])
        self.hybrid_bg_color = QComboBox()
        self.hybrid_bg_color.addItems(["黑色", "白色", "红色", "蓝色", "绿色"])
        self.hybrid_generate_text_button = QPushButton("生成文字层作为图像 B")
        for row in [("文字内容", self.hybrid_text_input), ("字体大小", self.hybrid_font_size), ("文字粗细", self.hybrid_bold_check), ("文字颜色", self.hybrid_text_color), ("背景颜色", self.hybrid_bg_color), (self.hybrid_generate_text_button,)]:
            text_layout.addRow(*row)
        controls_layout.addWidget(text_group)

        param_group = QGroupBox("融合参数")
        param_layout = QFormLayout(param_group)
        self.hybrid_r_low = QSpinBox()
        self.hybrid_r_low.setRange(1, 512)
        self.hybrid_r_low.setValue(45)
        self.hybrid_r_high = QSpinBox()
        self.hybrid_r_high.setRange(1, 512)
        self.hybrid_r_high.setValue(30)
        self.hybrid_alpha = QDoubleSpinBox()
        self.hybrid_alpha.setRange(0.0, 3.0)
        self.hybrid_alpha.setSingleStep(0.1)
        self.hybrid_alpha.setValue(1.0)
        self.hybrid_beta = QDoubleSpinBox()
        self.hybrid_beta.setRange(0.0, 3.0)
        self.hybrid_beta.setSingleStep(0.1)
        self.hybrid_beta.setValue(1.0)
        self.hybrid_run_button = QPushButton("生成 Hybrid Image")
        for row in [("A 低通半径 r_low", self.hybrid_r_low), ("B 高通半径 r_high", self.hybrid_r_high), ("低频权重 alpha", self.hybrid_alpha), ("高频权重 beta", self.hybrid_beta), (self.hybrid_run_button,)]:
            param_layout.addRow(*row)
        controls_layout.addWidget(param_group)
        controls_layout.addStretch(1)
        top.addWidget(controls)

        views = QWidget()
        grid = QGridLayout(views)
        self.hybrid_a_view = ImageView("图像 A")
        self.hybrid_b_view = ImageView("图像 B / 文字层")
        self.hybrid_low_view = ImageView("A 的低频结果")
        self.hybrid_high_view = ImageView("B 的高频结果")
        self.hybrid_result_view = ImageView("融合结果")
        self.hybrid_preview_view = ImageView("缩小预览图")
        for i, view in enumerate([self.hybrid_a_view, self.hybrid_b_view, self.hybrid_low_view, self.hybrid_high_view, self.hybrid_result_view, self.hybrid_preview_view]):
            grid.addWidget(view, i // 2, i % 2)
        top.addWidget(views)
        top.setSizes([320, 1150])
        self.hybrid_explain_text = QTextEdit()
        self.hybrid_explain_text.setReadOnly(True)
        self.hybrid_explain_text.setMaximumHeight(125)
        self.hybrid_explain_text.setText("Hybrid Image 对素材要求较高，通常适合结构相似、位置对齐的两张图片，例如两张人脸或两个轮廓相似物体。普通照片与文字层融合时，更接近频域水印或高频叠加效果，因此本功能作为扩展实验保留。")
        layout.addWidget(self.hybrid_explain_text)
        return tab
