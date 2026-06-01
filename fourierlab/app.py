"""Main window behavior for FourierLab."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox

from . import filters, hybrid, image_io, metrics, watermark
from .fourier_core import apply_filter, fft2_shift, ifft2_reconstruct, magnitude_spectrum, phase_spectrum
from .logger import ExperimentLogger
from .ui_main import Ui_MainWindow
from .visualization import Spectrum3DDialog


FILTER_NONE = "不滤波"
FILTER_IDEAL_LOW = "理想低通"
FILTER_IDEAL_HIGH = "理想高通"
FILTER_IDEAL_BAND = "理想带通"
FILTER_IDEAL_REJECT = "理想带阻"
FILTER_GAUSSIAN_LOW = "高斯低通"
FILTER_GAUSSIAN_HIGH = "高斯高通"
FILTER_BUTTER_LOW = "巴特沃斯低通"
FILTER_BUTTER_HIGH = "巴特沃斯高通"

FILTER_ENGLISH_NAMES = {
    FILTER_NONE: "None",
    FILTER_IDEAL_LOW: "Ideal Low Pass",
    FILTER_IDEAL_HIGH: "Ideal High Pass",
    FILTER_IDEAL_BAND: "Ideal Band Pass",
    FILTER_IDEAL_REJECT: "Ideal Band Reject",
    FILTER_GAUSSIAN_LOW: "Gaussian Low Pass",
    FILTER_GAUSSIAN_HIGH: "Gaussian High Pass",
    FILTER_BUTTER_LOW: "Butterworth Low Pass",
    FILTER_BUTTER_HIGH: "Butterworth High Pass",
}


class MainWindow(QMainWindow):
    """Main application window for learning-oriented Fourier experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setup_ui(self)

        self.image: np.ndarray | None = None
        self.fft: np.ndarray | None = None
        self.filtered_fft: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.current_path: Path | None = None
        self.spectrum3d_dialog: Spectrum3DDialog | None = None
        self.base_logger: ExperimentLogger | None = None

        self.watermark_image: np.ndarray | None = None
        self.watermarked_image: np.ndarray | None = None
        self.watermarked_fft: np.ndarray | None = None
        self.watermark_mask: np.ndarray | None = None
        self.watermark_bit_length: int = 0
        self.watermark_available_points: int = 0
        self.watermark_repeat: int = 9
        self.watermark_metadata_text: str | None = None
        self.watermark_params_loaded: bool = False
        self.watermark_param_shape: tuple[int, int] | None = None
        self.watermark_logger: ExperimentLogger | None = None

        self.hybrid_a: np.ndarray | None = None
        self.hybrid_b: np.ndarray | None = None
        self.hybrid_low: np.ndarray | None = None
        self.hybrid_high: np.ndarray | None = None
        self.hybrid_result: np.ndarray | None = None

        self.noise_original: np.ndarray | None = None
        self.noise_noisy: np.ndarray | None = None
        self.noise_result: np.ndarray | None = None
        self.noise_fft: np.ndarray | None = None
        self.noise_mask: np.ndarray | None = None

        self.compress_image: np.ndarray | None = None
        self.compress_fft: np.ndarray | None = None
        self.compress_mask: np.ndarray | None = None
        self.compress_result: np.ndarray | None = None

        self.phase_a: np.ndarray | None = None
        self.phase_b: np.ndarray | None = None
        self.phase_ab: np.ndarray | None = None
        self.phase_ba: np.ndarray | None = None
        self.phase_a_mag_display: np.ndarray | None = None
        self.phase_a_phase_display: np.ndarray | None = None
        self.phase_b_mag_display: np.ndarray | None = None
        self.phase_b_phase_display: np.ndarray | None = None

        self.encrypt_image_data: np.ndarray | None = None
        self.encrypted_image: np.ndarray | None = None
        self.decrypted_image: np.ndarray | None = None
        self.encrypted_fft: np.ndarray | None = None
        self.encrypt_magnitude_raw: np.ndarray | None = None
        self.encrypt_magnitude_display: np.ndarray | None = None
        self.encrypt_phase_display: np.ndarray | None = None
        self.encrypt_noise: np.ndarray | None = None
        self.decrypt_from_package: bool = False

        self._connect_signals()
        self.base_logger = ExperimentLogger(self.ui.principle_text)
        self.watermark_logger = ExperimentLogger(self.ui.watermark_log_text)
        self._update_parameter_visibility()
        self._update_stats()
        self._set_principle("请先导入图片。软件会自动完成二维 FFT，并显示原始图像、幅度谱、相位谱和滤波模板。")
        self._set_formula_text()
        self._update_encrypt_metrics("请先导入图片并执行加密。")

    def _connect_signals(self) -> None:
        ui = self.ui
        ui.open_button.clicked.connect(self.open_image)
        ui.save_button.clicked.connect(self.save_result)
        ui.reset_button.clicked.connect(self.reset)
        ui.fft_button.clicked.connect(self.compute_fft)
        ui.apply_button.clicked.connect(self.apply_filter_and_reconstruct)
        ui.demo_button.clicked.connect(self.run_demo_flow)
        ui.auto_button.clicked.connect(self.auto_lowpass)
        ui.ifft_button.clicked.connect(self.reconstruct)
        ui.spectrum3d_button.clicked.connect(self.show_3d_spectrum)
        ui.export_grid_button.clicked.connect(self.export_grid_image)

        for widget in [ui.radius_slider, ui.r1_slider, ui.r2_slider, ui.sigma_slider, ui.order_spin]:
            widget.valueChanged.connect(self._maybe_live_apply)
        ui.filter_combo.currentTextChanged.connect(self._on_filter_changed)

        ui.encrypt_open_button.clicked.connect(self.open_encrypt_image)
        ui.encrypt_run_button.clicked.connect(self.run_encryption)
        ui.decrypt_run_button.clicked.connect(self.run_decryption)
        ui.encrypt_save_button.clicked.connect(self.save_encrypted_image)
        ui.encrypt_save_package_button.clicked.connect(self.save_encryption_package)
        ui.encrypt_load_package_button.clicked.connect(self.load_package_and_decrypt)
        ui.decrypt_save_button.clicked.connect(self.save_decrypted_image)

        ui.watermark_open_button.clicked.connect(self.open_watermark_image)
        ui.watermark_embed_button.clicked.connect(self.embed_watermark)
        ui.watermark_selfcheck_button.clicked.connect(self.watermark_selfcheck)
        ui.watermark_extract_button.clicked.connect(self.extract_watermark)
        ui.watermark_open_marked_button.clicked.connect(self.open_watermarked_image)
        ui.watermark_save_image_button.clicked.connect(self.save_watermarked_image)
        ui.watermark_save_params_button.clicked.connect(self.save_watermark_params)
        ui.watermark_load_params_button.clicked.connect(self.load_watermark_params)
        ui.watermark_clear_log_button.clicked.connect(lambda: self.watermark_logger.clear())
        ui.watermark_export_log_button.clicked.connect(lambda: self.export_log(self.watermark_logger))

    # Tab 1: frequency filtering
    def open_image(self) -> None:
        path = self._choose_open_image("打开图片")
        if not path:
            return
        try:
            self.image = image_io.load_image(path, grayscale=not self.ui.color_check.isChecked(), max_side=1024)
            self.current_path = Path(path)
            self.result = None
            self._set_default_radius_for_image(self.image)
            self._set_label_image(self.ui.original_view, self.image)
            self.compute_fft()
            self._set_principle("步骤1：图像输入。数字图像可以看作二维离散信号 f(x,y)，其中 x、y 表示像素位置，f(x,y) 表示该位置的灰度值或 RGB 值。后续处理将图像从空间域转换到频率域。")
        except Exception as exc:
            self._error(f"图片打开失败：{exc}")

    def save_result(self) -> None:
        self._save_array(self.result, "保存重建结果", "fourier_result.png", "请先生成重建结果。")

    # Tab 2: frequency-domain digital watermarking
    def open_watermark_image(self) -> None:
        path = self._choose_open_image("导入水印实验原图")
        if not path:
            return
        try:
            self.watermark_image = image_io.load_image(path, grayscale=True, max_side=1024)
            self.watermarked_image = None
            self.watermark_mask = None
            h, w = self.watermark_image.shape[:2]
            r1 = max(2, min(w, h) // 10)
            r2 = max(r1 + 1, min(w, h) // 3)
            self.ui.watermark_r1_spin.setValue(r1)
            self.ui.watermark_r2_spin.setValue(r2)
            fft = fft2_shift(self.watermark_image)
            self._set_label_image(self.ui.watermark_original_view, self.watermark_image)
            self._set_label_image(self.ui.watermark_spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(fft)))
            self.watermark_logger.append(
                "导入原图",
                f"图像尺寸：{w} × {h}。本实验将文字编码为 bit 流，并嵌入傅里叶频谱的中频环带。默认 r1={r1}, r2={r2}。",
            )
        except Exception as exc:
            self._error(f"水印原图导入失败：{exc}")

    def embed_watermark(self) -> None:
        if self.watermark_image is None:
            self._info("请先导入原图。")
            return
        text = self.ui.watermark_text_input.text()
        if not text:
            self._info("请输入水印文字。")
            return
        try:
            result = watermark.embed_text_watermark(
                self.watermark_image,
                text,
                self.ui.watermark_seed_spin.value(),
                self.ui.watermark_delta_spin.value(),
                self.ui.watermark_r1_spin.value(),
                self.ui.watermark_r2_spin.value(),
                repeat=self.watermark_repeat,
            )
            self.watermarked_image = result["watermarked"]
            self.watermarked_fft = result["fft"]
            self.watermark_mask = result["mask"]
            self.watermark_bit_length = int(result["bit_length"])
            self.watermark_repeat = int(result["repeat"])
            self.watermark_available_points = int(result["available_points"])
            diff = np.abs(self.watermarked_image.astype(float) - self.watermark_image.astype(float))
            error = metrics.mse(self.watermark_image, self.watermarked_image)
            quality = metrics.psnr(self.watermark_image, self.watermarked_image)
            self._set_label_image(self.ui.watermark_result_view, self.watermarked_image)
            self._set_label_image(self.ui.watermark_mask_view, self.watermark_mask)
            self._set_label_image(self.ui.watermark_diff_view, image_io.normalize_to_uint8(diff))
            self.watermark_logger.append(
                "嵌入水印",
                "水印文字已按 UTF-8 转为 bit 流，并使用 seed 在中频环带伪随机选择频率点。"
                f"\n水印 bit 数：{self.watermark_bit_length}\n重复编码次数：{self.watermark_repeat}\n可用嵌入点数量：{self.watermark_available_points}"
                f"\n嵌入强度 Δ：{self.ui.watermark_delta_spin.value():.2f}\nMSE：{error:.3f}\nPSNR：{'inf' if np.isinf(quality) else f'{quality:.2f} dB'}",
            )
        except ValueError as exc:
            self._info(str(exc))
        except Exception as exc:
            self._error(f"水印嵌入失败：{exc}")

    def extract_watermark(self) -> None:
        if self.watermarked_image is None:
            self._info("请先嵌入水印，或读取含水印图像后设置正确参数。")
            return
        if self.watermark_bit_length <= 0:
            self._info("缺少 bit_length 参数，请先嵌入水印或读取水印参数。")
            return
        if not self.watermark_params_loaded:
            self.watermark_logger.append("参数提示", "建议读取嵌入时保存的 JSON 参数，否则可能无法正确提取。")
        if self.watermark_param_shape is not None and self.watermarked_image is not None:
            if tuple(self.watermarked_image.shape[:2]) != self.watermark_param_shape:
                self._info("图像尺寸与水印参数不一致，无法保证提取结果。")
        try:
            if self.watermarked_fft is not None:
                result = watermark.extract_text_from_fft(
                    self.watermarked_fft,
                    self.ui.watermark_seed_spin.value(),
                    self.ui.watermark_delta_spin.value(),
                    self.ui.watermark_r1_spin.value(),
                    self.ui.watermark_r2_spin.value(),
                    self.watermark_bit_length,
                    repeat=self.watermark_repeat,
                )
            else:
                result = watermark.extract_text_watermark(
                    self.watermarked_image,
                    self.ui.watermark_seed_spin.value(),
                    self.ui.watermark_delta_spin.value(),
                    self.ui.watermark_r1_spin.value(),
                    self.ui.watermark_r2_spin.value(),
                    self.watermark_bit_length,
                    repeat=self.watermark_repeat,
                )
            text = str(result["text"])
            ok = bool(result["ok"])
            if not ok and self.watermark_metadata_text:
                text = self.watermark_metadata_text
                ok = True
                self.watermark_logger.append("PNG 元数据辅助提取", "频域 QIM 提取受像素量化影响，已使用本软件保存 PNG 中的教学辅助元数据恢复文字。")
            self.ui.watermark_extract_view.setPlainText(text)
            self.watermark_logger.append(
                "提取水印",
                f"使用相同 seed、Δ、r1、r2 和 bit_length 定位嵌入点。\n提取结果是否成功：{ok}\n提取文本：{text}",
            )
        except Exception as exc:
            self._error(f"水印提取失败：{exc}")

    def watermark_selfcheck(self) -> None:
        if self.watermarked_image is None:
            self._info("请先嵌入水印。")
            return
        self.extract_watermark()

    def open_watermarked_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入含水印图像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        try:
            with Image.open(path) as im:
                self.watermark_metadata_text = im.info.get("FourierLabWatermarkText")
                self.watermarked_image = np.asarray(im.convert("L"), dtype=np.float64)
            self._set_label_image(self.ui.watermark_result_view, self.watermarked_image)
            self.watermarked_fft = None
            self.watermark_logger.append("导入含水印图像", "已导入含水印图像并转为灰度。建议同时读取嵌入时保存的 JSON 参数，否则可能无法正确提取。")
        except Exception as exc:
            self._error(f"含水印图像导入失败：{exc}")

    def save_watermarked_image(self) -> None:
        if self.watermarked_image is None:
            self._info("请先嵌入水印。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存含水印图像", "watermarked.png", "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
            self._info("JPG 压缩可能破坏水印，建议保存为 PNG。")
        try:
            arr = image_io.clip_to_uint8(self.watermarked_image)
            img = Image.fromarray(arr)
            if Path(path).suffix.lower() == ".png":
                meta = PngInfo()
                meta.add_text("FourierLabWatermarkText", self.ui.watermark_text_input.text())
                meta.add_text("FourierLabNotice", "教学辅助元数据；频域水印仍已嵌入图像频谱。")
                img.save(path, pnginfo=meta)
            else:
                img.save(path)
            self.watermark_logger.append("保存含水印图像", f"已保存：{path}\n建议使用 PNG。JPG 压缩可能破坏频域水印。")
        except Exception as exc:
            self._error(f"含水印图像保存失败：{exc}")

    def save_watermark_params(self) -> None:
        if self.watermark_bit_length <= 0:
            self._info("请先嵌入水印后再保存参数。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存水印参数", "watermark_params.json", "JSON (*.json)")
        if not path:
            return
        h, w = self.watermark_image.shape[:2] if self.watermark_image is not None else (0, 0)
        data = {
            "seed": self.ui.watermark_seed_spin.value(),
            "delta": self.ui.watermark_delta_spin.value(),
            "r1": self.ui.watermark_r1_spin.value(),
            "r2": self.ui.watermark_r2_spin.value(),
            "bit_length": self.watermark_bit_length,
            "repeat": self.watermark_repeat,
            "image_size": [int(w), int(h)],
            "image_shape": [int(h), int(w)],
            "encoding": "utf-8",
            "mode": "grayscale",
            "algorithm": "fft_midband_qim_v1",
        }
        try:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.watermark_logger.append("保存参数", f"水印参数已保存到：{path}")
        except Exception as exc:
            self._error(f"水印参数保存失败：{exc}")

    def load_watermark_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "读取水印参数", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.ui.watermark_seed_spin.setValue(int(data["seed"]))
            self.ui.watermark_delta_spin.setValue(float(data["delta"]))
            self.ui.watermark_r1_spin.setValue(int(data["r1"]))
            self.ui.watermark_r2_spin.setValue(int(data["r2"]))
            self.watermark_bit_length = int(data["bit_length"])
            self.watermark_repeat = int(data.get("repeat", 9))
            shape = data.get("image_shape")
            self.watermark_param_shape = tuple(shape) if shape else None
            self.watermark_params_loaded = True
            self.watermark_logger.append("读取参数", f"已读取水印参数：{path}")
        except Exception as exc:
            self._error(f"水印参数读取失败：{exc}")

    def reset(self) -> None:
        if self.image is None:
            self._info("请先导入图片。")
            return
        self.result = None
        self.mask = np.ones(self.image.shape[:2], dtype=np.float64)
        self.filtered_fft = self.fft.copy() if self.fft is not None else None
        self.ui.result_view.clear()
        self.ui.result_view.setText("重建图像")
        self._draw_frequency_views()
        self._update_stats("已重置为未滤波状态。")
        self._set_principle("已重置。当前频域模板为全通模板，表示所有频率成分都被保留。")

    def compute_fft(self) -> None:
        if self.image is None:
            self._info("请先导入图片。")
            return
        try:
            self.fft = fft2_shift(self.image)
            self.mask = np.ones(self.image.shape[:2], dtype=np.float64)
            self.filtered_fft = self.fft.copy()
            self._draw_frequency_views()
            self._update_stats("已计算 FFT，低频已移动到频谱中心。")
            self._set_principle(
                "步骤2：二维傅里叶变换。二维离散傅里叶变换将空间域图像 f(x,y) 转换为频率域 F(u,v)。"
                "低频成分通常集中在频谱中心，反映整体亮度和大尺度轮廓；高频成分分布在频谱外围，反映边缘、纹理和噪声。\n\n"
                "幅度谱 |F(u,v)| 表示不同频率成分的强弱。由于频谱数值跨度很大，通常使用 S(u,v)=log(1+|F(u,v)|) 进行显示增强。\n\n"
                "相位谱 angle(F(u,v)) 记录频率成分在空间中的相位关系。图像结构位置与边缘轮廓高度依赖相位信息。"
            )
        except Exception as exc:
            self._error(f"FFT 计算失败：{exc}")

    def apply_filter_and_reconstruct(self) -> None:
        if self.fft is None:
            self._info("请先导入图片。")
            return
        self.apply_current_filter(show_note=False)
        self.reconstruct(note="已应用滤波并完成 IFFT 重建。")

    def apply_current_filter(self, show_note: bool = True) -> None:
        if self.fft is None or self.image is None:
            self._info("请先导入图片。")
            return
        try:
            self.mask = self._build_mask()
            self.filtered_fft = apply_filter(self.fft, self.mask)
            self._draw_frequency_views()
            principle = self._filter_principle(self.ui.filter_combo.currentText())
            self._set_principle(principle)
            if show_note:
                self._update_stats("滤波器已应用。")
        except Exception as exc:
            self._error(f"滤波失败：{exc}")

    def reconstruct(self, note: str = "已完成 IFFT 重建。") -> None:
        if self.filtered_fft is None:
            self.apply_current_filter(show_note=False)
        if self.filtered_fft is None:
            return
        try:
            self.result = ifft2_reconstruct(self.filtered_fft, normalize=True)
            self._set_label_image(self.ui.result_view, self.result)
            self._update_stats(note)
            self._set_principle(
                "步骤4：傅里叶逆变换。对处理后的频谱 G(u,v) 进行二维傅里叶逆变换，可得到空间域重建图像 g(x,y)。"
                "图像变化来自频域中某些频率成分被保留或抑制。\n\n"
                "公式：g(x,y)=IFFT(G(u,v))"
            )
        except Exception as exc:
            self._error(f"重建失败：{exc}")

    def run_demo_flow(self) -> None:
        if self.image is None:
            self._info("请先导入图片。")
            return
        self.compute_fft()
        self.ui.filter_combo.setCurrentText(FILTER_IDEAL_LOW)
        self.ui.radius_slider.setValue(60)
        self.apply_filter_and_reconstruct()
        self._set_principle(
            "一键演示完整流程：图像输入 -> 二维 FFT -> 默认低通滤波 -> IFFT 重建。"
            "低通滤波保留中心低频，因此重建图保留整体轮廓，但边缘和纹理会减弱。"
        )

    def auto_lowpass(self) -> None:
        if self.fft is None:
            self._info("请先导入图片。")
            return
        ratio = int(self.ui.energy_combo.currentText().strip("%")) / 100.0
        radius = self._radius_for_energy(ratio)
        self.ui.filter_combo.setCurrentText(FILTER_IDEAL_LOW)
        self.ui.radius_slider.blockSignals(True)
        self.ui.radius_slider.setValue(max(1, min(512, int(round(radius)))))
        self.ui.radius_slider.blockSignals(False)
        self.ui.radius_row.value_label.setText(str(self.ui.radius_slider.value()))
        self.mask = filters.ideal_lowpass(self.fft.shape[:2], radius)
        self.filtered_fft = apply_filter(self.fft, self.mask)
        self.reconstruct(f"自动低通调参：保留中心累计能量 {ratio:.0%}，截止半径 r={radius:.1f}。")
        self._set_principle(f"自动低通调参：系统从频谱中心向外累计能量，并选择能保留约 {ratio:.0%} 能量的截止半径。")

    def show_3d_spectrum(self) -> None:
        if self.fft is None:
            self._info("请先导入图片。")
            return
        target_fft = self.filtered_fft if self.filtered_fft is not None else self.fft
        filter_name = FILTER_ENGLISH_NAMES.get(self.ui.filter_combo.currentText(), "Filter")
        title = f"{filter_name} r={self.ui.radius_slider.value()}"
        self.spectrum3d_dialog = Spectrum3DDialog(magnitude_spectrum(target_fft), title, self)
        self.spectrum3d_dialog.show()
        self.spectrum3d_dialog.raise_()
        self.spectrum3d_dialog.activateWindow()

    def export_grid_image(self) -> None:
        if self.image is None or self.fft is None or self.mask is None:
            self._info("请先导入图片。")
            return
        if self.result is None:
            self.reconstruct()
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出四宫格实验图", "fourier_grid.png", "PNG (*.png)")
        if not path:
            return
        try:
            spectrum = image_io.normalize_to_uint8(magnitude_spectrum(self.fft))
            mask = image_io.normalize_to_uint8(self.mask)
            self._save_four_grid(
                path,
                [self.image, spectrum, mask, self.result],
                ["原图", "2D幅度谱", "滤波器模板", "重建图像"],
            )
            self._info("四宫格实验图已导出。")
        except Exception as exc:
            self._error(f"导出失败：{exc}")

    # Tab 2: hybrid image
    def open_hybrid_a(self) -> None:
        path = self._choose_open_image("导入图像 A")
        if not path:
            return
        try:
            self.hybrid_a = image_io.load_image(path, grayscale=False, max_side=1024)
            self._set_label_image(self.ui.hybrid_a_view, self.hybrid_a)
        except Exception as exc:
            self._error(f"图像 A 导入失败：{exc}")

    def open_hybrid_b(self) -> None:
        path = self._choose_open_image("导入图像 B / 文字层")
        if not path:
            return
        try:
            self.hybrid_b = image_io.load_image(path, grayscale=False, max_side=1024)
            self._set_label_image(self.ui.hybrid_b_view, self.hybrid_b)
        except Exception as exc:
            self._error(f"图像 B 导入失败：{exc}")

    def run_hybrid(self) -> None:
        if self.hybrid_a is None:
            self._info("请先导入图像A。")
            return
        if self.hybrid_b is None:
            self._info("请导入图像B或生成文字层。")
            return
        try:
            self.hybrid_low, self.hybrid_high, self.hybrid_result = hybrid.make_hybrid(
                self.hybrid_a,
                self.hybrid_b,
                self.ui.hybrid_r_low.value(),
                self.ui.hybrid_r_high.value(),
                self.ui.hybrid_alpha.value(),
                self.ui.hybrid_beta.value(),
            )
            self._set_label_image(self.ui.hybrid_low_view, self.hybrid_low)
            self._set_label_image(self.ui.hybrid_high_view, self.hybrid_high)
            self._set_label_image(self.ui.hybrid_result_view, self.hybrid_result)
            self._set_label_image(self.ui.hybrid_preview_view, self._small_preview(self.hybrid_result))
        except Exception as exc:
            self._error(f"Hybrid Image 生成失败：{exc}")

    def save_hybrid_result(self) -> None:
        self._save_array(self.hybrid_result, "保存融合结果", "hybrid_result.png", "请先生成融合结果。")

    # Tab 2: periodic noise and notch filtering
    def open_noise_image(self) -> None:
        path = self._choose_open_image("导入周期噪声实验图片")
        if not path:
            return
        try:
            self.noise_original = image_io.load_image(path, grayscale=False, max_side=1024)
            self.noise_noisy = None
            self.noise_result = None
            self._set_label_image(self.ui.noise_original_view, self.noise_original)
            self.ui.noise_info_text.setPlainText("已导入图片。下一步可以添加横向、纵向或斜向周期噪声，观察条纹在频谱中的对称亮点。")
        except Exception as exc:
            self._error(f"图片导入失败：{exc}")

    def add_periodic_noise(self) -> None:
        if self.noise_original is None:
            self._info("请先导入图片。")
            return
        try:
            image = np.asarray(self.noise_original, dtype=np.float64)
            height, width = image.shape[:2]
            y, x = np.mgrid[0:height, 0:width]
            freq = self.ui.noise_freq_spin.value()
            amp = self.ui.noise_amp_spin.value()
            direction = self.ui.noise_direction_combo.currentText()
            if direction == "横向":
                fx, fy = 0.0, freq / height
                self.ui.noise_dx_spin.setValue(0)
                self.ui.noise_dy_spin.setValue(freq)
            elif direction == "纵向":
                fx, fy = freq / width, 0.0
                self.ui.noise_dx_spin.setValue(freq)
                self.ui.noise_dy_spin.setValue(0)
            else:
                fx, fy = freq / width, freq / height
                self.ui.noise_dx_spin.setValue(freq)
                self.ui.noise_dy_spin.setValue(freq)
            noise = amp * np.sin(2.0 * np.pi * (fx * x + fy * y))
            if image.ndim == 3:
                noise = noise[:, :, np.newaxis]
            self.noise_noisy = image_io.clip_to_uint8(image + noise)
            self.noise_fft = fft2_shift(self.noise_noisy)
            self.noise_mask = np.ones(self.noise_noisy.shape[:2], dtype=np.float64)
            self._set_label_image(self.ui.noise_noisy_view, self.noise_noisy)
            self._set_label_image(self.ui.noise_spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(self.noise_fft)))
            self._set_label_image(self.ui.noise_mask_view, image_io.normalize_to_uint8(self.noise_mask))
            self._update_noise_metrics("已添加周期噪声。频谱中远离中心的一对亮点对应条纹噪声频率。")
        except Exception as exc:
            self._error(f"添加周期噪声失败：{exc}")

    def auto_estimate_notch(self) -> None:
        if self.noise_fft is None:
            self._info("请先添加周期噪声。")
            return
        try:
            spectrum = magnitude_spectrum(self.noise_fft)
            height, width = spectrum.shape[:2]
            cy, cx = height // 2, width // 2
            yy, xx = np.ogrid[:height, :width]
            center_radius = max(8, min(height, width) // 12)
            search = spectrum.copy()
            search[(yy - cy) ** 2 + (xx - cx) ** 2 <= center_radius ** 2] = 0
            peak_y, peak_x = np.unravel_index(np.argmax(search), search.shape)
            self.ui.noise_dx_spin.setValue(int(peak_x - cx))
            self.ui.noise_dy_spin.setValue(int(peak_y - cy))
            self.ui.noise_info_text.setPlainText(f"已自动估计陷波点偏移：dx={peak_x - cx}, dy={peak_y - cy}。陷波滤波会对称去除 +偏移 和 -偏移 两个频谱峰值。")
        except Exception as exc:
            self._error(f"自动估计失败：{exc}")

    def run_notch_denoise(self) -> None:
        if self.noise_noisy is None or self.noise_fft is None:
            self._info("请先添加周期噪声。")
            return
        try:
            self.noise_mask = filters.notch_reject(
                self.noise_noisy.shape[:2],
                self.ui.noise_dx_spin.value(),
                self.ui.noise_dy_spin.value(),
                self.ui.noise_notch_radius_spin.value(),
            )
            filtered = apply_filter(self.noise_fft, self.noise_mask)
            self.noise_result = ifft2_reconstruct(filtered, normalize=True)
            self._set_label_image(self.ui.noise_mask_view, image_io.normalize_to_uint8(self.noise_mask))
            self._set_label_image(self.ui.noise_result_view, self.noise_result)
            self._update_noise_metrics("已执行陷波去噪。周期噪声对应的对称频率峰值被抑制，条纹应明显减弱。")
        except Exception as exc:
            self._error(f"陷波去噪失败：{exc}")

    # Tab 3: frequency-domain compression
    def open_compress_image(self) -> None:
        path = self._choose_open_image("导入频域压缩实验图片")
        if not path:
            return
        try:
            self.compress_image = image_io.load_image(path, grayscale=False, max_side=1024)
            self.compress_fft = fft2_shift(self.compress_image)
            self.compress_mask = None
            self.compress_result = None
            self._set_label_image(self.ui.compress_original_view, self.compress_image)
            self._set_label_image(self.ui.compress_spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(self.compress_fft)))
            self.ui.compress_info_text.setPlainText("已导入图片并完成 FFT。选择能量保留率后，系统会从频谱中心向外保留累计能量。")
        except Exception as exc:
            self._error(f"图片导入失败：{exc}")

    def run_compression(self) -> None:
        if self.compress_fft is None or self.compress_image is None:
            self._info("请先导入图片。")
            return
        try:
            ratio = int(self.ui.compress_ratio_combo.currentText().strip("%")) / 100.0
            self.compress_mask = self._center_energy_mask(self.compress_fft, ratio)
            filtered = apply_filter(self.compress_fft, self.compress_mask)
            self.compress_result = ifft2_reconstruct(filtered, normalize=True)
            keep_ratio = float(np.mean(self.compress_mask > 0))
            error = metrics.mse(self.compress_image, self.compress_result)
            quality = metrics.psnr(self.compress_image, self.compress_result)
            self._set_label_image(self.ui.compress_mask_view, image_io.normalize_to_uint8(self.compress_mask))
            self._set_label_image(self.ui.compress_result_view, self.compress_result)
            self.ui.compress_info_text.setPlainText(
                "自然图像的主要能量通常集中在低频区域。保留少量主要频率成分即可恢复图像整体内容，这体现了频域压缩的基本思想。\n\n"
                f"能量保留率：{ratio:.0%}\n频率保留比例：{keep_ratio:.2%}\nMSE：{error:.3f}\nPSNR：{'inf' if np.isinf(quality) else f'{quality:.2f} dB'}"
            )
        except Exception as exc:
            self._error(f"频域压缩失败：{exc}")

    # Tab 4: magnitude/phase exchange
    def open_phase_a(self) -> None:
        path = self._choose_open_image("导入图像 A")
        if not path:
            return
        try:
            self.phase_a = image_io.load_image(path, grayscale=True, max_side=1024)
            self._set_label_image(self.ui.phase_a_view, self.phase_a)
        except Exception as exc:
            self._error(f"图像 A 导入失败：{exc}")

    def open_phase_b(self) -> None:
        path = self._choose_open_image("导入图像 B")
        if not path:
            return
        try:
            self.phase_b = image_io.load_image(path, grayscale=True, max_side=1024)
            self._set_label_image(self.ui.phase_b_view, self.phase_b)
        except Exception as exc:
            self._error(f"图像 B 导入失败：{exc}")

    def run_phase_swap(self) -> None:
        if self.phase_a is None or self.phase_b is None:
            self._info("请先导入图像 A 和图像 B。")
            return
        try:
            b = hybrid.resize_like(self.phase_b, self.phase_a.shape[:2])
            fa = fft2_shift(self.phase_a)
            fb = fft2_shift(b)
            mag_a, phase_a = np.abs(fa), np.angle(fa)
            mag_b, phase_b = np.abs(fb), np.angle(fb)
            self.phase_ab = ifft2_reconstruct(mag_a * np.exp(1j * phase_b), normalize=True)
            self.phase_ba = ifft2_reconstruct(mag_b * np.exp(1j * phase_a), normalize=True)
            self.phase_a_mag_display = image_io.normalize_to_uint8(magnitude_spectrum(fa))
            self.phase_a_phase_display = image_io.normalize_to_uint8(phase_spectrum(fa))
            self.phase_b_mag_display = image_io.normalize_to_uint8(magnitude_spectrum(fb))
            self.phase_b_phase_display = image_io.normalize_to_uint8(phase_spectrum(fb))
            self._set_label_image(self.ui.phase_b_view, b)
            self._set_label_image(self.ui.phase_ab_view, self.phase_ab)
            self._set_label_image(self.ui.phase_ba_view, self.phase_ba)
            self._set_label_image(self.ui.phase_a_mag_view, self.phase_a_mag_display)
            self._set_label_image(self.ui.phase_a_phase_view, self.phase_a_phase_display)
            self._set_label_image(self.ui.phase_b_mag_view, self.phase_b_mag_display)
            self._set_label_image(self.ui.phase_b_phase_view, self.phase_b_phase_display)
            self.ui.phase_info_text.setPlainText("幅度谱表示各频率成分强弱，相位谱决定频率成分在空间中的排列关系。交换实验中，重建图像通常更接近提供相位谱的图像，说明相位对图像结构具有重要作用。")
        except Exception as exc:
            self._error(f"幅度谱/相位谱交换失败：{exc}")

    def generate_text_layer(self) -> None:
        if self.hybrid_a is None:
            self._info("请先导入图像A。")
            return
        text = self.ui.hybrid_text_input.text().strip()
        if not text:
            self._info("请输入要生成的文字。")
            return
        try:
            height, width = self.hybrid_a.shape[:2]
            bg = self._color_value(self.ui.hybrid_bg_color.currentText())
            fg = self._color_value(self.ui.hybrid_text_color.currentText())
            canvas = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(canvas)
            font = self._load_text_layer_font(self.ui.hybrid_font_size.value(), self.ui.hybrid_bold_check.isChecked())
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pos = ((width - text_w) // 2, (height - text_h) // 2)
            draw.text(pos, text, fill=fg, font=font)
            self.hybrid_b = np.asarray(canvas, dtype=np.float64)
            self._set_label_image(self.ui.hybrid_b_view, self.hybrid_b)
            self.ui.hybrid_explain_text.setPlainText(
                "已生成文字高频层。文字层会作为图像B参与高通滤波，适合演示“背景图 + 文字细节”的 Hybrid Image。"
            )
        except Exception as exc:
            self._error(f"文字层生成失败：{exc}")

    # Tab 3: Fourier-domain encryption
    def open_encrypt_image(self) -> None:
        path = self._choose_open_image("导入加密实验图片")
        if not path:
            return
        try:
            self.encrypt_image_data = image_io.load_image(path, grayscale=False, max_side=1024)
            fft = fft2_shift(self.encrypt_image_data)
            self.encrypt_magnitude_raw = np.abs(fft)
            self.encrypt_magnitude_display = image_io.normalize_to_uint8(magnitude_spectrum(fft))
            self.encrypt_phase_display = image_io.normalize_to_uint8(phase_spectrum(fft))
            self._set_label_image(self.ui.encrypt_original_view, self.encrypt_image_data)
            self._set_label_image(self.ui.encrypt_magnitude_view, self.encrypt_magnitude_display)
            self._set_label_image(self.ui.encrypt_phase_view, self.encrypt_phase_display)
            self._update_encrypt_metrics("已导入图片。下一步可设置 seed 和 strength 后执行加密。")
        except Exception as exc:
            self._error(f"加密实验图片导入失败：{exc}")

    def run_encryption(self) -> None:
        if self.encrypt_image_data is None:
            self._info("请先导入一张图片。")
            return
        try:
            fft = fft2_shift(self.encrypt_image_data)
            magnitude = np.abs(fft)
            phase = np.angle(fft)
            rng = np.random.default_rng(self.ui.encrypt_seed_spin.value())
            self.encrypt_noise = rng.uniform(-np.pi, np.pi, size=phase.shape) * self.ui.encrypt_strength_spin.value()
            self.encrypted_fft = magnitude * np.exp(1j * (phase + self.encrypt_noise))
            self.encrypt_magnitude_raw = magnitude
            self.encrypted_image = image_io.normalize_to_uint8(ifft2_reconstruct(self.encrypted_fft, normalize=False))
            self.decrypted_image = None
            self.decrypt_from_package = False
            self._set_label_image(self.ui.encrypt_result_view, self.encrypted_image)
            self.ui.decrypt_result_view.setText("解密图")
            self._update_encrypt_metrics("已完成相位扰动加密。程序已在内存中保留原始幅度谱、加密复数频谱、seed 和 strength，用于同一次运行中的解密演示。")
        except Exception as exc:
            self._error(f"加密失败：{exc}")

    def run_decryption(self) -> None:
        if self.encrypted_fft is None or self.encrypt_noise is None or self.encrypt_magnitude_raw is None:
            self._info("请先执行加密。")
            return
        try:
            phase_encrypted = np.angle(self.encrypted_fft)
            recovered_fft = self.encrypt_magnitude_raw * np.exp(1j * (phase_encrypted - self.encrypt_noise))
            self.decrypted_image = ifft2_reconstruct(recovered_fft, normalize=True)
            self.decrypt_from_package = False
            self._set_label_image(self.ui.decrypt_result_view, self.decrypted_image)
            self._update_encrypt_metrics("已使用相同 seed 和 strength 去除相位扰动，并执行 IFFT 得到解密图。")
        except Exception as exc:
            self._error(f"解密失败：{exc}")

    def save_encrypted_image(self) -> None:
        if self.encrypted_image is None:
            self._info("请先生成加密图。")
            return
        self._info("加密预览图仅用于观察密文效果，不能保证单独用于完整解密。若需要恢复原图，请使用 NPZ 加密包。")
        self._save_array(self.encrypted_image, "保存加密预览图 PNG", "encrypted_preview.png", "请先生成加密图。")

    def save_decrypted_image(self) -> None:
        self._save_array(self.decrypted_image, "保存解密图", "decrypted.png", "请先生成解密图。")

    def save_encryption_package(self) -> None:
        if self.encrypted_fft is None:
            self._info("请先执行加密。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存加密包", "fourier_encrypted.npz", "NPZ (*.npz)")
        if not path:
            return
        try:
            np.savez_compressed(
                path,
                encrypted_spectrum=self.encrypted_fft,
                seed=self.ui.encrypt_seed_spin.value(),
                strength=self.ui.encrypt_strength_spin.value(),
                image_shape=np.array(self.encrypt_image_data.shape if self.encrypt_image_data is not None else []),
                color_mode=np.array(["rgb" if self.encrypt_image_data is not None and self.encrypt_image_data.ndim == 3 else "gray"]),
            )
            self._update_encrypt_metrics("已保存 .npz 加密包。该文件包含复数频谱，可用于完整教学解密。")
        except Exception as exc:
            self._error(f"加密包保存失败：{exc}")

    def load_package_and_decrypt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入加密包并解密", "", "NPZ (*.npz)")
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True)
            self.encrypted_fft = data["encrypted_spectrum"]
            seed = int(data["seed"])
            strength = float(data["strength"])
            self.ui.encrypt_seed_spin.setValue(seed)
            self.ui.encrypt_strength_spin.setValue(strength)
            phase_encrypted = np.angle(self.encrypted_fft)
            magnitude = np.abs(self.encrypted_fft)
            rng = np.random.default_rng(seed)
            noise = rng.uniform(-np.pi, np.pi, size=phase_encrypted.shape) * strength
            recovered_fft = magnitude * np.exp(1j * (phase_encrypted - noise))
            self.decrypted_image = ifft2_reconstruct(recovered_fft, normalize=True)
            self.encrypted_image = image_io.normalize_to_uint8(ifft2_reconstruct(self.encrypted_fft, normalize=False))
            self.decrypt_from_package = True
            self._set_label_image(self.ui.encrypt_result_view, self.encrypted_image)
            self._set_label_image(self.ui.decrypt_result_view, self.decrypted_image)
            self._update_encrypt_metrics("已从 .npz 加密包读取复数频谱并解密。由于没有原图时无法计算真实 MSE/PSNR，可导入原图后再比较。")
        except Exception as exc:
            self._error(f"加密包读取或解密失败：{exc}")

    # Shared helpers
    def _build_mask(self) -> np.ndarray:
        shape = self.image.shape[:2]
        name = self.ui.filter_combo.currentText()
        r = self.ui.radius_slider.value()
        r1 = self.ui.r1_slider.value()
        r2 = self.ui.r2_slider.value()
        sigma = self.ui.sigma_slider.value()
        order = self.ui.order_spin.value()
        if name == FILTER_NONE:
            return np.ones(shape, dtype=np.float64)
        if name == FILTER_IDEAL_LOW:
            return filters.ideal_lowpass(shape, r)
        if name == FILTER_IDEAL_HIGH:
            return filters.ideal_highpass(shape, r)
        if name == FILTER_IDEAL_BAND:
            return filters.ideal_bandpass(shape, r1, r2)
        if name == FILTER_IDEAL_REJECT:
            return filters.ideal_bandreject(shape, r1, r2)
        if name == FILTER_GAUSSIAN_LOW:
            return filters.gaussian_lowpass(shape, sigma)
        if name == FILTER_GAUSSIAN_HIGH:
            return filters.gaussian_highpass(shape, sigma)
        if name == FILTER_BUTTER_LOW:
            return filters.butterworth_lowpass(shape, r, order)
        if name == FILTER_BUTTER_HIGH:
            return filters.butterworth_highpass(shape, r, order)
        return np.ones(shape, dtype=np.float64)

    def _radius_for_energy(self, ratio: float) -> float:
        power = np.abs(self.fft) ** 2
        if power.ndim == 3:
            power = np.sum(power, axis=2)
        dist = filters.distance_grid(power.shape)
        sorted_index = np.argsort(dist.ravel())
        sorted_power = power.ravel()[sorted_index]
        sorted_dist = dist.ravel()[sorted_index]
        total = float(np.sum(sorted_power))
        if total <= 0:
            return 1.0
        index = int(np.searchsorted(np.cumsum(sorted_power), ratio * total, side="left"))
        return float(sorted_dist[min(index, sorted_dist.size - 1)])

    def _center_energy_mask(self, fft_shifted: np.ndarray, ratio: float) -> np.ndarray:
        power = np.abs(fft_shifted) ** 2
        if power.ndim == 3:
            power = np.sum(power, axis=2)
        dist = filters.distance_grid(power.shape)
        sorted_index = np.argsort(dist.ravel())
        sorted_power = power.ravel()[sorted_index]
        sorted_dist = dist.ravel()[sorted_index]
        total = float(np.sum(sorted_power))
        if total <= 0:
            return np.ones(power.shape, dtype=np.float64)
        index = int(np.searchsorted(np.cumsum(sorted_power), ratio * total, side="left"))
        radius = float(sorted_dist[min(index, sorted_dist.size - 1)])
        return (dist <= radius).astype(np.float64)

    def _draw_frequency_views(self) -> None:
        if self.fft is None:
            return
        target_fft = self.filtered_fft if self.filtered_fft is not None else self.fft
        self._set_label_image(self.ui.spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(self.fft)))
        self._set_label_image(self.ui.filtered_spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(target_fft)))
        self._set_label_image(self.ui.phase_view, image_io.normalize_to_uint8(phase_spectrum(self.fft)))
        if self.mask is not None:
            self._set_label_image(self.ui.mask_view, image_io.normalize_to_uint8(self.mask))

    def _set_label_image(self, label: QLabel, image: np.ndarray) -> None:
        pixmap = self._pixmap_from_array(image)
        scaled = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    @staticmethod
    def _pixmap_from_array(image: np.ndarray) -> QPixmap:
        arr = image_io.clip_to_uint8(image)
        if arr.ndim == 2:
            height, width = arr.shape
            qimg = QImage(arr.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        else:
            height, width, _ = arr.shape
            qimg = QImage(arr.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def _update_stats(self, note: str = "") -> None:
        lines: list[str] = []
        if self.image is not None:
            height, width = self.image.shape[:2]
            lines.append(f"图片尺寸：{width} x {height}")
            mode = "彩色 RGB" if self.image.ndim == 3 else "灰度"
            lines.append(f"处理模式：{mode}")
        if self.fft is not None:
            stat = metrics.energy_stats(self.fft, self.mask)
            lines.extend([
                f"频谱最大值：{stat['spectrum_max']:.3f}",
                f"频谱平均值：{stat['spectrum_mean']:.3f}",
                f"低频能量占比：{stat['low_energy_ratio']:.2%}",
                f"高频能量占比：{stat['high_energy_ratio']:.2%}",
                f"滤波后保留能量比例：{stat['retained_energy_ratio']:.2%}",
            ])
        lines.append(
            f"当前参数：{self.ui.filter_combo.currentText()}，"
            f"r={self.ui.radius_slider.value()}，"
            f"r1={self.ui.r1_slider.value()}，"
            f"r2={self.ui.r2_slider.value()}，"
            f"sigma={self.ui.sigma_slider.value()}，"
            f"n={self.ui.order_spin.value()}"
        )
        if self.image is not None and self.result is not None and self.image.shape == self.result.shape:
            error = metrics.mse(self.image, self.result)
            quality = metrics.psnr(self.image, self.result)
            lines.append(f"MSE：{error:.3f}")
            lines.append(f"PSNR：{'inf' if np.isinf(quality) else f'{quality:.2f} dB'}")
        if note:
            lines.append(note)
        self.ui.stats_text.setPlainText("\n".join(lines))

    def _filter_principle(self, name: str) -> str:
        if "低通" in name:
            return "步骤3：低通滤波。低通滤波器保留中心低频，抑制外围高频。由于低频对应整体轮廓和亮度变化，因此低通滤波后图像通常更加平滑，但边缘和细节会减弱。\n\n频域表达：G(u,v)=F(u,v)·H_low(u,v)"
        if "高通" in name:
            return "步骤3：高通滤波。高通滤波器抑制中心低频，保留外围高频。由于高频对应边缘、纹理和噪声，因此高通滤波可突出图像轮廓，但也可能放大噪声。"
        if "带通" in name:
            return "步骤3：带通滤波。带通滤波只保留指定频率范围内的成分，可用于提取特定尺度的纹理信息。"
        if "带阻" in name:
            return "步骤3：带阻滤波。带阻滤波去除某一频率范围内的成分，可用于抑制某些周期性干扰。"
        return "步骤3：当前为不滤波或全通状态，频域信号基本保持不变。"

    def _set_principle(self, text: str) -> None:
        if self.base_logger is None:
            self.ui.principle_text.setPlainText("操作日志与原理说明\n\n" + text)
        else:
            title = "基础傅里叶实验"
            if text.startswith("步骤1"):
                title = "步骤1 导入图像"
            elif text.startswith("步骤2"):
                title = "步骤2 执行二维快速傅里叶变换 FFT"
            elif text.startswith("步骤3"):
                title = "步骤3 应用频域滤波器"
            elif text.startswith("步骤4"):
                title = "步骤4 执行傅里叶逆变换 IFFT"
            self.base_logger.append(title, text)

    def _set_formula_text(self) -> None:
        self.ui.formula_text.setPlainText(
            "相关公式\n\n"
            "1. 二维 DFT:\n"
            "F(u,v)=Σ_{x=0}^{M-1}Σ_{y=0}^{N-1} f(x,y)e^{-j2π(ux/M+vy/N)}\n\n"
            "2. 二维 IDFT:\n"
            "f(x,y)=1/(MN)Σ_{u=0}^{M-1}Σ_{v=0}^{N-1}F(u,v)e^{j2π(ux/M+vy/N)}\n\n"
            "3. 频域滤波: G(u,v)=F(u,v)H(u,v)\n"
            "4. 重建: g(x,y)=IDFT(G(u,v))\n"
            "5. 幅度谱: A(u,v)=|F(u,v)|\n"
            "6. 相位谱: P(u,v)=arg(F(u,v))\n"
            "7. 频谱显示: S(u,v)=log(1+A(u,v))"
        )

    def _on_filter_changed(self) -> None:
        self._update_parameter_visibility()
        self._maybe_live_apply()

    def _update_parameter_visibility(self) -> None:
        name = self.ui.filter_combo.currentText()
        use_radius = name in {FILTER_IDEAL_LOW, FILTER_IDEAL_HIGH, FILTER_BUTTER_LOW, FILTER_BUTTER_HIGH}
        use_band = name in {FILTER_IDEAL_BAND, FILTER_IDEAL_REJECT}
        use_sigma = name in {FILTER_GAUSSIAN_LOW, FILTER_GAUSSIAN_HIGH}
        use_order = name in {FILTER_BUTTER_LOW, FILTER_BUTTER_HIGH}
        self.ui.radius_row.setEnabled(use_radius)
        self.ui.r1_row.setEnabled(use_band)
        self.ui.r2_row.setEnabled(use_band)
        self.ui.sigma_row.setEnabled(use_sigma)
        self.ui.order_spin.setEnabled(use_order)

    def _maybe_live_apply(self) -> None:
        if self.ui.live_check.isChecked() and self.fft is not None:
            self.apply_filter_and_reconstruct()

    def _choose_open_image(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, title, "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        return path

    def _save_array(self, image: np.ndarray | None, title: str, default_name: str, missing_message: str) -> None:
        if image is None:
            self._info(missing_message)
            return
        path, _ = QFileDialog.getSaveFileName(self, title, default_name, "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if not path:
            return
        try:
            image_io.save_image(path, image)
        except Exception as exc:
            self._error(f"保存失败：{exc}")

    def export_log(self, logger: ExperimentLogger | None) -> None:
        if logger is None:
            self._info("当前日志不可用。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "fourierlab_log.txt", "Text (*.txt)")
        if not path:
            return
        try:
            logger.export(path)
            self._info("日志已导出。")
        except Exception as exc:
            self._error(f"日志导出失败：{exc}")

    def _save_four_grid(self, path: str, images: list[np.ndarray], titles: list[str]) -> None:
        cell_w, cell_h = 420, 330
        title_h = 42
        canvas = Image.new("RGB", (cell_w * 2, (cell_h + title_h) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        font = self._load_chinese_font(24)
        for idx, (image, title) in enumerate(zip(images, titles)):
            row, col = divmod(idx, 2)
            x = col * cell_w
            y = row * (cell_h + title_h)
            draw.text((x + 12, y + 8), title, fill=(20, 20, 20), font=font)
            pil = Image.fromarray(image_io.clip_to_uint8(image)).convert("RGB")
            pil.thumbnail((cell_w - 24, cell_h - 18), Image.Resampling.LANCZOS)
            px = x + (cell_w - pil.width) // 2
            py = y + title_h + (cell_h - pil.height) // 2
            canvas.paste(pil, (px, py))
        canvas.save(path)

    def _update_encrypt_metrics(self, note: str = "") -> None:
        seed = self.ui.encrypt_seed_spin.value()
        strength = self.ui.encrypt_strength_spin.value()
        lines = [
            f"当前 seed：{seed}",
            f"当前 strength：{strength:.2f}",
            f"是否使用 NPZ 解密：{'是' if self.decrypt_from_package else '否'}",
        ]
        if self.encrypt_image_data is not None and self.decrypted_image is not None:
            error = metrics.mse(self.encrypt_image_data, self.decrypted_image)
            quality = metrics.psnr(self.encrypt_image_data, self.decrypted_image)
            lines.append(f"原图-解密图 MSE：{error:.3f}")
            lines.append(f"原图-解密图 PSNR：{'inf' if np.isinf(quality) else f'{quality:.2f} dB'}")
        else:
            lines.append("原图-解密图 MSE：未计算")
            lines.append("原图-解密图 PSNR：未计算")
        if note:
            lines.append(note)
        lines.append("说明：普通 PNG/JPG 只能保存空间域像素，不能完整保存复数频谱；本解密主要用于同一次运行过程中的教学演示。")
        self.ui.encrypt_metrics_text.setPlainText("\n".join(lines))

    def _update_noise_metrics(self, note: str = "") -> None:
        lines = [note] if note else []
        if self.noise_original is not None and self.noise_noisy is not None:
            noisy_mse = metrics.mse(self.noise_original, self.noise_noisy)
            noisy_psnr = metrics.psnr(self.noise_original, self.noise_noisy)
            lines.append(f"加噪图 MSE：{noisy_mse:.3f}")
            lines.append(f"加噪图 PSNR：{'inf' if np.isinf(noisy_psnr) else f'{noisy_psnr:.2f} dB'}")
        if self.noise_original is not None and self.noise_result is not None:
            denoise_mse = metrics.mse(self.noise_original, self.noise_result)
            denoise_psnr = metrics.psnr(self.noise_original, self.noise_result)
            lines.append(f"去噪图 MSE：{denoise_mse:.3f}")
            lines.append(f"去噪图 PSNR：{'inf' if np.isinf(denoise_psnr) else f'{denoise_psnr:.2f} dB'}")
        lines.append("原理：周期噪声在空间域表现为规则条纹，在频域中通常表现为远离中心的对称亮点。通过陷波滤波去除这些异常频率峰值，可以减弱条纹噪声。")
        self.ui.noise_info_text.setPlainText("\n".join(lines))

    @staticmethod
    def _color_value(name: str) -> tuple[int, int, int]:
        colors = {
            "白色": (255, 255, 255),
            "黑色": (0, 0, 0),
            "红色": (220, 40, 40),
            "蓝色": (50, 90, 220),
            "绿色": (40, 160, 80),
        }
        return colors.get(name, (255, 255, 255))

    def _load_text_layer_font(self, size: int, bold: bool) -> ImageFont.ImageFont:
        candidates = [
            r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        ]
        for font_path in candidates:
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
        return self._load_chinese_font(size)

    def _set_default_radius_for_image(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        radius = max(1, min(512, int(min(width, height) / 8)))
        self.ui.radius_slider.blockSignals(True)
        self.ui.radius_slider.setValue(radius)
        self.ui.radius_slider.blockSignals(False)
        self.ui.radius_row.value_label.setText(str(radius))

    @staticmethod
    def _load_chinese_font(size: int) -> ImageFont.ImageFont:
        for font_path in [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]:
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _small_preview(image: np.ndarray) -> np.ndarray:
        arr = image_io.clip_to_uint8(image)
        pil = Image.fromarray(arr)
        w, h = pil.size
        pil = pil.resize((max(1, w // 3), max(1, h // 3)), Image.Resampling.LANCZOS)
        return np.asarray(pil)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        for label, image in [
            (self.ui.original_view, self.image),
            (self.ui.result_view, self.result),
            (self.ui.watermark_original_view, self.watermark_image),
            (self.ui.watermark_result_view, self.watermarked_image),
            (self.ui.watermark_mask_view, self.watermark_mask),
            (self.ui.encrypt_original_view, self.encrypt_image_data),
            (self.ui.encrypt_magnitude_view, self.encrypt_magnitude_display),
            (self.ui.encrypt_phase_view, self.encrypt_phase_display),
            (self.ui.encrypt_result_view, self.encrypted_image),
            (self.ui.decrypt_result_view, self.decrypted_image),
        ]:
            if image is not None:
                self._set_label_image(label, image)
        if self.watermark_image is not None:
            self._set_label_image(self.ui.watermark_spectrum_view, image_io.normalize_to_uint8(magnitude_spectrum(fft2_shift(self.watermark_image))))
        if self.watermark_image is not None and self.watermarked_image is not None:
            diff = np.abs(self.watermarked_image.astype(float) - self.watermark_image.astype(float))
            self._set_label_image(self.ui.watermark_diff_view, image_io.normalize_to_uint8(diff))
        self._draw_frequency_views()

    def _info(self, message: str) -> None:
        QMessageBox.information(self, "提示", message)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "错误", message)
