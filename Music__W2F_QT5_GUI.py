import sys
import os
import filetype
from pathlib import Path
import subprocess
import PyQt5
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QProgressBar, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

SUPPORTED_EXTENSIONS = (
    '.mp3', '.wav', '.aac', '.flac', '.ogg',
    '.m4a', '.wma', '.aiff', '.alac', '.opus'
)


class ConversionThread(QThread):
    progress_updated = pyqtSignal(int, int, str)
    conversion_finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    directory_scanned = pyqtSignal(str)  # 新增目录扫描信号

    def __init__(self, input_dir, output_dir):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.is_running = True
        self.scan_count = 0

    def run(self):
        try:
            input_path = Path(self.input_dir)
            output_path = Path(self.output_dir)

            # 新增目录结构预检查
            if not self.validate_directory_structure(input_path):
                self.log_message.emit("錯誤：輸入目錄結構異常")
                self.conversion_finished.emit(False, "目錄結構異常")
                return

            self.log_message.emit("開始掃描音頻文件...")
            audio_files = self.scan_audio_files(input_path)

            if not audio_files:
                self.log_message.emit("錯誤：未找到支援的音頻文件")
                self.conversion_finished.emit(False, "未找到支援的音頻文件")
                return

            self.log_message.emit(f"找到 {len(audio_files)} 個音頻文件")
            self.perform_conversion(audio_files, input_path, output_path)

        except Exception as e:
            self.conversion_finished.emit(False, f"嚴重錯誤：{str(e)}")

    def validate_directory_structure(self, path):
        """深度验证目录结构有效性"""
        try:
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if not os.access(full_path, os.R_OK):
                        self.log_message.emit(f"目錄權限異常：{full_path}")
                        return False
                return True
        except Exception as e:
            self.log_message.emit(f"目錄掃描失敗：{str(e)}")
            return False

    def scan_audio_files(self, input_path):
        """增强型文件扫描，包含子目录深度检测"""
        audio_files = []
        try:
            for root, dirs, files in os.walk(input_path):
                if not self.is_running:
                    break

                self.directory_scanned.emit(root)  # 发送目录扫描进度

                for f in files:
                    file_path = Path(root) / f
                    self.scan_count += 1

                    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        audio_files.append(file_path)
                        self.log_message.emit(f"找到支援文件：{file_path}")
                    elif self.is_audio_by_content(file_path):
                        audio_files.append(file_path)
                        self.log_message.emit(f"內容識別文件：{file_path}")

        except Exception as e:
            self.log_message.emit(f"掃描異常：{str(e)}")

        return audio_files

    def is_audio_by_content(self, file_path):
        """增强型文件内容检测"""
        try:
            # 双重验证机制
            if filetype.guess(file_path).mime.startswith('audio/'):
                return True
            # 新增FFprobe验证
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=format_name', '-of', 'csv=p=0', str(file_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return 'audio' in result.stdout.lower()
        except Exception as e:
            self.log_message.emit(f"內容檢測失敗：{file_path.name}")
            return False

    def perform_conversion(self, audio_files, input_path, output_path):
        """增强型转换流程，包含子目录完整性检查"""
        total = len(audio_files)
        success_count = 0

        for index, audio_path in enumerate(audio_files, 1):
            if not self.is_running:
                break

            try:
                # 验证输出目录结构
                relative_path = audio_path.relative_to(input_path).parent
                target_dir = output_path / relative_path
                if not self.ensure_directory(target_dir):
                    continue

                # 执行转换
                if self.convert_single_file(audio_path, target_dir):
                    success_count += 1

                self.progress_updated.emit(index, total, str(audio_path))

            except Exception as e:
                self.log_message.emit(f"轉換異常：{audio_path.name}")

        msg = f"轉換完成，成功 {success_count}/{total} 個文件"
        self.conversion_finished.emit(True, msg)

    def ensure_directory(self, target_dir):
        """确保目录创建并验证"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            if not target_dir.is_dir():
                self.log_message.emit(f"目錄創建失敗：{target_dir}")
                return False
            return True
        except Exception as e:
            self.log_message.emit(f"目錄錯誤：{target_dir} - {str(e)}")
            return False

    def convert_single_file(self, audio_path, target_dir):
        """增强型单文件转换"""
        flac_path = target_dir / f"{audio_path.stem}.flac"

        # 增强型存在性检查
        if flac_path.exists():
            self.log_message.emit(f"文件已存在，跳過：{flac_path.relative_to(target_dir)}")
            return False

        try:
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(audio_path),
                 '-c:a', 'flac', '-map_metadata', '0',
                 '-compression_level', '5', str(flac_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            # 结果验证
            if flac_path.stat().st_size == 0:
                self.log_message.emit(f"錯誤：生成空文件 {flac_path.name}")
                flac_path.unlink()
                return False
            return True
        except subprocess.CalledProcessError as e:
            self.log_message.emit(f"轉換失敗：{audio_path.name} (錯誤碼 {e.returncode})")
            return False


class AudioConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.conversion_thread = None
        self.init_ui()
        self.setMinimumSize(800, 600)

    def init_ui(self):
        layout = QVBoxLayout()

        # 路径输入区
        self.setup_path_controls(layout)

        # 进度显示
        self.setup_progress_controls(layout)

        # 日志显示
        self.setup_log_display(layout)

        # 控制按钮
        self.setup_action_buttons(layout)

        self.setLayout(layout)
        self.setWindowTitle("專業音頻轉換工具")

    def setup_path_controls(self, layout):
        """路径输入控件组"""
        path_group = QVBoxLayout()
        path_group.addWidget(QLabel("輸入目錄結構要求："))
        path_group.addWidget(QLabel("1. 支持多级子目录\n2. 自动保留目录结构\n3. 最小文件大小：1MB"))

        self.input_path = self.create_path_field('輸入目錄：', 'D:\\Music\\Input')
        path_group.addLayout(self.input_path)

        self.output_dir = self.create_path_field('輸出目錄：', 'D:\\Music\\Output')
        path_group.addLayout(self.output_dir)

        layout.addLayout(path_group)

    def create_path_field(self, label, default):
        """创建路径输入行"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        field = QLineEdit(default)
        browse_btn = QPushButton('瀏覽')
        browse_btn.clicked.connect(lambda: self.browse_directory(field))
        layout.addWidget(field)
        layout.addWidget(browse_btn)
        return layout

    def setup_progress_controls(self, layout):
        """进度显示控件组"""
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("處理進度：%v/%m 文件")
        layout.addWidget(self.progress_bar)

        self.dir_progress = QLabel("當前掃描目錄：")
        layout.addWidget(self.dir_progress)

    def setup_log_display(self, layout):
        """日志显示区"""
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        layout.addWidget(self.log_display)

    def setup_action_buttons(self, layout):
        """操作按钮组"""
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("開始轉換")
        self.run_btn.clicked.connect(self.start_conversion)
        self.cancel_btn = QPushButton("取消轉換")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def browse_directory(self, field):
        """目录浏览通用方法"""
        path = QFileDialog.getExistingDirectory(self, "選擇目錄", field.text())
        if path:
            field.setText(path)

    def start_conversion(self):
        """启动转换流程"""
        input_dir = self.input_path.itemAt(1).widget().text()
        output_dir = self.output_dir.itemAt(1).widget().text()

        if not self.validate_paths(input_dir, output_dir):
            return

        self.prepare_for_conversion()
        self.start_conversion_thread(input_dir, output_dir)

    def validate_paths(self, input_dir, output_dir):
        """路径验证增强版"""
        checks = [
            (os.path.isdir, input_dir, "輸入路徑無效"),
            (os.path.isdir, output_dir, "輸出路徑無效"),
            (lambda x: x != output_dir, input_dir, "輸入輸出目錄不可相同")
        ]

        for check_func, path, msg in checks:
            if not check_func(path):
                self.show_error("路徑錯誤", f"{msg}：{path}")
                return False
        return True

    def prepare_for_conversion(self):
        """准备转换环境"""
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.reset()
        self.log_display.clear()
        self.dir_progress.setText("準備掃描目錄結構...")

    def start_conversion_thread(self, input_dir, output_dir):
        """启动转换线程"""
        self.conversion_thread = ConversionThread(input_dir, output_dir)
        self.conversion_thread.progress_updated.connect(self.update_progress)
        self.conversion_thread.conversion_finished.connect(self.handle_conversion_finish)
        self.conversion_thread.log_message.connect(self.append_log)
        self.conversion_thread.directory_scanned.connect(
            lambda d: self.dir_progress.setText(f"掃描目錄：{d}")
        )
        self.conversion_thread.start()

    def update_progress(self, current, total, filename):
        """更新进度显示"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.append_log(f"正在處理：{os.path.basename(filename)}")

    def append_log(self, message, time=None):
        """添加日志条目"""
        self.log_display.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_display.verticalScrollBar().setValue(
            self.log_display.verticalScrollBar().maximum()
        )

    def cancel_conversion(self):
        """取消转换操作"""
        if self.conversion_thread and self.conversion_thread.isRunning():
            self.conversion_thread.stop()
            self.append_log("用戶請求取消轉換...")
            self.cancel_btn.setEnabled(False)

    def handle_conversion_finish(self, success, message):
        """处理转换完成"""
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.dir_progress.setText("轉換完成" if success else "轉換中斷")
        self.append_log(message)
        QMessageBox.information(self, "操作完成", message)

    def show_error(self, title, message):
        """显示错误对话框"""
        QMessageBox.critical(self, title, message)
        self.append_log(f"[錯誤] {title}：{message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioConverter()
    window.show()
    sys.exit(app.exec_())