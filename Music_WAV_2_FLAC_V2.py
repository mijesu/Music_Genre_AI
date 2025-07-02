import os
import subprocess
import filetype
from pathlib import Path

# === 設定區 ===
INPUT_DIR = Path(r"D:\Music\Pascal\TBD_FLAC")
OUTPUT_DIR = Path(r"D:\Music\FLAC")
FFMPEG_PATH = r"D:\Music\PS_Code\ID3_CLI\FFmpeg\ffmpeg.exe"
FFPROBE_PATH = r"D:\Music\PS_Code\ID3_CLI\FFmpeg\ffprobe.exe"

SUPPORTED_EXTENSIONS = (
    '.mp3', '.wav', '.aac', '.flac', '.ogg',
    '.m4a', '.wma', '.aiff', '.alac', '.opus'
)


def validate_directory_structure(path: Path) -> bool:
    """確認目錄是否可讀取"""
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            full_path = Path(root) / d
            if not os.access(full_path, os.R_OK):
                print(f"❌ 無法讀取目錄：{full_path}")
                return False
    return True


def scan_audio_files(input_path: Path) -> list:
    """掃描支援的音頻文件"""
    audio_files = []
    for root, _, files in os.walk(input_path):
        for f in files:
            file_path = Path(root) / f
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS or is_audio_by_content(file_path):
                audio_files.append(file_path)
    return audio_files


def run_ffprobe(file_path: Path) -> str:
    """呼叫 ffprobe 並取得格式資訊"""
    try:
        result = subprocess.run(
            [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=format_name',
             '-of', 'csv=p=0', str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        return result.stdout.strip().lower()
    except Exception as e:
        print(f"⚠️ ffprobe 執行錯誤：{file_path.name} ({type(e).__name__}: {e})")
        return ""


def is_audio_by_content(file_path: Path) -> bool:
    """使用 filetype 與 ffprobe 檢查是否為音頻"""
    try:
        kind = filetype.guess(file_path)
        if kind and kind.mime.startswith('audio/'):
            return True
        return 'audio' in run_ffprobe(file_path)
    except Exception:
        return False


def ensure_directory(path: Path) -> bool:
    """建立目標資料夾"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.is_dir()
    except Exception as e:
        print(f"❌ 建立目錄失敗：{path} ({e})")
        return False


def convert_single_file(audio_path: Path, output_path: Path) -> bool:
    """轉換單一音頻為 FLAC"""
    flac_file = output_path / f"{audio_path.stem}.flac"

    if flac_file.exists():
        print(f"⚠️ 跳過已存在：{flac_file.relative_to(OUTPUT_DIR)}")
        return False

    try:
        subprocess.run(
            [FFMPEG_PATH, '-y', '-i', str(audio_path),
             '-c:a', 'flac', '-map_metadata', '0',
             '-compression_level', '5', str(flac_file)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        if flac_file.stat().st_size == 0:
            print(f"❌ 空文件刪除：{flac_file.name}")
            flac_file.unlink()
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 轉換失敗：{audio_path.name} (錯誤碼 {e.returncode})")
        return False


def main():
    print(f"📂 輸入目錄：{INPUT_DIR}")
    print(f"📁 輸出目錄：{OUTPUT_DIR}")

    if not INPUT_DIR.is_dir() or not OUTPUT_DIR.is_dir():
        print("❌ 請確認輸入與輸出目錄都存在")
        return

    if not validate_directory_structure(INPUT_DIR):
        print("❌ 目錄結構驗證失敗")
        return

    files = scan_audio_files(INPUT_DIR)
    if not files:
        print("⚠️ 未找到支援的音頻文件")
        return

    print(f"🔍 找到 {len(files)} 個音頻文件，開始轉換")

    success_count = 0
    for i, audio_file in enumerate(files, 1):
        relative_dir = audio_file.relative_to(INPUT_DIR).parent
        output_dir = OUTPUT_DIR / relative_dir
        if not ensure_directory(output_dir):
            continue

        if convert_single_file(audio_file, output_dir):
            success_count += 1
            print(f"[{i}/{len(files)}] ✅ 成功：{audio_file.name}")
        else:
            print(f"[{i}/{len(files)}] ❌ 失敗：{audio_file.name}")

    print(f"\n🎉 完成！成功轉換 {success_count}/{len(files)} 個文件")


if __name__ == "__main__":
    main()
