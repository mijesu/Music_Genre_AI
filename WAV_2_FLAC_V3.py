import os
import subprocess
import filetype
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 設定區 ===
INPUT_DIR = Path(r"D:\Music\Pascal\TBD_FLAC") # M:\(My Song)\Name\ ; D:\Music\Pascal\TBD_FLAC ; T:\FLAC
OUTPUT_FLAC_DIR = Path(r"D:\Music\FLAC")
OUTPUT_WAV_DIR = Path(r"D:\Music\WAV")
FFMPEG_PATH = r"D:\Music\PS_Code\ID3_CLI\FFmpeg\ffmpeg.exe"
FFPROBE_PATH = r"D:\Music\PS_Code\ID3_CLI\FFmpeg\ffprobe.exe"

SUPPORTED_EXTENSIONS = (
    '.mp3', '.wav', '.aac', '.flac', '.ogg',
    '.m4a', '.wma', '.aiff', '.alac', '.opus'
)

def validate_directory_structure(path: Path) -> bool:
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if not os.access(Path(root) / d, os.R_OK):
                print(f"❌ 無法讀取目錄：{Path(root) / d}")
                return False
    return True

def scan_audio_files(input_path: Path) -> list:
    audio_files = []
    for root, _, files in os.walk(input_path):
        for f in files:
            file_path = Path(root) / f
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS or is_audio_by_content(file_path):
                audio_files.append(file_path)
    return audio_files

def run_ffprobe(file_path: Path) -> str:
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
        print(f"⚠️ ffprobe 錯誤：{file_path.name} ({e})")
        return ""

def is_audio_by_content(file_path: Path) -> bool:
    try:
        kind = filetype.guess(file_path)
        if kind and kind.mime.startswith('audio/'):
            return True
        return 'audio' in run_ffprobe(file_path)
    except Exception:
        return False

def ensure_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.is_dir()
    except Exception as e:
        print(f"❌ 建立目錄失敗：{path} ({e})")
        return False

def convert_to_flac(audio_path: Path, output_path: Path) -> bool:
    flac_file = output_path / f"{audio_path.stem}.flac"
    if flac_file.exists():
        print(f"⚠️ 跳過已存在：{flac_file.relative_to(OUTPUT_FLAC_DIR)}")
        return False
    try:
        subprocess.run([
            FFMPEG_PATH, '-y', '-i', str(audio_path),
            '-c:a', 'flac', '-map_metadata', '0',
            '-compression_level', '5', str(flac_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if flac_file.stat().st_size == 0:
            flac_file.unlink()
            return False
        return True
    except subprocess.CalledProcessError:
        return False

def convert_flac_to_wav(flac_path: Path, output_path: Path) -> bool:
    wav_file = output_path / f"{flac_path.stem}.wav"
    try:
        subprocess.run([
            FFMPEG_PATH, '-y', '-i', str(flac_path),
            '-c:a', 'pcm_s16le', '-map_metadata', '0', str(wav_file)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if wav_file.stat().st_size == 0:
            wav_file.unlink()
            return False
        return True
    except subprocess.CalledProcessError:
        return False

def process_file(audio_file: Path) -> str:
    relative_dir = audio_file.relative_to(INPUT_DIR).parent
    if audio_file.suffix.lower() == '.flac':
        output_dir = OUTPUT_WAV_DIR / relative_dir
        if ensure_directory(output_dir) and convert_flac_to_wav(audio_file, output_dir):
            return f"✅ FLAC 轉 WAV：{audio_file.name}"
        return f"❌ FLAC 轉 WAV 失敗：{audio_file.name}"
    else:
        output_dir = OUTPUT_FLAC_DIR / relative_dir
        if ensure_directory(output_dir) and convert_to_flac(audio_file, output_dir):
            return f"✅ 音訊轉 FLAC：{audio_file.name}"
        return f"❌ 音訊轉 FLAC 失敗：{audio_file.name}"

def main():
    print(f"📂 掃描來源：{INPUT_DIR}")
    if not INPUT_DIR.is_dir():
        print("❌ 輸入目錄不存在")
        return

    if not validate_directory_structure(INPUT_DIR):
        print("❌ 目錄驗證失敗")
        return

    files = scan_audio_files(INPUT_DIR)
    print(f"🔍 共發現 {len(files)} 個音頻文件，啟動轉換...")

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(futures):
            print(future.result())

    print("\n🎉 所有轉換作業完成！")

if __name__ == "__main__":
    main()
