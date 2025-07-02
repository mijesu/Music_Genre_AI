import os
import sys
import shutil
import subprocess
import re
import patoolib
import chardet
from pathlib import Path
from mutagen.wave import WAVE
from mutagen.id3 import TIT2, TPE1, TALB, TRCK
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed

# === 設定 ===
FFMPEG_PATH = r"D:\Music\PS_Code\ID3_CLI\FFmpeg\ffmpeg.exe"
WORK_DIR = Path(r"D:\Music\Pascal\Work")
SUPPORTED_ARCHIVES = ('.zip', '.rar', '.7z', '.tar.gz', '.tar', '.gz')
converter = OpenCC('s2t')

def convert_folder_to_traditional(directory: Path):
    for path in sorted(directory.rglob('*'), key=lambda p: -len(str(p))):
        new_name = converter.convert(path.name)
        if new_name != path.name:
            new_path = path.parent / new_name
            try:
                path.rename(new_path)
            except Exception:
                continue

def is_utf8_bom(file_path: Path) -> bool:
    with open(file_path, 'rb') as f:
        return f.read(3) == b'\xef\xbb\xbf'

def detect_and_convert_cue(cue_path: Path):
    if is_utf8_bom(cue_path):
        with open(cue_path, 'r', encoding='utf-8-sig') as f:
            return f.read(), 'utf-8-sig'
    with open(cue_path, 'rb') as f:
        raw = f.read()
        enc = chardet.detect(raw)['encoding'] or 'utf-8'
        try:
            decoded = raw.decode(enc)
        except UnicodeDecodeError:
            decoded = raw.decode('gb18030', errors='replace')
    return converter.convert(decoded), enc

def write_utf8_bom(file_path: Path, content: str):
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        f.write(content)
def parse_cue(content: str):
    lines = content.splitlines()
    file_line = next((l for l in lines if l.upper().startswith('FILE')), None)
    match = re.match(r'FILE\s+"?(.*?)"?\s+WAVE', file_line, re.IGNORECASE) if file_line else None
    audio_file = match.group(1).strip() if match else None
    performer = ""
    album = ""
    date = ""
    tracks = []
    current = {}
    for line in lines:
        line = line.strip()
        if line.upper().startswith("REM DATE") and not date:
            date = line.partition("DATE")[2].strip()
        elif line.upper().startswith("PERFORMER") and not performer:
            performer = line.partition(" ")[2].strip('" ')
        elif line.upper().startswith("TITLE") and not album:
            album = line.partition(" ")[2].strip('" ')
        elif line.upper().startswith("TRACK"):
            if current:
                tracks.append(current)
            current = {'number': line.split()[1], 'title': '', 'index': ''}
        elif line.upper().startswith("TITLE") and current and not current['title']:
            current['title'] = line.partition(" ")[2].strip('" ')
        elif line.upper().startswith("INDEX 01") and current:
            current['index'] = line.split()[2]
    if current:
        tracks.append(current)
    return {
        'audio_file': audio_file,
        'tracks': tracks,
        'performer': performer,
        'album': album,
        'date': date
    }

def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [FFMPEG_PATH, '-i', str(audio_path)],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        text=True, errors='ignore'
    )
    match = re.search(r'Duration: (\d+):(\d+):([\d.]+)', result.stderr)
    if not match:
        return 0.0
    h, m, s = map(float, match.groups())
    return h * 3600 + m * 60 + s

def cue_time_to_seconds(timestamp: str) -> float:
    mm, ss, ff = map(int, timestamp.split(":"))
    return mm * 60 + ss + ff / 75

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '', name).strip()

def get_unique_file_path(directory: Path, filename: str) -> Path:
    base = directory / filename
    if not base.exists():
        return base
    name = base.stem
    ext = base.suffix
    count = 1
    while True:
        new_name = f"{name} ({count}){ext}"
        candidate = directory / new_name
        if not candidate.exists():
            return candidate
        count += 1
def write_tags(wav_file: Path, title: str, artist: str, album: str, track_no: str):
    try:
        audio = WAVE(wav_file)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.add(TIT2(encoding=3, text=title))
        audio.tags.add(TPE1(encoding=3, text=artist))
        audio.tags.add(TALB(encoding=3, text=album))
        audio.tags.add(TRCK(encoding=3, text=track_no))
        audio.save()
    except Exception as e:
        print(f"⚠️ TAG 寫入錯誤: {wav_file.name}: {e}")

def ffmpeg_split_and_tag(cue_info, cue_path: Path):
    audio_file = cue_info['audio_file']
    full_audio_path = cue_path.parent / audio_file
    if not full_audio_path.exists():
        wav_candidates = list(cue_path.parent.glob("*.wav"))
        if not wav_candidates:
            print(f"❌ 無可用音訊: {cue_path}")
            return
        full_audio_path = max(wav_candidates, key=lambda p: p.stat().st_size)
        print(f"⚠️ 找不到 {audio_file}，改用最大 WAV：{full_audio_path.name}")
    total_duration = get_audio_duration(full_audio_path)
    for i, track in enumerate(cue_info['tracks']):
        start = cue_time_to_seconds(track['index'])
        end = cue_time_to_seconds(cue_info['tracks'][i+1]['index']) if i+1 < len(cue_info['tracks']) else total_duration
        duration = end - start
        title = sanitize_filename(track['title']) if track['title'] else f"Track {track['number']}"
        out_path = get_unique_file_path(cue_path.parent, f"{cue_info['performer']} - {title}.wav")
        cmd = [
            FFMPEG_PATH,
            '-ss', str(start),
            '-t', f"{duration:.3f}",
            '-i', str(full_audio_path),
            '-acodec', 'pcm_s16le',
            '-y', str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        write_tags(out_path, title, cue_info['performer'], cue_info['album'], track['number'])
        print(f"🎧 輸出: {out_path.name}")

def process_cue_file(cue_path: Path):
    content, encoding = detect_and_convert_cue(cue_path)
    print(f"🔍 處理 CUE: {cue_path.name}（編碼: {encoding}）")
    if not is_utf8_bom(cue_path):
        backup = cue_path.with_suffix('.bak')
        if backup.exists(): backup.unlink()
        shutil.move(cue_path, backup)
        write_utf8_bom(cue_path, content)
    cue_info = parse_cue(content)
    if not cue_info or not cue_info['tracks']:
        print("❌ cue 結構異常，略過")
        return

    # ➕ 目錄重命名
    performer = cue_info.get("performer", "").strip()
    title = cue_info.get("album", "").strip()
    date = cue_info.get("date", "").strip()
    if performer and title and date:
        current_folder = cue_path.parent
        new_name = f"{performer} - {date} - {title}"
        new_path = current_folder.parent / new_name
        if new_path != current_folder:
            if new_path.exists():
                counter = 1
                while (current_folder.parent / f"{new_name} ({counter})").exists():
                    counter += 1
                new_path = current_folder.parent / f"{new_name} ({counter})"
            try:
                cue_path = new_path / cue_path.name
                shutil.move(str(current_folder), str(new_path))
                print(f"📁 已重新命名資料夾: {new_path.name}")
            except Exception as e:
                print(f"⚠️ 資料夾重命名失敗: {e}")
    ffmpeg_split_and_tag(cue_info, cue_path)

def scan_and_process(directory: Path):
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir():
            convert_folder_to_traditional(subdir)
    cue_files = [Path(root) / f for root, _, files in os.walk(directory) for f in files if f.lower().endswith('.cue')]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_cue_file, cue) for cue in cue_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ 執行緒錯誤: {e}")

def decompress_archives(root_dir: Path):
    for root, _, files in os.walk(root_dir):
        for file in files:
            fpath = Path(root) / file
            if fpath.suffix.lower() not in SUPPORTED_ARCHIVES:
                continue
            target = fpath.with_suffix('')
            temp_dir = fpath.parent / "_temp_"
            try:
                temp_dir.mkdir(exist_ok=True)
                patoolib.extract_archive(str(fpath), outdir=str(temp_dir))
                entries = list(temp_dir.iterdir())
                if len(entries) == 1 and entries[0].is_dir():
                    shutil.move(str(entries[0]), str(target))
                    temp_dir.rmdir()
                else:
                    shutil.move(str(temp_dir), str(target))
                print(f"✅ 解壓縮完成: {fpath.name}")
            except Exception as e:
                print(f"❌ 解壓縮失敗: {fpath.name} - {e}")
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    if not WORK_DIR.is_dir():
        print(f"❌ 無效目錄: {WORK_DIR}")
        sys.exit(1)
    decompress_archives(WORK_DIR)
    scan_and_process(WORK_DIR)
    print("\n✅ 所有作業完成")
