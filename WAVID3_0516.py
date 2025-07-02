import os
import logging
from pathlib import Path
from mutagen.id3 import (
    ID3, TPE2, TALB, TYER, TPE1, TIT2, APIC,
    TCOM, COMM, TPOS, TCON, TDRC
)
import mutagen
from mutagen.wave import WAVE
import sys


def configure_system_encoding():
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'UTF-8':
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)


configure_system_encoding()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tagging.log', encoding='utf-8-sig'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_directory_structure(work_dir):
    base_path = Path(work_dir)
    validate_directory_structure(base_path)

    for album_artist_dir in base_path.iterdir():
        if not album_artist_dir.is_dir():
            continue

        album_artist = album_artist_dir.name
        logger.info(f"處理專輯藝術家: {album_artist}")

        for sub_dir in album_artist_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            try:
                process_sub_directory(sub_dir, album_artist)
            except Exception as e:
                logger.error(f"子目錄處理失敗: {sub_dir} - {str(e)}")


def validate_directory_structure(base_path):
    required_dirs = [d for d in base_path.iterdir() if d.is_dir() and ' - ' not in d.name]
    if not required_dirs:
        raise ValueError("根目錄缺少專輯藝術家子目錄")


def process_sub_directory(sub_dir, album_artist):
    dir_parts = sub_dir.name.split(' - ')
    if len(dir_parts) != 3:
        logger.warning(f"無效目錄格式: {sub_dir.name}")
        return

    artist, year, album = dir_parts
    validate_year_format(year)

    logger.info(f"解析目錄信息: 藝人={artist}, 年份={year}, 專輯={album}")

    cover_path = find_cover_image(sub_dir)

    for audio_file in sub_dir.glob('*.*'):
        if audio_file.suffix.lower() not in ['.wav', '.mp3', '.flac']:
            continue

        process_audio_file(
            audio_file=audio_file,
            album_artist=album_artist,
            year=year,
            album=album,
            cover_path=cover_path
        )


def validate_year_format(year):
    if not (year.isdigit() and len(year) == 4):
        raise ValueError(f"無效年份格式: {year}")


def find_cover_image(directory):
    image_files = []
    priority_files = []

    for path in directory.iterdir():
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            stem_lower = path.stem.lower()
            if stem_lower in {'cover', '封面'}:
                priority_files.append(path)
            image_files.append(path)

    if priority_files:
        return priority_files[0]
    return image_files[0] if image_files else None


def process_audio_file(audio_file, album_artist, year, album, cover_path):
    try:
        filename_artist, title = parse_artist_title(audio_file.stem)
        logger.info(f"處理文件: {audio_file.name} | 藝人={filename_artist} | 標題={title}")

        audio = init_wav_id3_tag(audio_file)
        process_id3_tags(
            audio=audio,
            album_artist=album_artist,
            filename_artist=filename_artist,
            year=year,
            album=album,
            title=title,
            cover_path=cover_path
        )
        clear_unused_fields(audio)
        process_genre(audio)
        save_wav_changes(audio, audio_file)

    except Exception as e:
        logger.error(f"文件處理失敗: {audio_file} - {str(e)}")


def parse_artist_title(filename):
    if ' - ' in filename:
        parts = filename.split(' - ', 1)
        return parts[0].strip(), parts[1].strip()

    separators = ['—', '_', ' ']
    for sep in separators:
        if sep in filename:
            parts = filename.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    raise ValueError(f"無法解析文件名格式: {filename}")


def init_wav_id3_tag(audio_file):
    try:
        return WAVE(audio_file).tags or ID3()
    except mutagen.MutagenError:
        return ID3()


def process_id3_tags(audio, album_artist, filename_artist, year, album, title, cover_path):
    encoding = 3

    audio.add(TPE2(encoding=encoding, text=album_artist))
    audio.add(TPE1(encoding=encoding, text=filename_artist))
    audio.add(TALB(encoding=encoding, text=album))

    if not any(f for f in audio.keys() if f.startswith(('TDRC', 'TYER'))):
        audio.add(TYER(encoding=encoding, text=year))

    audio.add(TIT2(encoding=encoding, text=title))

    if not any(f for f in audio.keys() if f.startswith('APIC')):
        add_wav_album_cover(audio, cover_path)


def add_wav_album_cover(audio, cover_path):
    if not cover_path:
        return

    try:
        with open(cover_path, 'rb') as f:
            cover_data = f.read()

        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        mime_type = mime_map.get(cover_path.suffix.lower(), 'image/jpeg')

        audio.add(APIC(
            encoding=3,
            mime=mime_type,
            type=3,
            desc=u'Cover',
            data=cover_data
        ))
    except Exception as e:
        logger.warning(f"封面添加失敗: {str(e)}")


def clear_unused_fields(audio):
    """強化版欄位清除"""
    target_frames = {'TCOM', 'COMM', 'TPOS'}
    custom_descs = {'composer', 'comment', 'discnumber'}

    # 標準框架處理
    for frame_id in list(audio.keys()):
        # 刪除目標標準框架
        if frame_id in target_frames:
            del audio[frame_id]
            continue

        # 處理COMM框架的所有變體
        if frame_id.startswith('COMM:'):
            del audio[frame_id]
            continue

        # 處理自定義框架
        if frame_id.startswith('TXXX:'):
            frame = audio[frame_id]
            if frame.desc.lower() in custom_descs:
                del audio[frame_id]

    # 清除ID3v1殘留標籤
    if 'TAG' in audio:
        del audio['TAG']


def process_genre(audio):
    if 'TCON' in audio:
        current_genre = audio['TCON'].text[0]
        if current_genre == 'POP':
            audio['TCON'].text = ['Pop']


def save_wav_changes(audio, audio_file):
    try:
        wave_file = WAVE(audio_file)
        wave_file.tags = audio
        wave_file.save()
        logger.info(f"標籤更新成功: {audio_file}")
    except Exception as e:
        logger.error(f"保存失敗: {audio_file} - {str(e)}")


if __name__ == '__main__':
    work_directory = r"D:\Music\Pascal\Work"

    try:
        if not os.path.exists(work_directory):
            raise FileNotFoundError(f"工作目錄不存在: {work_directory}")

        logger.info(f"開始處理: {work_directory}")
        process_directory_structure(work_directory)
        logger.info("處理完成")
    except Exception as e:
        logger.critical(f"程序終止: {str(e)}")
        sys.exit(1)