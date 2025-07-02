import os
import logging
from pathlib import Path
import sys
from mutagen.id3 import (
    ID3, TPE2, TALB, TYER, TPE1, TIT2, APIC,
    TCOM, COMM, TPOS, TCON, TDRC
)
from mutagen.flac import FLAC, Picture
from mutagen.wave import WAVE
import mutagen


# 系统编码配置
def configure_system_encoding():
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'UTF-8':
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)


configure_system_encoding()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tagging.log', encoding='utf-8-sig'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioFileProcessor:
    """音频文件处理基类"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.audio = self._load_file()

    def _load_file(self):
        raise NotImplementedError

    def set_metadata(self, album_artist, artist, album, year, title):
        raise NotImplementedError

    def set_cover(self, cover_path):
        raise NotImplementedError

    def clean_metadata(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class WavProcessor(AudioFileProcessor):
    """WAV 文件处理器"""

    def _load_file(self):
        try:
            audio = WAVE(self.file_path)
            if not audio.tags:
                audio.add_tags()
            return audio
        except mutagen.MutagenError:
            return WAVE(self.file_path)

    def set_metadata(self, album_artist, artist, album, year, title):
        encoding = 3
        self.audio.tags.add(TPE2(encoding=encoding, text=album_artist))
        self.audio.tags.add(TPE1(encoding=encoding, text=artist))
        self.audio.tags.add(TALB(encoding=encoding, text=album))
        self.audio.tags.add(TYER(encoding=encoding, text=year))
        self.audio.tags.add(TIT2(encoding=encoding, text=title))

    def set_cover(self, cover_path):
        if not cover_path or not cover_path.exists():
            return

        try:
            with open(cover_path, 'rb') as f:
                cover_data = f.read()

            mime_type = 'image/jpeg' if cover_path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
            self.audio.tags.add(APIC(
                encoding=3,
                mime=mime_type,
                type=3,
                desc=u'Cover',
                data=cover_data
            ))
        except Exception as e:
            logger.warning(f"WAV封面添加失败: {str(e)}")

    def clean_metadata(self):
        remove_frames = {'TCOM', 'COMM', 'TPOS', 'TDRC'}
        for frame_id in list(self.audio.tags.keys()):
            if frame_id in remove_frames or frame_id.startswith(('COMM:', 'TXXX:')):
                del self.audio.tags[frame_id]

    def save(self):
        self.audio.save()


class FlacProcessor(AudioFileProcessor):
    """FLAC 文件处理器"""

    def _load_file(self):
        return FLAC(self.file_path)

    def set_metadata(self, album_artist, artist, album, year, title):
        self.audio['ALBUMARTIST'] = [album_artist]
        self.audio['ARTIST'] = [artist]
        self.audio['ALBUM'] = [album]
        self.audio['DATE'] = [year]
        self.audio['TITLE'] = [title]

    def set_cover(self, cover_path):
        if not cover_path or not cover_path.exists():
            return

        try:
            pic = Picture()
            pic.type = 3
            pic.mime = 'image/jpeg' if cover_path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
            with open(cover_path, 'rb') as f:
                pic.data = f.read()
            self.audio.add_picture(pic)
        except Exception as e:
            logger.warning(f"FLAC封面添加失败: {str(e)}")

    def clean_metadata(self):
        remove_fields = ['GENRE', 'COMMENT', 'COMPOSER']
        for field in remove_fields:
            if field in self.audio:
                del self.audio[field]

    def save(self):
        self.audio.save()


def process_directory_structure(work_dir):
    base_path = Path(work_dir)
    validate_directory_structure(base_path)

    for album_artist_dir in base_path.iterdir():
        if not album_artist_dir.is_dir():
            continue

        logger.info(f"处理专辑艺术家: {album_artist_dir.name}")

        for sub_dir in album_artist_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            try:
                process_sub_directory(sub_dir, album_artist_dir.name)
            except Exception as e:
                logger.error(f"子目录处理失败: {sub_dir} - {str(e)}")


def validate_directory_structure(base_path):
    required_dirs = [d for d in base_path.iterdir() if d.is_dir() and ' - ' not in d.name]
    if not required_dirs:
        raise ValueError("根目录缺少专辑艺术家子目录")


def process_sub_directory(sub_dir, album_artist):
    dir_parts = sub_dir.name.split(' - ')
    if len(dir_parts) != 3:
        logger.warning(f"无效目录格式: {sub_dir.name}")
        return

    artist, year, album = dir_parts
    validate_year_format(year)

    logger.info(f"解析目录信息: 艺人={artist}, 年份={year}, 专辑={album}")

    cover_path = find_cover_image(sub_dir)

    for audio_file in sub_dir.glob('*.*'):
        if audio_file.suffix.lower() not in ['.wav', '.flac']:
            continue

        process_audio_file(
            audio_file=audio_file,
            album_artist=album_artist,
            artist=artist,
            year=year,
            album=album,
            cover_path=cover_path
        )


def validate_year_format(year):
    if not (year.isdigit() and len(year) == 4):
        raise ValueError(f"无效年份格式: {year}")


def find_cover_image(directory):
    image_files = []
    priority_files = []

    for path in directory.iterdir():
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            stem_lower = path.stem.lower()
            if stem_lower in {'cover', '封面'}:
                priority_files.append(path)
            image_files.append(path)

    return priority_files[0] if priority_files else image_files[0] if image_files else None


def process_audio_file(audio_file, album_artist, artist, year, album, cover_path):
    try:
        # 解析文件名元数据
        filename_artist, title = parse_artist_title(audio_file.stem)
        logger.info(f"处理文件: {audio_file.name} | 艺人={filename_artist} | 标题={title}")

        # 创建处理器实例
        suffix = audio_file.suffix.lower()
        if suffix == '.wav':
            processor = WavProcessor(audio_file)
        elif suffix == '.flac':
            processor = FlacProcessor(audio_file)
        else:
            logger.warning(f"跳过不支持格式: {audio_file}")
            return

        # 设置元数据
        processor.set_metadata(
            album_artist=album_artist,
            artist=filename_artist,
            album=album,
            year=year,
            title=title
        )

        # 处理封面
        processor.set_cover(cover_path)

        # 清理元数据
        processor.clean_metadata()

        # 保存修改
        processor.save()
        logger.info(f"元数据更新成功: {audio_file}")

    except Exception as e:
        logger.error(f"文件处理失败: {audio_file} - {str(e)}")


def parse_artist_title(filename):
    separators = [' - ', '—', '_', ' ']
    for sep in separators:
        if sep in filename:
            parts = filename.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    raise ValueError(f"无法解析文件名格式: {filename}")


if __name__ == '__main__':
    work_directory = r"D:\Music\Pascal\Work"

    try:
        if not os.path.exists(work_directory):
            raise FileNotFoundError(f"工作目录不存在: {work_directory}")

        logger.info(f"开始处理: {work_directory}")
        process_directory_structure(work_directory)
        logger.info("处理完成")
    except Exception as e:
        logger.critical(f"程序终止: {str(e)}")
        sys.exit(1)