import os
import json
import subprocess
import threading
import time
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import cv2.aruco as aruco

# ============================================
# 0. 공통 설정
# ============================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = "gpt-4o-mini-tts"   # 음성 생성 모델
TEXT_MODEL = "gpt-4o-mini"      # 대사 생성 모델

ALLOWED_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
}

# 현재 상태 (배경 / 캐릭터)
CURRENT_BG_BOOK_CODE = None      # 현재 배경이 된 책 코드
CURRENT_BG_INFO = None           # 현재 배경 정보
CURRENT_CHA1_INFO = None         # 현재 cha1 캐릭터 dict
CURRENT_CHA2_INFO = None         # 현재 cha2 캐릭터 dict

# 비디오 플레이어 (스레드 기반)
class VideoPlayer:
    """OpenCV 기반 비디오 플레이어 (별도 스레드에서 무한 루프 재생, 오디오 포함)"""
    
    def __init__(self):
        self.current_video_path = None
        self.video_cap = None
        self.next_video_path = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.audio_process = None  # 오디오 재생 프로세스
        self.fade_alpha = 1.0  # 페이드 알파 값 (0.0 ~ 1.0)
        self.is_fading = False  # 페이드 중인지 여부
        self.fade_duration = 0.5  # 페이드 지속 시간 (초)
        self.fade_start_time = None
    
    def _play_loop(self):
        """비디오 재생 루프 (별도 스레드에서 실행)"""
        while self.running:
            with self.lock:
                # 페이드 효과 처리
                if self.is_fading and self.fade_start_time:
                    elapsed = time.time() - self.fade_start_time
                    if elapsed < self.fade_duration:
                        # 페이드아웃: 1.0 -> 0.0
                        self.fade_alpha = 1.0 - (elapsed / self.fade_duration)
                    elif elapsed < self.fade_duration * 2:
                        # 비디오 전환
                        if self.next_video_path and self.next_video_path != self.current_video_path:
                            self._switch_video_internal(self.next_video_path)
                            self.next_video_path = None
                        # 페이드인: 0.0 -> 1.0
                        self.fade_alpha = (elapsed - self.fade_duration) / self.fade_duration
                    else:
                        # 페이드 완료
                        self.is_fading = False
                        self.fade_alpha = 1.0
                        self.fade_start_time = None
                
                if self.video_cap is None or not self.video_cap.isOpened():
                    time.sleep(0.01)
                    continue
                
                ret, frame = self.video_cap.read()
                if not ret:
                    # 비디오 끝나면 처음으로 돌아가기 (무한 루프)
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_cap.read()
                
                if ret:
                    # 페이드 효과 적용
                    if self.is_fading and self.fade_alpha < 1.0:
                        # 검은 화면과 블렌딩
                        black_frame = frame.copy()
                        black_frame.fill(0)
                        frame = cv2.addWeighted(frame, self.fade_alpha, black_frame, 1.0 - self.fade_alpha, 0)
                    self.frame = frame
            
            # 프레임 레이트 맞추기 (약 30fps)
            time.sleep(0.033)
    
    def _switch_video_internal(self, video_path: str):
        """내부 비디오 전환 (페이드 중에 호출)"""
        # 기존 비디오 해제
        if self.video_cap:
            self.video_cap.release()
        
        # 새 비디오 열기
        self.current_video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            print(f"❌ 비디오를 열 수 없음: {video_path}")
            self.video_cap = None
        else:
            print(f"🎬 비디오 전환: {video_path}")
            
            # 오디오 재생 시작 (무한 루프)
            self._start_audio(video_path)
    
    def _start_audio(self, video_path: str):
        """비디오의 오디오를 무한 루프로 재생"""
        # 기존 오디오 프로세스 종료
        if self.audio_process:
            # 딕셔너리인 경우 (macOS 스레드)
            if isinstance(self.audio_process, dict):
                self.audio_process["ref"]["running"] = False
            # 프로세스인 경우
            elif hasattr(self.audio_process, "terminate"):
                try:
                    self.audio_process.terminate()
                    self.audio_process.wait(timeout=1.0)
                except:
                    try:
                        self.audio_process.kill()
                    except:
                        pass
            self.audio_process = None
        
        # ffmpeg를 사용하여 비디오의 오디오를 무한 루프로 재생
        import platform
        is_macos = platform.system() == "Darwin"
        
        try:
            if is_macos:
                # macOS: ffmpeg로 오디오를 추출하고 afplay로 재생 (무한 루프)
                # 별도 스레드에서 무한 루프로 재생
                audio_thread_ref = {"running": True}
                
                def play_audio_loop():
                    while audio_thread_ref["running"]:
                        try:
                            # ffmpeg로 오디오를 무한 루프로 재생 (stream_loop 사용)
                            proc = subprocess.Popen(
                                ["ffmpeg", "-re", "-stream_loop", "-1", "-i", video_path, 
                                 "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-f", "wav", "-"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                bufsize=0
                            )
                            # afplay로 재생 (stdin에서 읽기)
                            afplay_proc = subprocess.Popen(
                                ["afplay", "-"],
                                stdin=proc.stdout,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            afplay_proc.wait()
                            proc.wait()
                        except Exception as e:
                            print(f"⚠️ 오디오 재생 오류: {e}")
                            break
                
                # 오디오 재생 스레드 시작
                audio_thread = threading.Thread(target=play_audio_loop, daemon=True)
                audio_thread.start()
                self.audio_process = {"thread": audio_thread, "ref": audio_thread_ref}
            else:
                # Linux: ffplay 사용
                self.audio_process = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loop", "0", "-loglevel", "quiet", video_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        except FileNotFoundError:
            print("⚠️ ffmpeg/ffplay를 찾을 수 없습니다. 오디오는 재생되지 않습니다.")
            if is_macos:
                print("   macOS에서는 'brew install ffmpeg'로 설치하세요.")
            else:
                print("   Linux에서는 'sudo apt-get install ffmpeg' 또는 'sudo yum install ffmpeg'로 설치하세요.")
    
    def start(self):
        """플레이어 시작"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """플레이어 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            self.frame = None
        # 오디오 프로세스/스레드 종료
        if self.audio_process:
            # 딕셔너리인 경우 (macOS 스레드)
            if isinstance(self.audio_process, dict):
                self.audio_process["ref"]["running"] = False
            # 프로세스인 경우 (Linux)
            elif hasattr(self.audio_process, "terminate"):
                try:
                    self.audio_process.terminate()
                    self.audio_process.wait(timeout=1.0)
                except:
                    try:
                        self.audio_process.kill()
                    except:
                        pass
            self.audio_process = None
    
    def set_video(self, video_path: str):
        """비디오 파일 변경 (페이드 효과와 함께 부드러운 전환)"""
        with self.lock:
            if self.current_video_path is None:
                # 첫 번째 비디오는 페이드 없이 바로 시작
                self._switch_video_internal(video_path)
            else:
                # 다음 비디오로 전환 (페이드 효과)
                self.next_video_path = video_path
                self.is_fading = True
                self.fade_start_time = time.time()
    
    def get_frame(self):
        """현재 프레임 가져오기"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

# 전역 비디오 플레이어 인스턴스
VIDEO_PLAYER = VideoPlayer()

# 배경 비디오 설정
BG_VIDEO_DIR = "bg_video"
BOOK_TO_VIDEO = {
    "BJBJ": "10_BJBJ_matchedSize.mov",
    "PSJ": "11_PSJ_matchedSize.mov",
    "DGJ": "13_DGJ_matchedSize.mov",
    "HBJ": "17_HBJ_matchedSize.mov",
    "JWCJ": "19_JWCJ_matchedSize.mov",
    "KWJ": "3_KWJ_matchedSize.mov",
    "OGJJ": "5_OGJJ_matchedSize.mov",
    "JHHRJ": "6_JHHRJ_matchedSize.mov",
    "SCJ": "7_SCJ_matchedSize.mov",
}


# ============================================
# 1. 설정 파일 로드
# ============================================
def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일을 찾을 수 없습니다!")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CHARACTERS = load_json("characters.json")
BACKGROUNDS = load_json("backgrounds.json")

# 책 코드 → cha1 / cha2 역할 키 매핑
ROLE_MAP = {
    "SCJ": {"cha1": "simcheong",    "cha2": "simbongsa"},
    "HBJ": {"cha1": "heungbu",      "cha2": "nolbu"},
    "BJBJ": {"cha1": "turtle",      "cha2": "rabbit"},
    "OGJJ": {"cha1": "onggojip",    "cha2": "onggojip"},
    "JWCJ": {"cha1": "jeonwoochi",  "cha2": "jeonwoochi"},
    "JHHRJ": {"cha1": "sister_older",    "cha2": "ghost"},
    "PSJ": {"cha1": "ugly",         "cha2": "pretty"},
    "DGJ": {"cha1": "toad",         "cha2": "fox"},
    "KWJ": {"cha1": "kimwon",       "cha2": "monster"}
}

# ============================================
# ArUco 마커 설정
# ============================================
ARUCO_DICTIONARY = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
ARUCO_MARKER_SIZE = 200  # pixels

# 마커 ID → 책 코드 매핑
MARKER_TO_BOOK = {
    1: "KWJ",      # 03_KWJ
    2: "PSJ",      # 11_PSJ
    6: "DGJ",      # 13_DGJ
    7: "JHHRJ",    # 06_JHHRJ
    8: "JWCJ",     # 19_JWCJ
    9: "HBJ",      # 17_HBJ
    10: "OGJJ",    # 05_OGJJ
    11: "SCJ",     # 07_SCJ
    12: "BJBJ",    # 10_BJBJ
}

# 마커 ID → 파일명 매핑
MARKER_NAMES = {
    1: "03_KWJ",
    2: "11_PSJ",
    6: "13_DGJ",
    7: "06_JHHRJ",
    8: "19_JWCJ",
    9: "17_HBJ",
    10: "05_OGJJ",
    11: "07_SCJ",
    12: "10_BJBJ",
}


def generate_aruco_markers(output_dir: str = "markers"):
    """
    ArUco 마커 이미지들을 생성하여 지정된 폴더에 저장합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for marker_id, name in MARKER_NAMES.items():
        marker_image = aruco.generateImageMarker(ARUCO_DICTIONARY, marker_id, ARUCO_MARKER_SIZE)
        filename = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(filename, marker_image)
        print(f"✅ Saved ArUco marker: {filename}")
    
    print(f"\n🎯 총 {len(MARKER_NAMES)}개의 ArUco 마커가 '{output_dir}' 폴더에 저장되었습니다.")


def get_book_code_from_marker(marker_id: int) -> str | None:
    """
    ArUco 마커 ID로부터 책 코드를 반환합니다.
    """
    return MARKER_TO_BOOK.get(marker_id)

# ============================================
# 2. Background 관련
# ============================================
def get_background(book_code: str):
    return BACKGROUNDS.get(book_code)

# 배경 음악은 이제 비디오에 포함된 오디오를 사용하므로 이 함수는 사용하지 않음
# def play_background_music_if_exists(bg_info: dict):
#     pass


def stop_background_video():
    """현재 재생 중인 배경 비디오를 중지합니다."""
    VIDEO_PLAYER.stop()
    print("🎬 배경 비디오 중지됨")


def play_background_video(book_code: str):
    """
    책 코드에 해당하는 배경 비디오를 무한 루프로 재생합니다.
    같은 윈도우에서 부드럽게 전환됩니다.
    """
    # 비디오 파일 경로 확인
    video_file = BOOK_TO_VIDEO.get(book_code)
    if video_file is None:
        print(f"🎬 '{book_code}'에 해당하는 배경 비디오가 없습니다.")
        return
    
    video_path = os.path.join(BG_VIDEO_DIR, video_file)
    if not os.path.exists(video_path):
        print(f"🎬 비디오 파일을 찾을 수 없음: {video_path}")
        return
    
    # VideoPlayer를 통해 비디오 전환 (같은 윈도우에서 부드럽게)
    VIDEO_PLAYER.set_video(video_path)

def get_interaction_profile(bg_info: dict) -> dict:
    """
    backgrounds.json 안에 미리 정의해 둔
    interaction_label / interaction_summary / interaction_emotions를 가져온다.
    interaction_emotions는 10가지 감정 옵션 리스트로, LLM이 캐릭터 성격에 맞게 선택한다.
    """
    if bg_info is None:
        return {
            "label": "neutral",
            "summary": "A neutral situation with no special context",
            "emotion_options": ["mild curiosity", "calm observation", "quiet interest"]
        }

    # interaction_emotions는 이제 리스트
    emotions_data = bg_info.get("interaction_emotions", [])
    if isinstance(emotions_data, list):
        emotion_options = emotions_data
    else:
        # 혹시 문자열이면 리스트로 변환
        emotion_options = [emotions_data]

    return {
        "label": bg_info.get("interaction_label", "neutral"),
        "summary": bg_info.get(
            "interaction_summary",
            f"A scene involving '{bg_info.get('interaction', '')}'"
        ),
        "emotion_options": emotion_options
    }

# ============================================
# 3. Character 관련
# ============================================
def build_character(book_code: str, role_key: str) -> dict:
    data = CHARACTERS[book_code][role_key]

    gender = data["gender"]
    age = data["age"]
    base_desc = data["base_personality"]
    raw_voice = data.get("voice", "alloy")
    speed = data.get("speed", 1.0)

    voice = raw_voice if raw_voice in ALLOWED_VOICES else "alloy"

    personality = base_desc

    return {
        "book_code": book_code,
        "role_key": role_key,
        "gender": gender,
        "age": age,
        "voice": voice,
        "personality": personality,
        "speed": speed,
    }

def build_sisters_pair() -> tuple[dict, dict]:
    older = build_character("JHHRJ", "sister_older")
    younger = build_character("JHHRJ", "sister_younger")
    return older, younger

# ============================================
# 4. 텍스트 LLM으로 대사 생성
# ============================================
def _clean_line(text: str) -> str:
    if not text:
        return ""
    line = text.strip().splitlines()[0]
    line = line.strip().strip("「」\"'“”‘’")
    return line


def generate_action_line(character: dict, bg_info: dict) -> str:
    """
    배경/인터랙션을 보고 캐릭터가 그 행동을 하기 직전에 하는 한 마디.
    → 최대한 짧고 구어체, 사람 말처럼.
    """
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    # 캐릭터의 speech_patterns 가져오기
    char_data = CHARACTERS.get(character['book_code'], {}).get(character['role_key'], {})
    speech_patterns = char_data.get('speech_patterns', {})
    speaking_style = speech_patterns.get('speaking_style', '')

    system = (
        "당신은 한국 옛이야기 속 등장인물이 실제로 말하는 대사를 쓰는 작가입니다. "
        "대본이나 나레이션이 아니라, 사람이 입으로 툭 튀어나오게 말하는 한국어 구어체를 만드세요."
    )

    user = f"""
배경 장소: {place}
배경 인터랙션: {action}

장면 분위기:
- 요약: {profile['summary']}
- 가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

캐릭터 설정(영어): {character['personality']}
캐릭터 정보: {character['age']}살 {character['gender']}
캐릭터 말투 스타일: {speaking_style}

상황:
- 이 캐릭터가 지금 '{action}'을(를) 하기 직전입니다.
- 위 감정 옵션 중 이 캐릭터의 성격에 가장 어울리는 감정을 선택하고, 그 감정을 담아 짧게 한 마디를 합니다.

말투 규칙:
- 문어체(예: '~것이다', '~합니다')를 절대 쓰지 마세요.
- 자연스러운 구어체만 쓰세요. (예: '~거야', '~하는 건가?', '~해볼까', '~하네요' 등)
- 캐릭터의 말투 스타일을 정확히 따르세요. 특히 '~이기야' 같은 비문법적 표현을 쓰지 말고 '~이지' 같은 올바른 표현을 사용하세요.
- 너무 길게 설명하지 말고, 1~2초 안에 말할 수 있을 정도의 짧은 한 문장으로.
- 느낌표나 물음표는 써도 되지만, 문장은 하나만.
- 따옴표( ", 『 』 등)는 쓰지 마세요.

출력:
- 조건을 지키는 한국어 한 문장만 출력하세요.
"""

    resp = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_output_tokens=50,
        temperature=0.7  # 너무 튀지 않게 약간 낮춤
    )
    return _clean_line(resp.output_text)


def generate_dialogue_lines(char_a: dict, char_b: dict, bg_info: dict) -> tuple[str, str]:
    """
    같은 배경/인터랙션에서 char_a가 먼저 한 마디,
    char_b가 자연스럽게 이어서 한 마디.
    → 둘 다 짧고 구어체.
    Avoid any narration or book-style phrases. The line must sound like spontaneous spoken Korean, not a written script.
    Add small hesitations (예: '아...', '음...') when appropriate, only if it fits the character.

    """
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    # A의 첫 마디
    char_a_data = CHARACTERS.get(char_a['book_code'], {}).get(char_a['role_key'], {})
    char_a_speech = char_a_data.get('speech_patterns', {})
    char_a_style = char_a_speech.get('speaking_style', '')
    
    system_a = (
        "당신은 한국 옛이야기 속 등장인물이 실제로 말하는 대사를 쓰는 작가입니다. "
        "첫 번째 인물이 툭 내뱉는 짧은 한 마디를 만드세요."
    )
    user_a = f"""
배경 장소: {place}
배경 인터랙션: {action}

장면 분위기:
- 요약: {profile['summary']}
- 가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

첫 번째 인물 설정(영어): {char_a['personality']}
첫 번째 인물 정보: {char_a['age']}살 {char_a['gender']}
첫 번째 인물 말투 스타일: {char_a_style}

상황:
- 첫 번째 인물이 '{action}' 장면 속에서 위 감정 중 자신의 성격에 맞는 것을 느끼며 짧게 말합니다.
- 배경 장소와 인터랙션을 고려하여 자연스럽고 맥락에 맞는 대사를 생성하세요.
- 예를 들어, 토끼 같은 장난꾸러기 캐릭터는 '고백할 기회' 같은 이상한 표현을 쓰지 말고, 
  현재 상황과 배경에 맞는 자연스러운 말을 사용하세요.

말투 규칙:
- 문어체(예: '~것이다', '~합니다') 대신 자연스러운 구어체를 사용하세요.
- 캐릭터의 말투 스타일을 정확히 따르세요. 특히 '~이기야' 같은 비문법적 표현을 쓰지 말고 '~이지' 같은 올바른 표현을 사용하세요.
- 맥락에 맞지 않는 이상한 표현(예: '고백할 기회', '아버지한테 고백' 등)을 피하고, 
  현재 배경과 인터랙션에 자연스럽게 어울리는 대사를 생성하세요.
- 최대한 짧고 간단하게, 일상 대화처럼. (예: '~할까?', '~하는 거지', '~같은데' 등)
- 한 문장만, 1~2초에 말할 수 있는 길이.
- 따옴표는 쓰지 마세요.
"""
    resp_a = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": system_a},
            {"role": "user", "content": user_a}
        ],
        max_output_tokens=50,
        temperature=0.7
    )
    line_a = _clean_line(resp_a.output_text)

    # B의 응답
    char_b_data = CHARACTERS.get(char_b['book_code'], {}).get(char_b['role_key'], {})
    char_b_speech = char_b_data.get('speech_patterns', {})
    char_b_style = char_b_speech.get('speaking_style', '')
    
    system_b = (
        "당신은 한국 옛이야기 속 두 인물이 실제로 주고받는 대화를 쓰는 작가입니다. "
        "두 번째 인물이 첫 번째 인물의 말에 바로 반응하는 짧은 한 마디를 만드세요."
    )
    user_b = f"""
배경 장소: {place}
배경 인터랙션: {action}
장면 분위기 요약: {profile['summary']}
가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

첫 번째 인물의 말:
{line_a}

두 번째 인물 설정(영어): {char_b['personality']}
두 번째 인물 정보: {char_b['age']}살 {char_b['gender']}
두 번째 인물 말투 스타일: {char_b_style}

상황:
- 두 번째 인물이 위 말을 듣고, 자신의 성격에 맞는 감정으로 바로 이어서 한 마디를 합니다.

말투 규칙:
- 첫 번째 인물의 말에 자연스럽게 이어지는 반응이어야 합니다.
- 문어체 금지, 자연스러운 구어체만. (예: '~지?', '~잖아', '~라니까', '~해요' 등)
- 캐릭터의 말투 스타일을 정확히 따르세요. 특히 '~이기야' 같은 비문법적 표현을 쓰지 말고 '~이지' 같은 올바른 표현을 사용하세요.
- 한 문장만, 짧게.
- 따옴표는 쓰지 마세요.
"""
    resp_b = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": system_b},
            {"role": "user", "content": user_b}
        ],
        max_output_tokens=50,
        temperature=0.7
    )
    line_b = _clean_line(resp_b.output_text)

    return line_a, line_b


def generate_surprised_line(character: dict, bg_info: dict) -> str:
    """
    배경이 갑자기 바뀌었을 때 놀라는 한 마디.
    → 감탄 + 짧은 구어체.
    """
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    system = (
        "당신은 한국 옛이야기 속 등장인물이 갑자기 다른 장소로 이동했을 때의 반응을 쓰는 작가입니다. "
        "실제 사람이 놀라서 툭 내뱉는 짧은 한국어 한 마디를 만드세요."
    )

    # 캐릭터의 speech_patterns 가져오기
    char_data = CHARACTERS.get(character['book_code'], {}).get(character['role_key'], {})
    speech_patterns = char_data.get('speech_patterns', {})
    frequent_expressions = speech_patterns.get('frequent_expressions', [])
    speaking_style = speech_patterns.get('speaking_style', '')
    
    user = f"""
새 배경 장소: {place}
새 배경 인터랙션: {action}
장면 분위기 요약: {profile['summary']}
가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

캐릭터 설정(영어): {character['personality']}
캐릭터 정보: {character['age']}살 {character['gender']}
캐릭터 말투 스타일: {speaking_style}

상황:
- 이 캐릭터는 방금 전까지 전혀 다른 곳에 있었는데,
  갑자기 이 장면으로 순간이동하듯 옮겨졌습니다.
- 위 감정 중 자신의 성격에 맞는 것을 느끼며, 놀라거나 당황하거나 신기해서 감탄과 함께 한 마디를 합니다.

말투 규칙:
- 이 캐릭터의 성격과 말투 스타일에 맞는 구체적인 감탄사를 사용하세요.
  예: 겁많은 캐릭터는 '어? 여기가 어디지?', '무서운 곳이네...', '이상한 곳에 왔어' 등
      용감한 캐릭터는 '오? 이곳이 바로 그 곳인가?', '흠, 여기서 뭘 해야 하지?', '뭐지, 이 분위기는?' 등
      장난꾸러기는 '어? 이거 재밌겠는데!', '오호, 여기서 뭘 할 수 있을까?', '이런 곳이 있었구나!' 등
      차분한 캐릭터는 '어라, 여기가 어디일까?', '이상하네, 분위기가 달라', '음... 이곳은 뭔가 특별해' 등
- '어라, 이게 무슨 신기한 일인가?', '오호, 이거 재밌네!' 같은 일반적인 멘트는 피하고, 
  현재 배경 장소와 인터랙션을 구체적으로 언급하는 놀라움 표현을 사용하세요.
- 문어체 금지, 자연스러운 구어체.
- 한 문장만, 아주 짧게.
- 따옴표는 쓰지 마세요.
"""

    resp = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_output_tokens=40,
        temperature=0.7
    )
    return _clean_line(resp.output_text)


def generate_sisters_two_lines(sisters: dict, bg_info: dict) -> tuple[str, str]:
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    system = (
        "당신은 한국 옛이야기 '장화홍련전'의 자매가 실제로 주고받는 대사를 쓰는 작가입니다. "
        "언니와 동생이 서로에게 하는 짧은 구어체 한 마디씩, 두 문장을 만드세요."
    )

    user = f"""
배경 장소: {place}
배경 인터랙션: {action}

장면 분위기:
- 요약: {profile['summary']}
- 가능한 감정들 (각 자매의 성격에 맞는 것을 선택하세요):
{emotion_list}

자매 설정(영어): {sisters['personality']}
자매 정보: {sisters['age']}살 {sisters['gender']}

출력 규칙:
- 첫 번째 줄: 언니가 동생에게 말합니다. 반드시 '홍련아' 포함.
- 두 번째 줄: 동생이 언니에게 말합니다. 반드시 '언니' 포함.
- 두 문장 모두 자연스러운 구어체여야 합니다. (예: '~거야', '~하지 마', '~해볼까' 등)
- 특히 '~이기야' 같은 비문법적 표현을 쓰지 말고 '~이지' 같은 올바른 표현을 사용하세요.
- 문어체 금지, 설명 금지.
- 각각 한 문장씩만 출력하세요.
"""

    resp = client.responses.create(
        model=TEXT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_output_tokens=80,
        temperature=0.7
    )

    lines = [l.strip() for l in resp.output_text.splitlines() if l.strip()]
    if len(lines) >= 2:
        return _clean_line(lines[0]), _clean_line(lines[1])
    elif len(lines) == 1:
        return _clean_line(lines[0]), "언니, 나도 그런 기분이야."
    else:
        return "홍련아, 너무 걱정하지 마.", "언니, 그래도 좀 무서워."


# ============================================
# 5. TTS
# ============================================
def generate_tts(character: dict, text: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    speaker_tag = f"{character['book_code'].upper()}-{character['role_key'].upper()}"
    print(f"🎤 [{speaker_tag}] line: {text}")

    voice_speed = character.get("speed", 1.0)

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=character["voice"],
        input=text,
        response_format="wav",
        speed=voice_speed
    )

    audio_bytes = response.read()

    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"✅ Saved: {output_path}")

    return output_path



def play_audio(path: str):
    print(f"🔊 PLAY AUDIO: {path}")
    subprocess.run(["afplay", path])


# ============================================
# 6. 메인 진입점: 웹캠에서 book_code + 순서 넘겨줄 때
# ============================================
def handle_book_input(book_code: str, index_in_sequence: int):
    """
    index_in_sequence 규칙:

    1: 초기 배경 설정
    2: 초기 cha1 등장 + 한 줄 대사
    3: 초기 cha2 등장 + cha1/cha2 대화 (각 한 줄)

    이후 4부터는 3개 주기로 반복:
    4,7,10,... : 배경만 교체 → cha1/cha2 둘 다 놀라는 대사 한 줄씩
    5,8,11,... : cha1 교체     → 새 cha1 + 기존 cha2 대화 (각 한 줄)
                 (단, 새 cha1이 자매면 언니/동생 두 줄 + cha2 한 줄)
    6,9,12,... : cha2 교체     → 기존 cha1 + 새 cha2 대화 (각 한 줄)
    """
    global CURRENT_BG_BOOK_CODE, CURRENT_BG_INFO, CURRENT_CHA1_INFO, CURRENT_CHA2_INFO

    print("\n==============================")
    print(f"[handle_book_input] book_code={book_code}, index={index_in_sequence}")

    # -------------------------
    # 1) index 1: 초기 배경
    # -------------------------
    if index_in_sequence == 1:
        bg = get_background(book_code)
        if bg is None:
            print(f"⚠ BACKGROUNDS에 없는 book_code: {book_code}")
            return

        CURRENT_BG_BOOK_CODE = book_code
        CURRENT_BG_INFO = bg
        CURRENT_CHA1_INFO = None
        CURRENT_CHA2_INFO = None

        print(f"[BACKGROUND INIT] {book_code} → {bg.get('background')}")
        play_background_video(book_code)  # 배경 비디오 재생 (무한 루프, 오디오 포함)
        return

    # -------------------------
    # 2) index 2: 초기 cha1
    # -------------------------
    if index_in_sequence == 2:
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha1"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha1 정의 없음")
            return

        cha1 = build_character(book_code, role_key)
        CURRENT_CHA1_INFO = cha1

        # 장화홍련전의 경우 자매 둘 다 말하도록
        if book_code == "JHHRJ":
            older, younger = build_sisters_pair()
            CURRENT_CHA1_INFO = older
            CURRENT_CHA2_INFO = younger
            
            # 자매 둘 다 대사 생성
            line1, line2 = generate_sisters_two_lines(older, CURRENT_BG_INFO)
            if not line1 or not line2:
                # 대사 생성 실패 시 기본 대사 사용
                line1 = "홍련아, 여기가 어디지?"
                line2 = "언니, 나도 모르겠어."
            
            out1 = f"output/JHHRJ_sister_older_init_cha1.wav"
            out2 = f"output/JHHRJ_sister_younger_init_cha1.wav"
            generate_tts(older, line1, out1)
            generate_tts(younger, line2, out2)
            play_audio(out1)
            play_audio(out2)
        else:
            line = generate_action_line(cha1, CURRENT_BG_INFO)
            if not line:
                line = f"{CURRENT_BG_INFO.get('interaction', '')}, 한번 해볼까?"

            out_path = f"output/{book_code}_{role_key}_init_cha1.wav"
            generate_tts(cha1, line, out_path)
            play_audio(out_path)
        return

    # -------------------------
    # 3) index 3: 초기 cha2 + 대화
    # -------------------------
    if index_in_sequence == 3:
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha2"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha2 정의 없음")
            return

        cha2 = build_character(book_code, role_key)
        CURRENT_CHA2_INFO = cha2

        if CURRENT_CHA1_INFO is None:
            print("⚠ cha1이 아직 설정되지 않아 cha2만 한 줄 대사")
            line2 = generate_action_line(cha2, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_init_cha2_only.wav"
            generate_tts(cha2, line2, out2)
            play_audio(out2)
            return

        # 장화홍련전의 경우: 박씨(cha2)가 먼저 말하고, 장화(cha1의 언니)가 말하고, 홍련(cha1의 동생)이 말함
        if CURRENT_CHA1_INFO['book_code'] == "JHHRJ" and book_code == "PSJ":
            # 박씨가 먼저 말
            line_psj = generate_action_line(cha2, CURRENT_BG_INFO)
            out_psj = f"output/{book_code}_{role_key}_init_dialog1.wav"
            generate_tts(cha2, line_psj, out_psj)
            
            # 장화가 말
            older, younger = build_sisters_pair()
            line_older = generate_action_line(older, CURRENT_BG_INFO)
            out_older = f"output/JHHRJ_sister_older_init_dialog2.wav"
            generate_tts(older, line_older, out_older)
            
            # 홍련이 말
            line_younger = generate_action_line(younger, CURRENT_BG_INFO)
            out_younger = f"output/JHHRJ_sister_younger_init_dialog3.wav"
            generate_tts(younger, line_younger, out_younger)
            
            play_audio(out_psj)
            play_audio(out_older)
            play_audio(out_younger)
        else:
            # 새로 등장하는 cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            line2, line1 = generate_dialogue_lines(cha2, CURRENT_CHA1_INFO, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_init_dialog1.wav"
            out1 = f"output/{CURRENT_CHA1_INFO['book_code']}_{CURRENT_CHA1_INFO['role_key']}_init_dialog2.wav"
            generate_tts(cha2, line2, out2)
            generate_tts(CURRENT_CHA1_INFO, line1, out1)
            play_audio(out2)
            play_audio(out1)
        return

    # -------------------------
    # 4) 이후: 3개 주기 (배경 / cha1 / cha2 교체)
    # -------------------------
    if CURRENT_BG_INFO is None or CURRENT_CHA1_INFO is None or CURRENT_CHA2_INFO is None:
        print("⚠ 아직 초기 1~3번 셋업이 완료되지 않았습니다.")
        return

    offset = (index_in_sequence - 4) % 3  # 0,1,2 반복

    # ---- 4,7,10,... : 배경 교체 + 두 캐릭터 놀람 ----
    if offset == 0:
        bg = get_background(book_code)
        if bg is None:
            print(f"⚠ BACKGROUNDS에 없는 book_code: {book_code}")
            return

        CURRENT_BG_BOOK_CODE = book_code
        CURRENT_BG_INFO = bg

        print(f"[BACKGROUND SWAP] {book_code} → {bg.get('background')}")
        play_background_video(book_code)  # 배경 비디오 교체 (무한 루프, 오디오 포함, 페이드 효과)

        line1 = generate_surprised_line(CURRENT_CHA1_INFO, CURRENT_BG_INFO)
        line2 = generate_surprised_line(CURRENT_CHA2_INFO, CURRENT_BG_INFO)

        out1 = f"output/{CURRENT_CHA1_INFO['book_code']}_{CURRENT_CHA1_INFO['role_key']}_surprised.wav"
        out2 = f"output/{CURRENT_CHA2_INFO['book_code']}_{CURRENT_CHA2_INFO['role_key']}_surprised.wav"
        generate_tts(CURRENT_CHA1_INFO, line1, out1)
        generate_tts(CURRENT_CHA2_INFO, line2, out2)
        play_audio(out1)
        play_audio(out2)
        return

        # ---- 5,8,11,... : cha1 교체 ----
    if offset == 1:
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha1"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha1 정의 없음")
            return

        cha1 = build_character(book_code, role_key)
        CURRENT_CHA1_INFO = cha1

        # 🔸 장화홍련 자매인 경우: 언니 + 동생이 각각 한 줄씩 말하고,
        #    기존 cha2(예: 토끼, 귀신 등)가 한 줄 더 대답.
        if book_code == "JHHRJ" and role_key == "sister_older":
            sister_older, sister_younger = build_sisters_pair()

            # 언니 → 동생 순서로 서로 한 줄씩 대사 생성
            lineA, lineB = generate_dialogue_lines(sister_older, sister_younger, CURRENT_BG_INFO)
            reply = generate_action_line(CURRENT_CHA2_INFO, CURRENT_BG_INFO)

            outA = "output/JHHRJ_sister_older_line.wav"
            outB = "output/JHHRJ_sister_younger_line.wav"
            outC = f"output/{CURRENT_CHA2_INFO['book_code']}_{CURRENT_CHA2_INFO['role_key']}_reply_to_sisters.wav"

            # 언니/동생이 서로 다른 voice로 각각 말하게 함
            generate_tts(sister_older, lineA, outA)
            generate_tts(sister_younger, lineB, outB)
            generate_tts(CURRENT_CHA2_INFO, reply, outC)

            play_audio(outA)
            play_audio(outB)
            play_audio(outC)
            return

        # 🔹 그 외 일반 캐릭터: 새 cha1 + 기존 cha2가 한 줄씩 대화
        line1, line2 = generate_dialogue_lines(cha1, CURRENT_CHA2_INFO, CURRENT_BG_INFO)
        out1 = f"output/{book_code}_{role_key}_swapcha1_dialog1.wav"
        out2 = f"output/{CURRENT_CHA2_INFO['book_code']}_{CURRENT_CHA2_INFO['role_key']}_swapcha1_dialog2.wav"
        generate_tts(cha1, line1, out1)
        generate_tts(CURRENT_CHA2_INFO, line2, out2)
        play_audio(out1)
        play_audio(out2)
        return

    # ---- 6,9,12,... : cha2 교체 ----
    if offset == 2:
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha2"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha2 정의 없음")
            return

        cha2 = build_character(book_code, role_key)
        CURRENT_CHA2_INFO = cha2

        # cha1이 장화홍련인 경우: cha2가 먼저 말하고, older가 말하고, younger가 말함
        if CURRENT_CHA1_INFO['book_code'] == "JHHRJ":
            older, younger = build_sisters_pair()
            
            # cha2가 먼저 말
            line_cha2 = generate_action_line(cha2, CURRENT_BG_INFO)
            out_cha2 = f"output/{book_code}_{role_key}_swapcha2_dialog1.wav"
            generate_tts(cha2, line_cha2, out_cha2)
            
            # older가 말
            line_older = generate_action_line(older, CURRENT_BG_INFO)
            out_older = f"output/JHHRJ_sister_older_swapcha2_dialog2.wav"
            generate_tts(older, line_older, out_older)
            
            # younger가 말
            line_younger = generate_action_line(younger, CURRENT_BG_INFO)
            out_younger = f"output/JHHRJ_sister_younger_swapcha2_dialog3.wav"
            generate_tts(younger, line_younger, out_younger)
            
            play_audio(out_cha2)
            play_audio(out_older)
            play_audio(out_younger)
        else:
            # cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            line2, line1 = generate_dialogue_lines(cha2, CURRENT_CHA1_INFO, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_swapcha2_dialog1.wav"
            out1 = f"output/{CURRENT_CHA1_INFO['book_code']}_{CURRENT_CHA1_INFO['role_key']}_swapcha2_dialog2.wav"
            generate_tts(cha2, line2, out2)
            generate_tts(CURRENT_CHA1_INFO, line1, out1)
            play_audio(out2)
            play_audio(out1)
        return


# ============================================
# 7. 웹캠 ArUco 마커 감지
# ============================================
def run_webcam_detection():
    """
    웹캠으로 ArUco 마커를 감지하고, 감지된 마커에 따라 handle_book_input을 호출합니다.
    배경 비디오는 별도의 윈도우에서 부드럽게 전환되며 무한 루프로 재생됩니다.
    TTS 생성은 별도 스레드에서 실행되어 비디오가 끊기지 않습니다.
    """
    global CURRENT_BG_BOOK_CODE
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다!")
        return
    
    print("📷 Camera . Press 'q' to quit.")
    print("📚 Show your book to camera...")
    
    # 비디오 플레이어 시작
    VIDEO_PLAYER.start()
    
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(ARUCO_DICTIONARY, detector_params)
    
    sequence_index = 0  # 현재 시퀀스 인덱스
    last_detected_marker = None  # 마지막으로 감지된 마커 (중복 방지)
    handler_thread = None  # handle_book_input 실행 스레드
    is_processing = False  # 현재 처리 중인지 여부
    
    # 비디오 윈도우 생성 (전체 화면)
    cv2.namedWindow("Background Video", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Background Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def run_handler_async(book_code, seq_idx):
        """handle_book_input을 별도 스레드에서 실행"""
        nonlocal is_processing
        is_processing = True
        try:
            handle_book_input(book_code, seq_idx)
        finally:
            is_processing = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다!")
            break
        
        # ArUco 마커 감지
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # 감지된 마커가 있으면 표시
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 첫 번째로 감지된 마커 처리
            marker_id = ids[0][0]
            book_code = get_book_code_from_marker(marker_id)
            
            # 이전 핸들러가 완료된 경우에만 새 마커 처리
            if book_code and marker_id != last_detected_marker and not is_processing:
                last_detected_marker = marker_id
                sequence_index += 1
                
                # 한글 책 이름 가져오기
                book_info = BACKGROUNDS.get(book_code, {})
                book_name_kr = book_info.get("book", book_code)
                
                print(f"\n🎯 Marker Detected! ID: {marker_id} → {book_name_kr} ({book_code}) (Num of books: {sequence_index})")
                
                # 별도 스레드에서 handle_book_input 실행 (비디오가 끊기지 않도록)
                handler_thread = threading.Thread(
                    target=run_handler_async, 
                    args=(book_code, sequence_index),
                    daemon=True
                )
                handler_thread.start()
        
        # 화면에 정보 표시 (웹캠 윈도우)
        cv2.putText(frame, f"Sequence: {sequence_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if last_detected_marker is not None:
            book = get_book_code_from_marker(last_detected_marker) or "Unknown"
            cv2.putText(frame, f"Last: {book}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if is_processing:
            cv2.putText(frame, "Processing...", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        
        cv2.imshow("ArUco Marker Detection", frame)
        
        # 배경 비디오 프레임 표시 (같은 윈도우에서 부드럽게 전환)
        video_frame = VIDEO_PLAYER.get_frame()
        if video_frame is not None:
            cv2.imshow("Background Video", video_frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    VIDEO_PLAYER.stop()
    cv2.destroyAllWindows()
    print("\n📷 웹캠 종료됨.")


# ============================================
# 8. 단독 실행
# ============================================
if __name__ == "__main__":
    import sys
    
    # ArUco 마커 생성 옵션
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-markers":
        generate_aruco_markers()
        sys.exit(0)
    
    # 기본: 웹캠 감지 모드 실행
    run_webcam_detection()