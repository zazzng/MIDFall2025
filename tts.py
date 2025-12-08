import os
import json
import subprocess
import threading
import time
import random
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
        self.bgm_process = None  # BGM 재생 프로세스 (스레드 또는 프로세스)
        self.bgm_proc_ref = None  # BGM 프로세스 참조 (실제 종료용)
        self.pending_bgm_path = None  # 제목 말하기 후 재생할 BGM 경로
        self.fade_alpha = 1.0  # 페이드 알파 값 (0.0 ~ 1.0)
        self.is_fading = False  # 페이드 중인지 여부
        self.fade_duration = 0.5  # 페이드 지속 시간 (초)
        self.fade_start_time = None
        self.overlay_video_cap = None  # 오버레이 비디오 ch1 (캐릭터 움직임)
        self.overlay_video_path = None  # 오버레이 비디오 ch1 경로
        self.overlay_video_cap2 = None  # 오버레이 비디오 ch2 (캐릭터 움직임)
        self.overlay_video_path2 = None  # 오버레이 비디오 ch2 경로
        self.bg_fps = 30.0  # 배경 비디오 FPS (기본값)
        self.overlay_fps = 30.0  # 오버레이 비디오 ch1 FPS (기본값)
        self.overlay_fps2 = 30.0  # 오버레이 비디오 ch2 FPS (기본값)
        self.last_frame_time = None  # 마지막 프레임 표시 시간
    
    def _play_loop(self):
        """비디오 재생 루프 (별도 스레드에서 실행)"""
        while self.running:
            loop_start_time = time.time()
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
                    
                    # 오버레이 비디오 ch1이 있으면 배경 위에 합성
                    if self.overlay_video_cap is not None and self.overlay_video_cap.isOpened():
                        overlay_ret, overlay_frame = self.overlay_video_cap.read()
                        if not overlay_ret:
                            # 오버레이 비디오 끝나면 처음으로 돌아가기
                            self.overlay_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            overlay_ret, overlay_frame = self.overlay_video_cap.read()
                        
                        if overlay_ret:
                            # 오버레이 프레임 크기를 배경 프레임 크기에 맞춤
                            if overlay_frame.shape[:2] != frame.shape[:2]:
                                overlay_frame = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]))
                            
                            # 알파 채널이 있으면 알파 블렌딩, 없으면 일반 오버레이
                            if overlay_frame.shape[2] == 4:
                                # RGBA -> RGB 변환 및 알파 블렌딩
                                overlay_rgb = overlay_frame[:, :, :3]
                                overlay_alpha = overlay_frame[:, :, 3:4] / 255.0
                                frame = (frame * (1 - overlay_alpha) + overlay_rgb * overlay_alpha).astype(frame.dtype)
                            else:
                                # 알파 채널이 없으면 일반 오버레이 (투명도 가정)
                                # 배경 위에 오버레이 합성
                                mask = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2GRAY)
                                mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
                                mask = mask.astype(float) / 255.0
                                mask = cv2.merge([mask, mask, mask])
                                frame = (frame * (1 - mask) + overlay_frame * mask).astype(frame.dtype)
                    
                    # 오버레이 비디오 ch2가 있으면 배경 위에 합성 (ch1 위에)
                    if self.overlay_video_cap2 is not None and self.overlay_video_cap2.isOpened():
                        overlay_ret2, overlay_frame2 = self.overlay_video_cap2.read()
                        if not overlay_ret2:
                            # 오버레이 비디오 끝나면 처음으로 돌아가기
                            self.overlay_video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            overlay_ret2, overlay_frame2 = self.overlay_video_cap2.read()
                        
                        if overlay_ret2:
                            # 오버레이 프레임 크기를 배경 프레임 크기에 맞춤
                            if overlay_frame2.shape[:2] != frame.shape[:2]:
                                overlay_frame2 = cv2.resize(overlay_frame2, (frame.shape[1], frame.shape[0]))
                            
                            # 알파 채널이 있으면 알파 블렌딩, 없으면 일반 오버레이
                            if overlay_frame2.shape[2] == 4:
                                # RGBA -> RGB 변환 및 알파 블렌딩
                                overlay_rgb2 = overlay_frame2[:, :, :3]
                                overlay_alpha2 = overlay_frame2[:, :, 3:4] / 255.0
                                frame = (frame * (1 - overlay_alpha2) + overlay_rgb2 * overlay_alpha2).astype(frame.dtype)
                            else:
                                # 알파 채널이 없으면 일반 오버레이 (투명도 가정)
                                # 배경 위에 오버레이 합성
                                mask2 = cv2.cvtColor(overlay_frame2, cv2.COLOR_BGR2GRAY)
                                mask2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)[1]
                                mask2 = mask2.astype(float) / 255.0
                                mask2 = cv2.merge([mask2, mask2, mask2])
                                frame = (frame * (1 - mask2) + overlay_frame2 * mask2).astype(frame.dtype)
                    
                    self.frame = frame
            
            # 실제 비디오 FPS에 맞춰 프레임 간격 조정
            # 배경 비디오의 FPS를 기준으로 사용 (오버레이가 있으면 더 높은 FPS 사용)
            target_fps = max(self.bg_fps, 
                           self.overlay_fps if self.overlay_video_cap else 0,
                           self.overlay_fps2 if self.overlay_video_cap2 else 0)
            if target_fps <= 0:
                target_fps = 30.0  # 기본값
            
            frame_interval = 1.0 / target_fps
            
            # 프레임 처리 시간 고려하여 정확한 타이밍으로 재생
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _switch_video_internal(self, video_path: str):
        """내부 비디오 전환 (페이드 중에 호출)"""
        # 배경이 바뀔 때 이전 BGM 즉시 종료
        self._stop_bgm_immediately()
        
        # 기존 비디오 해제
        if self.video_cap:
            self.video_cap.release()
        
        # 새 비디오 열기
        self.current_video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            print(f"❌ 비디오를 열 수 없음: {video_path}")
            self.video_cap = None
            self.bg_fps = 30.0  # 기본값
        else:
            # 실제 비디오 FPS 읽기
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.bg_fps = fps
            else:
                self.bg_fps = 30.0  # 기본값
            print(f"🎬 비디오 전환: {video_path} (FPS: {self.bg_fps:.2f})")
            
            # 비디오 파일명에서 책 코드 추출하여 BGM 경로 저장 (제목 말하기 후 재생)
            video_filename = os.path.basename(video_path)
            book_code = None
            for code, vfile in BOOK_TO_VIDEO.items():
                if vfile == video_filename:
                    book_code = code
                    break
            
            if book_code:
                bgm_file = BOOK_TO_BGM.get(book_code)
                if bgm_file:
                    bgm_path = os.path.join(BGM_DIR, bgm_file)
                    if os.path.exists(bgm_path):
                        # BGM 경로를 저장 (제목 말하기 후 재생)
                        self.pending_bgm_path = bgm_path
                    else:
                        print(f"⚠️ BGM 파일을 찾을 수 없음: {bgm_path}")
                else:
                    print(f"⚠️ '{book_code}'에 해당하는 BGM이 없습니다.")
            else:
                print(f"⚠️ 비디오 파일명에서 책 코드를 찾을 수 없음: {video_filename}")
    
    def _stop_bgm_immediately(self):
        """기존 BGM을 즉시 종료"""
        old_bgm_process = self.bgm_process
        old_bgm_proc_ref = self.bgm_proc_ref
        if old_bgm_process or old_bgm_proc_ref:
            try:
                if old_bgm_proc_ref:
                    if isinstance(old_bgm_proc_ref, dict):
                        # 딕셔너리인 경우 (macOS bgm_control)
                        old_bgm_proc_ref["running"] = False
                        if old_bgm_proc_ref.get("current_proc"):
                            try:
                                old_bgm_proc_ref["current_proc"].terminate()
                                old_bgm_proc_ref["current_proc"].wait(timeout=0.1)
                            except:
                                try:
                                    old_bgm_proc_ref["current_proc"].kill()
                                except:
                                    pass
                    elif isinstance(old_bgm_proc_ref, list):
                        # 리스트인 경우 [proc, afplay_proc]
                        for p in old_bgm_proc_ref:
                            if p and hasattr(p, "terminate"):
                                try:
                                    p.terminate()
                                    p.wait(timeout=0.1)
                                except:
                                    try:
                                        p.kill()
                                    except:
                                        pass
                    elif hasattr(old_bgm_proc_ref, "terminate"):
                        try:
                            old_bgm_proc_ref.terminate()
                            old_bgm_proc_ref.wait(timeout=0.1)
                        except:
                            try:
                                old_bgm_proc_ref.kill()
                            except:
                                pass
            except Exception as e:
                print(f"⚠️ BGM 종료 오류: {e}")
            
            self.bgm_process = None
            self.bgm_proc_ref = None
    
    def _start_bgm(self, bgm_path: str):
        """BGM을 무한 루프로 재생 (페이드인 효과 포함)"""
        # 기존 BGM을 즉시 종료
        self._stop_bgm_immediately()
        
        # BGM을 무한 루프로 재생 (페이드인 효과 포함)
        import platform
        is_macos = platform.system() == "Darwin"
        
        try:
            if is_macos:
                # macOS: afplay를 직접 사용하여 무한 루프 재생 (더 안정적)
                fade_duration = 0.5
                
                # 종료 플래그를 위한 딕셔너리
                bgm_control = {"running": True, "current_proc": None}
                
                def play_bgm_with_fade():
                    try:
                        # 임시 파일에 페이드인 효과를 적용한 BGM 생성 (첫 루프만)
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_bgm = os.path.join(temp_dir, f"bgm_fade_{os.getpid()}.wav")
                        
                        # 첫 루프에만 페이드인 적용 + 음량 50%
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", bgm_path,
                             "-af", f"afade=t=in:st=0:d={fade_duration},volume=0.5",
                             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                             temp_bgm],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True
                        )
                        
                        # 원본 파일도 음량 50%로 조정한 임시 파일 생성
                        temp_bgm_loop = os.path.join(temp_dir, f"bgm_loop_{os.getpid()}.wav")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", bgm_path,
                             "-af", "volume=0.5",
                             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                             temp_bgm_loop],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True
                        )
                        
                        # 첫 번째는 페이드인 적용된 파일 재생
                        first_play = True
                        while bgm_control["running"]:
                            if first_play:
                                proc = subprocess.Popen(
                                    ["afplay", temp_bgm],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                                bgm_control["current_proc"] = proc
                                proc.wait()
                                first_play = False
                            else:
                                # 이후는 음량 조정된 파일 무한 루프
                                proc = subprocess.Popen(
                                    ["afplay", temp_bgm_loop],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                                bgm_control["current_proc"] = proc
                                proc.wait()
                                
                                # 종료 신호 확인
                                if not bgm_control["running"]:
                                    break
                        
                        # 임시 파일 삭제
                        try:
                            os.remove(temp_bgm)
                            os.remove(temp_bgm_loop)
                        except:
                            pass
                    except Exception as e:
                        print(f"⚠️ BGM 재생 오류: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 별도 스레드에서 BGM 재생
                bgm_thread = threading.Thread(target=play_bgm_with_fade, daemon=True)
                bgm_thread.start()
                self.bgm_process = bgm_thread
                # 프로세스 참조는 제어 딕셔너리로 관리
                with self.lock:
                    self.bgm_proc_ref = bgm_control
                print(f"🎵 BGM 재생 시작 (페이드인): {bgm_path}")
            else:
                # Linux: ffplay 사용
                proc = subprocess.Popen(
                    ["ffmpeg", "-stream_loop", "-1", "-i", bgm_path,
                     "-af", "afade=t=in:st=0:d=0.5,volume=0.5",
                     "-f", "wav", "-"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                ffplay_proc = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loop", "0", "-loglevel", "quiet", "-"],
                    stdin=proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self.bgm_process = ffplay_proc
                self.bgm_proc_ref = [proc, ffplay_proc]
                print(f"🎵 BGM 재생 시작 (페이드인): {bgm_path}")
        except FileNotFoundError:
            print("⚠️ ffmpeg/afplay를 찾을 수 없습니다. BGM은 재생되지 않습니다.")
            if is_macos:
                print("   macOS에서는 'brew install ffmpeg'로 설치하세요.")
            else:
                print("   Linux에서는 'sudo apt-get install ffmpeg' 또는 'sudo yum install ffmpeg'로 설치하세요.")
    
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
    
    def set_overlay_video(self, overlay_path: str):
        """오버레이 비디오 ch1 설정 (배경 위에 표시될 캐릭터 움직임)"""
        with self.lock:
            # 기존 오버레이 비디오 해제
            if self.overlay_video_cap:
                self.overlay_video_cap.release()
            
            if overlay_path and os.path.exists(overlay_path):
                self.overlay_video_path = overlay_path
                self.overlay_video_cap = cv2.VideoCapture(overlay_path)
                if not self.overlay_video_cap.isOpened():
                    print(f"❌ 오버레이 비디오를 열 수 없음: {overlay_path}")
                    self.overlay_video_cap = None
                    self.overlay_video_path = None
                    self.overlay_fps = 30.0  # 기본값
                else:
                    # 실제 비디오 FPS 읽기
                    fps = self.overlay_video_cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        self.overlay_fps = fps
                    else:
                        self.overlay_fps = 30.0  # 기본값
                    print(f"🎬 오버레이 비디오 ch1 설정: {overlay_path} (FPS: {self.overlay_fps:.2f})")
            else:
                self.overlay_video_cap = None
                self.overlay_video_path = None
    
    def set_overlay_video2(self, overlay_path: str):
        """오버레이 비디오 ch2 설정 (배경 위에 표시될 캐릭터 움직임)"""
        with self.lock:
            # 기존 오버레이 비디오 ch2 해제
            if self.overlay_video_cap2:
                self.overlay_video_cap2.release()
            
            if overlay_path and os.path.exists(overlay_path):
                self.overlay_video_path2 = overlay_path
                self.overlay_video_cap2 = cv2.VideoCapture(overlay_path)
                if not self.overlay_video_cap2.isOpened():
                    print(f"❌ 오버레이 비디오 ch2를 열 수 없음: {overlay_path}")
                    self.overlay_video_cap2 = None
                    self.overlay_video_path2 = None
                    self.overlay_fps2 = 30.0  # 기본값
                else:
                    # 실제 비디오 FPS 읽기
                    fps = self.overlay_video_cap2.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        self.overlay_fps2 = fps
                    else:
                        self.overlay_fps2 = 30.0  # 기본값
                    print(f"🎬 오버레이 비디오 ch2 설정: {overlay_path} (FPS: {self.overlay_fps2:.2f})")
            else:
                self.overlay_video_cap2 = None
                self.overlay_video_path2 = None
    
    def clear_overlay_video(self):
        """오버레이 비디오 모두 제거"""
        with self.lock:
            if self.overlay_video_cap:
                self.overlay_video_cap.release()
                self.overlay_video_cap = None
                self.overlay_video_path = None
            if self.overlay_video_cap2:
                self.overlay_video_cap2.release()
                self.overlay_video_cap2 = None
                self.overlay_video_path2 = None
                print("🎬 오버레이 비디오 모두 제거")
    
    def stop(self):
        """플레이어 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            if self.overlay_video_cap:
                self.overlay_video_cap.release()
                self.overlay_video_cap = None
            if self.overlay_video_cap2:
                self.overlay_video_cap2.release()
                self.overlay_video_cap2 = None
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
        
        # BGM 프로세스 종료
        if self.bgm_proc_ref:
            if isinstance(self.bgm_proc_ref, dict):
                # 딕셔너리인 경우 (macOS bgm_control)
                self.bgm_proc_ref["running"] = False
                if self.bgm_proc_ref.get("current_proc"):
                    try:
                        self.bgm_proc_ref["current_proc"].terminate()
                        self.bgm_proc_ref["current_proc"].wait(timeout=0.5)
                    except:
                        try:
                            self.bgm_proc_ref["current_proc"].kill()
                        except:
                            pass
            elif isinstance(self.bgm_proc_ref, list):
                # 리스트인 경우 [proc, afplay_proc]
                for p in self.bgm_proc_ref:
                    if p and hasattr(p, "terminate"):
                        try:
                            p.terminate()
                            p.wait(timeout=0.5)
                        except:
                            try:
                                p.kill()
                            except:
                                pass
            elif hasattr(self.bgm_proc_ref, "terminate"):
                try:
                    self.bgm_proc_ref.terminate()
                    self.bgm_proc_ref.wait(timeout=0.5)
                except:
                    try:
                        self.bgm_proc_ref.kill()
                    except:
                        pass
            self.bgm_proc_ref = None
        self.bgm_process = None
    
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
BGM_DIR = "bgm"
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
# 오버레이 비디오 파일명 매핑 (inter_video 폴더의 파일명 형식)
BOOK_TO_OVERLAY_CODE = {
    "BJBJ": "BJBJ",
    "PSJ": "BSJ",  # 박씨전 -> BSJ
    "DGJ": "DCJ",  # 덕캐비전 -> DCJ
    "HBJ": "HBJ",
    "JWCJ": "JWCJ",
    "KWJ": "KWJ",
    "OGJJ": "OGJJ",
    "JHHRJ": "JHHRJ",
    "SCJ": "SCJ",
}

BOOK_TO_BGM = {
    "BJBJ": "10_BJBJ_audioExtracted.wav",
    "PSJ": "11_BSJ_audioExtracted.wav",  # 파일명이 BSJ로 되어 있음
    "DGJ": "13_DGJ_audioExtracted.wav",
    "HBJ": "17_HBJ_audioExtracted.wav",
    "JWCJ": "19_JWCJ_audioExtracted.wav",
    "KWJ": "3_KWJ_audioExtracted.wav",
    "OGJJ": "5_OGJJ_audioExtracted.wav",
    "JHHRJ": "6_JHHRJ_audioExtracted.wav",
    "SCJ": "7_SCJ_audioExtracted.wav",
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


def generate_first_dialogue_line(char_a: dict, bg_info: dict) -> str:
    """
    같은 배경/인터랙션에서 char_a가 먼저 한 마디를 생성.
    → 짧고 구어체.
    Avoid any narration or book-style phrases. The line must sound like spontaneous spoken Korean, not a written script.
    Add small hesitations (예: '아...', '음...') when appropriate, only if it fits the character.
    """
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

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
- '~이기에요' 같은 비문법적 표현을 쓰지 말고 '~이에요' 같은 올바른 표현을 사용하세요.
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
    return line_a


def generate_second_dialogue_line(char_b: dict, line_a: str, bg_info: dict) -> str:
    """
    char_b가 char_a의 말(line_a)에 반응하는 한 마디를 생성.
    → 짧고 구어체.
    """
    place = bg_info.get("background", "")
    action = bg_info.get("interaction", "")
    profile = get_interaction_profile(bg_info)
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    char_b_data = CHARACTERS.get(char_b['book_code'], {}).get(char_b['role_key'], {})
    char_b_speech = char_b_data.get('speech_patterns', {})
    char_b_style = char_b_speech.get('speaking_style', '')
    
    system_b = (
        "당신은 한국 옛이야기 속 두 인물이 실제로 주고받는 대화를 쓰는 작가입니다. "
        "두 번째 인물이 첫 번째 인물의 말을 듣고 직접적으로 반응하는 짧은 한 마디를 만드세요. "
        "반드시 첫 번째 인물의 말에 대한 응답이어야 하며, 혼잣말이 아닌 대화여야 합니다."
    )
    user_b = f"""
배경 장소: {place}
배경 인터랙션: {action}
장면 분위기 요약: {profile['summary']}
가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

첫 번째 인물의 말:
"{line_a}"

두 번째 인물 설정(영어): {char_b['personality']}
두 번째 인물 정보: {char_b['age']}살 {char_b['gender']}
두 번째 인물 말투 스타일: {char_b_style}

중요한 상황:
- 두 번째 인물은 위의 첫 번째 인물의 말을 직접 듣고 있습니다.
- 첫 번째 인물의 말에 대해 반응하는 대답을 해야 합니다.
- 혼잣말이 아니라 첫 번째 인물에게 말하는 대화여야 합니다.
- 첫 번째 인물의 말의 내용, 톤, 의도를 고려하여 적절히 반응하세요.
- 동의, 반박, 질문, 제안, 놀람 등 첫 번째 인물의 말에 대한 자연스러운 반응을 보여주세요.

말투 규칙:
- 첫 번째 인물의 말에 직접적으로 반응하는 대답이어야 합니다.
- 첫 번째 인물의 말의 내용을 언급하거나 참조하는 것이 좋습니다.
- 예: 첫 번째가 "~할까?"라고 물으면 → "그래, 해보자" / "안 돼" / "~하는 게 좋겠어" 등
- 예: 첫 번째가 "~해야 해"라고 말하면 → "맞아" / "그렇지 않아" / "~하는 게 나을 것 같은데" 등
- 예: 첫 번째가 "~했어"라고 말하면 → "정말?" / "그래?" / "~했구나" 등
- 문어체 금지, 자연스러운 구어체만. (예: '~지?', '~잖아', '~라니까', '~해요' 등)
- 캐릭터의 말투 스타일을 정확히 따르세요. 특히 '~이기야' 같은 비문법적 표현을 쓰지 말고 '~이지' 같은 올바른 표현을 사용하세요.
- '~이기에요' 같은 비문법적 표현을 쓰지 말고 '~이에요' 같은 올바른 표현을 사용하세요.
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
    return line_b


def generate_dialogue_lines(char_a: dict, char_b: dict, bg_info: dict) -> tuple[str, str]:
    """
    같은 배경/인터랙션에서 char_a가 먼저 한 마디,
    char_b가 자연스럽게 이어서 한 마디.
    → 둘 다 짧고 구어체.
    (하위 호환성을 위해 유지, 하지만 순차 생성/재생을 위해 generate_first_dialogue_line과 generate_second_dialogue_line을 사용하는 것을 권장)
    """
    line_a = generate_first_dialogue_line(char_a, bg_info)
    line_b = generate_second_dialogue_line(char_b, line_a, bg_info)
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
- 배경 장소와 인터랙션을 파악하고, 이곳에서 무슨 일이 일어나는지 이해한 후 놀라움을 표현합니다.

말투 규칙:
- 이 캐릭터의 성격과 말투 스타일에 맞는 구체적인 감탄사를 사용하세요.
  예: 겁많은 캐릭터는 '어? 여기가 어디지?', '무서운 곳이네...', '이상한 곳에 왔어' 등
      용감한 캐릭터는 '오? 이곳이 바로 그 곳인가?', '흠, 여기서 뭘 해야 하지?', '뭐지, 이 분위기는?' 등
      장난꾸러기는 '어? 이거 재밌겠는데!', '오호, 여기서 뭘 할 수 있을까?', '이런 곳이 있었구나!' 등
      차분한 캐릭터는 '어라, 여기가 어디일까?', '이상하네, 분위기가 달라', '음... 이곳은 뭔가 특별해' 등
- '어라, 이게 무슨 신기한 일인가?', '오호, 이거 재밌네!' 같은 일반적인 멘트는 피하고, 
  현재 배경 장소와 인터랙션을 구체적으로 언급하는 놀라움 표현을 사용하세요.
- 배경 장소나 인터랙션을 언급하면서 놀라움을 표현하세요.
- 문어체 금지, 자연스러운 구어체.
- 한 문장, 적당한 길이 (1~2초에 말할 수 있는 길이).
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
    
    # 영어 번역 생성
    try:
        translation_resp = client.responses.create(
            model=TEXT_MODEL,
            input=[
                {"role": "system", "content": "You are a translator. Translate the given Korean dialogue to natural English, preserving the character's tone and emotion."},
                {"role": "user", "content": f"Translate this Korean dialogue to English: {text}"}
            ],
            max_output_tokens=50,
            temperature=0.3
        )
        english_text = _clean_line(translation_resp.output_text)
        print(f"🎤 [{speaker_tag}] line: {text} | {english_text}")
    except Exception as e:
        print(f"🎤 [{speaker_tag}] line: {text}")
        print(f"⚠️ Translation failed: {e}")

    voice_speed = character.get("speed", 1.0)

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=character["voice"],
        input=text,
        response_format="wav",
        speed=voice_speed
    )

    audio_bytes = response.read()

    # 임시 파일에 원본 오디오 저장
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, f"tts_temp_{os.getpid()}_{id(character)}.wav")
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    
    # 특수 캐릭터 오디오 효과 적용
    book_code = character.get("book_code", "")
    role_key = character.get("role_key", "")
    
    if (book_code == "JHHRJ" and role_key == "ghost") or (book_code == "KWJ" and role_key == "monster"):
        # reverb 효과 적용 (aecho 필터 사용)
        # ghost의 경우: 더 서글프고 울먹거리는 효과를 위해 tremolo와 pitch 조정도 추가
        if book_code == "JHHRJ" and role_key == "ghost":
            # ghost: 구슬프고 우울하고 한이 서린 처녀귀신 목소리
            # 효과: 깊은 reverb + 강한 tremolo (울먹거림) + 낮은 pitch (어둡고 우울) + 느린 속도 + 고주파 필터링 + delay + equalizer
            audio_filter = (
                "asetrate=44100*0.92,aresample=44100,"
                "atempo=0.95,"
                "lowpass=f=3000,"
                "aecho=1.0:0.9:120:0.5,"
                "adelay=50|50,"
                "tremolo=f=3.0:d=0.4,"
                "equalizer=f=200:width_type=h:width=300:g=2,"
                "equalizer=f=5000:width_type=h:width=2000:g=-3"
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", audio_filter,
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        else:
            # monster: reverb만 적용
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "aecho=0.8:0.88:60:0.4",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        # 임시 파일 삭제
        try:
            os.remove(temp_input)
        except:
            pass
    elif book_code == "SCJ" and role_key == "simcheong":
        # 심청: 어리고 명랑하고 결연에 가득 찬 목소리
        # 효과: 높은 pitch (어리고 밝게) + 빠른 속도 (명랑함) + 고주파 강조 (맑고 밝게) + vibrato (생동감) + 저주파 억제 (가볍고 밝게)
        # 오디오 필터 체인을 하나의 문자열로 합침
        audio_filter = (
            "asetrate=44100*1.12,aresample=44100,"  # pitch 올림 (더 어리고 밝게)
            "atempo=1.08,"  # 속도 빠르게 (명랑하고 활기차게)
            "equalizer=f=3000:width_type=h:width=2000:g=3,"  # 고주파 강조 (맑고 밝게)
            "equalizer=f=5000:width_type=h:width=1500:g=2,"  # 더 높은 고주파 강조 (명랑함)
            "equalizer=f=200:width_type=h:width=300:g=-2,"  # 저주파 억제 (가볍고 밝게)
            "vibrato=f=5.5:d=0.15,"  # 약간의 vibrato (생동감과 결연함)
            "highpass=f=100"  # 매우 낮은 주파수 제거 (더 맑게)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_input,
             "-af", audio_filter,
             output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        # 임시 파일 삭제
        try:
            os.remove(temp_input)
        except:
            pass
    elif book_code == "DGJ":
        if role_key == "fox":
            # 여우: 교활하고 가는 목소리, 간신배 느낌
            # pitch를 약간 올려서 더 가늘게, tremolo를 약간 추가해서 교활한 느낌
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "asetrate=44100*1.15,aresample=44100,tremolo=f=3.0:d=0.2",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        elif role_key == "toad":
            # 두꺼비: 현명하고 총명하고 뭉툭하고 묵직한 목소리
            # pitch를 약간 낮춰서 더 묵직하게, bass boost로 더 깊고 뭉툭한 느낌
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "asetrate=44100*0.9,aresample=44100,equalizer=f=100:width_type=h:width=200:g=3",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        else:
            # 다른 DGJ 캐릭터는 원본 그대로
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
        # 임시 파일 삭제
        try:
            os.remove(temp_input)
        except:
            pass
    else:
        # 일반 캐릭터는 원본 그대로 저장
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        # 임시 파일 삭제
        try:
            os.remove(temp_input)
        except:
            pass
    
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
        
        # 배경이 바뀔 때 사운드 이펙트만 재생 (제목 말하기는 마커 감지 시에만 재생)
        sound_effect_path = "soundeffect/ES_Dream, Harp - Epidemic Sound.wav"
        
        def play_sound():
            # ES_Dream 사운드 이펙트를 음량 20%로 처리한 임시 파일 생성 및 재생
            if os.path.exists(sound_effect_path):
                try:
                    os.makedirs("title_saying", exist_ok=True)
                    temp_sound = f"title_saying/temp_sound_{book_code}.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", sound_effect_path,
                         "-af", "volume=0.2",
                         temp_sound],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                    # 사운드 이펙트 재생
                    subprocess.run(["afplay", temp_sound],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                    os.remove(temp_sound)
                    print(f"🔊 사운드 효과 재생 (음량 20%): {sound_effect_path}")
                except Exception as e:
                    print(f"⚠️ 사운드 효과 재생 실패: {e}")
        
        # 비동기로 재생 (블로킹 방지)
        threading.Thread(target=play_sound, daemon=True).start()
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
        
        # 첫 번째 책이 심청전(SCJ)이고 두 번째 책이 감지되면 오버레이 비디오 설정
        if CURRENT_BG_BOOK_CODE == "SCJ":
            # inter_video/inter_bgSCJ/bgSCJ_ch1_{overlay_code}.mov 파일 찾기
            overlay_code = BOOK_TO_OVERLAY_CODE.get(book_code, book_code)
            overlay_path = f"inter_video/inter_bgSCJ/bgSCJ_ch1_{overlay_code}.mov"
            if os.path.exists(overlay_path):
                VIDEO_PLAYER.set_overlay_video(overlay_path)
                print(f"🎬 오버레이 비디오 설정: {overlay_path}")
            else:
                print(f"⚠️ 오버레이 비디오를 찾을 수 없음: {overlay_path}")

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
            
            # 랜덤으로 순서 결정
            if random.random() < 0.5:
                # 장화 먼저
                out1 = f"output/JHHRJ_sister_older_init_cha1.wav"
                out2 = f"output/JHHRJ_sister_younger_init_cha1.wav"
                generate_tts(older, line1, out1)
                generate_tts(younger, line2, out2)
                play_audio(out1)
                play_audio(out2)
            else:
                # 홍련 먼저
                out1 = f"output/JHHRJ_sister_younger_init_cha1.wav"
                out2 = f"output/JHHRJ_sister_older_init_cha1.wav"
                generate_tts(younger, line2, out1)
                generate_tts(older, line1, out2)
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
        
        # 첫 번째 책이 심청전(SCJ)이고 세 번째 책이 감지되면 ch2 오버레이 비디오 설정
        if CURRENT_BG_BOOK_CODE == "SCJ":
            # inter_video/inter_bgSCJ/bgSCJ_ch2_{overlay_code}.mov 파일 찾기
            overlay_code = BOOK_TO_OVERLAY_CODE.get(book_code, book_code)
            overlay_path2 = f"inter_video/inter_bgSCJ/bgSCJ_ch2_{overlay_code}.mov"
            if os.path.exists(overlay_path2):
                VIDEO_PLAYER.set_overlay_video2(overlay_path2)
                print(f"🎬 오버레이 비디오 ch2 설정: {overlay_path2}")
            else:
                print(f"⚠️ 오버레이 비디오 ch2를 찾을 수 없음: {overlay_path2}")

        if CURRENT_CHA1_INFO is None:
            print("⚠ cha1이 아직 설정되지 않아 cha2만 한 줄 대사")
            line2 = generate_action_line(cha2, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_init_cha2_only.wav"
            generate_tts(cha2, line2, out2)
            play_audio(out2)
            return

        # 장화홍련전의 경우: 자매가 랜덤 순서로 말함
        if CURRENT_CHA1_INFO['book_code'] == "JHHRJ":
            older, younger = build_sisters_pair()
            
            # 랜덤으로 순서 결정
            if random.random() < 0.5:
                # 장화 먼저
                line_older = generate_action_line(older, CURRENT_BG_INFO)
                out_older = f"output/JHHRJ_sister_older_init_dialog1.wav"
                generate_tts(older, line_older, out_older)
                play_audio(out_older)
                
                line_younger = generate_second_dialogue_line(younger, line_older, CURRENT_BG_INFO)
                out_younger = f"output/JHHRJ_sister_younger_init_dialog2.wav"
                generate_tts(younger, line_younger, out_younger)
                play_audio(out_younger)
            else:
                # 홍련 먼저
                line_younger = generate_action_line(younger, CURRENT_BG_INFO)
                out_younger = f"output/JHHRJ_sister_younger_init_dialog1.wav"
                generate_tts(younger, line_younger, out_younger)
                play_audio(out_younger)
                
                line_older = generate_second_dialogue_line(older, line_younger, CURRENT_BG_INFO)
                out_older = f"output/JHHRJ_sister_older_init_dialog2.wav"
                generate_tts(older, line_older, out_older)
                play_audio(out_older)
            
            # cha2가 자매의 대화에 반응
            line_cha2 = generate_second_dialogue_line(cha2, line_younger if random.random() < 0.5 else line_older, CURRENT_BG_INFO)
            out_cha2 = f"output/{book_code}_{role_key}_init_dialog3.wav"
            generate_tts(cha2, line_cha2, out_cha2)
            play_audio(out_cha2)
        else:
            # 새로 등장하는 cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            # 첫 번째 대화 생성 및 재생
            line2 = generate_first_dialogue_line(cha2, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_init_dialog1.wav"
            generate_tts(cha2, line2, out2)
            play_audio(out2)
            
            # 두 번째 대화 생성 및 재생
            line1 = generate_second_dialogue_line(CURRENT_CHA1_INFO, line2, CURRENT_BG_INFO)
            out1 = f"output/{CURRENT_CHA1_INFO['book_code']}_{CURRENT_CHA1_INFO['role_key']}_init_dialog2.wav"
            generate_tts(CURRENT_CHA1_INFO, line1, out1)
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
        # 배경만 교체하고 오버레이 비디오(ch1, ch2)는 그대로 유지
        play_background_video(book_code)  # 배경 비디오 교체 (무한 루프, 오디오 포함, 페이드 효과)

        # 배경이 바뀔 때 사운드 이펙트만 재생 (제목 말하기는 마커 감지 시에만 재생)
        sound_effect_path = "soundeffect/ES_Dream, Harp - Epidemic Sound.wav"
        
        def play_sound():
            # ES_Dream 사운드 이펙트를 음량 20%로 처리한 임시 파일 생성 및 재생
            if os.path.exists(sound_effect_path):
                try:
                    os.makedirs("title_saying", exist_ok=True)
                    temp_sound = f"title_saying/temp_sound_{book_code}.wav"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", sound_effect_path,
                         "-af", "volume=0.2",
                         temp_sound],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True
                    )
                    # 사운드 이펙트 재생
                    subprocess.run(["afplay", temp_sound],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                    os.remove(temp_sound)
                    print(f"🔊 사운드 효과 재생 (음량 20%): {sound_effect_path}")
                except Exception as e:
                    print(f"⚠️ 사운드 효과 재생 실패: {e}")
        
        # 비동기로 재생 (블로킹 방지)
        threading.Thread(target=play_sound, daemon=True).start()

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

        # 🔸 장화홍련 자매인 경우: 랜덤 순서로 각각 한 줄씩 말하고,
        #    기존 cha2(예: 토끼, 귀신 등)가 한 줄 더 대답.
        if book_code == "JHHRJ" and role_key == "sister_older":
            sister_older, sister_younger = build_sisters_pair()

            # 랜덤으로 순서 결정
            if random.random() < 0.5:
                # 언니 → 동생 순서
                lineA = generate_first_dialogue_line(sister_older, CURRENT_BG_INFO)
                outA = "output/JHHRJ_sister_older_line.wav"
                generate_tts(sister_older, lineA, outA)
                play_audio(outA)
                
                lineB = generate_second_dialogue_line(sister_younger, lineA, CURRENT_BG_INFO)
                outB = "output/JHHRJ_sister_younger_line.wav"
                generate_tts(sister_younger, lineB, outB)
                play_audio(outB)
            else:
                # 동생 → 언니 순서
                lineB = generate_first_dialogue_line(sister_younger, CURRENT_BG_INFO)
                outB = "output/JHHRJ_sister_younger_line.wav"
                generate_tts(sister_younger, lineB, outB)
                play_audio(outB)
                
                lineA = generate_second_dialogue_line(sister_older, lineB, CURRENT_BG_INFO)
                outA = "output/JHHRJ_sister_older_line.wav"
                generate_tts(sister_older, lineA, outA)
                play_audio(outA)
            
            # cha2가 자매의 대화에 반응
            reply = generate_second_dialogue_line(CURRENT_CHA2_INFO, lineA if random.random() < 0.5 else lineB, CURRENT_BG_INFO)
            outC = f"output/{CURRENT_CHA2_INFO['book_code']}_{CURRENT_CHA2_INFO['role_key']}_reply_to_sisters.wav"
            generate_tts(CURRENT_CHA2_INFO, reply, outC)
            play_audio(outC)
            return

        # 🔹 그 외 일반 캐릭터: 새 cha1 + 기존 cha2가 한 줄씩 대화
        # 첫 번째 대화 생성 및 재생
        line1 = generate_first_dialogue_line(cha1, CURRENT_BG_INFO)
        out1 = f"output/{book_code}_{role_key}_swapcha1_dialog1.wav"
        generate_tts(cha1, line1, out1)
        play_audio(out1)
        
        # 두 번째 대화 생성 및 재생
        line2 = generate_second_dialogue_line(CURRENT_CHA2_INFO, line1, CURRENT_BG_INFO)
        out2 = f"output/{CURRENT_CHA2_INFO['book_code']}_{CURRENT_CHA2_INFO['role_key']}_swapcha1_dialog2.wav"
        generate_tts(CURRENT_CHA2_INFO, line2, out2)
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

        # cha1이 장화홍련인 경우: 자매가 랜덤 순서로 말함
        if CURRENT_CHA1_INFO['book_code'] == "JHHRJ":
            older, younger = build_sisters_pair()
            
            # 랜덤으로 순서 결정
            if random.random() < 0.5:
                # 장화 먼저
                line_older = generate_action_line(older, CURRENT_BG_INFO)
                out_older = f"output/JHHRJ_sister_older_swapcha2_dialog1.wav"
                generate_tts(older, line_older, out_older)
                play_audio(out_older)
                
                line_younger = generate_second_dialogue_line(younger, line_older, CURRENT_BG_INFO)
                out_younger = f"output/JHHRJ_sister_younger_swapcha2_dialog2.wav"
                generate_tts(younger, line_younger, out_younger)
                play_audio(out_younger)
            else:
                # 홍련 먼저
                line_younger = generate_action_line(younger, CURRENT_BG_INFO)
                out_younger = f"output/JHHRJ_sister_younger_swapcha2_dialog1.wav"
                generate_tts(younger, line_younger, out_younger)
                play_audio(out_younger)
                
                line_older = generate_second_dialogue_line(older, line_younger, CURRENT_BG_INFO)
                out_older = f"output/JHHRJ_sister_older_swapcha2_dialog2.wav"
                generate_tts(older, line_older, out_older)
                play_audio(out_older)
            
            # cha2가 자매의 대화에 반응
            line_cha2 = generate_second_dialogue_line(cha2, line_younger if random.random() < 0.5 else line_older, CURRENT_BG_INFO)
            out_cha2 = f"output/{book_code}_{role_key}_swapcha2_dialog3.wav"
            generate_tts(cha2, line_cha2, out_cha2)
            play_audio(out_cha2)
        else:
            # cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            # 첫 번째 대화 생성 및 재생
            line2 = generate_first_dialogue_line(cha2, CURRENT_BG_INFO)
            out2 = f"output/{book_code}_{role_key}_swapcha2_dialog1.wav"
            generate_tts(cha2, line2, out2)
            play_audio(out2)
            
            # 두 번째 대화 생성 및 재생
            line1 = generate_second_dialogue_line(CURRENT_CHA1_INFO, line2, CURRENT_BG_INFO)
            out1 = f"output/{CURRENT_CHA1_INFO['book_code']}_{CURRENT_CHA1_INFO['role_key']}_swapcha2_dialog2.wav"
            generate_tts(CURRENT_CHA1_INFO, line1, out1)
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
    
    # 비디오 윈도우 생성 (팝업창)
    cv2.namedWindow("Background Video", cv2.WINDOW_NORMAL)
    # 창 크기 설정 (예: 1280x720)
    cv2.resizeWindow("Background Video", 1280, 720)
    
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
                
                # 마커 감지 즉시 제목 말하기 재생 (배경이 바뀔 때만 사운드 이펙트 포함)
                title_saying_path = f"title_saying/{book_code}_title.wav"
                
                def play_title():
                    # 제목 말하기 재생 (음량 150%)
                    if os.path.exists(title_saying_path):
                        # 음량 150%로 조정한 임시 파일 생성
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        temp_title = os.path.join(temp_dir, f"title_{os.getpid()}_{id(title_saying_path)}.wav")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", title_saying_path,
                             "-af", "volume=1.5",
                             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                             temp_title],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=True
                        )
                        # 조정된 파일 재생
                        subprocess.run(["afplay", temp_title],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                        # 임시 파일 삭제
                        try:
                            os.remove(temp_title)
                        except:
                            pass
                        print(f"📚 제목 말하기 재생 (음량 150%): {title_saying_path}")
                    
                    # 제목 말하기가 끝난 후 BGM 재생
                    if VIDEO_PLAYER.pending_bgm_path:
                        bgm_path = VIDEO_PLAYER.pending_bgm_path
                        VIDEO_PLAYER.pending_bgm_path = None
                        VIDEO_PLAYER._start_bgm(bgm_path)
                
                # 비동기로 재생 (블로킹 방지)
                threading.Thread(target=play_title, daemon=True).start()
                
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