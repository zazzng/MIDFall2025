import os
import json
import subprocess
import threading
import time
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import cv2.aruco as aruco
from PIL import Image, ImageDraw, ImageFont

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

# 현재 재생 중인 오디오 프로세스 추적
_current_audio_processes = []  # 현재 재생 중인 모든 오디오 프로세스
_audio_processes_lock = threading.Lock()  # 오디오 프로세스 리스트 보호용 락
_should_stop_audio = False  # 오디오 재생 중단 플래그
_stop_audio_lock = threading.Lock()  # 중단 플래그 보호용 락

# 비디오 플레이어 (스레드 기반)
class VideoPlayer:
    """OpenCV 기반 비디오 플레이어 (별도 스레드에서 무한 루프 재생)"""
    
    def __init__(self):
        self.current_video_path = None
        self.video_cap = None
        self.next_video_path = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
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
        self.frame_accumulator = 0.0  # 프레임 누적 시간 (드롭 보상용)
        self.current_subtitle_text = None  # 현재 자막 텍스트 (예: "toad: Haha")
        self.current_subtitle_lock = threading.Lock()  # 자막 정보 보호용 락
        
        # 폰트 캐시 (성능 최적화)
        self._subtitle_cache = {}  # (text, width, font_scale) -> lines
        self._last_frame_size = None  # 마지막 프레임 크기 (폰트 재계산 방지)
    
    def _play_loop(self):
        """비디오 재생 루프 (별도 스레드에서 실행)"""
        import time as time_module
        while self.running:
            loop_start_time = time_module.perf_counter()
            
            # 비디오 전환 처리 (페이드와 독립적으로, 즉시 처리)
            next_path = None
            old_cap_to_release = None
            with self.lock:
                if self.next_video_path == "":
                    # 페이드 아웃 요청
                    elapsed = time_module.time() - self.fade_start_time if self.fade_start_time else 0
                    if elapsed >= self.fade_duration:
                        # 페이드 아웃 완료: 비디오 해제
                        if self.video_cap:
                            old_cap_to_release = self.video_cap
                            self.video_cap = None
                            self.current_video_path = None
                        self.next_video_path = None
                        self.is_fading = False
                        self.fade_alpha = 1.0
                        self.fade_start_time = None
                elif self.next_video_path is not None:
                    # 비디오 전환 요청
                    next_path = self.next_video_path
                    self.next_video_path = None  # 즉시 클리어하여 중복 처리 방지
            
            # lock 밖에서 비디오 해제 (페이드 아웃)
            if old_cap_to_release is not None:
                try:
                    if old_cap_to_release.isOpened():
                        old_cap_to_release.release()
                except:
                    pass
            
            if next_path is not None:
                # 비디오 전환 즉시 처리
                old_cap = None
                with self.lock:
                    old_cap = self.video_cap
                    self.video_cap = None  # 먼저 None으로 설정하여 _play_loop가 검은 프레임 표시
                
                # lock 밖에서 기존 비디오 해제
                if old_cap is not None:
                    try:
                        if old_cap.isOpened():
                            old_cap.release()
                    except:
                        pass
                
                # 새 비디오 열기 (lock 밖에서, 시간이 걸릴 수 있음)
                new_cap = cv2.VideoCapture(next_path)
                if new_cap.isOpened():
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fps = new_cap.get(cv2.CAP_PROP_FPS)
                    # 비디오가 성공적으로 열린 후에만 경로와 캡처 객체 설정
                    with self.lock:
                        self.current_video_path = next_path
                        self.bg_fps = fps if fps > 0 else 30.0
                        self.video_cap = new_cap
                    print(f"🎬 비디오 전환 완료: {os.path.basename(next_path)} (FPS: {self.bg_fps:.2f})")
                else:
                    print(f"❌ 비디오를 열 수 없음: {next_path}")
                    with self.lock:
                        self.video_cap = None
                        self.current_video_path = None
                        self.bg_fps = 30.0
            
            # 페이드 효과 계산 (시각 효과만)
            fade_alpha = 1.0
            if self.is_fading and self.fade_start_time:
                elapsed = time_module.time() - self.fade_start_time
                if elapsed < self.fade_duration:
                    # 페이드 아웃: 1.0 -> 0.0
                    fade_alpha = 1.0 - (elapsed / self.fade_duration)
                elif elapsed < self.fade_duration * 2:
                    # 페이드 인: 0.0 -> 1.0
                    fade_alpha = (elapsed - self.fade_duration) / self.fade_duration
                else:
                    # 페이드 완료
                    with self.lock:
                        self.is_fading = False
                        self.fade_alpha = 1.0
                        self.fade_start_time = None
            
            # 프레임 읽기 및 처리 (lock 최소화)
            frame = None
            with self.lock:
                # 비디오 캡처 객체 참조만 가져오기 (lock 안에서 최소한만)
                # 오버레이 비디오는 직접 참조하지 않고, 매번 lock 안에서 확인
                video_cap = self.video_cap
                self.fade_alpha = fade_alpha
            
            # 비디오가 없으면 검은 프레임 생성
            if video_cap is None:
                # 검은 프레임 생성 (기본 해상도 1280x720)
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # 페이드 효과 적용 (페이드 아웃 중이면 검은 화면 유지)
                if self.is_fading and fade_alpha < 1.0:
                    # 페이드 아웃 중이면 검은 화면
                    pass  # 이미 검은 프레임이므로 추가 처리 불필요
                # 검은 프레임은 오버레이 없이 바로 저장
                with self.lock:
                    self.frame = frame
                # 기본 프레임 간격 설정 (30 FPS)
                frame_interval = 1.0 / 30.0
                # 프레임 처리 시간 고려하여 정확한 타이밍으로 재생
                elapsed = time_module.perf_counter() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    if sleep_time < 0.001:
                        time_module.sleep(0)
                    else:
                        time_module.sleep(sleep_time)
                continue  # 다음 루프로
            
            # video_cap이 있지만 열려있지 않은 경우도 체크
            try:
                is_opened = video_cap.isOpened()
            except:
                is_opened = False
            
            if not is_opened:
                # 검은 프레임 생성 (기본 해상도 1280x720)
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                # 페이드 효과 적용 (페이드 아웃 중이면 검은 화면 유지)
                if self.is_fading and fade_alpha < 1.0:
                    # 페이드 아웃 중이면 검은 화면
                    pass  # 이미 검은 프레임이므로 추가 처리 불필요
                # 검은 프레임은 오버레이 없이 바로 저장
                with self.lock:
                    self.frame = frame
                # 기본 프레임 간격 설정 (30 FPS)
                frame_interval = 1.0 / 30.0
                # 프레임 처리 시간 고려하여 정확한 타이밍으로 재생
                elapsed = time_module.perf_counter() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    if sleep_time < 0.001:
                        time_module.sleep(0)
                    else:
                        time_module.sleep(sleep_time)
                continue  # 다음 루프로
            else:
                # 실제 비디오 FPS에 맞춰 프레임 간격 조정 (먼저 계산)
                with self.lock:
                    overlay_cap = self.overlay_video_cap
                    overlay_cap2 = self.overlay_video_cap2
                
                # 오버레이가 없을 때는 배경 비디오 FPS만 사용
                if overlay_cap is None and overlay_cap2 is None:
                    # 오버레이가 없으면 배경 비디오의 실제 FPS 사용
                    target_fps = self.bg_fps if self.bg_fps > 0 else 30.0
                else:
                    # 오버레이가 있으면 가장 높은 FPS 사용 (동기화를 위해)
                    target_fps = max(self.bg_fps, 
                                   self.overlay_fps if overlay_cap and overlay_cap.isOpened() else 0,
                                   self.overlay_fps2 if overlay_cap2 and overlay_cap2.isOpened() else 0)
                    if target_fps <= 0:
                        target_fps = self.bg_fps if self.bg_fps > 0 else 30.0  # 기본값은 배경 비디오 FPS
                
                frame_interval = 1.0 / target_fps
                
                # lock 밖에서 프레임 읽기 (비디오 I/O는 느릴 수 있음)
                ret, frame = video_cap.read()
                if not ret:
                    # 비디오 끝나면 처음으로 돌아가기 (무한 루프)
                    with self.lock:
                        if self.video_cap:
                            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = video_cap.read()
                
                if ret:
                    # 페이드 효과 적용
                    if self.is_fading and fade_alpha < 1.0:
                        black_frame = frame.copy()
                        black_frame.fill(0)
                        frame = cv2.addWeighted(frame, fade_alpha, black_frame, 1.0 - fade_alpha, 0)
                    
                    # 페이드 중일 때는 오버레이를 표시하지 않음 (까만 화면에 캐릭터가 보이지 않도록)
                    if not (self.is_fading and fade_alpha < 1.0):
                        # 오버레이 비디오 처리 순서: ch2 먼저 (뒤 레이어), ch1 나중 (앞 레이어)
                        # ch2 오버레이 비디오 처리 (뒤 레이어) - 매번 lock에서 최신 참조 가져오기
                        overlay_cap2 = None
                        overlay_ret2 = False
                        overlay_frame2 = None
                        
                        with self.lock:
                            if self.overlay_video_cap2 is not None:
                                try:
                                    # 참조를 가져오고 즉시 유효성 확인 (안전하게)
                                    cap2_ref = self.overlay_video_cap2
                                    if cap2_ref is not None:
                                        try:
                                            if cap2_ref.isOpened():
                                                overlay_cap2 = cap2_ref
                                            else:
                                                self.overlay_video_cap2 = None
                                        except:
                                            # isOpened() 호출 중 오류 (비디오가 해제되는 중일 수 있음)
                                            self.overlay_video_cap2 = None
                                except:
                                    self.overlay_video_cap2 = None
                        
                        if overlay_cap2 is not None:
                            try:
                                # 비디오 캡처가 여전히 유효한지 확인
                                try:
                                    if not overlay_cap2.isOpened():
                                        overlay_ret2 = False
                                        overlay_frame2 = None
                                        with self.lock:
                                            if self.overlay_video_cap2 == overlay_cap2:
                                                self.overlay_video_cap2 = None
                                    else:
                                        overlay_ret2, overlay_frame2 = overlay_cap2.read()
                                        if not overlay_ret2:
                                            try:
                                                overlay_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                                overlay_ret2, overlay_frame2 = overlay_cap2.read()
                                            except:
                                                overlay_ret2 = False
                                                overlay_frame2 = None
                                except:
                                    # isOpened() 또는 read() 중 오류 (비디오가 해제되는 중일 수 있음)
                                    overlay_ret2 = False
                                    overlay_frame2 = None
                                    with self.lock:
                                        if self.overlay_video_cap2 == overlay_cap2:
                                            self.overlay_video_cap2 = None
                            except Exception as e:
                                # 오류 발생 시 안전하게 처리
                                overlay_ret2 = False
                                overlay_frame2 = None
                                with self.lock:
                                    if self.overlay_video_cap2 == overlay_cap2:
                                        self.overlay_video_cap2 = None
                            
                            if overlay_ret2 and overlay_cap2 is not None and overlay_frame2 is not None:
                                try:
                                    # 오버레이 프레임 크기를 배경 프레임 크기에 맞춤
                                    if overlay_frame2.shape[:2] != frame.shape[:2]:
                                        overlay_frame2 = cv2.resize(overlay_frame2, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    
                                    # 알파 채널이 있으면 알파 블렌딩, 없으면 일반 오버레이
                                    if len(overlay_frame2.shape) == 3 and overlay_frame2.shape[2] == 4:
                                        # RGBA -> BGR 변환
                                        overlay_bgr2 = overlay_frame2[:, :, :3]
                                        # 알파 마스크 추출 (uint8)
                                        alpha2 = overlay_frame2[:, :, 3]
                                        # 알파가 0이 아닌 영역만 블렌딩 (성능 최적화)
                                        mask2_alpha = alpha2 > 0
                                        if np.any(mask2_alpha):
                                            # 알파를 float로 변환 (0-1 범위)
                                            alpha2_f = alpha2.astype(np.float32) / 255.0
                                            alpha_3d2 = alpha2_f[:, :, None]  # np.newaxis 대신 None 사용
                                            # 알파 블렌딩 (벡터화된 연산)
                                            frame = (frame.astype(np.float32) * (1 - alpha_3d2) + overlay_bgr2.astype(np.float32) * alpha_3d2).astype(np.uint8)
                                    elif len(overlay_frame2.shape) == 3:
                                        # 그레이스케일 마스크 생성 및 블렌딩
                                        mask2 = cv2.cvtColor(overlay_frame2, cv2.COLOR_BGR2GRAY)
                                        _, mask2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)
                                        # 마스크가 있는 영역만 오버레이 복사 (더 빠름)
                                        cv2.copyTo(overlay_frame2, mask2, frame)
                                except Exception as e:
                                    print(f"⚠️ ch2 오버레이 처리 중 오류: {e}")
                        
                        # ch1 오버레이 비디오 처리 (앞 레이어 - 마지막에 적용하여 항상 앞에 표시)
                        # 매번 lock에서 최신 참조 가져오기
                        overlay_cap = None
                        overlay_ret = False
                        overlay_frame = None
                        
                        with self.lock:
                            if self.overlay_video_cap is not None:
                                try:
                                    # 참조를 가져오고 즉시 유효성 확인 (안전하게)
                                    cap_ref = self.overlay_video_cap
                                    if cap_ref is not None:
                                        try:
                                            if cap_ref.isOpened():
                                                overlay_cap = cap_ref
                                            else:
                                                self.overlay_video_cap = None
                                        except:
                                            # isOpened() 호출 중 오류 (비디오가 해제되는 중일 수 있음)
                                            self.overlay_video_cap = None
                                except:
                                    self.overlay_video_cap = None
                        
                        if overlay_cap is not None:
                            try:
                                # 비디오 캡처가 여전히 유효한지 확인
                                try:
                                    if not overlay_cap.isOpened():
                                        overlay_ret = False
                                        overlay_frame = None
                                        with self.lock:
                                            if self.overlay_video_cap == overlay_cap:
                                                self.overlay_video_cap = None
                                    else:
                                        overlay_ret, overlay_frame = overlay_cap.read()
                                        if not overlay_ret:
                                            try:
                                                overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                                overlay_ret, overlay_frame = overlay_cap.read()
                                            except:
                                                overlay_ret = False
                                                overlay_frame = None
                                except:
                                    # isOpened() 또는 read() 중 오류 (비디오가 해제되는 중일 수 있음)
                                    overlay_ret = False
                                    overlay_frame = None
                                    with self.lock:
                                        if self.overlay_video_cap == overlay_cap:
                                            self.overlay_video_cap = None
                            except Exception as e:
                                # 오류 발생 시 안전하게 처리
                                overlay_ret = False
                                overlay_frame = None
                                with self.lock:
                                    if self.overlay_video_cap == overlay_cap:
                                        self.overlay_video_cap = None
                            
                            if overlay_ret and overlay_cap is not None and overlay_frame is not None:
                                try:
                                    # 오버레이 프레임 크기를 배경 프레임 크기에 맞춤
                                    if overlay_frame.shape[:2] != frame.shape[:2]:
                                        overlay_frame = cv2.resize(overlay_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    
                                    # 알파 채널이 있으면 알파 블렌딩, 없으면 일반 오버레이
                                    if len(overlay_frame.shape) == 3 and overlay_frame.shape[2] == 4:
                                        # RGBA -> BGR 변환
                                        overlay_bgr = overlay_frame[:, :, :3]
                                        # 알파 마스크 추출 (uint8)
                                        alpha = overlay_frame[:, :, 3]
                                        # 알파가 0이 아닌 영역만 블렌딩 (성능 최적화)
                                        mask = alpha > 0
                                        if np.any(mask):
                                            # 알파를 float로 변환 (0-1 범위)
                                            alpha_f = alpha.astype(np.float32) / 255.0
                                            alpha_3d = alpha_f[:, :, None]  # np.newaxis 대신 None 사용
                                            # 알파 블렌딩 (벡터화된 연산) - ch1은 항상 앞 레이어
                                            frame = (frame.astype(np.float32) * (1 - alpha_3d) + overlay_bgr.astype(np.float32) * alpha_3d).astype(np.uint8)
                                    elif len(overlay_frame.shape) == 3:
                                        # 그레이스케일 마스크 생성 및 블렌딩
                                        mask = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2GRAY)
                                        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                                        # 마스크가 있는 영역만 오버레이 복사 (더 빠름) - ch1은 항상 앞 레이어
                                        cv2.copyTo(overlay_frame, mask, frame)
                                except Exception as e:
                                    print(f"⚠️ ch1 오버레이 처리 중 오류: {e}")
                            elif overlay_cap is None:
                                # 디버깅: ch1 오버레이가 None인 경우 (첫 프레임에서만 출력)
                                pass
                    
                    # 최종 프레임 저장 (lock 안에서)
                    with self.lock:
                        self.frame = frame
            
                # 프레임 처리 시간 고려하여 정확한 타이밍으로 재생 (perf_counter 사용)
                elapsed = time_module.perf_counter() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
            
            # 프레임 드롭 보상: 처리 시간이 프레임 간격보다 길면 다음 프레임을 즉시 읽기
            if elapsed > frame_interval * 1.5:
                # 프레임이 너무 늦으면 누적 시간 초기화하고 계속 진행
                self.frame_accumulator = 0.0
                # 다음 프레임을 즉시 읽기 위해 sleep 건너뛰기
            else:
                # 정상적인 경우 sleep
                if sleep_time > 0:
                    # 작은 sleep 시간은 더 정확하게 처리
                    if sleep_time < 0.001:
                        time_module.sleep(0)  # yield to other threads
                    else:
                        time_module.sleep(sleep_time)
    
    def start(self):
        """플레이어 시작"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()
    
    def set_overlay_video(self, overlay_path: str):
        """오버레이 비디오 ch1 설정 (배경 위에 표시될 캐릭터 움직임)"""
        # 기존 오버레이 비디오 해제 (lock 밖에서 먼저 해제)
        old_cap = None
        with self.lock:
            old_cap = self.overlay_video_cap
            self.overlay_video_cap = None  # 먼저 None으로 설정하여 재생 루프에서 사용하지 않도록
            self.overlay_video_path = None
        
        # lock 밖에서 해제 (재생 루프와의 충돌 방지)
        # 짧은 대기로 재생 루프가 참조를 놓도록 함
        import time
        time.sleep(0.1)  # 재생 루프가 현재 프레임 처리를 완료할 시간 제공
        
        if old_cap is not None:
            try:
                # 안전하게 해제
                if hasattr(old_cap, 'isOpened'):
                    try:
                        if old_cap.isOpened():
                            old_cap.release()
                    except:
                        pass  # 이미 해제되었거나 오류 발생
                else:
                    try:
                        old_cap.release()
                    except:
                        pass
            except:
                pass  # 해제 중 오류는 무시
            finally:
                old_cap = None
        
        if overlay_path and os.path.exists(overlay_path):
            with self.lock:
                self.overlay_video_path = overlay_path
                self.overlay_video_cap = cv2.VideoCapture(overlay_path)
                if not self.overlay_video_cap.isOpened():
                    print(f"❌ 오버레이 비디오를 열 수 없음: {overlay_path}")
                    self.overlay_video_cap = None
                    self.overlay_video_path = None
                    self.overlay_fps = 30.0  # 기본값
                else:
                    # 비디오 캡처 최적화 설정
                    self.overlay_video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # 비디오를 처음부터 재생하도록 설정
                    self.overlay_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # FPS 정보를 ffprobe로 먼저 시도
                    try:
                        probe_cmd = [
                            "ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate",
                            "-of", "default=noprint_wrappers=1:nokey=1",
                            overlay_path
                        ]
                        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=2)
                        if result.returncode == 0:
                            fps_str = result.stdout.strip()
                            if '/' in fps_str:
                                num, den = map(int, fps_str.split('/'))
                                self.overlay_fps = num / den if den > 0 else 30.0
                            else:
                                self.overlay_fps = float(fps_str) if fps_str else 30.0
                        else:
                            fps = self.overlay_video_cap.get(cv2.CAP_PROP_FPS)
                            self.overlay_fps = fps if fps > 0 else 30.0
                    except:
                        fps = self.overlay_video_cap.get(cv2.CAP_PROP_FPS)
                        self.overlay_fps = fps if fps > 0 else 30.0
                    print(f"🎬 오버레이 비디오 ch1 설정 완료: {overlay_path} (FPS: {self.overlay_fps:.2f}, 열림: {self.overlay_video_cap.isOpened()})")
        else:
            with self.lock:
                self.overlay_video_cap = None
                self.overlay_video_path = None
    
    def set_overlay_video2(self, overlay_path: str):
        """오버레이 비디오 ch2 설정 (배경 위에 표시될 캐릭터 움직임)"""
        # 기존 오버레이 비디오 ch2 해제 (lock 밖에서 먼저 해제)
        old_cap2 = None
        with self.lock:
            old_cap2 = self.overlay_video_cap2
            self.overlay_video_cap2 = None  # 먼저 None으로 설정하여 재생 루프에서 사용하지 않도록
            self.overlay_video_path2 = None
        
        # lock 밖에서 해제 (재생 루프와의 충돌 방지)
        # 짧은 대기로 재생 루프가 참조를 놓도록 함
        import time
        time.sleep(0.1)  # 재생 루프가 현재 프레임 처리를 완료할 시간 제공
        
        if old_cap2 is not None:
            try:
                # 안전하게 해제
                if hasattr(old_cap2, 'isOpened'):
                    try:
                        if old_cap2.isOpened():
                            old_cap2.release()
                    except:
                        pass  # 이미 해제되었거나 오류 발생
                else:
                    try:
                        old_cap2.release()
                    except:
                        pass
            except:
                pass  # 해제 중 오류는 무시
            finally:
                old_cap2 = None
        
        if overlay_path and os.path.exists(overlay_path):
            with self.lock:
                self.overlay_video_path2 = overlay_path
                self.overlay_video_cap2 = cv2.VideoCapture(overlay_path)
                if not self.overlay_video_cap2.isOpened():
                    print(f"❌ 오버레이 비디오 ch2를 열 수 없음: {overlay_path}")
                    self.overlay_video_cap2 = None
                    self.overlay_video_path2 = None
                    self.overlay_fps2 = 30.0  # 기본값
                else:
                    # 비디오 캡처 최적화 설정
                    self.overlay_video_cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # 비디오를 처음부터 재생하도록 설정
                    self.overlay_video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # FPS 정보를 ffprobe로 먼저 시도
                    try:
                        probe_cmd = [
                            "ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate",
                            "-of", "default=noprint_wrappers=1:nokey=1",
                            overlay_path
                        ]
                        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=2)
                        if result.returncode == 0:
                            fps_str = result.stdout.strip()
                            if '/' in fps_str:
                                num, den = map(int, fps_str.split('/'))
                                self.overlay_fps2 = num / den if den > 0 else 30.0
                            else:
                                self.overlay_fps2 = float(fps_str) if fps_str else 30.0
                        else:
                            fps = self.overlay_video_cap2.get(cv2.CAP_PROP_FPS)
                            self.overlay_fps2 = fps if fps > 0 else 30.0
                    except:
                        fps = self.overlay_video_cap2.get(cv2.CAP_PROP_FPS)
                        self.overlay_fps2 = fps if fps > 0 else 30.0
                    print(f"🎬 오버레이 비디오 ch2 설정 완료: {overlay_path} (FPS: {self.overlay_fps2:.2f}, 열림: {self.overlay_video_cap2.isOpened()})")
        else:
            with self.lock:
                self.overlay_video_cap2 = None
                self.overlay_video_path2 = None
    
    def clear_overlay_video(self):
        """오버레이 비디오 모두 제거"""
        # 기존 오버레이 비디오 해제 (lock 밖에서 먼저 해제)
        old_cap = None
        old_cap2 = None
        with self.lock:
            old_cap = self.overlay_video_cap
            old_cap2 = self.overlay_video_cap2
            self.overlay_video_cap = None  # 먼저 None으로 설정하여 재생 루프에서 사용하지 않도록
            self.overlay_video_cap2 = None
            self.overlay_video_path = None
            self.overlay_video_path2 = None
        
        # lock 밖에서 해제 (재생 루프와의 충돌 방지)
        import time
        time.sleep(0.1)  # 재생 루프가 현재 프레임 처리를 완료할 시간 제공
        
        if old_cap is not None:
            try:
                if hasattr(old_cap, 'isOpened'):
                    try:
                        if old_cap.isOpened():
                            old_cap.release()
                    except:
                        pass
                else:
                    try:
                        old_cap.release()
                    except:
                        pass
            except:
                pass
            finally:
                old_cap = None
        
        if old_cap2 is not None:
            try:
                if hasattr(old_cap2, 'isOpened'):
                    try:
                        if old_cap2.isOpened():
                            old_cap2.release()
                    except:
                        pass
                else:
                    try:
                        old_cap2.release()
                    except:
                        pass
            except:
                pass
            finally:
                old_cap2 = None
        
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
    
    def set_video(self, video_path: str):
        """비디오 파일 변경 (페이드 효과와 함께 부드러운 전환). None을 전달하면 페이드 아웃 (검은 화면)"""
        if video_path is None:
            # None이면 페이드 아웃 (검은 화면)
            with self.lock:
                self.next_video_path = ""  # 빈 문자열로 페이드 아웃 표시
                self.is_fading = True
                self.fade_start_time = time.time()
            return
        
        # 첫 번째 비디오인지 확인
        with self.lock:
            is_first = (self.current_video_path is None)
        
        if is_first:
            # 첫 번째 비디오는 페이드 없이 바로 시작
            new_cap = cv2.VideoCapture(video_path)
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fps = new_cap.get(cv2.CAP_PROP_FPS)
                with self.lock:
                    self.current_video_path = video_path
                    self.bg_fps = fps if fps > 0 else 30.0
                    self.video_cap = new_cap
                print(f"🎬 첫 비디오 시작: {os.path.basename(video_path)} (FPS: {self.bg_fps:.2f})")
            else:
                print(f"❌ 비디오를 열 수 없음: {video_path}")
                with self.lock:
                    self.video_cap = None
                    self.bg_fps = 30.0
        else:
            # 다음 비디오로 전환 (페이드 효과)
            with self.lock:
                self.next_video_path = video_path
                self.is_fading = True
                self.fade_start_time = time.time()
    
    def set_subtitle(self, subtitle_text: str):
        """자막 텍스트를 설정합니다."""
        with self.current_subtitle_lock:
            self.current_subtitle_text = subtitle_text
    
    def clear_subtitle(self):
        """자막을 지웁니다."""
        with self.current_subtitle_lock:
            self.current_subtitle_text = None
    
    def _wrap_text_cv2(self, text, font_scale, thickness, max_width):
        """OpenCV를 사용하여 텍스트를 화면 너비에 맞게 줄바꿈 (캐시 사용)."""
        cache_key = (text, max_width, font_scale)
        if cache_key in self._subtitle_cache:
            return self._subtitle_cache[cache_key]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (text_width, text_height), baseline = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        if len(lines) == 0:
            lines = [text]
        elif len(lines) > 2:
            lines = lines[:2]
        
        self._subtitle_cache[cache_key] = lines
        return lines
    
    def _draw_subtitle(self, frame):
        """프레임에 자막을 그립니다 (제일 위 레이어). OpenCV 기본 폰트 사용 (최고 성능)."""
        with self.current_subtitle_lock:
            subtitle_text = self.current_subtitle_text
        
        # 자막이 없으면 프레임 그대로 반환
        if subtitle_text is None or subtitle_text == "":
            return frame
        
        h, w = frame.shape[:2]
        frame_size = (h, w)
        
        # 프레임 크기가 바뀌면 캐시 클리어
        if self._last_frame_size != frame_size:
            self._subtitle_cache.clear()
            self._last_frame_size = frame_size
        
        frame_with_subtitle = frame.copy()
        
        # OpenCV 기본 폰트 사용 (PIL보다 훨씬 빠름)
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA
        
        # 일반 자막 그리기 (하단)
        if subtitle_text and subtitle_text != "":
            font_scale = h / 720.0 * 0.8  # 720p 기준으로 스케일링
            thickness = max(1, int(h / 360.0))
            max_width = int(w * 0.9)
            
            lines = self._wrap_text_cv2(subtitle_text, font_scale, thickness, max_width)
            
            # 각 줄의 높이 계산
            line_height = 0
            for line in lines:
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                line_height = max(line_height, text_height)
            
            line_spacing = int(line_height * 0.3)
            total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing
            y_start = h - 64 - total_height
            
            # 각 줄을 그리기
            for i, line in enumerate(lines):
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                x = (w - text_width) // 2
                y = y_start + i * (line_height + line_spacing) + text_height
                
                # 검은색 stroke (외곽선) 그리기 - 8방향만
                stroke_width = 2
                stroke_offsets = [
                    (-stroke_width, -stroke_width), (-stroke_width, 0), (-stroke_width, stroke_width),
                    (0, -stroke_width), (0, stroke_width),
                    (stroke_width, -stroke_width), (stroke_width, 0), (stroke_width, stroke_width)
                ]
                for dx, dy in stroke_offsets:
                    cv2.putText(frame_with_subtitle, line, (x + dx, y + dy), font, font_scale,
                               (0, 0, 0), thickness + 1, line_type)
                
                # 흰색 fill (본문) 그리기
                cv2.putText(frame_with_subtitle, line, (x, y), font, font_scale,
                           (255, 255, 255), thickness, line_type)
        
        return frame_with_subtitle
    
    def get_frame(self):
        """현재 프레임 가져오기"""
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()
                # 자막 그리기 (제일 위 레이어)
                frame = self._draw_subtitle(frame)
                return frame
        return None

# 전역 비디오 플레이어 인스턴스
VIDEO_PLAYER = VideoPlayer()

# 배경 비디오 설정
BG_VIDEO_DIR = "bg_video"
BOOK_TO_VIDEO = {
    "BJBJ": "10bgBJBJ.mov",
    "PSJ": "11bgPSJ.mov",
    "DGJ": "13bgDGJ.mov",
    "HBJ": "17bgHBJ.mov",
    "JWCJ": "19bgJWCJ.mov",
    "KWJ": "3bgKWJ.mov",
    "OGJJ": "5bgOGJJ.mov",
    "JHHRJ": "6bgJHHRJ.mov",
    "SCJ": "7bgSCJ.mov",
}
# 오버레이 비디오 파일명 매핑 (파일명과 일치)
BOOK_TO_OVERLAY_CODE = {
    "BJBJ": "BJBJ",
    "PSJ": "PSJ",  # 박씨전 -> PSJ (파일명과 일치)
    "DGJ": "DGJ",  # 두껍전 -> DGJ (파일명과 일치)
    "HBJ": "HBJ",
    "JWCJ": "JWCJ",
    "KWJ": "KWJ",
    "OGJJ": "OGJJ",
    "JHHRJ": "JHHRJ",
    "SCJ": "SCJ",
}

# Interactions 폴더 경로
INTERACTIONS_DIR = "Interactions"


def get_overlay_video_path(bg_book_code: str, char_num: int, char_book_code: str) -> str:
    """
    오버레이 비디오 파일 경로를 반환합니다.
    
    Args:
        bg_book_code: 배경 책 코드 (예: "SCJ", "HBJ")
        char_num: 캐릭터 번호 (1 또는 2)
        char_book_code: 캐릭터 책 코드 (예: "SCJ", "HBJ")
    
    Returns:
        오버레이 비디오 파일 경로 (예: "Interactions/bgSCJ/bgSCJ_ch1_HBJ.mov")
    """
    overlay_code = BOOK_TO_OVERLAY_CODE.get(char_book_code, char_book_code)
    filename = f"bg{bg_book_code}_ch{char_num}_{overlay_code}.mov"
    return os.path.join(INTERACTIONS_DIR, f"bg{bg_book_code}", filename)

def measure_character_height(overlay_path: str) -> tuple[int, int]:
    """
    캐릭터 오버레이 비디오의 높이와 키 중앙점을 측정합니다 (투명 부분 제외).
    
    Args:
        overlay_path: 오버레이 비디오 파일 경로
    
    Returns:
        (캐릭터의 실제 높이, 키 중앙점 Y 좌표) 튜플
    """
    if not os.path.exists(overlay_path):
        return (0, 0)
    
    try:
        cap = cv2.VideoCapture(overlay_path)
        if not cap.isOpened():
            return (0, 0)
        
        # 첫 프레임 읽기
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return (0, 0)
        
        frame_height = frame.shape[0]
        
        # RGBA 또는 BGR 확인
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA: 알파 채널 사용
            alpha = frame[:, :, 3]
            # 각 행에서 알파가 0이 아닌 픽셀이 있는지 확인
            rows_with_content = np.any(alpha > 0, axis=1)
        else:
            # BGR: 그레이스케일로 변환하여 검은색이 아닌 부분 찾기
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rows_with_content = np.any(gray > 0, axis=1)
        
        # 내용이 있는 첫 행과 마지막 행 찾기
        if not np.any(rows_with_content):
            # 내용이 없으면 전체 높이 반환, 중앙점은 프레임 중앙
            return (frame_height, frame_height // 2)
        
        first_row = np.argmax(rows_with_content)
        last_row = len(rows_with_content) - 1 - np.argmax(rows_with_content[::-1])
        
        height = last_row - first_row + 1
        # 키 중앙점: first_row와 last_row의 중간점
        center_y = (first_row + last_row) // 2
        
        return (height, center_y)
    except Exception as e:
        print(f"⚠️ 캐릭터 높이 측정 오류: {e}")
        return (0, 0)

# ============================================
# 1. 설정 파일 로드
# ============================================
def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일을 찾을 수 없습니다!")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CHARACTERS = load_json("characters_tone.json")
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

# 배경 사운드 및 음악 재생
BG_SOUND_DIR = "bg_sound"
BG_MUSIC_DIR = "bg_music"
# bg_sound 파일명 매핑
BOOK_TO_BG_SOUND = {
    "BJBJ": "10_BJBJ_bg_sound.wav",
    "PSJ": "11_BSJ_bg_sound.wav",
    "DGJ": "13_DGJ_bg_sound.wav",
    "HBJ": "17_HBJ_bg_sound.wav",
    "JWCJ": "19_JWCJ_bg_sound.wav",
    "KWJ": "3_KWJ_bg_sound.wav",
    "OGJJ": "5_OGJJ_bg_sound.wav",
    "JHHRJ": "6_JHHRJ_bg_sound.wav",
    "SCJ": "7_SCJ_bg_sound.wav"
}
# bg_music 파일명 매핑
BOOK_TO_BG_MUSIC = {
    "BJBJ": "10_BJBJ_bg_music.wav",
    "PSJ": "11_BSJ_bg_music.wav",
    "DGJ": "13_DGJ_bg_music.wav",
    "HBJ": "17_HBJ_bg_music.wav",
    "JWCJ": "19_JWCJ_bg_music.wav",
    "KWJ": "3_KWJ_bg_music.wav",
    "JHHRJ": "6_JHHRJ_bg_music.wav",
    "SCJ": "7_SCJ_bg_music.wav"
}

# 현재 재생 중인 bg 오디오 프로세스 및 스레드
_current_bg_sound_process = None
_current_bg_music_process = None
_bg_audio_playing = False  # bg 오디오 재생 루프 플래그
_bg_audio_thread = None  # bg 오디오 재생 스레드

# 현재 재생 중인 배경 추적
_current_bg_music_book_code = None

def play_background_music(book_code: str):
    """
    책 코드에 해당하는 bg_sound와 bg_music을 동시에 무한 루프로 재생합니다.
    이미 같은 배경이 재생 중이면 재생하지 않습니다.
    """
    global _current_bg_sound_process, _current_bg_music_process, _bg_audio_playing, _bg_audio_thread, _current_bg_music_book_code
    
    # 이미 같은 배경이 재생 중이면 재생하지 않음
    if _current_bg_music_book_code == book_code and _bg_audio_playing:
        print(f"🎵 배경 음악이 이미 재생 중입니다: {book_code} (재생하지 않음)")
        return
    
    # 기존 bg 오디오 중지
    _bg_audio_playing = False
    if _current_bg_sound_process is not None:
        try:
            _current_bg_sound_process.terminate()
            _current_bg_sound_process.wait(timeout=1)
        except:
            try:
                _current_bg_sound_process.kill()
            except:
                pass
        _current_bg_sound_process = None
    
    if _current_bg_music_process is not None:
        try:
            _current_bg_music_process.terminate()
            _current_bg_music_process.wait(timeout=1)
        except:
            try:
                _current_bg_music_process.kill()
            except:
                pass
        _current_bg_music_process = None
    
    # bg_sound 파일 경로 확인
    bg_sound_file = BOOK_TO_BG_SOUND.get(book_code)
    bg_sound_path = None
    if bg_sound_file:
        bg_sound_path = os.path.join(BG_SOUND_DIR, bg_sound_file)
        if not os.path.exists(bg_sound_path):
            print(f"⚠️ bg_sound 파일을 찾을 수 없음: {bg_sound_path}")
            bg_sound_path = None
    
    # bg_music 파일 경로 확인
    bg_music_file = BOOK_TO_BG_MUSIC.get(book_code)
    bg_music_path = None
    if bg_music_file:
        bg_music_path = os.path.join(BG_MUSIC_DIR, bg_music_file)
        if not os.path.exists(bg_music_path):
            print(f"⚠️ bg_music 파일을 찾을 수 없음: {bg_music_path}")
            bg_music_path = None
    
    if bg_sound_path is None and bg_music_path is None:
        print(f"🎵 '{book_code}'에 해당하는 배경 오디오가 없습니다.")
        return
    
    # 음량을 절반으로 조절한 임시 파일 생성
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_bg_sound = None
    temp_bg_music = None
    
    if bg_sound_path:
        temp_bg_sound = os.path.join(temp_dir, f"bg_sound_{book_code}_{os.getpid()}.wav")
        try:
            # ffmpeg로 음량 50%로 조절
            subprocess.run(
                ["ffmpeg", "-y", "-i", bg_sound_path,
                 "-af", "volume=0.5",
                 temp_bg_sound],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5
            )
        except Exception as e:
            print(f"⚠️ bg_sound 음량 조절 실패, 원본 파일 사용: {e}")
            temp_bg_sound = bg_sound_path  # 실패 시 원본 사용
    
    if bg_music_path:
        temp_bg_music = os.path.join(temp_dir, f"bg_music_{book_code}_{os.getpid()}.wav")
        try:
            # ffmpeg로 음량 50%로 조절
            subprocess.run(
                ["ffmpeg", "-y", "-i", bg_music_path,
                 "-af", "volume=0.5",
                 temp_bg_music],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=5
            )
        except Exception as e:
            print(f"⚠️ bg_music 음량 조절 실패, 원본 파일 사용: {e}")
            temp_bg_music = bg_music_path  # 실패 시 원본 사용
    
    # 현재 배경 책 코드 기록
    _current_bg_music_book_code = book_code
    
    # afplay로 무한 루프 재생 (별도 스레드에서)
    def play_bg_audio_loop():
        global _current_bg_sound_process, _current_bg_music_process, _bg_audio_playing, _current_bg_music_book_code
        _bg_audio_playing = True
        
        bg_sound_file_to_play = temp_bg_sound
        bg_music_file_to_play = temp_bg_music
        
        while _bg_audio_playing:
            try:
                # bg_sound와 bg_music을 동시에 재생
                if bg_sound_file_to_play:
                    _current_bg_sound_process = subprocess.Popen(
                        ["afplay", bg_sound_file_to_play],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                
                if bg_music_file_to_play:
                    _current_bg_music_process = subprocess.Popen(
                        ["afplay", bg_music_file_to_play],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                
                # 두 프로세스 중 하나라도 끝나면 다시 시작 (무한 루프)
                if _current_bg_sound_process:
                    _current_bg_sound_process.wait()
                if _current_bg_music_process:
                    _current_bg_music_process.wait()
                
                # 재생이 끝나면 다시 시작 (무한 루프) - 단, _bg_audio_playing이 True일 때만
            except Exception as e:
                if _bg_audio_playing:  # 중지 요청이 아닌 경우에만 오류 출력
                    print(f"⚠️ 배경 오디오 재생 오류: {e}")
                break
        
        # 스레드 종료 시 임시 파일 삭제
        if temp_bg_sound and temp_bg_sound != bg_sound_path and os.path.exists(temp_bg_sound):
            try:
                os.remove(temp_bg_sound)
            except:
                pass
        if temp_bg_music and temp_bg_music != bg_music_path and os.path.exists(temp_bg_music):
            try:
                os.remove(temp_bg_music)
            except:
                pass
    
    _bg_audio_thread = threading.Thread(target=play_bg_audio_loop, daemon=False)
    _bg_audio_thread.start()
    
    if bg_sound_path:
        print(f"🎵 bg_sound 재생 시작 (음량 50%): {bg_sound_path}")
    if bg_music_path:
        print(f"🎵 bg_music 재생 시작 (음량 50%): {bg_music_path}")

def stop_background_music():
    """bg_sound와 bg_music을 중지합니다."""
    global _current_bg_sound_process, _current_bg_music_process, _bg_audio_playing, _bg_audio_thread, _current_bg_music_book_code
    
    # 재생 루프 중지
    _bg_audio_playing = False
    _current_bg_music_book_code = None
    
    # 현재 재생 중인 프로세스 중지
    if _current_bg_sound_process is not None:
        try:
            _current_bg_sound_process.terminate()
            _current_bg_sound_process.wait(timeout=1)
        except:
            try:
                _current_bg_sound_process.kill()
            except:
                pass
        _current_bg_sound_process = None
    
    if _current_bg_music_process is not None:
        try:
            _current_bg_music_process.terminate()
            _current_bg_music_process.wait(timeout=1)
        except:
            try:
                _current_bg_music_process.kill()
            except:
                pass
        _current_bg_music_process = None
    
    # 스레드가 종료될 때까지 대기 (최대 2초)
    if _bg_audio_thread is not None and _bg_audio_thread.is_alive():
        _bg_audio_thread.join(timeout=2)
        if _bg_audio_thread.is_alive():
            # 스레드가 종료되지 않으면 강제 종료는 불가능하지만, 프로세스는 이미 중지됨
            pass
        _bg_audio_thread = None
    
    print("🎵 배경 음악 중지됨")


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
    
    # 비디오가 성공적으로 설정되고 재생 중인지 확인 (재시도 로직)
    # 주의: set_video가 next_video_path만 설정하고 _play_loop에서 나중에 
    # 비디오 전환이 완료될 때까지 충분히 기다려야 함
    import time
    max_retries = 20  # 최대 2초 대기 (페이드 효과 고려)
    retry_delay = 0.1  # 0.1초마다 확인
    is_playing = False
    
    for attempt in range(max_retries):
        time.sleep(retry_delay)
        with VIDEO_PLAYER.lock:
            video_cap = VIDEO_PLAYER.video_cap
            current_path = VIDEO_PLAYER.current_video_path
            next_path = VIDEO_PLAYER.next_video_path
        
        # next_video_path가 설정되어 있으면 아직 전환 중
        if next_path == video_path:
            # 전환 대기 중
            continue
        
        # 비디오 캡처 객체가 있고 열려있는지 확인
        if video_cap is not None:
            try:
                if video_cap.isOpened() and current_path == video_path:
                    is_playing = True
                    break
            except:
                # isOpened() 호출 중 오류 발생 시 다음 시도
                pass
        elif current_path == video_path:
            # 경로는 맞지만 video_cap이 아직 None (전환 중)
            continue
    
    if is_playing:
        print(f"✅ 배경 비디오 재생 중: {video_file} (무한 루프)")
    else:
        # 최종 확인 (경로만 확인)
        with VIDEO_PLAYER.lock:
            final_path = VIDEO_PLAYER.current_video_path
            final_next = VIDEO_PLAYER.next_video_path
        if final_path == video_path:
            print(f"✅ 배경 비디오 설정 완료: {video_file} (재생 확인 중...)")
        elif final_next == video_path:
            print(f"⏳ 배경 비디오 전환 대기 중: {video_file} (페이드 효과 진행 중...)")
        else:
            print(f"⚠️ 배경 비디오 재생 확인 실패: {video_file} (현재: {final_path}, 다음: {final_next})")

def get_interaction_profile(bg_info: dict, character: dict = None, is_cha1: bool = False) -> dict:
    """
    backgrounds.json 안에 미리 정의해 둔
    interaction_label / interaction_summary / interaction_emotions를 가져온다.
    interaction_emotions는 10가지 감정 옵션 리스트로, LLM이 캐릭터 성격에 맞게 선택한다.
    
    Args:
        bg_info: 배경 정보 딕셔너리
        character: 캐릭터 정보 딕셔너리 (선택적)
        is_cha1: True면 cha1, False면 cha2 (선택적)
    """
    if bg_info is None:
        return {
            "label": "neutral",
            "summary": "A neutral situation with no special context",
            "emotion_options": ["mild curiosity", "calm observation", "quiet interest"]
        }

    # interaction 문자열에서 Character1/Character2 구분 추출
    interaction_str = bg_info.get("interaction", "")
    interaction_summary = bg_info.get("interaction_summary", "")
    
    # Character1/Character2 구분이 있는 경우 파싱
    if character is not None:
        # cha1인지 cha2인지 확인
        if is_cha1:
            char_marker = "Character1"
        else:
            char_marker = "Character2"
        
        # interaction 문자열에서 해당 캐릭터의 interaction 추출
        if char_marker in interaction_str:
            # "(Character1)...(Character2)" 형식 파싱
            import re
            # Character1과 Character2 부분 추출
            pattern1 = r'\(Character1\)([^,)]+?)(?:\(Character2\)|$)'
            pattern2 = r'\(Character2\)([^,)]+?)(?:\(Character1\)|$)'
            
            if is_cha1:
                match = re.search(pattern1, interaction_str)
                if match:
                    interaction_str = match.group(1).strip()
            else:
                match = re.search(pattern2, interaction_str)
                if match:
                    interaction_str = match.group(1).strip()
            
            # summary도 캐릭터에 맞게 조정 (간단한 버전)
            if is_cha1 and "Character1" in interaction_summary:
                # Character1 관련 부분만 추출하거나 요약
                pass  # summary는 그대로 사용하되, interaction_str이 더 중요
            elif not is_cha1 and "Character2" in interaction_summary:
                pass  # summary는 그대로 사용

    # interaction_emotions는 이제 리스트
    emotions_data = bg_info.get("interaction_emotions", [])
    if isinstance(emotions_data, list):
        emotion_options = emotions_data
    else:
        # 혹시 문자열이면 리스트로 변환
        emotion_options = [emotions_data]

    return {
        "label": bg_info.get("interaction_label", "neutral"),
        "summary": interaction_summary,  # 원본 summary 사용
        "interaction": interaction_str,  # 파싱된 interaction 추가
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
    frequent_expressions = speech_patterns.get('frequent_expressions', [])[:15]  # 상위 15개만
    endings_from_dialogues = speech_patterns.get('endings_from_dialogues', [])[:10]  # 상위 10개만
    common_words = speech_patterns.get('common_words', [])[:10]  # 상위 10개만
    analysis = char_data.get('analysis_from_dialogues', {})
    
    # 분석 결과 포맷팅
    formality_info = ', '.join(analysis.get('formality_indicators', [])[:3]) if analysis.get('formality_indicators') else ''
    emotional_keywords = ', '.join([e.split(':')[0] for e in analysis.get('emotional_keywords', [])[:5]]) if analysis.get('emotional_keywords') else ''
    dialect_info = ', '.join([d.split(':')[0] for d in analysis.get('dialect_indicators', [])]) if analysis.get('dialect_indicators') else ''

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

캐릭터 말투 특징 (실제 대사 분석 결과):
{f"- 자주 사용하는 표현: {', '.join(frequent_expressions) if frequent_expressions else '없음'}"}
{f"- 실제 대사에서 자주 쓰는 어미: {', '.join(endings_from_dialogues) if endings_from_dialogues else '없음'}"}
{f"- 자주 사용하는 단어: {', '.join(common_words) if common_words else '없음'}"}
{f"- 격식/공손도: {formality_info if formality_info else '없음'}"}
{f"- 감정 톤: {emotional_keywords if emotional_keywords else '없음'}"}
{f"- 방언 특징: {dialect_info if dialect_info else '없음'}"}

상황:
- 이 캐릭터가 지금 '{action}'을(를) 하기 직전입니다.
- 위 감정 옵션 중 이 캐릭터의 성격에 가장 어울리는 감정을 선택하고, 그 감정을 담아 짧게 한 마디를 합니다.

말투 규칙:
- 위의 "캐릭터 말투 특징"에 나온 실제 대사 분석 결과를 반드시 참고하여 말투를 정확히 재현하세요.
- 자주 사용하는 표현과 어미를 자연스럽게 활용하세요.
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


def generate_first_dialogue_line(char_a: dict, bg_info: dict, is_cha1: bool = False) -> str:
    """
    같은 배경/인터랙션에서 char_a가 먼저 한 마디를 생성.
    → 짧고 구어체.
    Avoid any narration or book-style phrases. The line must sound like spontaneous spoken Korean, not a written script.
    Add small hesitations (예: '아...', '음...') when appropriate, only if it fits the character.

    Args:
        char_a: 첫 번째 캐릭터 정보 딕셔너리
        bg_info: 배경 정보 딕셔너리
        is_cha1: True면 cha1, False면 cha2
    """
    place = bg_info.get("background", "")
    profile = get_interaction_profile(bg_info, char_a, is_cha1)
    # 파싱된 interaction 사용 (없으면 원본 사용)
    action = profile.get("interaction", bg_info.get("interaction", ""))
    emotion_list = "\n".join([f"  - {e}" for e in profile['emotion_options']])

    char_a_data = CHARACTERS.get(char_a['book_code'], {}).get(char_a['role_key'], {})
    char_a_speech = char_a_data.get('speech_patterns', {})
    char_a_style = char_a_speech.get('speaking_style', '')
    char_a_expressions = char_a_speech.get('frequent_expressions', [])[:15]
    char_a_endings = char_a_speech.get('endings_from_dialogues', [])[:10]
    char_a_words = char_a_speech.get('common_words', [])[:10]
    char_a_analysis = char_a_data.get('analysis_from_dialogues', {})
    
    system_a = (
        "당신은 한국 옛이야기 속 등장인물이 실제로 말하는 대사를 쓰는 작가입니다. "
        "첫 번째 인물이 다른 인물(두 번째 인물)에게 말을 건네는 짧은 한 마디를 만드세요. "
        "혼잣말이 아니라 상대방에게 말을 거는 대화여야 합니다."
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

첫 번째 인물 말투 특징 (실제 대사 분석 결과):
{f"- 자주 사용하는 표현: {', '.join(char_a_expressions) if char_a_expressions else '없음'}"}
{f"- 실제 대사에서 자주 쓰는 어미: {', '.join(char_a_endings) if char_a_endings else '없음'}"}
{f"- 자주 사용하는 단어: {', '.join(char_a_words) if char_a_words else '없음'}"}

상황:
- 첫 번째 인물이 '{action}' 장면 속에서 위 감정 중 자신의 성격에 맞는 것을 느끼며 짧게 말합니다.
- 혼잣말이 아니라, 같은 장면에 있는 두 번째 인물에게 말을 거는 대화입니다.
- 두 번째 인물이 듣고 반응할 수 있도록, 질문이나 제안, 관찰 등을 포함하는 것이 좋습니다.
- 배경 장소와 인터랙션을 고려하여 자연스럽고 맥락에 맞는 대사를 생성하세요.
- 예를 들어, 토끼 같은 장난꾸러기 캐릭터는 '고백할 기회' 같은 이상한 표현을 쓰지 말고, 
  현재 상황과 배경에 맞는 자연스러운 말을 사용하세요.

말투 규칙:
- 위의 "첫 번째 인물 말투 특징"에 나온 실제 대사 분석 결과를 반드시 참고하여 말투를 정확히 재현하세요.
- 자주 사용하는 표현과 어미를 자연스럽게 활용하세요.
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
    char_b_expressions = char_b_speech.get('frequent_expressions', [])[:15]
    char_b_endings = char_b_speech.get('endings_from_dialogues', [])[:10]
    char_b_words = char_b_speech.get('common_words', [])[:10]
    char_b_analysis = char_b_data.get('analysis_from_dialogues', {})
    
    system_b = (
        "당신은 한국 옛이야기 속 두 인물이 실제로 주고받는 대화를 쓰는 작가입니다. "
        "두 번째 인물이 첫 번째 인물의 말을 듣고 직접적으로 반응하는 짧은 한 마디를 만드세요. "
        "반드시 첫 번째 인물에게 말을 거는 대답이어야 하며, 혼잣말이 아닌 대화여야 합니다."
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

두 번째 인물 말투 특징 (실제 대사 분석 결과):
{f"- 자주 사용하는 표현: {', '.join(char_b_expressions) if char_b_expressions else '없음'}"}
{f"- 실제 대사에서 자주 쓰는 어미: {', '.join(char_b_endings) if char_b_endings else '없음'}"}
{f"- 자주 사용하는 단어: {', '.join(char_b_words) if char_b_words else '없음'}"}

중요한 상황:
- 두 번째 인물은 위의 첫 번째 인물의 말을 직접 듣고 있습니다.
- 첫 번째 인물이 두 번째 인물에게 말을 건넨 것에 대해, 두 번째 인물이 첫 번째 인물에게 직접 대답해야 합니다.
- 혼잣말이 아니라 첫 번째 인물에게 말을 거는 대화여야 합니다.
- 첫 번째 인물의 말의 내용, 톤, 의도를 고려하여 적절히 반응하세요.
- 동의, 반박, 질문, 제안, 놀람 등 첫 번째 인물의 말에 대한 자연스러운 반응을 보여주세요.
- 첫 번째 인물의 말에 직접적으로 응답하는 느낌이 강하게 드러나야 합니다.

말투 규칙:
- 위의 "두 번째 인물 말투 특징"에 나온 실제 대사 분석 결과를 반드시 참고하여 말투를 정확히 재현하세요.
- 자주 사용하는 표현과 어미를 자연스럽게 활용하세요.
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
    frequent_expressions = speech_patterns.get('frequent_expressions', [])[:15]
    endings_from_dialogues = speech_patterns.get('endings_from_dialogues', [])[:10]
    common_words = speech_patterns.get('common_words', [])[:10]
    speaking_style = speech_patterns.get('speaking_style', '')
    analysis = char_data.get('analysis_from_dialogues', {})

    user = f"""
새 배경 장소: {place}
새 배경 인터랙션: {action}
장면 분위기 요약: {profile['summary']}
가능한 감정들 (캐릭터 성격에 맞는 것을 선택하세요):
{emotion_list}

캐릭터 설정(영어): {character['personality']}
캐릭터 정보: {character['age']}살 {character['gender']}
캐릭터 말투 스타일: {speaking_style}

캐릭터 말투 특징 (실제 대사 분석 결과):
{f"- 자주 사용하는 표현: {', '.join(frequent_expressions) if frequent_expressions else '없음'}"}
{f"- 실제 대사에서 자주 쓰는 어미: {', '.join(endings_from_dialogues) if endings_from_dialogues else '없음'}"}
{f"- 자주 사용하는 단어: {', '.join(common_words) if common_words else '없음'}"}

상황:
- 이 캐릭터는 방금 전까지 전혀 다른 곳에 있었는데,
  갑자기 이 장면으로 순간이동하듯 옮겨졌습니다.
- 위 감정 중 자신의 성격에 맞는 것을 느끼며, 놀라거나 당황하거나 신기해서 감탄과 함께 한 마디를 합니다.
- 배경 장소와 인터랙션을 파악하고, 이곳에서 무슨 일이 일어나는지 이해한 후 놀라움을 표현합니다.

말투 규칙:
- 위의 "캐릭터 말투 특징"에 나온 실제 대사 분석 결과를 반드시 참고하여 말투를 정확히 재현하세요.
- 자주 사용하는 표현과 어미를 자연스럽게 활용하세요.
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
def apply_audio_effects(character: dict, input_path: str, output_path: str):
    """
    캐릭터에 맞는 오디오 효과를 적용합니다.
    이 함수는 tts.py와 test_character_voice.py에서 공통으로 사용됩니다.
    모든 캐릭터의 음량을 동일하게 정규화합니다.
    
    Args:
        character: 캐릭터 정보 딕셔너리 (book_code, role_key 포함)
        input_path: 원본 오디오 파일 경로
        output_path: 효과가 적용된 오디오 파일 저장 경로
    """
    book_code = character.get("book_code", "")
    role_key = character.get("role_key", "")
    
    # 음량 정규화를 위한 임시 파일 경로
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_normalized = os.path.join(temp_dir, f"normalized_{os.getpid()}_{id(character)}.wav")
    
    if (book_code == "JHHRJ" and role_key == "ghost") or (book_code == "KWJ" and role_key == "monster"):
        # reverb 효과 적용 (aecho 필터 사용)
        # ghost의 경우: 구슬프고 우울하지만 자연스러운 처녀귀신 목소리
        if book_code == "JHHRJ" and role_key == "ghost":
            # ghost: 구슬프고 우울하고 한이 서린 처녀귀신 목소리
            # 효과: 자연스러운 reverb + 약간의 pitch 조정 (어둡고 우울) + 고주파 필터링 + equalizer
            # tremolo와 delay 제거하여 음성변조 느낌 최소화
            audio_filter = (
                "lowpass=f=4000,"
                "aecho=0.8:0.7:80:0.3,"
                "equalizer=f=200:width_type=h:width=300:g=1.5,"
                "equalizer=f=5000:width_type=h:width=2000:g=-2"
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", audio_filter,
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        else:
            # monster: 중후하고 무게감 있는 괴물 목소리 효과
            # 효과: 매우 낮은 피치 + 강한 리버브/에코 + 저주파 강조 + 중후한 느낌
            # 1단계: 피치를 매우 낮춤 (속도를 0.65배로 낮춰서 피치 낮춤 - 더 중후하게)
            temp_pitch = os.path.join(os.path.dirname(output_path), f"monster_pitch_{os.getpid()}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", "atempo=0.65,asetrate=44100*0.65,aresample=44100",
                 temp_pitch],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            # 2단계: 중후하고 무게감 있는 효과 적용
            audio_filter = (
                "equalizer=f=60:width_type=h:width=80:g=10,"  # 매우 낮은 저주파 강조 (깊고 중후한 느낌)
                "equalizer=f=120:width_type=h:width=150:g=8,"  # 저주파 강조 (무게감)
                "equalizer=f=250:width_type=h:width=200:g=6,"  # 중저주파 강조 (중후함)
                "equalizer=f=4000:width_type=h:width=3000:g=-5,"  # 고주파 억제 (어둡고 무거운 느낌)
                "equalizer=f=6000:width_type=h:width=2000:g=-6,"  # 더 높은 고주파 억제
                "lowpass=f=2500,"  # 고주파 필터링 (더 어둡게)
                "aecho=0.95:0.95:120:0.6,"  # 매우 강한 리버브 (중후한 공간감)
                #"aecho=0.8:0.8:250:0.4,"  # 추가 리버브 레이어 (깊은 공간감)
                #"aecho=0.6:0.6:400:0.2"  # 더 긴 리버브 (중후한 느낌)
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_pitch,
                 "-af", audio_filter,
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            # 임시 파일 삭제
            try:
                os.remove(temp_pitch)
            except:
                pass
    elif book_code == "SCJ" and role_key == "simcheong":
        # 심청: 어리고 명랑하고 결연에 가득 찬 목소리
        # 효과: 높은 pitch (어리고 밝게) + 빠른 속도 (명랑함) + 고주파 강조 (맑고 밝게) + vibrato (생동감) + 저주파 억제 (가볍고 밝게)
        audio_filter = (
            "equalizer=f=3000:width_type=h:width=2000:g=3,"  # 고주파 강조 (맑고 밝게)
            "equalizer=f=5000:width_type=h:width=1500:g=2,"  # 더 높은 고주파 강조 (명랑함)
            "equalizer=f=200:width_type=h:width=300:g=-2,"  # 저주파 억제 (가볍고 밝게)
            "vibrato=f=5.5:d=0.15,"  # 약간의 vibrato (생동감과 결연함)
            "highpass=f=100"  # 매우 낮은 주파수 제거 (더 맑게)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-af", audio_filter,
             output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    elif book_code == "DGJ":
        if role_key == "fox":
            # 여우: 교활하고 매우 가는 목소리, 간신배 느낌, 잘난체
            # pitch를 더 올려서 매우 가늘게, tremolo 추가, 고주파 강조로 더 가는 느낌
            audio_filter = (
                "aresample=44100,"  # pitch를 30% 올려서 더 가늘게
                "equalizer=f=3000:width_type=h:width=2000:g=3,"  # 고주파 강조 (가는 느낌)
                "equalizer=f=5000:width_type=h:width=1500:g=2,"  # 더 높은 고주파 강조
                "equalizer=f=200:width_type=h:width=300:g=-2,"  # 저주파 억제 (가볍고 가는 느낌)
                "tremolo=f=3.0:d=0.2"  # tremolo로 교활한 느낌
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", audio_filter,
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        elif role_key == "toad":
            # 두꺼비: 현명하고 총명하고 뭉툭하고 묵직한 목소리
            # pitch를 약간 낮춰서 더 묵직하게, bass boost로 더 깊고 뭉툭한 느낌
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", "aresample=44100,equalizer=f=100:width_type=h:width=200:g=3",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        else:
            # 다른 DGJ 캐릭터는 원본 그대로 저장
            import shutil
            shutil.copy2(input_path, output_path)
    elif book_code == "OGJJ" and role_key == "onggojip":
        # 옹고집: 매우 나이든 남자 목소리 (72세)
        # 효과: 저주파 강조 (깊고 중후한 느낌)
        audio_filter = (
            "equalizer=f=80:width_type=h:width=100:g=4,"  # 매우 낮은 저주파 강조 (깊고 나이든 느낌)
            "equalizer=f=150:width_type=h:width=200:g=3,"  # 저주파 강조 (중후함)
            "equalizer=f=300:width_type=h:width=250:g=2,"  # 중저주파 강조 (깊은 목소리)
            "equalizer=f=4000:width_type=h:width=3000:g=-3,"  # 고주파 약간 억제 (나이든 느낌)
            "lowpass=f=3500"  # 고주파 필터링 (나이든 느낌)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-af", audio_filter,
             output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    elif book_code == "HBJ" and role_key == "nolbu":
        # 놀부: 나이든 남자 목소리 (58세)
        # 효과: 저주파 강조 (깊고 중후한 느낌)
        audio_filter = (
            "equalizer=f=80:width_type=h:width=100:g=4,"  # 매우 낮은 저주파 강조 (깊고 나이든 느낌)
            "equalizer=f=150:width_type=h:width=200:g=3,"  # 저주파 강조 (중후함)
            "equalizer=f=300:width_type=h:width=250:g=2,"  # 중저주파 강조 (깊은 목소리)
            "equalizer=f=4000:width_type=h:width=3000:g=-3,"  # 고주파 약간 억제 (나이든 느낌)
            "lowpass=f=3500"  # 고주파 필터링 (나이든 느낌)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-af", audio_filter,
             output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    else:
        # 일반 캐릭터는 원본 그대로 저장
        import shutil
        shutil.copy2(input_path, output_path)
    
    # 모든 캐릭터에 대해 음량 정규화 적용 (효과 적용 후)
    # loudnorm 필터를 사용하여 모든 오디오의 음량을 동일하게 맞춤
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path,
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             temp_normalized],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10
        )
        # 정규화된 파일을 최종 출력 경로로 복사
        import shutil
        shutil.move(temp_normalized, output_path)
    except Exception as e:
        # 음량 정규화 실패 시 원본 파일 사용 (오류 무시)
        try:
            if os.path.exists(temp_normalized):
                os.remove(temp_normalized)
        except:
            pass


def generate_tts(character: dict, text: str, output_path: str):
    """
    TTS를 생성하고 임시 파일로 저장합니다.
    output_path는 호환성을 위해 유지하지만 실제로는 임시 파일 경로를 반환합니다.
    반환값: (오디오 파일 경로, 영어 번역 텍스트, 캐릭터 이름)
    """
    speaker_tag = f"{character['book_code'].upper()}-{character['role_key'].upper()}"
    
    # 캐릭터 이름 가져오기 (영어로)
    role_key = character.get('role_key', '')
    # role_key를 영어 이름으로 변환
    character_name_map = {
        'simcheong': 'Simcheong', 'simbongsa': 'Simbongsa',
        'heungbu': 'Heungbu', 'nolbu': 'Nolbu',
        'turtle': 'Turtle', 'rabbit': 'Rabbit',
        'onggojip': 'Onggojip',
        'jeonwoochi': 'Jeonwoochi',
        'sister_older': 'Janghwa', 'sister_younger': 'Hongryeon', 'ghost': 'Ghost',
        'ugly': 'Ugly', 'pretty': 'Pretty',
        'toad': 'Toad', 'fox': 'Fox',
        'kimwon': 'Kimwon', 'monster': 'Monster'
    }
    character_name = character_name_map.get(role_key, role_key.capitalize())
    
    # 영어 번역 생성
    english_text = ""
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
        english_text = text  # 번역 실패 시 원문 사용

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
    import uuid
    temp_dir = tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, f"tts_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav")
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    
    # 최종 출력도 임시 파일로 (재생 후 삭제됨)
    temp_output = os.path.join(temp_dir, f"tts_output_{os.getpid()}_{uuid.uuid4().hex[:8]}.wav")
    
    # 특수 캐릭터 오디오 효과 적용 (공통 함수 사용)
    apply_audio_effects(character, temp_input, temp_output)
    
    # 임시 입력 파일 삭제
    try:
        os.remove(temp_input)
    except:
        pass

    return temp_output, english_text, character_name



def stop_all_audio():
    """모든 재생 중인 오디오를 즉시 중단합니다."""
    global _current_audio_processes, _should_stop_audio
    # 중단 플래그 설정 (play_audio_sequence가 다음 오디오를 재생하지 않도록)
    with _stop_audio_lock:
        _should_stop_audio = True
    
    # 추적 중인 프로세스 종료
    processes_to_kill = []
    with _audio_processes_lock:
        for process in _current_audio_processes:
            try:
                if process.poll() is None:  # 프로세스가 아직 실행 중이면
                    processes_to_kill.append(process)
            except:
                pass
        _current_audio_processes.clear()
    
    # 프로세스 강제 종료 (terminate + kill)
    # 주의: bg_sound와 bg_music은 별도로 관리되므로 여기서는 TTS 프로세스만 종료
    for process in processes_to_kill:
        try:
            process.terminate()
            # 즉시 kill (대기하지 않음)
            try:
                process.kill()
            except:
                pass
        except:
            pass
    
    # 자막 지우기
    VIDEO_PLAYER.clear_subtitle()
    
    # 잠시 대기 후 플래그 리셋 (다음 시퀀스가 시작될 수 있도록)
    import time
    time.sleep(0.05)  # 대기 시간 단축
    with _stop_audio_lock:
        _should_stop_audio = False
    print("🔇 모든 오디오 중단됨")

def play_audio(path: str, blocking: bool = False, subtitle_text: str = None):
    """
    오디오를 재생합니다.
    
    Args:
        path: 오디오 파일 경로
        blocking: True면 동기적으로 재생 (다음 오디오가 재생 완료될 때까지 대기), False면 비동기로 재생
        subtitle_text: 자막 텍스트 (예: "toad: Haha")
    """
    # 중단 플래그 확인
    with _stop_audio_lock:
        if _should_stop_audio:
            return  # 중단 플래그가 설정되어 있으면 재생하지 않음
    
    print(f"🔊 PLAY AUDIO: {path}")
    
    # 자막 설정
    if subtitle_text:
        VIDEO_PLAYER.set_subtitle(subtitle_text)
    
    def play():
        import tempfile
        temp_dir = tempfile.gettempdir()
        is_temp_file = path.startswith(temp_dir)
        
        try:
            # 재생 시작 전 다시 한 번 확인
            with _stop_audio_lock:
                if _should_stop_audio:
                    # 중단된 경우 임시 파일 삭제
                    if is_temp_file:
                        try:
                            os.remove(path)
                        except:
                            pass
                    # 자막 지우기
                    VIDEO_PLAYER.clear_subtitle()
                    return
            
            process = subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # 프로세스를 리스트에 추가
            with _audio_processes_lock:
                _current_audio_processes.append(process)
            
            process.wait()  # 재생 완료 대기
            
            # 재생 완료 후 리스트에서 제거
            with _audio_processes_lock:
                if process in _current_audio_processes:
                    _current_audio_processes.remove(process)
            
            # 재생 완료 후 자막 지우기
            VIDEO_PLAYER.clear_subtitle()
            
            # 재생 완료 후 임시 파일 삭제
            if is_temp_file:
                try:
                    os.remove(path)
                except:
                    pass
        except Exception as e:
            print(f"⚠️ 오디오 재생 오류: {e}")
            with _audio_processes_lock:
                if 'process' in locals() and process in _current_audio_processes:
                    _current_audio_processes.remove(process)
            # 오류 발생 시에도 임시 파일 삭제 시도
            if is_temp_file:
                try:
                    os.remove(path)
                except:
                    pass
    
    if blocking:
        # 동기적으로 재생 (순차 재생용)
        play()
    else:
        # 비동기로 재생
        threading.Thread(target=play, daemon=True).start()

def play_audio_sequence(paths: list[str], subtitles: list[str] = None):
    """
    여러 오디오 파일을 순차적으로 재생합니다 (겹치지 않게).
    
    Args:
        paths: 재생할 오디오 파일 경로 리스트
        subtitles: 각 오디오에 대한 자막 텍스트 리스트 (None이면 자막 없음)
    """
    if subtitles is None:
        subtitles = [None] * len(paths)
    
    def play_sequence():
        for i, path in enumerate(paths):
            # 각 오디오 재생 전에 중단 플래그 확인
            with _stop_audio_lock:
                if _should_stop_audio:
                    remaining = len(paths) - i
                    print(f"🔇 오디오 시퀀스 중단됨 (남은 파일: {remaining})")
                    VIDEO_PLAYER.clear_subtitle()  # 자막 지우기
                    return  # 중단 플래그가 설정되어 있으면 시퀀스 중단
            
            if not os.path.exists(path):
                print(f"⚠️ 오디오 파일을 찾을 수 없음: {path}")
                continue
            
            # 자막 설정
            subtitle = subtitles[i] if i < len(subtitles) else None
            play_audio(path, blocking=True, subtitle_text=subtitle)
            
            # 재생 후에도 중단 플래그 확인 (다음 오디오로 넘어가기 전)
            with _stop_audio_lock:
                if _should_stop_audio:
                    remaining = len(paths) - i - 1
                    print(f"🔇 오디오 시퀀스 중단됨 (남은 파일: {remaining})")
                    return  # 중단 플래그가 설정되어 있으면 시퀀스 중단
    
    # 별도 스레드에서 순차 재생 (다른 작업을 블로킹하지 않음)
    threading.Thread(target=play_sequence, daemon=True).start()


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
    
    # 새 마커 감지 시 즉시 모든 오디오 중단 (가장 먼저 실행)
    stop_all_audio()

    print("\n==============================")
    print(f"[handle_book_input] book_code={book_code}, index={index_in_sequence}")
    

    # -------------------------
    # 1) index 1: 초기 배경
    # -------------------------
    if index_in_sequence == 1:
        # 이전 진행 상황 중단: 모든 오디오 및 bgm 중단
        stop_all_audio()
        stop_background_music()
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
        play_background_music(book_code)  # 배경 음악 재생 (무한 루프)
        
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
        # (stop_all_audio는 함수 시작 부분에서 이미 호출됨)
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha1"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha1 정의 없음")
            return

        cha1 = build_character(book_code, role_key)
        CURRENT_CHA1_INFO = cha1

        # ch1 오버레이 비디오 설정 (배경에 맞는 폴더에서 찾기)
        if CURRENT_BG_BOOK_CODE:
            overlay_path = get_overlay_video_path(CURRENT_BG_BOOK_CODE, 1, book_code)
            print(f"🔍 [index 2] ch1 오버레이 비디오 경로: {overlay_path}")
            print(f"🔍 [index 2] 배경: {CURRENT_BG_BOOK_CODE}, 캐릭터: {book_code}")
            if os.path.exists(overlay_path):
                print(f"✅ 파일 존재 확인, 오버레이 비디오 설정 중...")
                VIDEO_PLAYER.set_overlay_video(overlay_path)
                # 비디오가 제대로 설정되었는지 확인
                import time
                time.sleep(0.15)  # 비디오 초기화를 위한 대기
                with VIDEO_PLAYER.lock:
                    is_set = VIDEO_PLAYER.overlay_video_cap is not None and VIDEO_PLAYER.overlay_video_cap.isOpened()
                print(f"🎬 오버레이 비디오 ch1 설정 완료: {overlay_path} (설정됨: {is_set})")
            else:
                print(f"⚠️ 오버레이 비디오를 찾을 수 없음: {overlay_path}")
                VIDEO_PLAYER.set_overlay_video(None)

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
            
            # 랜덤으로 순서 결정 - 모든 대화 생성 후 순차 재생
            if random.random() < 0.5:
                # 장화 먼저
                out1, out1_eng, out1_name = generate_tts(older, line1, "")
                out2, out2_eng, out2_name = generate_tts(younger, line2, "")
                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([out1, out2], [f"{out1_name}: {out1_eng}", f"{out2_name}: {out2_eng}"])
            else:
                # 홍련 먼저
                out1, out1_eng, out1_name = generate_tts(younger, line2, "")
                out2, out2_eng, out2_name = generate_tts(older, line1, "")
                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([out1, out2], [f"{out1_name}: {out1_eng}", f"{out2_name}: {out2_eng}"])
        else:
            line = generate_action_line(cha1, CURRENT_BG_INFO)
            if not line:
                line = f"{CURRENT_BG_INFO.get('interaction', '')}, 한번 해볼까?"

            out_path, eng, name = generate_tts(cha1, line, "")
            # 순차 재생 (단일 파일이지만 일관성을 위해)
            play_audio_sequence([out_path], [f"{name}: {eng}"])
        return

    # -------------------------
    # 3) index 3: 초기 cha2 + 대화
    # -------------------------
    if index_in_sequence == 3:
        # (stop_all_audio는 함수 시작 부분에서 이미 호출됨)
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha2"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha2 정의 없음")
            return

        cha2 = build_character(book_code, role_key)
        CURRENT_CHA2_INFO = cha2
        
        # ch2 오버레이 비디오 설정 (배경에 맞는 폴더에서 찾기)
        if CURRENT_BG_BOOK_CODE:
            overlay_path2 = get_overlay_video_path(CURRENT_BG_BOOK_CODE, 2, book_code)
            if os.path.exists(overlay_path2):
                VIDEO_PLAYER.set_overlay_video2(overlay_path2)
                print(f"🎬 오버레이 비디오 ch2 설정: {overlay_path2}")
                import time
                time.sleep(0.15)
                with VIDEO_PLAYER.lock:
                    is_set = VIDEO_PLAYER.overlay_video_cap2 is not None and VIDEO_PLAYER.overlay_video_cap2.isOpened()
                print(f"🎬 오버레이 비디오 ch2 업데이트 완료: {overlay_path2} (설정됨: {is_set})")
            else:
                print(f"⚠️ 오버레이 비디오 ch2를 찾을 수 없음: {overlay_path2}")

        if CURRENT_CHA1_INFO is None:
            print("⚠ cha1이 아직 설정되지 않아 cha2만 한 줄 대사")
            line2 = generate_action_line(cha2, CURRENT_BG_INFO)
            out2, out2_eng, out2_name = generate_tts(cha2, line2, "")
            # 순차 재생 (단일 파일이지만 일관성을 위해)
            play_audio_sequence([out2], [f"{out2_name}: {out2_eng}"])
            return

        # 장화홍련전의 경우: 자매가 랜덤 순서로 말함
        if CURRENT_CHA1_INFO is not None and CURRENT_CHA1_INFO.get('book_code') == "JHHRJ":
            older, younger = build_sisters_pair()
            
            # 랜덤으로 순서 결정 - 모든 대화 생성 후 순차 재생
            if random.random() < 0.5:
                # 장화 먼저
                line_older = generate_action_line(older, CURRENT_BG_INFO)
                out_older, out_older_eng, out_older_name = generate_tts(older, line_older, "")
                
                line_younger = generate_second_dialogue_line(younger, line_older, CURRENT_BG_INFO)
                out_younger, out_younger_eng, out_younger_name = generate_tts(younger, line_younger, "")
                
                # cha2가 자매의 대화에 반응
                line_cha2 = generate_second_dialogue_line(cha2, line_older, CURRENT_BG_INFO)
                out_cha2, out_cha2_eng, out_cha2_name = generate_tts(cha2, line_cha2, "")
                
                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([out_older, out_younger, out_cha2], 
                                  [f"{out_older_name}: {out_older_eng}", 
                                   f"{out_younger_name}: {out_younger_eng}", 
                                   f"{out_cha2_name}: {out_cha2_eng}"])
            else:
                # 홍련 먼저
                line_younger = generate_action_line(younger, CURRENT_BG_INFO)
                out_younger, out_younger_eng, out_younger_name = generate_tts(younger, line_younger, "")
                
                line_older = generate_second_dialogue_line(older, line_younger, CURRENT_BG_INFO)
                out_older, out_older_eng, out_older_name = generate_tts(older, line_older, "")
                
                # cha2가 자매의 대화에 반응
                line_cha2 = generate_second_dialogue_line(cha2, line_younger, CURRENT_BG_INFO)
                out_cha2, out_cha2_eng, out_cha2_name = generate_tts(cha2, line_cha2, "")
                
                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([out_younger, out_older, out_cha2],
                                  [f"{out_younger_name}: {out_younger_eng}", 
                                   f"{out_older_name}: {out_older_eng}", 
                                   f"{out_cha2_name}: {out_cha2_eng}"])
        else:
            # 새로 등장하는 cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            # 첫 번째 대화 생성
            line2 = generate_first_dialogue_line(cha2, CURRENT_BG_INFO)
            out2, out2_eng, out2_name = generate_tts(cha2, line2, "")
            
            # 두 번째 대화 생성
            line1 = generate_second_dialogue_line(CURRENT_CHA1_INFO, line2, CURRENT_BG_INFO)
            out1, out1_eng, out1_name = generate_tts(CURRENT_CHA1_INFO, line1, "")
            
            # 순차적으로 재생 (겹치지 않게)
            play_audio_sequence([out2, out1], [f"{out2_name}: {out2_eng}", f"{out1_name}: {out1_eng}"])
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
        # 이전 진행 상황 중단: 모든 오디오 및 bgm 중단
        stop_all_audio()
        stop_background_music()
        bg = get_background(book_code)
        if bg is None:
            print(f"⚠ BACKGROUNDS에 없는 book_code: {book_code}")
            return

        CURRENT_BG_BOOK_CODE = book_code
        CURRENT_BG_INFO = bg

        print(f"[BACKGROUND SWAP] {book_code} → {bg.get('background')}")
        # 배경 교체 및 오버레이 비디오도 새 배경에 맞게 업데이트
        play_background_video(book_code)  # 배경 비디오 교체 (무한 루프, 오디오 포함, 페이드 효과)
        play_background_music(book_code)  # 배경 음악 교체 (무한 루프)
        
        # 현재 캐릭터들의 오버레이 비디오를 새 배경에 맞게 업데이트
        if CURRENT_CHA1_INFO is not None and CURRENT_CHA1_INFO.get('book_code'):
            try:
                overlay_path_ch1 = get_overlay_video_path(book_code, 1, CURRENT_CHA1_INFO['book_code'])
                print(f"🔍 [배경 교체] ch1 오버레이 비디오 경로: {overlay_path_ch1}")
                if os.path.exists(overlay_path_ch1):
                    print(f"✅ 파일 존재 확인, 오버레이 비디오 설정 중...")
                    VIDEO_PLAYER.set_overlay_video(overlay_path_ch1)
                    # 비디오가 제대로 설정되었는지 확인
                    import time
                    time.sleep(0.15)  # 비디오 초기화를 위한 대기
                    with VIDEO_PLAYER.lock:
                        is_set = VIDEO_PLAYER.overlay_video_cap is not None and VIDEO_PLAYER.overlay_video_cap.isOpened()
                    print(f"🎬 오버레이 비디오 ch1 업데이트 완료 (새 배경): {overlay_path_ch1} (설정됨: {is_set})")
                else:
                    print(f"⚠️ 오버레이 비디오 ch1를 찾을 수 없음: {overlay_path_ch1}")
                    VIDEO_PLAYER.set_overlay_video(None)
            except Exception as e:
                print(f"❌ ch1 오버레이 비디오 설정 중 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 발생 시에도 계속 진행
                try:
                    VIDEO_PLAYER.set_overlay_video(None)
                except:
                    pass
        
        if CURRENT_CHA2_INFO is not None and CURRENT_CHA2_INFO.get('book_code'):
            try:
                overlay_path_ch2 = get_overlay_video_path(book_code, 2, CURRENT_CHA2_INFO['book_code'])
                print(f"🔍 [배경 교체] ch2 오버레이 비디오 경로: {overlay_path_ch2}")
                if os.path.exists(overlay_path_ch2):
                    print(f"✅ 파일 존재 확인, 오버레이 비디오 설정 중...")
                    VIDEO_PLAYER.set_overlay_video2(overlay_path_ch2)
                    # 비디오가 제대로 설정되었는지 확인
                    import time
                    time.sleep(0.15)  # 비디오 초기화를 위한 대기
                    with VIDEO_PLAYER.lock:
                        is_set = VIDEO_PLAYER.overlay_video_cap2 is not None and VIDEO_PLAYER.overlay_video_cap2.isOpened()
                    print(f"🎬 오버레이 비디오 ch2 업데이트 완료 (새 배경): {overlay_path_ch2} (설정됨: {is_set})")
                else:
                    print(f"⚠️ 오버레이 비디오 ch2를 찾을 수 없음: {overlay_path_ch2}")
                    VIDEO_PLAYER.set_overlay_video2(None)
            except Exception as e:
                print(f"❌ ch2 오버레이 비디오 설정 중 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 발생 시에도 계속 진행
                try:
                    VIDEO_PLAYER.set_overlay_video2(None)
                except:
                    pass

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

        # 배경이 바뀌었을 때 놀란 대사
        line1 = generate_surprised_line(CURRENT_CHA1_INFO, CURRENT_BG_INFO)
        line2 = generate_surprised_line(CURRENT_CHA2_INFO, CURRENT_BG_INFO)

        out1, out1_eng, out1_name = generate_tts(CURRENT_CHA1_INFO, line1, "")
        out2, out2_eng, out2_name = generate_tts(CURRENT_CHA2_INFO, line2, "")
        
        # 순차적으로 재생 (겹치지 않게)
        play_audio_sequence([out1, out2], [f"{out1_name}: {out1_eng}", f"{out2_name}: {out2_eng}"])
        return

        # ---- 5,8,11,... : cha1 교체 ----
    if offset == 1:
        # (stop_all_audio는 함수 시작 부분에서 이미 호출됨)
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha1"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha1 정의 없음")
            return

        cha1 = build_character(book_code, role_key)
        CURRENT_CHA1_INFO = cha1

        # ch1 오버레이 비디오 업데이트 (배경에 맞는 폴더에서 찾기)
        if CURRENT_BG_BOOK_CODE:
            overlay_path = get_overlay_video_path(CURRENT_BG_BOOK_CODE, 1, book_code)
            print(f"🔍 [cha1 교체] ch1 오버레이 비디오 경로: {overlay_path}")
            print(f"🔍 [cha1 교체] 배경: {CURRENT_BG_BOOK_CODE}, 새 캐릭터: {book_code}")
            if os.path.exists(overlay_path):
                print(f"✅ 파일 존재 확인됨, 오버레이 비디오 설정 중...")
                # 오버레이 비디오 즉시 설정
                VIDEO_PLAYER.set_overlay_video(overlay_path)
                import time
                time.sleep(0.1)
                print(f"🎬 오버레이 비디오 ch1 업데이트 완료: {overlay_path}")
            else:
                print(f"⚠️ 오버레이 비디오를 찾을 수 없음: {overlay_path}")
                # 파일이 없으면 오버레이 비디오를 None으로 설정
                VIDEO_PLAYER.set_overlay_video(None)

        # 🔸 장화홍련 자매인 경우: 랜덤 순서로 각각 한 줄씩 말하고,
        #    기존 cha2(예: 토끼, 귀신 등)가 한 줄 더 대답.
        if book_code == "JHHRJ" and role_key == "sister_older":
            sister_older, sister_younger = build_sisters_pair()

            # 랜덤으로 순서 결정 - 모든 대화 생성 후 순차 재생
            if random.random() < 0.5:
                # 언니 → 동생 순서
                lineA = generate_first_dialogue_line(sister_older, CURRENT_BG_INFO)
                outA, outA_eng, outA_name = generate_tts(sister_older, lineA, "")
                
                lineB = generate_second_dialogue_line(sister_younger, lineA, CURRENT_BG_INFO)
                outB, outB_eng, outB_name = generate_tts(sister_younger, lineB, "")
                
                # cha2가 자매의 대화에 반응
                reply = generate_second_dialogue_line(CURRENT_CHA2_INFO, lineA, CURRENT_BG_INFO)
                outC, outC_eng, outC_name = generate_tts(CURRENT_CHA2_INFO, reply, "")
                
                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([outA, outB, outC], [f"{outA_name}: {outA_eng}", f"{outB_name}: {outB_eng}", f"{outC_name}: {outC_eng}"])
            else:
                # 동생 → 언니 순서
                lineB = generate_first_dialogue_line(sister_younger, CURRENT_BG_INFO)
                outB, outB_eng, outB_name = generate_tts(sister_younger, lineB, "")
                
                lineA = generate_second_dialogue_line(sister_older, lineB, CURRENT_BG_INFO)
                outA, outA_eng, outA_name = generate_tts(sister_older, lineA, "")
                
                # cha2가 자매의 대화에 반응
                reply = generate_second_dialogue_line(CURRENT_CHA2_INFO, lineB, CURRENT_BG_INFO)
                outC, outC_eng, outC_name = generate_tts(CURRENT_CHA2_INFO, reply, "")

                # 순차적으로 재생 (겹치지 않게)
                play_audio_sequence([outB, outA, outC], [f"{outB_name}: {outB_eng}", f"{outA_name}: {outA_eng}", f"{outC_name}: {outC_eng}"])
            return

        # 🔹 그 외 일반 캐릭터: 새 cha1 + 기존 cha2가 한 줄씩 대화
        # 모든 대화 생성 후 순차 재생
        line1 = generate_first_dialogue_line(cha1, CURRENT_BG_INFO)
        out1, out1_eng, out1_name = generate_tts(cha1, line1, "")
        
        line2 = generate_second_dialogue_line(CURRENT_CHA2_INFO, line1, CURRENT_BG_INFO)
        out2, out2_eng, out2_name = generate_tts(CURRENT_CHA2_INFO, line2, "")
        
        # 순차적으로 재생 (겹치지 않게)
        play_audio_sequence([out1, out2], [f"{out1_name}: {out1_eng}", f"{out2_name}: {out2_eng}"])
        return

    # ---- 6,9,12,... : cha2 교체 ----
    if offset == 2:
        # (stop_all_audio는 함수 시작 부분에서 이미 호출됨)
        if book_code not in ROLE_MAP:
            print(f"⚠ ROLE_MAP에 없는 book_code: {book_code}")
            return
        role_key = ROLE_MAP[book_code]["cha2"]
        if role_key is None:
            print(f"⚠ {book_code}에 cha2 정의 없음")
            return

        cha2 = build_character(book_code, role_key)
        CURRENT_CHA2_INFO = cha2

        # ch2 오버레이 비디오 업데이트 (배경에 맞는 폴더에서 찾기)
        if CURRENT_BG_BOOK_CODE:
            overlay_path2 = get_overlay_video_path(CURRENT_BG_BOOK_CODE, 2, book_code)
            print(f"🔍 [cha2 교체] ch2 오버레이 비디오 경로: {overlay_path2}")
            print(f"🔍 [cha2 교체] 배경: {CURRENT_BG_BOOK_CODE}, 새 캐릭터: {book_code}")
            if os.path.exists(overlay_path2):
                print(f"✅ 파일 존재 확인, 오버레이 비디오 설정 중...")
                # 오버레이 비디오 즉시 설정
                VIDEO_PLAYER.set_overlay_video2(overlay_path2)
                import time
                time.sleep(0.15)
                with VIDEO_PLAYER.lock:
                    is_set = VIDEO_PLAYER.overlay_video_cap2 is not None and VIDEO_PLAYER.overlay_video_cap2.isOpened()
                print(f"🎬 오버레이 비디오 ch2 업데이트 완료: {overlay_path2} (설정됨: {is_set})")
            else:
                print(f"⚠️ 오버레이 비디오 ch2를 찾을 수 없음: {overlay_path2}")
                VIDEO_PLAYER.set_overlay_video2(None)

        # cha1이 장화홍련인 경우: cha2가 먼저 말하고, 자매가 랜덤 순서로 각각 한 번씩 말함
        if CURRENT_CHA1_INFO is not None and CURRENT_CHA1_INFO.get('book_code') == "JHHRJ":
            older, younger = build_sisters_pair()
            
            # cha2가 먼저 말함
            line_cha2 = generate_first_dialogue_line(cha2, CURRENT_BG_INFO)
            out_cha2, out_cha2_eng, out_cha2_name = generate_tts(cha2, line_cha2, "")
            
            # 랜덤으로 순서 결정 - 모든 대화 생성 후 순차 재생
            if random.random() < 0.5:
                # 장화 먼저
                line_older = generate_second_dialogue_line(older, line_cha2, CURRENT_BG_INFO)
                out_older, out_older_eng, out_older_name = generate_tts(older, line_older, "")
                
                line_younger = generate_second_dialogue_line(younger, line_older, CURRENT_BG_INFO)
                out_younger, out_younger_eng, out_younger_name = generate_tts(younger, line_younger, "")
                
                # 순차적으로 재생 (겹치지 않게): cha2 -> 장화 -> 홍련
                play_audio_sequence([out_cha2, out_older, out_younger], [f"{out_cha2_name}: {out_cha2_eng}", f"{out_older_name}: {out_older_eng}", f"{out_younger_name}: {out_younger_eng}"])
            else:
                # 홍련 먼저
                line_younger = generate_second_dialogue_line(younger, line_cha2, CURRENT_BG_INFO)
                out_younger, out_younger_eng, out_younger_name = generate_tts(younger, line_younger, "")
                
                line_older = generate_second_dialogue_line(older, line_younger, CURRENT_BG_INFO)
                out_older, out_older_eng, out_older_name = generate_tts(older, line_older, "")
                
                # 순차적으로 재생 (겹치지 않게): cha2 -> 홍련 -> 장화
                play_audio_sequence([out_cha2, out_younger, out_older], [f"{out_cha2_name}: {out_cha2_eng}", f"{out_younger_name}: {out_younger_eng}", f"{out_older_name}: {out_older_eng}"])
        else:
            # cha2가 먼저 말하고, cha1이 대답하도록 순서 변경
            # 첫 번째 대화 생성
            line2 = generate_first_dialogue_line(cha2, CURRENT_BG_INFO)
            out2, out2_eng, out2_name = generate_tts(cha2, line2, "")
            
            # 두 번째 대화 생성
            line1 = generate_second_dialogue_line(CURRENT_CHA1_INFO, line2, CURRENT_BG_INFO)
            out1, out1_eng, out1_name = generate_tts(CURRENT_CHA1_INFO, line1, "")
            
            # 순차적으로 재생 (겹치지 않게)
            play_audio_sequence([out2, out1], [f"{out2_name}: {out2_eng}", f"{out1_name}: {out1_eng}"])
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
    last_marker_time = None  # 마지막으로 마커가 감지된 시간
    no_marker_timeout = 3.0  # 마커가 감지되지 않을 때 페이드 아웃까지의 대기 시간 (초)
    fade_out_triggered = False  # 페이드 아웃이 이미 트리거되었는지 여부
    
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
        
        current_time = time.time()
        
        # 마커가 감지되지 않았을 때 처리 (타임아웃 버퍼 적용)
        if ids is None or len(ids) == 0:
            # 이전에 마커가 있었는데 지금 없으면 타임아웃 체크
            if last_detected_marker is not None and not fade_out_triggered:
                # 마커가 감지되지 않은 시간 계산
                if last_marker_time is None:
                    # 마커가 처음으로 사라진 시점 기록
                    last_marker_time = current_time
                
                time_since_last_marker = current_time - last_marker_time
                
                # 타임아웃 시간이 지났고 아직 페이드 아웃이 트리거되지 않았으면 페이드 아웃
                if time_since_last_marker >= no_marker_timeout:
                    # 모든 오디오 중단
                    stop_all_audio()
                    stop_background_music()
                    # 비디오 페이드 아웃 (검은 화면으로)
                    VIDEO_PLAYER.set_video(None)  # None을 전달하면 페이드 아웃
                    # 오버레이 비디오 제거
                    VIDEO_PLAYER.clear_overlay_video()
                    # 상태 초기화 (전역 변수도 리셋)
                    global CURRENT_BG_BOOK_CODE, CURRENT_BG_INFO, CURRENT_CHA1_INFO, CURRENT_CHA2_INFO
                    last_detected_marker = None
                    sequence_index = 0
                    fade_out_triggered = True
                    last_marker_time = None
                    CURRENT_BG_BOOK_CODE = None
                    CURRENT_BG_INFO = None
                    CURRENT_CHA1_INFO = None
                    CURRENT_CHA2_INFO = None
                    print(f"🔇 마커가 {no_marker_timeout}초 동안 감지되지 않아 모든 오디오 중단 및 페이드 아웃 (리셋 완료)")
        # 감지된 마커가 있으면 표시
        if ids is not None and len(ids) > 0:
            # 마커가 감지되면 타임아웃 리셋 (같은 마커든 새 마커든)
            last_marker_time = current_time
            fade_out_triggered = False
            
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 첫 번째로 감지된 마커 처리
            marker_id = ids[0][0]
            book_code = get_book_code_from_marker(marker_id)
            
            # 새 마커 감지 처리
            if book_code and marker_id != last_detected_marker:
                # 3n-2, 3n-1, 3n번째 책은 즉시 전환 (이전 진행 상황 중단)
                should_interrupt = (sequence_index + 1) % 3 in [1, 2, 0]  # 1,2,0 -> 3n-2, 3n-1, 3n
                
                if should_interrupt or not is_processing:
                    # 즉시 전환이 필요한 경우 모든 오디오 중단
                    if should_interrupt:
                        stop_all_audio()
                    
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
        else:
            # 비디오가 없을 때 까만 화면 표시 (아무것도 감지되지 않았을 때)
            # 비디오가 시작되기 전에도 윈도우를 유지하기 위해 까만 화면 표시
            black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 기본 해상도
            cv2.imshow("Background Video", black_frame)
        
        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    VIDEO_PLAYER.stop()
    stop_background_music()  # bgm 중지
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