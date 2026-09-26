"""anc_core.py — 实时前馈主动降噪（ANC）音频引擎核心。

原理
----
把普通耳机变成降噪耳机，本质是"前馈主动降噪"（feedforward ANC）：

    环境噪声 ──► 麦克风采样 ──► 相位反转 180°（取负）──► 耳机播放
                                                              │
    原始噪声 ──────────────────────────► 耳内 ◄───────────────┘
                                        （反相波与噪声叠加相消）

当反相波与原始噪声在耳内同幅、同相（差 180°）叠加时，两者抵消，
人耳听到的噪声大幅减弱。

两个决定成败的物理约束
----------------------
1. 低延迟：软件往返延迟（麦克风 → 处理 → 播放）越低，能抵消的频率越高。
   粗略估算：延迟 τ 秒时，可有效降噪的频率上限约为 1/(4τ)。
   例如 τ≈10ms 时约能抵消 25Hz 以下的低频；τ≈1ms 时约 250Hz。
   因此：块大小尽量小、采样率适中、优先用 "low" 延迟，并尽量用
   WASAPI 独占 / ASIO 驱动（本程序通过 PortAudio 使用共享模式，
   延迟通常在 10~30ms 量级，主要抵消低频噪声）。

2. 麦克风位置：麦克风必须贴近耳朵（塞进耳罩内 / 耳塞旁、朝外），
   让采到的信号近似"即将进入耳朵"的噪声。用笔记本自带麦克风或桌面麦克风
   基本无效（它采到的不是耳内噪声），甚至可能增噪。

三种模式
--------
* off      —— 静音输出（耳机不通声音）。
* monitor  —— 透传监听（把麦克风采到的声音原样放给你听），
              用于确认麦克风确实采到了环境噪声、摆放位置对不对。
* anc      —— 输出反相波，进行降噪；可叠加音乐一起听。

依赖：sounddevice, numpy
"""

from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

# 运行模式
MODE_OFF = "off"
MODE_MONITOR = "monitor"
MODE_ANC = "anc"
MODES = (MODE_OFF, MODE_MONITOR, MODE_ANC)


@dataclass
class DeviceInfo:
    """设备枚举结果，方便 GUI/CLI 展示。"""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    is_default: bool

    def __str__(self) -> str:
        tag = " [默认]" if self.is_default else ""
        return f"[{self.index}] {self.name} ({self.hostapi}){tag}"


def list_devices(kind: str) -> list[DeviceInfo]:
    """列出输入或输出设备。kind: 'input' | 'output'。"""
    if kind not in ("input", "output"):
        raise ValueError("kind 必须是 'input' 或 'output'")
    devices = sd.query_devices()
    default = sd.default.device
    result: list[DeviceInfo] = []
    for idx, d in enumerate(devices):
        max_ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if max_ch <= 0:
            continue
        is_default = default[0 if kind == "input" else 1] == idx
        hostapi = sd.query_hostapis(d["hostapi"])["name"]
        result.append(
            DeviceInfo(
                index=idx,
                name=d["name"],
                hostapi=hostapi,
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=int(d["default_samplerate"]),
                is_default=is_default,
            )
        )
    return result


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件为 float32 单声道（立体声自动下混），返回 (data, samplerate)。"""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 3:
        # 24-bit：补齐成 32-bit 再缩放
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        x = (a[:, 0].astype(np.int32)
             | (a[:, 1].astype(np.int32) << 8)
             | (a[:, 2].astype(np.int32) << 16))
        x = np.where(x & 0x800000, x - 0x1000000, x)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype=np.int32)
    else:
        raise ValueError(f"不支持的采样位宽：{sampwidth * 8} bit")

    x = x.reshape(-1, channels)
    if channels > 1:
        x = x.mean(axis=1)  # 立体声下混为单声道
    x = x.astype(np.float32) / float(1 << (sampwidth * 8 - 1))
    return np.ascontiguousarray(x, dtype=np.float32), framerate


class BiquadLPF:
    """二阶低通滤波器（RBJ cookbook，Q=0.707 巴特沃斯）。

    作用：把反相波限制在"软件 ANC 真正能抵消"的低频段。
    延迟决定了可抵消频率上限（约 1/(4τ)），高于该上限的频率不仅抵消不了，
    还会被当成额外噪声叠加到耳朵里 —— 这正是"扩音器感"的来源。
    因此只保留低频反相波、砍掉中高频，既更接近真实 ANC 的效果，也更安静。
    """

    def __init__(self, samplerate: int, cutoff: float) -> None:
        self.samplerate = samplerate
        self.z1 = 0.0
        self.z2 = 0.0
        self.set_cutoff(cutoff)

    def set_cutoff(self, cutoff: float) -> None:
        cutoff = float(min(max(cutoff, 10.0), self.samplerate * 0.45))
        omega = 2.0 * np.pi * cutoff / self.samplerate
        alpha = np.sin(omega) / (2.0 * 0.7071067811865476)
        cosw = np.cos(omega)
        b0 = (1.0 - cosw) / 2.0
        b1 = 1.0 - cosw
        b2 = (1.0 - cosw) / 2.0
        a0 = 1.0 + alpha
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = (-2.0 * cosw) / a0
        self.a2 = (1.0 - alpha) / a0

    def process(self, x: np.ndarray) -> np.ndarray:
        """对输入块做滤波，返回同形状 float32 结果。"""
        out = np.empty_like(x)
        for i in range(len(x)):
            xi = float(x[i])
            y = self.b0 * xi + self.z1
            self.z1 = self.b1 * xi - self.a1 * y + self.z2
            self.z2 = self.b2 * xi - self.a2 * y
            out[i] = y
        return out


class ANCEngine:
    """基于 sounddevice 双工流的实时降噪引擎。

    一个对象同时开输入 + 输出（duplex），保证输入/输出采样同步，
    从而获得最短的往返延迟。音频回调运行在 PortAudio 的独立线程里，
    与 GUI 主线程互不阻塞。
    """

    def __init__(self) -> None:
        self._stream: Optional[sd.Stream] = None
        self._lock = threading.Lock()

        # ---- 流参数 ----
        self.samplerate: int = 48000
        self.blocksize: int = 128
        self.input_device: Optional[int] = None   # None = 系统默认
        self.output_device: Optional[int] = None
        self.input_channels: int = 1
        self.output_channels: int = 2
        self.latency: str = "low"                 # 'low' | 'high'

        # ---- 降噪 / 播放参数 ----
        self.mode: str = MODE_OFF
        self.gain: float = 1.0                    # 反相波（或监听）增益
        self.playback_gain: float = 1.0           # 音乐音量
        self._playback: Optional[np.ndarray] = None
        self._playback_pos: int = 0
        self._playback_loop: bool = True
        # 反相波低通：只保留可抵消的低频段（延迟越短，上限可调越高）
        self.lpf_enabled: bool = True
        self.lpf_cutoff: float = 150.0            # 截止频率(Hz)，>0.45*sr 视为禁用
        self._lpf: Optional[BiquadLPF] = None
        self._lpf_sr: int = 0
        self._lpf_cut: float = 0.0

        # ---- 监测 ----
        self.in_level: float = 0.0                # 输入 RMS 电平
        self.out_level: float = 0.0               # 输出 RMS 电平
        self._clip_count: int = 0
        self._xrun_count: int = 0

        # ---- 状态 ----
        self.running: bool = False
        self.error: Optional[str] = None
        self.started_at: float = 0.0

    # ------------------------------------------------------------------ #
    # 播放源
    # ------------------------------------------------------------------ #
    def set_playback(self, data: Optional[np.ndarray], loop: bool = True) -> None:
        """设置要混合进输出的音乐（单声道 float32）。传 None 表示停止播放。"""
        with self._lock:
            self._playback = (
                np.ascontiguousarray(data, dtype=np.float32) if data is not None else None
            )
            self._playback_pos = 0
            self._playback_loop = loop

    def load_playback_file(self, path: str, loop: bool = True) -> None:
        """从 WAV 文件加载音乐并开始混合播放。"""
        data, sr = load_wav_mono(path)
        if sr != self.samplerate:
            # 简单重采样到当前采样率（线性插值）
            xp = np.arange(len(data)) * sr / self.samplerate
            xi = np.arange(int(len(data) * self.samplerate / sr))
            data = np.interp(xi, xp, data).astype(np.float32)
        self.set_playback(data, loop)

    def _read_playback(self, frames: int) -> np.ndarray:
        """从播放缓冲区取 frames 个样本（循环播放），返回 float32。"""
        with self._lock:
            data = self._playback
            if data is None or len(data) == 0:
                return np.zeros(frames, dtype=np.float32)
            n = len(data)
            loop = self._playback_loop
            pos = self._playback_pos
            out = np.zeros(frames, dtype=np.float32)
            remaining = frames
            start = 0
            while remaining > 0 and pos < n:
                take = min(remaining, n - pos)
                out[start : start + take] = data[pos : pos + take]
                pos += take
                start += take
                remaining -= take
                if pos >= n and loop:
                    pos = 0
            self._playback_pos = pos
            return out

    # ------------------------------------------------------------------ #
    # 设备枚举（静态辅助，供 GUI/CLI 使用）
    # ------------------------------------------------------------------ #
    @staticmethod
    def list_input_devices() -> list[DeviceInfo]:
        return list_devices("input")

    @staticmethod
    def list_output_devices() -> list[DeviceInfo]:
        return list_devices("output")

    # ------------------------------------------------------------------ #
    # 音频回调（运行在 PortAudio 线程）
    # ------------------------------------------------------------------ #
    def _get_lpf(self) -> BiquadLPF:
        """返回与当前采样率/截止频率匹配的滤波器（自动重建）。"""
        if (self._lpf is None or self._lpf_sr != self.samplerate
                or self._lpf_cut != self.lpf_cutoff):
            self._lpf = BiquadLPF(self.samplerate, self.lpf_cutoff)
            self._lpf_sr = self.samplerate
            self._lpf_cut = self.lpf_cutoff
        return self._lpf

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            self._xrun_count += 1
        try:
            # 取第一路输入作为噪声参考
            noise = indata[:, 0]
            self.in_level = float(np.sqrt(np.mean(noise * noise)) + 1e-12)

            if self.mode == MODE_ANC:
                out = -self.gain * noise          # 相位反转 → 反相波
                if self.lpf_enabled:
                    out = self._get_lpf().process(out)  # 只保留可抵消的低频段
            elif self.mode == MODE_MONITOR:
                out = self.gain * noise           # 透传监听
            else:
                out = np.zeros(frames, dtype=np.float32)

            # 叠加音乐（若已加载）
            if self._playback is not None:
                out = out + self._read_playback(frames) * self.playback_gain

            # 电平与削波监测
            self.out_level = float(np.sqrt(np.mean(out * out)) + 1e-12)
            if float(np.max(np.abs(out))) > 0.999:
                self._clip_count += 1

            # 广播到所有输出通道
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = out
        except Exception as exc:  # 兜底：任何异常都静音，不让流崩掉
            self.error = str(exc)
            outdata.fill(0)

    # ------------------------------------------------------------------ #
    # 启停
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self.running:
            return
        self._xrun_count = 0
        self._clip_count = 0
        self.error = None
        self._lpf = None  # 重置滤波器状态，避免残留旧延迟
        self._stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=(self.input_device, self.output_device),
            channels=(self.input_channels, self.output_channels),
            dtype="float32",
            latency=self.latency,
            callback=self._callback,
        )
        self._stream.start()
        self.running = True
        self.started_at = time.time()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self.running = False
        self.in_level = 0.0
        self.out_level = 0.0

    def toggle(self) -> bool:
        if self.running:
            self.stop()
        else:
            self.start()
        return self.running

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #
    def latency_info(self) -> Optional[dict]:
        """返回实际输入/输出延迟（秒），未启动时返回 None。"""
        if self._stream is None:
            return None
        return {
            "input": float(self._stream.latency[0]),
            "output": float(self._stream.latency[1]),
            "roundtrip_ms": round(
                (float(self._stream.latency[0]) + float(self._stream.latency[1])) * 1000.0, 2
            ),
        }

    def status_summary(self) -> str:
        mode_label = {MODE_OFF: "关闭", MODE_MONITOR: "监听", MODE_ANC: "降噪"}.get(self.mode, self.mode)
        parts = [f"{mode_label}", f"运行中" if self.running else "已停止"]
        lat = self.latency_info()
        if lat:
            parts.append(f"往返延迟 {lat['roundtrip_ms']} ms")
        if self.mode == MODE_ANC and self.lpf_enabled:
            parts.append(f"低通 {int(self.lpf_cutoff)}Hz")
        parts.append(f"输入电平 {self.in_level * 100:.1f}%")
        parts.append(f"输出电平 {self.out_level * 100:.1f}%")
        if self._xrun_count:
            parts.append(f"欠载/溢出 {self._xrun_count} 次")
        if self._clip_count:
            parts.append(f"削波 {self._clip_count} 次")
        if self.error:
            parts.append(f"错误: {self.error}")
        return " | ".join(parts)
