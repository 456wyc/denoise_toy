"""test_anc.py — 不依赖真实声卡的核心逻辑冒烟测试。

用 sys.modules 注入一个 sounddevice 桩，从而在无 PortAudio 的环境下
也能 import anc_core 并验证：WAV 读取/下混、相位反转、音乐循环混合。
"""

from __future__ import annotations

import sys
import types
import wave
import os
import tempfile

import numpy as np

# ---- 注入 sounddevice 桩（仅当真实库不可用时）----
try:
    import sounddevice  # noqa: F401
except ImportError:
    fake = types.ModuleType("sounddevice")
    fake.Stream = None
    fake.query_devices = lambda *a, **k: []
    fake.query_hostapis = lambda *a, **k: {}
    fake.default = types.SimpleNamespace(device=(None, None))
    sys.modules["sounddevice"] = fake

from anc_core import (  # noqa: E402
    MODE_ANC,
    MODE_MONITOR,
    MODE_OFF,
    ANCEngine,
    BiquadLPF,
    load_wav_mono,
)


def _write_wav(path: str, channels: int, data: np.ndarray, sampwidth: int = 2) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(48000)
        wf.writeframes(data.astype(np.int16).tobytes() if sampwidth == 2 else data.tobytes())


def test_load_wav_mono_stereo_downmix() -> None:
    # 左右声道分别全 1.0 和全 -1.0，下混应为 0
    full = (1 << 15) - 1
    left = np.full(100, full, dtype=np.int16)
    right = np.full(100, -full, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)  # (100, 2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.wav")
        _write_wav(path, 2, stereo)
        data, sr = load_wav_mono(path)
    assert sr == 48000, sr
    assert data.shape == (100,), data.shape
    assert np.max(np.abs(data)) < 1e-3, f"下混未抵消: {np.max(np.abs(data))}"
    print("PASS load_wav_mono 立体声下混")


def test_load_wav_mono_amplitude() -> None:
    full = (1 << 15) - 1
    mono = np.full(200, full, dtype=np.int16)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.wav")
        _write_wav(path, 1, mono)
        data, _ = load_wav_mono(path)
    assert abs(data[0] - 1.0) < 1e-3, data[0]  # 满幅 ≈ 1.0
    print("PASS load_wav_mono 幅度标定")


def test_callback_phase_inversion() -> None:
    eng = ANCEngine()
    eng.mode = MODE_ANC
    eng.gain = 1.0
    eng.lpf_enabled = False  # 关闭低通，测试纯相位反转
    frames = 64
    t = np.linspace(0, 1, frames, endpoint=False)
    indata = (0.5 * np.sin(2 * np.pi * 5 * t)).astype(np.float32).reshape(-1, 1)
    outdata = np.zeros((frames, 2), dtype=np.float32)
    eng._callback(indata, outdata, frames, None, None)
    expected = -indata[:, 0]
    assert np.allclose(outdata[:, 0], expected, atol=1e-5), "反相波应等于 -输入"
    assert np.allclose(outdata[:, 1], expected, atol=1e-5), "两个输出通道应一致"
    print("PASS callback 相位反转 + 双声道广播")


def test_callback_off_silent() -> None:
    eng = ANCEngine()
    eng.mode = MODE_OFF
    indata = np.ones((32, 1), dtype=np.float32)
    outdata = np.ones((32, 2), dtype=np.float32)
    eng._callback(indata, outdata, 32, None, None)
    assert np.allclose(outdata, 0.0), "off 模式应静音"
    print("PASS callback off 静音")


def test_callback_monitor_passthrough() -> None:
    eng = ANCEngine()
    eng.mode = MODE_MONITOR
    eng.gain = 0.5
    indata = (np.ones((16, 1)) * 0.4).astype(np.float32)
    outdata = np.zeros((16, 2), dtype=np.float32)
    eng._callback(indata, outdata, 16, None, None)
    assert np.allclose(outdata[:, 0], 0.2, atol=1e-6), "monitor 应透传 × 增益"
    print("PASS callback monitor 透传")


def test_playback_loop() -> None:
    eng = ANCEngine()
    eng.set_playback(np.array([1.0, 2.0, 3.0], dtype=np.float32), loop=True)
    out = eng._read_playback(8)  # 3 样本循环取 8 个
    expected = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.float32)
    assert np.allclose(out, expected), out
    # 再取 2 个，应从上一位置继续（上一轮已取 8，pos=2 → 下一个是 3）
    out2 = eng._read_playback(2)
    assert np.allclose(out2, np.array([3, 1], dtype=np.float32)), out2
    print("PASS 音乐循环播放 + 位置续接")


def test_playback_mix_into_anc() -> None:
    eng = ANCEngine()
    eng.mode = MODE_ANC
    eng.gain = 1.0
    eng.lpf_enabled = False  # 关闭低通，单独验证混合
    eng.playback_gain = 0.5
    eng.set_playback(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32), loop=True)
    indata = (np.ones((4, 1)) * 0.2).astype(np.float32)
    outdata = np.zeros((4, 2), dtype=np.float32)
    eng._callback(indata, outdata, 4, None, None)
    expected = -0.2 + 0.5 * 1.0  # 反相波 + 音乐×音量 = 0.3
    assert np.allclose(outdata[:, 0], expected, atol=1e-6), outdata[:, 0]
    print("PASS 音乐混合进降噪输出")


def _rms_tail(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x[len(x) // 2 :] ** 2)))


def test_lpf_passes_low_attenuates_high() -> None:
    sr = 48000
    lpf = BiquadLPF(sr, 150.0)
    t = np.arange(4000) / sr
    low = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    out_low = lpf.process(low)
    lpf2 = BiquadLPF(sr, 150.0)
    out_high = lpf2.process(high)
    # 50Hz 基本通过（增益≈1），2000Hz 被大幅衰减
    assert _rms_tail(out_low) > 0.5, f"低频应通过: {_rms_tail(out_low)}"
    assert _rms_tail(out_high) < 0.05, f"高频应被衰减: {_rms_tail(out_high)}"
    print("PASS 低通：50Hz 通过 / 2000Hz 衰减")


def test_callback_anc_applies_lpf() -> None:
    eng = ANCEngine()
    eng.mode = MODE_ANC
    eng.gain = 1.0
    eng.lpf_enabled = True
    eng.lpf_cutoff = 150.0
    eng._lpf = None  # 强制重建
    sr = 48000
    t = np.arange(4000) / sr
    high = (np.sin(2 * np.pi * 2000 * t) * 0.5).astype(np.float32).reshape(-1, 1)
    outdata = np.zeros((4000, 1), dtype=np.float32)
    eng._callback(high, outdata, 4000, None, None)
    assert _rms_tail(outdata[:, 0]) < 0.05, f"ANC 高频应被低通砍掉: {_rms_tail(outdata[:, 0])}"
    print("PASS callback ANC 模式应用低通（高频被砍）")


def test_delay_line_ring_buffer() -> None:
    """延迟线环形缓冲：先推进去的样本，延迟 N 样本后能正确读出来。"""
    eng = ANCEngine()
    eng.blocksize = 10
    eng._ensure_delay_buf(20)
    # 写入 1-10
    eng._delay_push(np.arange(1, 11, dtype=np.float32))
    # 写入 11-20
    eng._delay_push(np.arange(11, 21, dtype=np.float32))
    # 读 delay=5 的 5 个样本，应为 11,12,13,14,15
    out = eng._delay_read_delayed(5, 5)
    expected = np.array([11, 12, 13, 14, 15], dtype=np.float32)
    assert np.allclose(out, expected), f"延迟读出不对: {out} vs {expected}"
    # 读 delay=15 的 5 个样本，应为 1,2,3,4,5
    out2 = eng._delay_read_delayed(5, 15)
    expected2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert np.allclose(out2, expected2), f"延迟读出不对: {out2} vs {expected2}"
    print("PASS 延迟线环形缓冲：写入/延迟读出正确")


def test_delay_line_wrap_around() -> None:
    """环形缓冲回绕时也能正确读出。"""
    eng = ANCEngine()
    eng.blocksize = 3
    eng._ensure_delay_buf(3)  # 缓冲大小会是 3 + 3*2 = 9
    buf_len = len(eng._delay_buf)
    # 多次写入让指针回绕
    for i in range(20):
        eng._delay_push(np.array([i * 3 + 1, i * 3 + 2, i * 3 + 3], dtype=np.float32))
    # delay=4 读 3 个：应该是最新位置 - 4 开始的 3 个
    # 第二次 push 后 pos = 60 % 9 = 6，最新样本值 60 在位置 5。
    # delay=4 个样本前，读 3 个 → 第 54, 55, 56 个样本（值 54, 55, 56）
    out = eng._delay_read_delayed(3, 4)
    expected = np.array([54, 55, 56], dtype=np.float32)
    assert np.allclose(out, expected), f"回绕读不对: {out} vs {expected}"
    print("PASS 延迟线环形缓冲：回绕后读取正确")


def test_callback_anc_respects_delay() -> None:
    """ANC 模式下设置 delay_samples > 0 时，输出应该是 N 个样本前的反相。"""
    eng = ANCEngine()
    eng.mode = MODE_ANC
    eng.gain = 1.0
    eng.lpf_enabled = False
    eng.blocksize = 16
    eng.delay_samples = 4
    # 第一次 callback：写入第一块
    indata1 = (np.arange(1, 17, dtype=np.float32) * 0.1).reshape(-1, 1)
    outdata1 = np.zeros((16, 1), dtype=np.float32)
    eng._callback(indata1, outdata1, 16, None, None)
    # delay=4 时，第一块的输出取的是 delay=4 前的输入。但第一块之前是 0。
    # 所以输出前 12 个是 delay 前 4 个（0s）+ 后 12 个用前 12 个输入？
    # 不对，第一次 callback 时 buffer 里只有第一块的 16 个样本（刚 push 的）。
    # delay=4 意味着读 "4 样本之前" 的值。刚 push 了 16 个，尾部是 16 个最新的。
    # read_delayed(16, 4) 从 pos - 4 - 16 = pos - 20 的位置读 16 个。
    # 第一次 push 后 pos = 16。16 - 20 = -4 → 模运算后是 buf_len - 4，那里都是 0。
    # 所以第一次输出应该是部分 0 部分有值，取决于缓冲大小。
    # 这个测试比较复杂，简化：跑两次 callback，第二次的输出应该包含第一次的输入的后半部分。
    # 第二次 callback：写入第二块
    indata2 = (np.arange(17, 33, dtype=np.float32) * 0.1).reshape(-1, 1)
    outdata2 = np.zeros((16, 1), dtype=np.float32)
    eng._callback(indata2, outdata2, 16, None, None)
    # 第二块 push 完后，缓冲末尾是 indata2。delay=4 时，读出 16 个 = 第 13..28 个样本的反相。
    # 但第一块是 1..16，第二块是 17..32。
    # push 顺序：先第一块，再第二块。缓冲末尾是第二块的最后一个（即 32）。
    # delay=4 前的样本 = 从末尾减 4 再往前数 16 个。
    # 末尾位置：第 32 个样本（索引 31 从 0 算起）。
    # 读 delay=4 的 16 个：起点 = 31 - 4 - 15 = 12？ 不对。
    # _delay_read_delayed(frames, delay): 读 delay 样本之前的 frames 个样本。
    # 最新样本位置 = _delay_pos - 1（刚 push 完）。
    # 起点 = (pos - delay - frames) % buf_len
    # 第二次 push 后 pos = 32。 frames=16, delay=4.
    # start = (32 - 4 - 16) % buf_len = 12 % buf_len = 12
    # 样本索引 12..27（共 16 个）的值：
    #   索引 12..15 = 第一块的最后 4 个（13..16），乘以 0.1 = 1.3..1.6
    #   索引 16..27 = 第二块的前 12 个（17..28），乘以 0.1 = 1.7..2.8
    # 反相后就是取负。
    expected = -np.concatenate([
        np.arange(13, 17, dtype=np.float32) * 0.1,
        np.arange(17, 29, dtype=np.float32) * 0.1,
    ])
    assert np.allclose(outdata2[:, 0], expected, atol=1e-5), f"延迟反相不对"
    print("PASS callback ANC 延迟补偿：输出为 N 样本前的反相")


def main() -> None:
    test_load_wav_mono_stereo_downmix()
    test_load_wav_mono_amplitude()
    test_callback_phase_inversion()
    test_callback_off_silent()
    test_callback_monitor_passthrough()
    test_playback_loop()
    test_playback_mix_into_anc()
    test_lpf_passes_low_attenuates_high()
    test_callback_anc_applies_lpf()
    test_delay_line_ring_buffer()
    test_delay_line_wrap_around()
    test_callback_anc_respects_delay()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
