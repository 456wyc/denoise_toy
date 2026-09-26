"""anc_cli.py — 主动降噪耳机小软件的命令行版本。

用法示例：
    # 列出设备
    python anc_cli.py --list

    # 使用默认设备启动降噪
    python anc_cli.py --mode anc

    # 指定设备并边降噪边听音乐
    python anc_cli.py --mode anc --input 2 --output 5 --gain 1.0 --music music.wav

    # 监听模式（确认麦克风摆位）
    python anc_cli.py --mode monitor

按 Ctrl+C 退出。
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from anc_core import MODES, ANCEngine


def _grouped_list(title: str, devices: list) -> None:
    """按 host API（驱动栈）分组打印设备，方便区分同一物理设备的重复条目。"""
    print(f"\n=== {title} ===")
    by_api: dict[str, list] = {}
    for d in devices:
        by_api.setdefault(d.hostapi, []).append(d)
    # 把 WASAPI 排最前（推荐），其余按名称
    order = sorted(by_api.keys(), key=lambda k: (0 if "wasapi" in k.lower() else 1, k.lower()))
    for api in order:
        items = by_api[api]
        tag = "  ← 推荐（延迟最低）" if "wasapi" in api.lower() else ""
        print(f"\n  [{api}]{tag}   共 {len(items)} 条")
        for d in items:
            print(f"    {d}")


def cmd_list(engine: ANCEngine) -> int:
    _grouped_list("输入设备（麦克风）", engine.list_input_devices())
    _grouped_list("输出设备（耳机）", engine.list_output_devices())
    print("\n说明：")
    print("  · 同一物理设备会在 MME / DirectSound / WASAPI / WDM-KS 等多个驱动栈下列出多次；")
    print("  · 优先选 WASAPI 条目（延迟最低）；[默认] 是系统默认设备；")
    print("  · 输入选真正的\"麦克风\"，不要选\"立体声混音\"、\"MIDI\"或\"内部 AUX 插座\"；")
    print("  · 输出选你实际插的耳机/扬声器，别选 HDMI/显示器输出（除非确实要用）。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="软件主动降噪（ANC）：麦克风采噪 → 相位反转 → 耳机播放反相波",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="列出输入/输出设备后退出")
    p.add_argument("--input", type=int, default=None, help="输入(麦克风)设备索引")
    p.add_argument("--output", type=int, default=None, help="输出(耳机)设备索引")
    p.add_argument("--mode", choices=MODES, default="anc",
                   help="off=静音 / monitor=透传监听 / anc=降噪")
    p.add_argument("--gain", type=float, default=1.0, help="反相波增益（降噪强度）")
    p.add_argument("--lpf-cutoff", type=float, default=150.0,
                   help="反相波低通截止频率(Hz)，只保留可抵消的低频段；0 表示关闭")
    p.add_argument("--samplerate", type=int, default=48000, help="采样率")
    p.add_argument("--blocksize", type=int, default=128, help="块大小（越小延迟越低）")
    p.add_argument("--latency", choices=("low", "high"), default="low", help="延迟档")
    p.add_argument("--music", type=str, default=None, help="边降噪边循环播放的 WAV 文件")
    p.add_argument("--music-gain", type=float, default=1.0, help="音乐音量")
    args = p.parse_args(argv)

    engine = ANCEngine()
    if args.list:
        return cmd_list(engine)

    engine.input_device = args.input
    engine.output_device = args.output
    engine.mode = args.mode
    engine.gain = args.gain
    engine.samplerate = args.samplerate
    engine.blocksize = args.blocksize
    engine.latency = args.latency
    engine.playback_gain = args.music_gain
    if args.lpf_cutoff <= 0:
        engine.lpf_enabled = False
    else:
        engine.lpf_enabled = True
        engine.lpf_cutoff = args.lpf_cutoff

    if args.music:
        engine.load_playback_file(args.music, loop=True)
        print(f"已加载音乐：{args.music}")

    try:
        engine.start()
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print("提示：可先用 --list 查看设备索引；确认设备未被占用。", file=sys.stderr)
        return 1

    print("降噪已启动，按 Ctrl+C 退出。")
    print(f"  模式={engine.mode}  增益={engine.gain:.2f}  "
          f"低通={'关' if not engine.lpf_enabled else f'{int(engine.lpf_cutoff)}Hz'}  "
          f"采样率={engine.samplerate}  块={engine.blocksize}  延迟档={engine.latency}")

    def on_int(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_int)

    try:
        while True:
            time.sleep(1.0)
            print(f"\r  {engine.status_summary()}      ", end="", flush=True)
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
