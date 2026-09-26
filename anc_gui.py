"""anc_gui.py — 主动降噪耳机小软件的图形界面（tkinter）。

用法：
    python anc_gui.py

依赖：sounddevice, numpy（见 requirements.txt）
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from anc_core import (
    MODE_ANC,
    MODE_MONITOR,
    MODE_OFF,
    ANCEngine,
    DeviceInfo,
)


class ANCApp:
    POLL_MS = 100  # 界面电平刷新间隔

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine = ANCEngine()

        self._in_devices: list[DeviceInfo] = []
        self._out_devices: list[DeviceInfo] = []
        self._in_map: dict[str, int] = {}
        self._out_map: dict[str, int] = {}
        self._playback_path: str | None = None

        self._build_ui()
        self._update_mode_label()
        self._refresh_devices()
        self._sync_running_state()
        self.root.after(self.POLL_MS, self._poll)

    # ------------------------------------------------------------------ #
    # 界面搭建
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.root.title("降噪耳机 · 软件主动降噪 (ANC)")
        self.root.geometry("640x640")
        self.root.minsize(560, 600)

        pad = {"padx": 10, "pady": 4}
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        # ---- 提示 ----
        tip = (
            "⚠ 麦克风必须贴近耳朵（塞进耳罩 / 耳塞旁、朝外）才能有效降噪。\n"
            "  用笔记本/桌面麦克风基本无效。先选 Monitor 模式确认麦克风采到了噪声。"
        )
        ttk.Label(outer, text=tip, foreground="#8a5a00",
                  wraplength=600, justify="left").pack(fill="x", **pad)

        # ---- 设备区 ----
        dev = ttk.LabelFrame(outer, text="1. 选择音频设备", padding=8)
        dev.pack(fill="x", **pad)

        row = ttk.Frame(dev)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="麦克风(输入)").grid(row=0, column=0, sticky="w")
        self.in_var = tk.StringVar()
        self.in_combo = ttk.Combobox(row, textvariable=self.in_var, state="readonly", width=48)
        self.in_combo.grid(row=0, column=1, sticky="we", padx=6)
        row.columnconfigure(1, weight=1)

        row2 = ttk.Frame(dev)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="耳机(输出)").grid(row=0, column=0, sticky="w")
        self.out_var = tk.StringVar()
        self.out_combo = ttk.Combobox(row2, textvariable=self.out_var, state="readonly", width=48)
        self.out_combo.grid(row=0, column=1, sticky="we", padx=6)
        row2.columnconfigure(1, weight=1)

        btns = ttk.Frame(dev)
        btns.pack(fill="x", pady=(4, 0))
        self.refresh_btn = ttk.Button(btns, text="刷新设备", command=self._refresh_devices)
        self.refresh_btn.pack(side="left")

        ttk.Label(dev, text="提示：同一物理设备会在多个驱动栈下列出多次，优先选带 WASAPI 的条目（延迟最低）。",
                  foreground="#666", wraplength=560).pack(fill="x", pady=(6, 0))

        # ---- 流参数 ----
        par = ttk.LabelFrame(outer, text="2. 流参数（改后需重新点击启动）", padding=8)
        par.pack(fill="x", **pad)

        self.sr_var = tk.StringVar(value="48000")
        self.bs_var = tk.StringVar(value="128")
        self.lat_var = tk.StringVar(value="low")

        row3 = ttk.Frame(par)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="采样率").grid(row=0, column=0, sticky="w")
        self.sr_combo = ttk.Combobox(row3, textvariable=self.sr_var, state="readonly",
                                     values=("44100", "48000", "96000"), width=10)
        self.sr_combo.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(row3, text="块大小(越小延迟越低)").grid(row=0, column=2, sticky="w", padx=(16, 0))
        self.bs_combo = ttk.Combobox(row3, textvariable=self.bs_var, state="readonly",
                                     values=("64", "128", "256", "512", "1024"), width=8)
        self.bs_combo.grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(row3, text="延迟档").grid(row=0, column=4, sticky="w", padx=(16, 0))
        self.lat_combo = ttk.Combobox(row3, textvariable=self.lat_var, state="readonly",
                                      values=("low", "high"), width=8)
        self.lat_combo.grid(row=0, column=5, sticky="w", padx=6)

        # ---- 降噪控制 ----
        anc = ttk.LabelFrame(outer, text="3. 降噪控制（可实时调整）", padding=8)
        anc.pack(fill="x", **pad)

        self.mode_var = tk.StringVar(value=MODE_OFF)
        mode_row = ttk.Frame(anc)
        mode_row.pack(fill="x", pady=2)
        ttk.Label(mode_row, text="模式").pack(side="left")
        for value, label in ((MODE_OFF, "关闭"), (MODE_MONITOR, "监听"), (MODE_ANC, "降噪 ANC")):
            ttk.Radiobutton(mode_row, text=label, value=value, variable=self.mode_var,
                            command=self._on_mode_change).pack(side="left", padx=8)
        self.mode_state_lbl = ttk.Label(mode_row, text="", font=("", 11, "bold"))
        self.mode_state_lbl.pack(side="right")
        self.mode_var.trace_add("write", lambda *_: setattr(self.engine, "mode", self.mode_var.get()))
        self.mode_var.trace_add("write", lambda *_: self._update_mode_label())

        self.gain_var = tk.DoubleVar(value=1.0)
        gain_row = ttk.Frame(anc)
        gain_row.pack(fill="x", pady=2)
        ttk.Label(gain_row, text="降噪强度").pack(side="left")
        gain_scale = ttk.Scale(gain_row, from_=0.0, to=2.0, variable=self.gain_var,
                               command=self._on_gain_change)
        gain_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.gain_lbl = ttk.Label(gain_row, text="1.00")
        self.gain_lbl.pack(side="left")

        # 反相波低通：只保留可抵消的低频段，避免中高频被当成额外噪声放大
        self.lpf_var = tk.BooleanVar(value=True)
        self.lpf_cut_var = tk.DoubleVar(value=150.0)
        lpf_row = ttk.Frame(anc)
        lpf_row.pack(fill="x", pady=2)
        ttk.Checkbutton(lpf_row, text="低通(只抵消低频)", variable=self.lpf_var,
                        command=self._on_lpf_toggle).pack(side="left")
        ttk.Scale(lpf_row, from_=30.0, to=800.0, variable=self.lpf_cut_var,
                  command=self._on_lpf_change).pack(side="left", fill="x", expand=True, padx=8)
        self.lpf_lbl = ttk.Label(lpf_row, text="150Hz")
        self.lpf_lbl.pack(side="left")

        # ---- 音乐混合 ----
        mus = ttk.LabelFrame(outer, text="4. 边降噪边听音乐（可选，WAV）", padding=8)
        mus.pack(fill="x", **pad)
        mrow = ttk.Frame(mus)
        mrow.pack(fill="x", pady=2)
        self.music_btn = ttk.Button(mrow, text="选择 WAV…", command=self._choose_music)
        self.music_btn.pack(side="left")
        self.music_stop_btn = ttk.Button(mrow, text="停止音乐", command=self._stop_music,
                                         state="disabled")
        self.music_stop_btn.pack(side="left", padx=6)
        self.music_lbl = ttk.Label(mrow, text="(未加载)", foreground="#666")
        self.music_lbl.pack(side="left", padx=6)

        self.pgain_var = tk.DoubleVar(value=1.0)
        prow = ttk.Frame(mus)
        prow.pack(fill="x", pady=2)
        ttk.Label(prow, text="音乐音量").pack(side="left")
        ttk.Scale(prow, from_=0.0, to=1.5, variable=self.pgain_var,
                  command=self._on_pgain_change).pack(side="left", fill="x", expand=True, padx=8)

        # ---- 启动/停止 ----
        act = ttk.Frame(outer)
        act.pack(fill="x", pady=(8, 4))
        self.start_btn = ttk.Button(act, text="▶ 启动降噪", command=self._on_start_stop)
        self.start_btn.pack(side="left", ipadx=12, ipady=6)

        # ---- 电平表 ----
        lev = ttk.LabelFrame(outer, text="实时电平", padding=8)
        lev.pack(fill="x", **pad)
        ttk.Label(lev, text="输入").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.in_bar = ttk.Progressbar(lev, maximum=100, length=200)
        self.in_bar.grid(row=0, column=1, sticky="we", padx=6)
        ttk.Label(lev, text="输出").grid(row=0, column=2, sticky="w", padx=(12, 6))
        self.out_bar = ttk.Progressbar(lev, maximum=100, length=200)
        self.out_bar.grid(row=0, column=3, sticky="we", padx=6)
        lev.columnconfigure(1, weight=1)
        lev.columnconfigure(3, weight=1)

        # ---- 状态栏 ----
        self.status_var = tk.StringVar(value="已停止")
        ttk.Label(outer, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=6).pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------------ #
    # 设备 / 参数
    # ------------------------------------------------------------------ #
    def _refresh_devices(self) -> None:
        try:
            self._in_devices = self.engine.list_input_devices()
            self._out_devices = self.engine.list_output_devices()
        except Exception as exc:
            messagebox.showerror("获取设备失败", str(exc))
            return
        # 排序：WASAPI 优先、默认设备优先，便于一眼选中推荐条目
        def pref(d: DeviceInfo):
            return (0 if "wasapi" in d.hostapi.lower() else 1,
                    0 if d.is_default else 1,
                    d.index)

        self._in_devices.sort(key=pref)
        self._out_devices.sort(key=pref)
        self._in_map = {str(d): d.index for d in self._in_devices}
        self._out_map = {str(d): d.index for d in self._out_devices}
        self.in_combo["values"] = list(self._in_map.keys())
        self.out_combo["values"] = list(self._out_map.keys())
        # 默认选中系统默认设备
        default_in = next((str(d) for d in self._in_devices if d.is_default), None)
        default_out = next((str(d) for d in self._out_devices if d.is_default), None)
        if default_in:
            self.in_var.set(default_in)
        if default_out:
            self.out_var.set(default_out)

    def _apply_params(self) -> None:
        """把界面上的流参数写进引擎（在 start 前调用）。"""
        e = self.engine
        e.samplerate = int(self.sr_var.get())
        e.blocksize = int(self.bs_var.get())
        e.latency = self.lat_var.get()
        in_sel = self._in_map.get(self.in_var.get())
        out_sel = self._out_map.get(self.out_var.get())
        e.input_device = in_sel if in_sel is not None else None
        e.output_device = out_sel if out_sel is not None else None
        # 依据所选设备能力确定声道数
        if in_sel is not None:
            for d in self._in_devices:
                if d.index == in_sel:
                    e.input_channels = 1  # ANC 只用一路参考，统一取 1
                    break
        if out_sel is not None:
            for d in self._out_devices:
                if d.index == out_sel:
                    e.output_channels = min(2, d.max_output_channels)
                    break
        if e.output_channels < 1:
            e.output_channels = 2

    # ------------------------------------------------------------------ #
    # 事件
    # ------------------------------------------------------------------ #
    def _on_start_stop(self) -> None:
        if self.engine.running:
            self.engine.stop()
        else:
            try:
                self._apply_params()
                self.engine.start()
            except Exception as exc:
                messagebox.showerror("启动失败", f"无法启动音频流：\n{exc}\n\n"
                                     "可能原因：设备被占用、采样率/声道不支持。")
                self.engine.stop()
        self._sync_running_state()

    def _on_mode_change(self) -> None:
        self.engine.mode = self.mode_var.get()

    def _update_mode_label(self) -> None:
        m = self.mode_var.get()
        text, color = {
            MODE_OFF: ("当前：关闭（静音）", "#666666"),
            MODE_MONITOR: ("当前：监听（麦克风直通=放大器）", "#b06000"),
            MODE_ANC: ("当前：降噪 ANC（输出反相波）", "#0a7d1f"),
        }.get(m, (m, "#000000"))
        self.mode_state_lbl.config(text=text, foreground=color)

    def _on_lpf_toggle(self) -> None:
        self.engine.lpf_enabled = bool(self.lpf_var.get())

    def _on_lpf_change(self, _val) -> None:
        try:
            self.engine.lpf_cutoff = float(self.lpf_cut_var.get())
        except (ValueError, tk.TclError):
            pass
        self.lpf_lbl.config(text=f"{int(self.engine.lpf_cutoff)}Hz")

    def _on_gain_change(self, _val) -> None:
        try:
            self.engine.gain = float(self.gain_var.get())
        except (ValueError, tk.TclError):
            pass
        self.gain_lbl.config(text=f"{self.engine.gain:.2f}")

    def _on_pgain_change(self, _val) -> None:
        try:
            self.engine.playback_gain = float(self.pgain_var.get())
        except (ValueError, tk.TclError):
            pass

    def _choose_music(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音乐文件",
            filetypes=[("WAV 音频", "*.wav"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            self.engine.load_playback_file(path, loop=True)
        except Exception as exc:
            messagebox.showerror("加载失败", f"无法读取音频文件：\n{exc}")
            return
        self._playback_path = path
        self.music_lbl.config(text=path.split("\\")[-1] if "\\" in path else path)
        self.music_stop_btn.config(state="normal")

    def _stop_music(self) -> None:
        self.engine.set_playback(None)
        self._playback_path = None
        self.music_lbl.config(text="(未加载)")
        self.music_stop_btn.config(state="disabled")

    def _sync_running_state(self) -> None:
        running = self.engine.running
        self.start_btn.config(text="■ 停止" if running else "▶ 启动降噪")
        # 运行时锁定流参数与设备（实时参数仍可调）
        for w in (self.refresh_btn, self.in_combo, self.out_combo,
                  self.sr_combo, self.bs_combo, self.lat_combo):
            w.config(state="disabled" if running else "readonly" if isinstance(w, ttk.Combobox) else "normal")
        self.refresh_btn.config(state="disabled" if running else "normal")

    # ------------------------------------------------------------------ #
    # 轮询刷新
    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        e = self.engine
        self.in_bar["value"] = min(100.0, e.in_level * 100.0)
        self.out_bar["value"] = min(100.0, e.out_level * 100.0)

        parts = []
        if e.running:
            parts.append("运行中")
            lat = e.latency_info()
            if lat:
                parts.append(f"往返延迟 {lat['roundtrip_ms']} ms")
                parts.append(f"(输入 {lat['input']*1000:.1f} / 输出 {lat['output']*1000:.1f} ms)")
        else:
            parts.append("已停止")
        if e.error:
            parts.append(f"错误: {e.error}")
        if e._xrun_count:
            parts.append(f"欠载/溢出 {e._xrun_count} 次")
        if e._clip_count:
            parts.append(f"削波 {e._clip_count} 次")
        self.status_var.set("  |  ".join(parts))
        self.root.after(self.POLL_MS, self._poll)

    def on_close(self) -> None:
        self.engine.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ANCApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
