import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import moderngl as mgl
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys
import subprocess
import shutil
import tempfile
import ctypes
import win32con
import win32gui
import win32api

kernel32 = ctypes.windll.kernel32
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class ShaderHandler(FileSystemEventHandler):
    def __init__(self, runner):
        self.runner = runner

    def on_modified(self, event):
        if event.src_path.endswith((".frag", ".glsl")):
            self.runner.needs_reload = True


class ShadertoyRunner:
    def __init__(self, shader_path, width=640, height=640):
        self.width = width
        self.height = height
        self.shader_path = shader_path
        self.custom_uniforms = {}
        self.custom_uniforms["iAnimation"] = 0.0
        self.anim_reset_time = 0.0
        self.paused = False
        self.pause_accum = 0.0
        self.pause_start = None
        self.needs_reload = False
        self.start_time = time.perf_counter()
        self.frame = 0
        self.mouse_pos = (0, 0)
        self.mouse_buttons = (0, 0, 0)
        self.always_on_top = False
        self.draw_overlay = True
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        pygame.init()
        kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)
        self.screen = pygame.display.set_mode(
            (width, height), pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME
        )
        self.toggle_always_on_top()
        hwnd = self.get_window_handle()
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)

        class MARGINS(ctypes.Structure):
            _fields_ = [
                ("cxLeftWidth", ctypes.c_int),
                ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int),
                ("cyBottomHeight", ctypes.c_int),
            ]

        dwmapi = ctypes.windll.dwmapi
        margins = MARGINS(-1, -1, -1, -1)  # -1 = extend over entire client area
        dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))
        self.ctx = mgl.create_context()
        self.load_shader()
        self.create_quad()
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)

        self.font = pygame.font.Font(None, 24)
        self.smoothing_alpha = 0.97  # 0..1 (higher = more smoothing)
        self.smoothed_render_ms = None
        # Create timer query for precise GPU timing
        self.timer_query = self.ctx.query(time=True)
        overlay_vert = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        overlay_frag = """
        #version 330
        uniform sampler2D tex;
        in vec2 v_uv;
        out vec4 fColor;
        void main() {
            fColor = texture(tex, v_uv);
        }
        """
        self.overlay_prog = self.ctx.program(
            vertex_shader=overlay_vert, fragment_shader=overlay_frag
        )
        # dynamic vbo for (x,y,u,v) * 4
        self.overlay_vbo = self.ctx.buffer(b"\x00" * (4 * 4 * 4))
        self.overlay_vao = self.ctx.vertex_array(
            self.overlay_prog, [(self.overlay_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self.text_tex = None
        self.text_size = (0, 0)
        self.dragging = False
        self.drag_start_mouse = None
        self.drag_start_window = None
        self.observer = Observer()
        watch_path = os.path.dirname(shader_path) or "."
        self.observer.schedule(ShaderHandler(self), path=watch_path, recursive=False)
        self.observer.start()

    def load_shader(self):
        try:
            with open(self.shader_path, "r") as f:
                frag_src = f.read()
            vert_src = """
            #version 450
            layout(location = 0) in vec2 in_vert;
            layout(location = 0) out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_uv = in_vert * 0.5 + 0.5;
            }
            """
            wrapped = f"""
            #version 450
            layout(location = 0) uniform vec3 iResolution;
            layout(location = 1) uniform float iTime;
            layout(location = 2) uniform vec4 iMouse;
            layout(location = 3) uniform int iFrame;
            """
            for name, value in self.custom_uniforms.items():
                if isinstance(value, float):
                    wrapped += f"uniform float {name};\n"
                elif isinstance(value, int):
                    wrapped += f"uniform int {name};\n"
                elif isinstance(value, (tuple, list)):
                    if len(value) == 2:
                        wrapped += f"uniform vec2 {name};\n"
                    elif len(value) == 3:
                        wrapped += f"uniform vec3 {name};\n"
                    elif len(value) == 4:
                        wrapped += f"uniform vec4 {name};\n"
            wrapped += f"""
            layout(location = 0) out vec4 fragColor;
            void mainImage(out vec4, in vec2);
            void main() {{
                mainImage(fragColor, gl_FragCoord.xy);
                fragColor.rgb *= fragColor.a;
            }}
            #line 0
            {frag_src}
            """
            self.program = self.ctx.program(
                vertex_shader=vert_src, fragment_shader=wrapped
            )
            # try:
            #     self._emit_spirv_debug(wrapped)
            # except Exception:
            #     pass
        except Exception as e:
            print(f"Error loading shader: {e}")
            if hasattr(self, "program"):
                pass
            else:
                sys.exit(1)

    def _emit_spirv_debug(self, wrapped_src):
        base_dir = os.path.dirname(self.shader_path) or "."
        disasm_path = os.path.join(base_dir, "shader.spv.txt")

        tmp_src = None
        compiled_spv = None
        opt_spv = None
        try:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".frag", mode="w", encoding="utf-8"
            )
            tmp.write(wrapped_src)
            tmp.close()
            tmp_src = tmp.name

            compiler = shutil.which("glslangValidator") or shutil.which("glslc")
            if not compiler:
                return

            compiled_spv = tmp_src + ".spv"
            cmd = [compiler, "-G", tmp_src, "-o", compiled_spv]

            p = subprocess.run(cmd, capture_output=True, text=True)

            if p.returncode != 0 or not os.path.exists(compiled_spv):
                with open(disasm_path, "w", encoding="utf-8") as f:
                    f.write("// SPIR-V compilation failed\n")
                    f.write((p.stdout or "") + "\n" + (p.stderr or ""))
                return

            spirv_opt = shutil.which("spirv-opt")
            chosen_spv = compiled_spv
            if spirv_opt:
                opt_spv = tmp_src + ".opt.spv"
                p2 = subprocess.run(
                    [spirv_opt, "-O", compiled_spv, "-o", opt_spv],
                    capture_output=True,
                    text=True,
                )
                if p2.returncode == 0 and os.path.exists(opt_spv):
                    chosen_spv = opt_spv

            disasm = shutil.which("spirv-dis")
            if disasm:
                subprocess.run(
                    [disasm, chosen_spv, "-o", disasm_path, "--comment"],
                    capture_output=True,
                    text=True,
                )

            spirv_cross = shutil.which("spirv-cross")
            if spirv_cross:
                try:
                    cross_out = os.path.join(base_dir, "shader.spv.glsl")
                    p4 = subprocess.run(
                        [spirv_cross, chosen_spv, "--version", "450"],
                        capture_output=True,
                        text=True,
                    )
                    if p4.returncode == 0 and p4.stdout:
                        with open(cross_out, "w", encoding="utf-8") as cf:
                            cf.write(p4.stdout)
                except Exception:
                    pass
        finally:
            for pth in (tmp_src, compiled_spv, opt_spv):
                try:
                    if pth and os.path.exists(pth):
                        os.remove(pth)
                except Exception:
                    pass

    def reload_shader(self):
        start_time = time.perf_counter()
        self.load_shader()
        self.create_quad()
        end_time = time.perf_counter()
        reload_time_ms = (end_time - start_time) * 1000
        print(f"Reloaded in {reload_time_ms:.3f} ms")

    def create_quad(self):
        vertices = np.array(
            [
                -1.0,
                -1.0,
                1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f", "in_vert")])

    def _set_uniform(self, name, value):
        try:
            self.program[name] = value
        except Exception:
            pass

    def _draw_text_overlay(self, text, margin=6):
        """Render antialiased text (with shadow) to a texture and draw it at bottom-left."""
        fg = self.font.render(text, True, (255, 255, 255))
        shadow = self.font.render(text, True, (0, 0, 0, 48))
        w, h = fg.get_size()
        surf = pygame.Surface((w + 5, h + 5), pygame.SRCALPHA)
        for x in range(5):
            for y in range(5):
                if x == 2 and y == 2 or x == 0 or y == 0:
                    continue
                surf.blit(shadow, (x, y))
        surf.blit(fg, (2, 2))
        data = pygame.image.tostring(surf, "RGBA", True)
        w2, h2 = surf.get_size()
        if self.text_tex is None or self.text_size != (w2, h2):
            if self.text_tex:
                self.text_tex.release()
            self.text_tex = self.ctx.texture((w2, h2), 4, data)
            self.text_tex.filter = (mgl.LINEAR, mgl.LINEAR)
            self.text_size = (w2, h2)
        else:
            self.text_tex.write(data)
        left = margin
        bottom_px = self.height - margin
        top_px = bottom_px - h2
        right = left + w2
        nl = left * 2.0 / self.width - 1.0
        nr = right * 2.0 / self.width - 1.0
        nt = 1.0 - top_px * 2.0 / self.height
        nb = 1.0 - bottom_px * 2.0 / self.height
        vertices = np.array(
            [
                nl,
                nb,
                0.0,
                0.0,
                nr,
                nb,
                1.0,
                0.0,
                nl,
                nt,
                0.0,
                1.0,
                nr,
                nt,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        self.overlay_vbo.write(vertices.tobytes())
        self.overlay_prog["tex"].value = 0
        self.text_tex.use(location=0)
        self.overlay_vao.render(mgl.TRIANGLE_STRIP)

    def get_window_handle(self):
        return pygame.display.get_wm_info()["window"]

    def is_mouse_over_window(self):
        px, py = win32gui.GetCursorPos()
        left, top, right, bottom = win32gui.GetWindowRect(self.get_window_handle())
        return left <= px < right and top <= py < bottom

    def move_window(self, x, y):
        hwnd = self.get_window_handle()
        win32gui.SetWindowPos(
            hwnd, 0, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER
        )

    def toggle_always_on_top(self):
        self.always_on_top = not self.always_on_top
        hwnd = self.get_window_handle()
        if self.always_on_top:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
        else:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )

    def run(self):
        running = True
        while running:
            if (
                win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000
                and self.is_mouse_over_window()
            ):
                if not self.dragging:
                    px, py = win32gui.GetCursorPos()
                    self.drag_start_mouse = (px, py)
                    left, top, right, bottom = win32gui.GetWindowRect(
                        self.get_window_handle()
                    )
                    self.drag_start_window = (left, top)
                    self.dragging = True
                else:
                    px, py = win32gui.GetCursorPos()
                    dx = px - self.drag_start_mouse[0]
                    dy = py - self.drag_start_mouse[1]
                    new_x = self.drag_start_window[0] + dx
                    new_y = self.drag_start_window[1] + dy
                    self.move_window(int(round(new_x)), int(round(new_y)))
            else:
                self.dragging = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.paused:
                            self.paused = True
                            self.pause_start = time.perf_counter()
                        else:
                            self.paused = False
                            self.pause_accum += time.perf_counter() - self.pause_start
                            self.pause_start = None
                    elif event.key == pygame.K_t:
                        self.toggle_always_on_top()
                    elif event.key == pygame.K_d:
                        self.draw_overlay = not self.draw_overlay
                    elif event.key == pygame.K_a:
                        if self.paused and self.pause_start is not None:
                            t_now = (
                                self.pause_start - self.start_time - self.pause_accum
                            )
                        else:
                            t_now = (
                                time.perf_counter() - self.start_time - self.pause_accum
                            )
                        self.anim_reset_time = t_now
                        self.custom_uniforms["iAnimation"] = 0.0
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_buttons = pygame.mouse.get_pressed()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.mouse_buttons = pygame.mouse.get_pressed()
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos
            if self.needs_reload:
                self.reload_shader()
                self.needs_reload = False

            if self.paused:
                if self.pause_start is not None:
                    current_time = self.pause_start - self.start_time - self.pause_accum
                else:
                    current_time = 0.0
            else:
                current_time = time.perf_counter() - self.start_time - self.pause_accum

            if not self.paused:
                self.frame += 1

            self._set_uniform("iResolution", (self.width, self.height, 1.0))
            self._set_uniform("iTime", current_time)
            self._set_uniform("iFrame", self.frame)
            anim_val = current_time - self.anim_reset_time
            if anim_val < 0.0:
                anim_val = 0.0
            self.custom_uniforms["iAnimation"] = float(anim_val)
            mx, my = self.mouse_pos
            buttons = sum(1 for b in self.mouse_buttons if b)
            self._set_uniform("iMouse", (mx, self.height - my, buttons, 0.0))
            for name, value in self.custom_uniforms.items():
                self._set_uniform(name, value)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            with self.timer_query:
                self.vao.render(mgl.TRIANGLE_STRIP)
            render_ns = self.timer_query.elapsed
            render_ms = render_ns / 1000000.0
            if self.smoothed_render_ms is None:
                self.smoothed_render_ms = render_ms
            else:
                a = self.smoothing_alpha
                self.smoothed_render_ms = (
                    a * self.smoothed_render_ms + (1.0 - a) * render_ms
                )
            if self.draw_overlay:
                try:
                    val = self.smoothed_render_ms
                    disp = f"{val:.3f} ms"
                    self._draw_text_overlay(disp)
                except Exception:
                    pass
            pygame.display.flip()
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        self.observer.stop()
        self.observer.join()
        pygame.quit()


if __name__ == "__main__":
    shader_path = "shader.glsl" if len(sys.argv) < 2 else sys.argv[1]
    runner = ShadertoyRunner(shader_path)
    # Add custom uniforms here, e.g.:
    # runner.custom_uniforms['iCustom'] = 1.0
    # runner.custom_uniforms['iVec2'] = (1.0, 2.0)
    runner.run()
