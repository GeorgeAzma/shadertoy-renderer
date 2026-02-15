import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from pygame.locals import *
import moderngl as mgl
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys
import ctypes
from ctypes import wintypes


# Windows API constants and functions for window manipulation
HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
WM_NCLBUTTONDOWN = 0x00A1
HTCAPTION = 2

user32 = ctypes.windll.user32
user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.UINT,
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND
user32.ReleaseCapture.argtypes = []
user32.ReleaseCapture.restype = wintypes.BOOL
user32.SendMessageW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    ctypes.c_long,
]
user32.SendMessageW.restype = ctypes.c_long
user32.GetAsyncKeyState.argtypes = [wintypes.INT]
user32.GetAsyncKeyState.restype = wintypes.SHORT

VK_LBUTTON = 0x01

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
        self.needs_reload = False
        self.start_time = time.perf_counter()
        self.frame = 0
        self.mouse_pos = (0, 0)
        self.mouse_buttons = (0, 0, 0)
        self.always_on_top = True
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        pygame.init()
        kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        self.screen = pygame.display.set_mode(
            (width, height), DOUBLEBUF | OPENGL | NOFRAME
        )
        self.toggle_always_on_top()
        self.ctx = mgl.create_context()
        self.load_shader()
        self.create_quad()
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        self.font = pygame.font.Font(None, 24)
        self.smoothing_alpha = 0.9  # 0..1 (higher = more smoothing)
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
            in vec2 in_vert;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_uv = in_vert * 0.5 + 0.5;
            }
            """
            wrapped = f"""
            #version 450
            uniform vec3 iResolution;
            uniform float iTime;
            uniform vec4 iMouse;
            uniform int iFrame;
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
            out vec4 fragColor;
            void mainImage(out vec4, in vec2);
            void main() {{
                mainImage(fragColor, gl_FragCoord.xy);
            }}
            {frag_src}
            """
            self.program = self.ctx.program(
                vertex_shader=vert_src, fragment_shader=wrapped
            )
        except Exception as e:
            print(f"Error loading shader: {e}")
            if hasattr(self, "program"):
                pass  # keep old
            else:
                sys.exit(1)

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
        shadow = self.font.render(text, True, (0, 0, 0))
        shadow.set_alpha(32)  # semi-transparent for soft shadow
        w, h = fg.get_size()
        surf = pygame.Surface((w + 5, h + 5), pygame.SRCALPHA)
        for x in range(5):
            for y in range(5):
                if x == 2 and y == 2:
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
        # compute NDC rect for bottom-left placement
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
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        rect = wintypes.RECT()
        user32.GetWindowRect(self.get_window_handle(), ctypes.byref(rect))
        return rect.left <= point.x < rect.right and rect.top <= point.y < rect.bottom

    def get_dpi_scale(self):
        MONITOR_DEFAULTTONEAREST = 2
        monitor = user32.MonitorFromWindow(
            self.get_window_handle(), MONITOR_DEFAULTTONEAREST
        )
        dpiX = ctypes.c_uint()
        dpiY = ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(
            monitor, 0, ctypes.byref(dpiX), ctypes.byref(dpiY)
        )
        return dpiX.value / 96.0

    def move_window(self, x, y):
        hwnd = self.get_window_handle()
        user32.SetWindowPos(hwnd, 0, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER)

    def toggle_always_on_top(self):
        self.always_on_top = not self.always_on_top
        hwnd = self.get_window_handle()
        if self.always_on_top:
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        else:
            user32.SetWindowPos(
                hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE
            )

    def run(self):
        running = True
        while running:
            # Handle dragging globally
            if user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000 and self.is_mouse_over_window():
                if not self.dragging:
                    point = wintypes.POINT()
                    user32.GetCursorPos(ctypes.byref(point))
                    self.drag_start_mouse = (point.x, point.y)
                    rect = wintypes.RECT()
                    user32.GetWindowRect(self.get_window_handle(), ctypes.byref(rect))
                    self.drag_start_window = (rect.left, rect.top)
                    self.dragging = True
                else:
                    point = wintypes.POINT()
                    user32.GetCursorPos(ctypes.byref(point))
                    dx = point.x - self.drag_start_mouse[0]
                    dy = point.y - self.drag_start_mouse[1]
                    new_x = self.drag_start_window[0] + dx
                    new_y = self.drag_start_window[1] + dy
                    self.move_window(int(round(new_x)), int(round(new_y)))
            else:
                self.dragging = False

            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_t:
                        self.toggle_always_on_top()
                elif event.type == MOUSEBUTTONDOWN:
                    self.mouse_buttons = pygame.mouse.get_pressed()
                elif event.type == MOUSEBUTTONUP:
                    self.mouse_buttons = pygame.mouse.get_pressed()
                elif event.type == MOUSEMOTION:
                    self.mouse_pos = event.pos
            if self.needs_reload:
                self.reload_shader()
                self.needs_reload = False
            current_time = time.perf_counter() - self.start_time
            self.frame += 1
            self._set_uniform("iResolution", (self.width, self.height, 1.0))
            self._set_uniform("iTime", current_time)
            self._set_uniform("iFrame", self.frame)
            mx, my = self.mouse_pos
            buttons = sum(1 for b in self.mouse_buttons if b)
            self._set_uniform("iMouse", (mx, self.height - my, buttons, 0.0))
            for name, value in self.custom_uniforms.items():
                self._set_uniform(name, value)
            self.ctx.clear()
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
            try:
                val = self.smoothed_render_ms
                disp = f"{val:.3f} ms" if val >= 0.01 else "<0.01 ms"
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
