#!/usr/bin/env python3
"""
CoMuRoS Chat GUI — with integrated camera viewer
Subscribes : /chat_output          (planner feedback → chat bubbles)
             /robot_status         (JSON robot states → status panel)
             Image topics          (live camera feeds → camera panel)
Publishes  : /chat_input           (human text → planner)
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import customtkinter as ctk
from PIL import Image as PILImage
import numpy as np
from threading import Thread
from datetime import datetime


SENDER_COLORS = {
    "human":   "#3F88C5",
    "planner": "#20B2AA",
    "system":  "#888888",
    "sr1":     "#008080",
    "sr2":     "#9932a8",
    "sr3":     "#4E5734",
    "sr4":     "#6B6DE6",
}

STATUS_IDLE  = "#2ecc71"
STATUS_BUSY  = "#e67e22"
STATUS_ERROR = "#e74c3c"

BG_MAIN    = "#f0f0f0"
BG_PANEL   = "#3F88C5"
BG_HEADER  = "#3F88C5"
BG_ENTRY   = "#3F88C5"
ACCENT     = "#e94560"
TEXT_LIGHT = "#0d1b2a"
TEXT_DIM   = "#0d1b2a"


CAM_W = 300
CAM_H = 220




class ChatGUI(Node):

    def __init__(self):
        super().__init__("human_gui")

        self._bridge = CvBridge()

        self._pub_input = self.create_publisher(String, "/chat_input", 10)
        self.create_subscription(String, "/chat_output",  self._on_output,       10)
        self.create_subscription(String, "/robot_status", self._on_robot_status, 10)

       
        self._known_robots: list = []   
        self._cam_slots: dict = {}
        self._available_topics: list = ["— select topic —"]
        self._selected_topics:  dict = {}

       
        self.create_timer(1.0, self._scan_image_topics)

       
        self._robot_labels: dict = {}
        self._robot_frames: dict = {}

       
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("CoMuRoS — Command Interface")
        self.window.geometry("1300x720")
        self.window.configure(fg_color=BG_MAIN)
        self.window.resizable(True, True)

        self._build_layout()
        self.get_logger().info("[ChatGUI] Initialised.")

    
    
    def _build_layout(self):
        self.window.columnconfigure(0, weight=1, uniform="group1")
        self.window.columnconfigure(1, weight=1, uniform="group1")
        self.window.rowconfigure(0, weight=0)
        self.window.rowconfigure(1, weight=1)
        self.window.rowconfigure(2, weight=0)
        

        self._build_header()
        self._build_chat_area()
        self._build_entry_bar()
        
        self._build_camera_panel()
       
     

    def _refresh_save_robot_dropdown(self):
        values = self._known_robots if self._known_robots else ["— robot —"]
        self._save_robot_dropdown.configure(values=values)
        if self._save_robot_var.get() not in values:
            self._save_robot_var.set(values[0])

    def _build_header(self):
        hdr = ctk.CTkFrame(self.window, fg_color=BG_HEADER,
                           corner_radius=0, height=10)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        hdr.grid_propagate(False)

        ctk.CTkLabel(hdr, text=" CoMuRoS",
                     font=("Courier New", 22, "bold"),
                     text_color=ACCENT
                     ).pack(side="left", padx=20, pady=10)

        ctk.CTkLabel(hdr, text="Multi-Robot Command Interface",
                     font=("Courier New", 13),
                     text_color=TEXT_DIM
                     ).pack(side="left", padx=0, pady=10)

        self._conn_label = ctk.CTkLabel(hdr, text="● LIVE",
                                        font=("Courier New", 12, "bold"),
                                        text_color=STATUS_IDLE)
        self._conn_label.pack(side="right", padx=20)

    def _build_chat_area(self):
        self._chat_scroll = ctk.CTkScrollableFrame(
            self.window, fg_color=BG_MAIN, corner_radius=0)
        self._chat_scroll.grid(row=1, column=0, sticky="nsew",
                               padx=(12, 4), pady=(8, 0))
        self._chat_scroll.columnconfigure(0, weight=1)
        self._append_system("System ready. Type a command below.")

    def _build_entry_bar(self):
        bar = ctk.CTkFrame(
            self.window,
            fg_color=BG_PANEL,
            corner_radius=12
        )
        bar.grid(row=2, column=0, sticky="ew",
                padx=(12,4), pady=(6,12))

        bar.columnconfigure(0, weight=1)

    

        self._entry = ctk.CTkEntry(
            bar,
            placeholder_text="Enter robot command...",
            height=36
        )
        self._entry.grid(row=0, column=0, padx=10, pady=8, sticky="ew")
        self._entry.bind("<Return>", self._send)

        send_btn = ctk.CTkButton(
            bar,
            text="SEND ▶",
            command=self._send,
            width=100
        )
        send_btn.grid(row=0, column=1, padx=10, pady=8)

        self._save_robot_var = ctk.StringVar(value="— robot —")
        self._save_robot_dropdown = ctk.CTkOptionMenu(
    bar,
    variable=self._save_robot_var,
    values=["— robot —"],
    width=90,
)
        self._save_robot_dropdown.grid(row=1, column=2, padx=(0, 10), pady=(0, 8))

        self._location_entry = ctk.CTkEntry(
            bar,
            placeholder_text="Location name..."
        )

        self._location_entry = ctk.CTkEntry(
            bar,
            placeholder_text="Location name..."
        )
        self._location_entry.grid(row=1, column=0,
                                padx=10, pady=(0,8),
                                sticky="ew")
        self._save_btn = ctk.CTkButton(
        bar,
        text="SAVE LOCATION",
        command=self._save_location,
        width=140,
        height=32,
        fg_color="#2a6090",
        hover_color="#1e4d7a",
    )
        self._save_btn.grid(row=1, column=1, padx=10, pady=(0,8))

    # def _save_location(self):
    #     name = self._location_entry.get().strip()

    #     if not name:
    #         self._append_system("Please enter a location name.")
    #         return

    #     msg = String()
    #     msg.data = f"human|save location {name}"
    #     self._pub_input.publish(msg)

    #     self._append_system(f"Saving location '{name}'")
    #     self._location_entry.delete(0, "end")   
    def _save_location(self):
        name = self._location_entry.get().strip()
        robot = self._save_robot_var.get().strip()

        if not name:
            self._append_system("Please enter a location name.")
            return

        if not robot:
            self._append_system("Please select a robot.")
            return

        msg = String()
        msg.data = f"human|save location {name}|{robot}"
        self._pub_input.publish(msg)

        self._append_system(f"Saving '{name}' using {robot.upper()}'s current pose...")
        self._location_entry.delete(0, "end")
    def _build_camera_panel(self):
        self._cam_panel = ctk.CTkFrame(self.window, fg_color=BG_PANEL,
                                       corner_radius=0)
        self._cam_panel.grid(row=1, column=1, rowspan=2, sticky="nsew",
                             padx=(4,12), pady=(8, 12))
        self._cam_panel.grid_propagate(False)
        self._cam_panel.columnconfigure(0, weight=1)

        ctk.CTkLabel(self._cam_panel, text="CAMERA FEEDS",
                     font=("Courier New", 12, "bold"),
                     text_color="#ffffff").pack(pady=(10, 4))

        ctk.CTkFrame(self._cam_panel, fg_color="#ffffff",
                     height=1).pack(fill="x", padx=8)

        
        self._cam_scroll = ctk.CTkScrollableFrame(
            self._cam_panel, fg_color=BG_PANEL, corner_radius=0)
        self._cam_scroll.pack(fill="both", expand=True, padx=4, pady=4)

       
        ctk.CTkButton(self._cam_panel, text="＋  Add feed",
                      command=self._add_cam_slot,
                      font=("Courier New", 11, "bold"),
                      fg_color="#2a6090", hover_color="#1e4d7a",
                      height=28, corner_radius=6
                      ).pack(pady=(4, 8), padx=12, fill="x")

        
        self._add_cam_slot()
        self._add_cam_slot()

    def _add_cam_slot(self):
        slot_name = f"cam{len(self._cam_slots)}"

        card = ctk.CTkFrame(self._cam_scroll, fg_color="#dce8f5",
                            corner_radius=8)
        card.pack(fill="x", pady=4, padx=2)
        card.columnconfigure(0, weight=1)

      
        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=6, pady=(6, 2))

        ctk.CTkLabel(top, text=slot_name.upper(),
                     font=("Courier New", 10, "bold"),
                     text_color=TEXT_DIM).pack(side="left")

        ctk.CTkButton(top, text="✕", width=24, height=20,
                      fg_color="#c0392b", hover_color="#922b21",
                      font=("Courier New", 10), text_color="#ffffff",
                      command=lambda s=slot_name, c=card: self._remove_cam_slot(s, c)
                      ).pack(side="right")

        
        topic_var = ctk.StringVar(value="— select topic —")
        self._selected_topics[slot_name] = topic_var

        dropdown = ctk.CTkOptionMenu(
            card,
            variable=topic_var,
            values=self._available_topics,
            font=("Courier New", 10),
            fg_color="#2a6090",
            button_color="#1e4d7a",
            button_hover_color="#163d63",
            dropdown_fg_color="#dce8f5",
            text_color="#ffffff",
            dropdown_text_color=TEXT_DIM,
            command=lambda t, s=slot_name: self._on_topic_selected(s, t),
            dynamic_resizing=False,
            width=300,
        )
        dropdown.pack(fill="x",padx=6, pady=(2, 4))

        # image display
        img_label = ctk.CTkLabel(card, text="No feed",
                                 font=("Courier New", 11),
                                 text_color=TEXT_DIM,
                                 fg_color="#b0c8e0",
                                  height=CAM_H,
                                 corner_radius=4)
        img_label.pack(fill="x",padx=6, pady=(0, 2))

        # fps
        fps_label = ctk.CTkLabel(card, text="",
                                 font=("Courier New", 9),
                                 text_color=TEXT_DIM)
        fps_label.pack(pady=(0, 4))

        self._cam_slots[slot_name] = {
            "card":      card,
            "dropdown":  dropdown,
            "img_label": img_label,
            "fps_label": fps_label,
            "topic_var": topic_var,
            "sub":       None,
            "fps_ts":    [],
        }

    def _remove_cam_slot(self, slot_name: str, card):
        slot = self._cam_slots.pop(slot_name, None)
        if slot and slot["sub"]:
            self.destroy_subscription(slot["sub"])
        self._selected_topics.pop(slot_name, None)
        card.destroy()

    def _on_topic_selected(self, slot_name: str, topic: str):
        if topic == "— select topic —":
            return
        slot = self._cam_slots.get(slot_name)
        if slot is None:
            return

        # unsub old
        if slot["sub"]:
            self.destroy_subscription(slot["sub"])
            slot["sub"] = None

        slot["fps_ts"] = []
        slot["img_label"].configure(text=f"Waiting …", image=None)

        # sub new
        sub = self.create_subscription(
            Image, topic,
            lambda msg, s=slot_name: self._on_image(msg, s),
            2)
        slot["sub"] = sub
        self.get_logger().info(f"[CAM] {slot_name} → {topic}")

    def _on_image(self, msg: Image, slot_name: str):
        slot = self._cam_slots.get(slot_name)
        if slot is None:
            return

        try:
            import cv2
            enc = msg.encoding.lower()

            if "16uc" in enc or "32fc" in enc:
                # depth image → normalise to greyscale
                cv_img = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="passthrough").astype(float)
                valid = cv_img[cv_img > 0]
                if valid.size:
                    mn, mx = valid.min(), valid.max()
                    cv_img = np.clip(
                        (cv_img - mn) / (mx - mn + 1e-6) * 255, 0, 255
                    ).astype(np.uint8)
                else:
                    cv_img = np.zeros(
                        (msg.height, msg.width), dtype=np.uint8)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

            elif "mono" in enc or "8uc1" in enc:
                cv_img = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="mono8")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

            else:
                cv_img = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="rgb8")

        except CvBridgeError as e:
            self.get_logger().warn(f"[CAM] cv_bridge {slot_name}: {e}")
            return

        pil_img = PILImage.fromarray(cv_img).resize(
            (CAM_W, CAM_H), PILImage.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil_img,
                               dark_image=pil_img,
                               size=(CAM_W, CAM_H))

        # FPS
        now = datetime.now().timestamp()
        slot["fps_ts"] = [t for t in slot["fps_ts"] if now - t < 2.0]
        slot["fps_ts"].append(now)
        fps = len(slot["fps_ts"]) / 2.0

        def update(img=ctk_img, f=fps):
            s = self._cam_slots.get(slot_name)
            if s is None:
                return
            s["img_label"].configure(image=img, text="")
            s["fps_label"].configure(text=f"{f:.1f} fps")

        self.window.after(0, update)

   

    def _scan_image_topics(self):
        try:
            topic_list = self.get_topic_names_and_types()
        except Exception:
            return

        image_topics = sorted(
            t for t, types in topic_list
            if any("sensor_msgs/msg/Image" in ty for ty in types)
        )
        new_list = ["— select topic —"] + image_topics
        if new_list == self._available_topics:
            return

        self._available_topics = new_list

        def refresh():
            for slot in self._cam_slots.values():
                current = slot["topic_var"].get()
                slot["dropdown"].configure(values=new_list)
                if current not in new_list:
                    slot["topic_var"].set("— select topic —")

        self.window.after(0, refresh)
        self.get_logger().info(f"[CAM] Topics found: {image_topics}")



    def _build_status_panel(self):
        panel = ctk.CTkFrame(self.window, fg_color=BG_PANEL,
                             corner_radius=0, width=210)
        panel.grid(row=1, column=2, rowspan=2, sticky="nsew",
                   padx=(0, 12), pady=(8, 12))
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text="FLEET STATUS",
                     font=("Courier New", 12, "bold"),
                     text_color="#ffffff").pack(pady=(12, 6))

        ctk.CTkFrame(panel, fg_color="#ffffff",
                     height=1).pack(fill="x", padx=8)

       

        legend_frame = ctk.CTkFrame(panel, fg_color="#dce8f5", corner_radius=8)
        legend_frame.pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkLabel(
            legend_frame,
            text=("COMMANDS\n"
                  "─────────────────\n"
                  "sr1 go to room1\n"
                  "go to room1\n"
                  "go to box object\n"
                  "sr1 go to box object\n"
                  "sr1 move left\n"
                  "stop sr1"),
            font=("Courier New", 10),
            text_color=TEXT_DIM,
            justify="left",
        ).pack(padx=8, pady=6)

  

    def _on_output(self, msg: String):
        line = msg.data
        if "]" in line:
            line = line.split("] ", 1)[-1]
        parts  = line.split("|", 1)
        sender = parts[0].strip() if len(parts) == 2 else "system"
        text   = parts[1].strip() if len(parts) == 2 else line
        self.window.after(0, lambda s=sender, t=text: self._append_bubble(s, t))

    def _on_robot_status(self, msg: String):
        try:
            data: dict = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        robots = sorted(data.keys())
        if robots != self._known_robots:
            self._known_robots = robots
            self.window.after(0, self._refresh_save_robot_dropdown)

    def _append_bubble(self, sender: str, text: str):
        color    = SENDER_COLORS.get(sender.lower(), "#555577")
        is_human = sender.lower() == "human"
        ts       = datetime.now().strftime("%H:%M:%S")

        row = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        row.pack(fill="x", pady=3, padx=6)

        ctk.CTkLabel(row, text=ts, font=("Courier New", 9),
                     text_color=TEXT_DIM,
                     anchor="e" if is_human else "w"
                     ).pack(anchor="e" if is_human else "w")

        bubble = ctk.CTkFrame(row, fg_color=color, corner_radius=10)
        bubble.pack(anchor="e" if is_human else "w",
                    padx=(60 if is_human else 0, 0 if is_human else 60))

        ctk.CTkLabel(bubble,
                     text=f"  {sender.upper()}: {text}  ",
                     font=("Courier New", 12),
                     text_color="#ffffff",
                     wraplength=420,
                     justify="left", anchor="w"
                     ).pack(padx=4, pady=4)

        self.window.after(50,
            lambda: self._chat_scroll._parent_canvas.yview_moveto(1.0))

    def _append_system(self, text: str):
        ts  = datetime.now().strftime("%H:%M:%S")
        row = ctk.CTkFrame(self._chat_scroll, fg_color="transparent")
        row.pack(fill="x", pady=2, padx=6)
        ctk.CTkLabel(row, text=f"[{ts}]  ⚙  {text}",
                     font=("Courier New", 11),
                     text_color=TEXT_DIM,
                     anchor="center").pack(anchor="center")
 
    def _send(self, event=None):
        text = self._entry.get().strip()
        if not text:
            return
        self._append_bubble("human", text)
        out      = String()
        out.data = f"human|{text}"
        self._pub_input.publish(out)
        self.get_logger().info(f"[ChatGUI] → {text}")
        self._entry.delete(0, "end")

  
    def run_gui(self):
        self.get_logger().info("[ChatGUI] Starting mainloop.")
        self.window.mainloop()



def main():
    rclpy.init()
    node = ChatGUI()

    def spin_bg():
        try:
            rclpy.spin(node)
        except (rclpy.executors.ExternalShutdownException, KeyboardInterrupt):
            pass

    Thread(target=spin_bg, daemon=True).start()

    def on_closing():
        node.get_logger().info("[ChatGUI] Shutting down …")
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        node.window.destroy()

    node.window.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        node.run_gui()
    except KeyboardInterrupt:
        on_closing()


if __name__ == "__main__":
    main()