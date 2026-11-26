import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.dates as mdates
import datetime

OUT = "doc_shots"

# ---------------------------------------------------------------
# 1. Architecture diagram
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 9)
ax.axis("off")

def box(x, y, w, h, text, color="#0f766e", textcolor="white", fontsize=10, weight="bold"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                           linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             color=textcolor, fontsize=fontsize, fontweight=weight, wrap=True)
    return (x, y, w, h)

def arrow(b1, b2, side1="bottom", side2="top", color="#374151"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {"top": (x1 + w1/2, y1 + h1), "bottom": (x1 + w1/2, y1),
           "left": (x1, y1 + h1/2), "right": (x1 + w1, y1 + h1/2)}
    pts2 = {"top": (x2 + w2/2, y2 + h2), "bottom": (x2 + w2/2, y2),
            "left": (x2, y2 + h2/2), "right": (x2 + w2/2 + w2/2, y2 + h2/2)}
    p1 = pts[side1]
    p2 = {"top": (x2 + w2/2, y2 + h2), "bottom": (x2 + w2/2, y2),
          "left": (x2, y2 + h2/2), "right": (x2 + w2, y2 + h2/2)}[side2]
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                           linewidth=1.4, color=color, connectionstyle="arc3,rad=0.0")
    ax.add_patch(arr)

ax.text(5.5, 8.6, "CanineCare AI - System Architecture", ha="center", fontsize=15, fontweight="bold", color="#111827")

client = box(2.9, 7.2, 5.2, 0.9, "Client Browser\nHTML5 + CSS3 + Bootstrap 5 + JS (fetch/AJAX) + Three.js", color="#1d4ed8", fontsize=9)

django = box(2.9, 5.9, 5.2, 0.9, "Django Web Server\nURLs -> Views -> Templates", color="#0f766e", fontsize=10)

apps_y = 4.55
apps = [
    ("accounts\n(Auth)", 0.3),
    ("dashboard\n(Home/Dataset)", 2.35),
    ("prediction\n(Detect Disease)", 4.4),
    ("chatbot\n(AI Assistant)", 6.45),
    ("clinics\n(Pet Clinics)", 8.5),
]
app_boxes = []
for label, x in apps:
    b = box(x, apps_y, 1.9, 0.95, label, color="#334155", fontsize=8)
    app_boxes.append(b)

sqlite = box(0.2, 3.0, 2.1, 0.85, "SQLite3\n(Users, Sessions)", color="#7c3aed", fontsize=8.5)
mlmodel = box(2.5, 3.0, 2.15, 0.85, "PyTorch Model\nService (.pt)", color="#b45309", fontsize=8.2)
kb = box(4.85, 3.0, 2.15, 0.85, "disease_info.json\nclass_names.json", color="#0369a1", fontsize=7.8)
media = box(7.2, 3.0, 2.15, 0.85, "Media / Static\n(uploads, plots)", color="#4d7c0f", fontsize=7.8)

gemini = box(3.65, 1.4, 2.4, 0.85, "Gemini API\n(Chat completion)", color="#db2777", fontsize=8.5)
geoapify = box(6.35, 1.4, 2.4, 0.85, "Geoapify API\n(Reverse geocoding)", color="#ea580c", fontsize=8.5)

arrow(client, django, "bottom", "top")
for b in app_boxes:
    arrow(django, b, "bottom", "top")

arrow(app_boxes[0], sqlite, "bottom", "top")
arrow(app_boxes[2], mlmodel, "bottom", "top")
arrow(app_boxes[2], kb, "bottom", "top")
arrow(app_boxes[1], media, "bottom", "top")
arrow(app_boxes[3], gemini, "bottom", "top")
arrow(app_boxes[4], geoapify, "bottom", "top")

plt.tight_layout()
plt.savefig(f"{OUT}/architecture_diagram.png", dpi=160, bbox_inches="tight")
plt.close()
print("saved architecture_diagram.png")

# ---------------------------------------------------------------
# 2. Gantt chart
# ---------------------------------------------------------------
tasks = [
    ("Dataset collection & analysis", "2026-08-20", "2026-08-24", "#0f766e"),
    ("Data preparation & augmentation", "2026-08-23", "2026-08-24", "#0f766e"),
    ("Baseline model training (MobileNetV2, ResNet50, EfficientNetB0)", "2026-08-24", "2026-08-25", "#14b8a6"),
    ("Proposed model design & training", "2026-08-24", "2026-08-25", "#14b8a6"),
    ("Model evaluation & comparison", "2026-08-25", "2026-08-26", "#14b8a6"),
    ("Django project setup & auth", "2026-08-26", "2026-08-27", "#f97316"),
    ("Dashboard & dataset pages", "2026-08-26", "2026-08-27", "#f97316"),
    ("Detect Disease page + ML integration", "2026-08-27", "2026-08-28", "#f97316"),
    ("Pet Clinics (Geoapify integration)", "2026-08-27", "2026-08-28", "#f97316"),
    ("AI Chatbot (Gemini integration)", "2026-08-27", "2026-08-28", "#f97316"),
    ("UI/UX polish, 3D animations, responsive fixes", "2026-08-27", "2026-08-28", "#7c3aed"),
    ("Testing & documentation", "2026-08-28", "2026-08-28", "#334155"),
]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, start, end, color) in enumerate(tasks):
    s = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    duration = max((e - s).days, 1)
    ax.barh(i, duration, left=s, height=0.55, color=color, edgecolor="white")
    ax.text(s, i, "", va="center")

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_title("Project Timeline (Gantt Chart)", fontsize=14, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/gantt_chart.png", dpi=160, bbox_inches="tight")
plt.close()
print("saved gantt_chart.png")
