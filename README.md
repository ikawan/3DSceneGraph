# 🤖 Real-time Scene Understanding for Robot Manipulation

This project explores how machines can *understand* what's happening in a scene — not just see it.

The goal?  
Turn raw visual input into a structured representation that a robot could actually reason about.

---

## 🧠 What’s the idea?

We take video (or RGB-D data) of human actions — like cutting or stirring — and try to break it down into:

- 🧍 Human pose (hands)
- 🧊 Objects in the scene
- 🔗 Relationships between them (e.g. *touching*, *above*, *moving together*)

All of this is combined into a **scene graph** — basically a map of *who is doing what to what*.

---

## ⚙️ What’s inside?

This repo experiments with combining:

- YOLO → object detection  
- Pose estimation → tracking human movement  
- Segmentation → understanding object shapes  
- Custom scripts → benchmarking + analysis  

---

## 🎯 Why does this matter?

Robots don’t just need to see — they need to **understand interactions**.

Scene graphs help bridge the gap between:
> pixels → meaning → action

---

## 🚧 Status

Work in progress.  
Lots of experiments, benchmarks, and “does this even work?” moments.

---

## 🧪 Bonus

There are also:
- test scripts
- benchmarking pipelines
- random outputs from experiments (some questionable 👀)

---

## 📝 Note

This is part of a course project exploring **real-time 3D semantic scene graph generation for robot manipulation**.

---

## 🚀 Future direction

- Real-time pipeline integration  
- Better tracking consistency  
- Eventually: something a robot could actually use  

---
