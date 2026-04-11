# 🛡 Missile Defense Simulation

A physics-based missile interception simulation using Python.

## 🎥 Demo

![Simulation Demo](assets/demo.gif)

## 🚀 Features
- Ballistic missile trajectories
- Proportional Navigation guidance
- Multi-missile interception
- Animated visualization
     

- Simulated real ballistic missile physics using Newtonian kinematics 
  (s = ut + ½at², v = u + at) with gravity, thrust modeling, and aerodynamic drag

- Implemented Augmented Proportional Navigation (APN) guidance — the same 
  algorithm used in Patriot PAC-3 and THAAD — for real-time interceptor steering

- Built analytical intercept geometry solver using quadratic impact prediction 
  and binary search refinement to compute optimal launch windows per threat

- Engineered multi-phase interceptor lifecycle (boost → coast → intercept) with 
  probabilistic hit modeling and LOS rate-based abort logic

## ▶️ Run

```bash
python main.py