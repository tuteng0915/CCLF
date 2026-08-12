# EXP-98 Spec — On-Policy Coupling Trajectory Distillation

**Status:** CLOSED BY EXP-94 / DO NOT LAUNCH
**Purpose:** compress a verified inference-time coupling schedule without
repeating the straight-endpoint target rejected by EXP-91/92.

The teacher is the best frozen Plaid coupling schedule. Training examples are
actual teacher states before admission, at the coupling event, and during
joint refinement. The student preserves the teacher's native curved vector
field using state/velocity and predicted-clean matching:

```text
L = L_native
  + lambda_v ||v_student - stopgrad(v_teacher)||^2_normalized
  + lambda_x ||xhat0_student - stopgrad(xhat0_teacher)||^2.
```

Half of each batch remains native synchronous training. Do not supervise
mixed states directly toward the clean endpoint, and do not introduce local
token clocks. First target is reducing a verified 56-call schedule to at most
40 calls.

Use a matched continued-training control and require healthy Standard
generation before evaluating compression. Promote only if the compressed
student preserves the teacher's compute-matched C-PPL/boundary advantage in
`2/3` seeds without worsening prompt gain, D1, Rep-4, or degeneration.

If EXP-94 finds no advantage over Parallel-44, or EXP-97 finds no scalable
schedule, this experiment remains closed.
