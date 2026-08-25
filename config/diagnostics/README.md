# Mahjong Solo prompt-phase diagnostics

These configs isolate the prompt-phase validation from the formal Mahjong runs.
They write only under `runs/diagnostics` and do not modify the 15-task experiment
configs.

Run the one-turn terminal-recognition checks first:

```bash
for config in \
  config/diagnostics/mahjong_terminal_direct_phasefix.yaml \
  config/diagnostics/mahjong_terminal_cot_phasefix.yaml \
  config/diagnostics/mahjong_terminal_tot_phasefix.yaml \
  config/diagnostics/mahjong_terminal_plan_then_solve_phasefix.yaml \
  config/diagnostics/mahjong_terminal_critic_refine_phasefix.yaml
do
  python -m minibench.cli run-config "$config"
done
```

If ToT, plan-then-solve, and critic-refine all recognize the terminal hand, run
the five-task rollout checks:

```bash
for config in \
  config/diagnostics/mahjong_solo5_direct_phasefix.yaml \
  config/diagnostics/mahjong_solo5_cot_phasefix.yaml \
  config/diagnostics/mahjong_solo5_tot_phasefix.yaml \
  config/diagnostics/mahjong_solo5_plan_then_solve_phasefix.yaml \
  config/diagnostics/mahjong_solo5_critic_refine_phasefix.yaml
do
  python -m minibench.cli run-config "$config"
done
```

Summarize the saved internal stage outputs and token usage:

```bash
python scripts/summarize_mahjong_solo_traces.py runs/diagnostics/*-phasefix
```

Use unique run names if repeating a diagnostic because Mahjong snapshots replace
files in an existing run directory.
