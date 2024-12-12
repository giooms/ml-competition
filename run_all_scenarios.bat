@echo off
REM Scenarios: A, B, C, D, E
set SCENARIOS=A B C D E
set FOLDS=1 2 3 4 5

echo Running Random Forest scenarios...
for %%S in (%SCENARIOS%) do (
  for %%F in (%FOLDS%) do (
    python main.py --scenario %%S --fold %%F --model_type rf
  )
)

echo Running XGBoost (GPU) scenarios...
for %%S in (%SCENARIOS%) do (
  for %%F in (%FOLDS%) do (
    python main.py --scenario %%S --fold %%F --model_type xgb
  )
)

echo All runs completed.
