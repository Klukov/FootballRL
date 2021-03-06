ALGORITHM_NAME='PPO2'
SCENARIOS='13 14 17'
NUMBER_OF_PARALLEL_ENVS='8'
EVALUATION_ACCURACY='1000'
TIME_STEPS='10000 20000 50000 100000 200000 500000 1000000 2000000 5000000'

for scenario_number in $SCENARIOS; do
  dir_name="algorithm-${ALGORITHM_NAME}_scenario-${scenario_number}"
  project_path=$(pwd)
  mkdir "$dir_name"
  cd "$dir_name" || exit
  for time_steps in $TIME_STEPS; do
    number=$(echo -e "import re\nprint(str($time_steps / 1e6).replace(""'\.'"", ""''"") \
    if $time_steps < 1e6 \
    else re.sub(""'\..*'"", ""''"", str($time_steps / 1e6)))" | python)
    name="${number}M"
    mkdir "$name"
    cd ./"$name" || exit
    python "${project_path}"/baselines_run.py \
      --scenario_number="${scenario_number}" \
      --algorithm="${ALGORITHM_NAME}" \
      --number_of_steps="${time_steps}" \
      --number_of_envs="${NUMBER_OF_PARALLEL_ENVS}"
    zip_file_name=$(find . -iname \*.zip)
    python "${project_path}"/baselines_evaluator.py \
      --scenario_number="${scenario_number}" \
      --algorithm="${ALGORITHM_NAME}" \
      --accuracy=$EVALUATION_ACCURACY \
      --path=./"${zip_file_name}"
    cd ..
  done
  cd ..
done
