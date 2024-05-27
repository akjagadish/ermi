#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=ERMI_Baselines
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18

cd ~/ermi/categorisation/
module load anaconda/3/2021.11
pip3 install --user python-dotenv ipdb accelerate tqdm anthropic

# python baselines/run_llm.py --mode 'human' --dataset 'devraj2022' --start-participant 56
#python baselines/run_llm.py --mode 'human' --dataset 'devraj2022' --start-participant 75
# python baselines/run_llm.py --mode 'human' --dataset 'devraj2022' --start-participant 77
python baselines/run_llm.py --mode 'human' --dataset 'devraj2022' --start-participant 55
# python baselines/run_llm.py --mode 'human' --dataset 'devraj2022' --start-participant 110


# python baselines/run_llm.py --mode 'human' --dataset 'badham2017'
# python baselines/run_llm.py --mode 'match_ermi' --dataset 'badham2017' --start-participant 72
# python baselines.simulate_llm --mode 'match_ermi' --dataset 'badham2017' --from-env --start-participant 3
