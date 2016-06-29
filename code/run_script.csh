#! /bin/csh -f

#if ($#argv != 1) then
#        echo "Usage: $0 star"
#        exit 0
#endif
#
#
#PBS -l select=15:ncpus=24:mpiprocs=24:model=has
#PBS -l walltime=120:00:00
#PBS -M barclay.astro@gmail.com
#PBS -m bae
#PBS -N epic2113
#PBS -q long
#PBS -W group_list=s1089
#PBS -j oe


set starnum=koi2113

cd /nobackupp8/tsbarcl2/redgiantGP/code/

rm -f "$starnum/time_special.dat"

mpiexec.hydra -machinefile $PBS_NODEFILE -np 360 python /nobackupp8/tsbarcl2/redgiantGP/code/run_epic.py  >& time_special.dat

#mpiexec -np 96 /nobackupp8/tsbarcl2/redgiantGP/code/run_epic.py  >& time_special.dat
exit 0
