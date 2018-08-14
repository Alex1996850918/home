#!/bin/sh

JOB=$1 
HERE=`pwd`
USER=`whoami`
JOBID=$$

if [ -e $JOB.vasp ] 
then
 /bin/rm $JOB.vasp
fi
echo -e "\033[40;33m --------------------------------------------------------------------------------------- \033[0m"
echo "This script is for running a single VASP job"
echo "To run a series of VASP jobs, you have to use"
echo "command"
echo "qsubs JOB"
echo "where script JOB describes how to run these"
echo "jobs in sequence"
echo "All input files have to be in this directory"
echo "Thank you!"
echo -e "\033[40;33m --------------------------------------------------------------------------------------- \033[0m"


QUE=1cpu
NCPUS=1
NPROC=1
MPI=1
echo "How many CPUs do you want to run your job? "
echo "1cpu, 20cpu, 40cpu, 80cpu, 120cpu, 160cpu, 200cpu, 300cpu, 500cpu, 1000cpu"
echo "For simple job or test ,please try: 20cput, 40cput, 80cput"
echo -e "\033[44;33m Not recommended using more than 80cpu \033[0m"
echo "Please don't waste our resources since someone have to pay for it, THX!! :))) [$QUE] "
read QUE
echo "setting your mpi [$MPI]"
read MPI
echo -e "\033[40;33m --------------------------------------------------------------------------------------- \033[0m"
exe='/home/u0/j31lee00/vasp.5.4.4/bin/intel_2017/vasp_std'
echo "This script was modified first by Shih-Kuang Lee on May 26 2018"

if [ -z "$QUE" ]
  then
  QUEUE=serial
  NCPUS=1
  NPROC=1
fi

if [ "$QUE" = 1cpu ]
then
  QUEUE=serial
  NCPUS=1
  NPROC=1
fi
if [ "$QUE" = 20cpu ]
then
  QUEUE=cf40
  NCPUS=20
  NPROC=1
fi
if [ "$QUE" = 40cpu ]
then
  QUEUE=cf40
  NCPUS=40
  NPROC=1
fi
if [ "$QUE" = 80cpu ]
then
  QUEUE=cf160
  NCPUS=40
  NPROC=2
fi
if [ "$QUE" = 120cpu ]
then
  QUEUE=cf160
  NCPUS=40
  NPROC=3
fi
if [ "$QUE" = 160cpu ]
then
  QUEUE=cf160
  NCPUS=40
  NPROC=4
fi
if [ "$QUE" = 200cpu ]
then
  QUEUE=ct400
  NCPUS=40
  NPROC=5
fi
if [ "$QUE" = 500cpu ]
then
  QUEUE=cf1200
  NCPUS=40
  NPROC=13
fi
if [ "$QUE" = 1000cpu ]
then
  QUEUE=cf1200
  NCPUS=40
  NPROC=25
fi
if [ "$QUE" = 20cput ]
then
  QUEUE=ctest
  NCPUS=20
  NPROC=1
fi
if [ "$QUE" = 40cput ]
then
  QUEUE=ctest
  NCPUS=40
  NPROC=1
fi
if [ "$QUE" = 80cput ]
then
  QUEUE=ctest
  NCPUS=40
  NPROC=2
fi 
cat << END_OF_CAT > $JOB.vasp
#!/bin/bash
#PBS -N ${JOB}
#PBS -e ${JOB}.stderr
#PBS -o ${JOB}.stdout
#PBS -l select=$NPROC:ncpus=$NCPUS:mpiprocs=$MPI
#PBS -q $QUEUE
#PBS -P ACD107023
#PBS -M alex850918@gmail.com
#PBS -m be 

cd `pwd`

module purge
module load intel/2017_u4 
export I_MPI_HYDRA_BRANCH_COUNT=-1

echo "Your VASP job starts at  \`date\` "

mpiexec.hydra -PSM2 $exe

wait


echo "Your VASP job completed at  \`date\` "


END_OF_CAT

#
chmod +x $JOB.vasp
#
qsub $JOB.vasp



