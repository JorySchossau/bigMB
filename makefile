# the below commented line was used for dynamic parallelism features
#GPUFLAGS= -lcuda -lcurand -gencode arch=compute_35,code=sm_35 -rdc=true -G # -lcudadevrt
## the -x cu means to compile all files specified as if they were .cu files

all:
	nvcc -std=c++14 -w -lcuda -lcurand -Icub -o markov main.cpp timer/Timer.cpp -x cu -arch=sm_35
	@#nvcc -std=c++14 -w -lcuda -Icub-1.8.0 -lcurand -o mnist main.cpp -x cu -O3 --expt-extended-lambda #&& ./mnist
	@#nvcc -w -lcuda -Ihemi -lcurand -o mnist main.cpp -x cu -rdc=true -arch=sm_35 -O3 --expt-extended-lambda && ./mnist

clean:
	rm -rf markov
