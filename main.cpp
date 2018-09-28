// c++14 required

// ALGORITHM DESCRIPTION
//
// To achieve computations of large networks of gates, a single card will execute 1
// contiguous network (one brain) at a time, out of a population of 100 or more such
// networks. All the data for the population is stored in device global memory
// to minimize host-device data transfer. All evolution is performed on the device
// side to minimize host-device data transfer.
//
// Previous iterations of Markov Brains used genomes that were transcribed
// to their phenotypes before any fitness evaluation could be performed.
// This is costly, so here we directly encode everything. In other words,
// all gates are only built once, then "mutated" probabilistically. Mutations
// like this occur on a feature by feature basis.
//
// To allow for heterogeneous mixtures of gates in a signle network, we
// evaluate all gates of the same type at once by calling their associated kernel,
// then proceed to the next type. At the end of this processing, all gates have
// performed their update for a single time step.
//
// Data for gates are stored in several large global device memory arrays:
// (`idx` here is shorthand for `threadIdx.x+blockDim.x*blockIdx.x`)
// gateType[idx]: which type of gate idx is (0 for deterministic)
// numIns[idx]: number of inpus gate idx has
// numOuts[idx]: number of outputs gate idx has
// insList[idx*4] through insList[idx*4+4]: 4 indices from the state buffer (numIns determines how many are used)
// outsList[idx*4] through outsList[idx*4+4]: 4 indices from the state buffer (numOuts determines how many are used)
// DGDIM1 = 16, DGDIM2 = 4 // maximum size for 4-in 4-out
// detData[idx*DGDIM1*DGDIM2] through detData[idx*DGDIM1*DGDIM2+DGDIM1*DGDIM2]: deterministic logic table
//
// Because of this layout and direct encoding, we can bias mutations toward structures like the logic table.

#define RELEASE
#include <cstdio>
#include <cstdlib> // RAND_MAX, 
#include <iostream>
#include <vector>
#include <numeric> // iota
#include <algorithm> // for ... and sort
#include <functional> // greater, lambdas
#include <utility> // swap
#include <iterator> // distance
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <csignal> // sigint
#include "timer/Timer.h" // stopwatch lib from ./timer/Timer.{h,cpp} [https://github.com/wbinventor/timer]
#include <cub/cub.cuh> // for gpu radix sort
//#include <test/test_util.h> // for/in cub library

namespace ccu { // Custom CUda namespace
    // array, with lazy JIT cudaMemcpy-ing between host and device (inspired by HEMI, but HEMI's is broken)
    template <typename T>
        class array {
            size_t size_;
            private:
            T *host_data, *device_data; // the usual host-side and dev-side pointer pair
            bool host_dirtyYN, device_dirtyYN;
            public:
            inline void memcpyD2H() {
                host_dirtyYN = false;
                cudaMemcpy(host_data, device_data, sizeof(T)*size_, cudaMemcpyDeviceToHost);
            }
            inline void memcpyH2D() {
                device_dirtyYN = false;
                cudaMemcpy(device_data, host_data, sizeof(T)*size_, cudaMemcpyHostToDevice);
            }
            unsigned int inline size() { return size_; }
            array(size_t size, bool pinnedYN=false) : size_(size), host_dirtyYN(false), device_dirtyYN(false) {
                switch( pinnedYN ) {
                    case true:
                        fprintf(stderr, "Pinned memory not implemented yet, but it's easy: cudaHostAlloc(...)\n");
                        exit(1);
                        break;
                    case false:
                        // init host pointer
                        host_data = new T[size_];
                        break;
                }
                // init device pointer
                cudaMalloc( (void**)&device_data, sizeof(T)*size_ ); 
                cudaDeviceSynchronize();
            }
            array(const array &other) {
                printf("copy ctor being called! Probably shouldn't be... you might be doing something very costly.\n");
                size_ = other.size_;
                host_data = other.host_data;
                device_data = other.device_data;
                host_dirtyYN = other.host_dirtyYN;
                device_dirtyYN = other.device_dirtyYN;
            }
            array(array &&other) {
                size_ = other.size_;
                host_data = other.host_data;
                device_data = other.device_data;
                host_dirtyYN = other.host_dirtyYN;
                device_dirtyYN = other.device_dirtyYN;
                other.host_data = nullptr;
                other.device_data = nullptr;
            }
            array& operator=(array&) noexcept = default;
            array& operator=(array&&) noexcept = default;
            inline T* host() { // return host pointer
                if (host_dirtyYN) memcpyD2H();
                device_dirtyYN = true;
                return host_data;
            }
            inline T* device() { // return device pointer
                if (device_dirtyYN) memcpyH2D();
                host_dirtyYN = true;
                return device_data;
            }
            inline void sync() { // force synchronization to undirty the buffers (not usually necessary)
                if (host_dirtyYN) memcpyD2H();
                else if (device_dirtyYN) memcpyH2D();
            }
				void setDeviceData(T *other_device_data) {
					cudaFree(device_data);
					device_data = other_device_data;
					device_dirtyYN = false;
					memcpyD2H();
				}
            // using copy and move ctor wrong so destructor breaks things currently
        };
}

// kernel configuration helper for readability
struct ExecutionPolicy {
   public:
      size_t gridSize,blockSize,sharedMemBytes; // TODO: implement streams
      ExecutionPolicy() : gridSize(0), blockSize(0), sharedMemBytes(0) { }
};

// kernel launch helper for readability
template <typename... Arguments>
inline void launch(const ExecutionPolicy &p, void (*f)(Arguments...), Arguments... args) {
   f<<<p.gridSize, p.blockSize, p.sharedMemBytes>>>(args...);
}
__device__ inline unsigned int globalThreadIndex() { return threadIdx.x + blockIdx.x*blockDim.x; }
__device__ inline unsigned int localThreadIndex() { return threadIdx.x; }

// checks cudaGetLastError and prints the associated message if any found, otherwise does nothing
inline void cudaShowErrors() {
#ifndef RELEASE
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess)
   {
      printf("CUDA error: %s\n", cudaGetErrorString(error));
      exit(-1);
   } 
#endif
}

// helper to benchmark stuff
void showElapsedTime(const char* label, std::function<void()> func) {
   cudaEvent_t startEvent, stopEvent;
   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);
   cudaEventRecord(startEvent, 0);
   func();
   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);
   float time;
   cudaEventElapsedTime(&time, startEvent, stopEvent);
   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);
   printf("%s finished in %.1f ms\n", label, time);
}

// helper macro to make and fill a vector representing the whole population
// name##_d holds the start address for each device array in vector<> name.
#define POP_LITE(name,typ,size,popsize) \
	std::vector<ccu::array<typ>> name; \
   name.reserve(popsize); \
   for (int i(0); i<popsize; i++) name.emplace_back( std::move(ccu::array<typ>(size)) );

   //for (int i(0); i<popsize; i++) cudaMalloc( (void**)&name##_d[i], sizeof(typ)*size);
#define POP(name,typ,size,popsize) \
	typ* name##_d; \
	cudaMalloc( (void**)&name##_d, sizeof(typ)*popsize*size ); \
	std::vector<ccu::array<typ>> name; \
   name.reserve(popsize); \
   for (int i(0); i<popsize; i++) name.emplace_back( std::move(ccu::array<typ>(size)) ); \
	for (int i(0); i<popsize; i++) name[i].setDeviceData(name##_d+i*size);

// DEFINES and MACROS
#define uint8 unsigned char
#define uint unsigned int

/* #############################
 * ### SIMULATION PARAMETERS ###
 * #############################
 */
// this arrangement is used to test the Max Ones standard genetic programming fitness function
#define NUM_INPUT_STATES 1 // # of locations in the state buffer are for input
#define NUM_HIDDEN_STATES 0 // # of locations in state buffer not hidden or input
#define NUM_OUTPUT_STATES 2000 // # of locations in the state buffer are for output
#define NUM_STATES NUM_INPUT_STATES+NUM_OUTPUT_STATES+NUM_HIDDEN_STATES // total state buffer size
#define PGDIM 16 // probabilistic gates logic table dimension (both sides typically 2^4)
#define DGDIM1 16 // deterministic gates logic table dimension (number of rows typically 2^4)
#define DGDIM2 4  // deterministic gates logic table second dimension (number of cols typically 4)

// gate type definitions
#define GATE_DETERMINISTIC 0
#define GATE_PROBABILISTIC 1
//... #define ANOTHER PREVIOUS+1
#define NUM_GATE_TYPES 1 // set to 1 to limit to deterministic gates

// hemiRand helper fn
__device__ inline
unsigned int gpurand(curandState* randstates) {
	return curand(&(randstates[globalThreadIndex()]));
}
#define GPU_RAND gpurand(randstates) /// convenience function, relies on randstates being passed to the kernel
__device__ inline
float gpurand_uniform(curandState* randstates) {
	return curand_uniform(&(randstates[globalThreadIndex()]));
}
#define GPU_RAND_FLOAT gpurand_uniform(randstates) /// convenience function, relies on randstates being passed to the kernel

/* ###############
 * ### KERNELS ###
 * ###############
 */

// seed the random number generator
__global__ void KernelRandInit(curandState* randstates, int seed) {
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
			globalThreadIndex(), /* the sequence number is only important with multiple cores */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&(randstates[globalThreadIndex()]));
}

// KERNEL InitGates
__global__ void KernelInitGates(curandState* randstates, uint8 *dnumIns, uint8 *dnumOuts, uint *dinsList, uint *doutsList, uint8 *dgateType, uint8 *dProb, uint *dProbRowMarginals, uint8 *dDetData) {
	/// create local memory (hopefully as registers??)
	uint8 numIns(GPU_RAND&0b11);
	uint8 numOuts(GPU_RAND&0b11);
	uint *rowMarginals(dProbRowMarginals+PGDIM*globalThreadIndex());
	uint8 gateType(GPU_RAND % NUM_GATE_TYPES);
	/// save to global memory
	dnumIns[globalThreadIndex()] = numIns;
	dnumOuts[globalThreadIndex()] = numOuts;
	dinsList[globalThreadIndex()*4+0] =  GPU_RAND % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);
	dinsList[globalThreadIndex()*4+1] =  GPU_RAND % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);
	dinsList[globalThreadIndex()*4+2] =  GPU_RAND % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);
	dinsList[globalThreadIndex()*4+3] =  GPU_RAND % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);
	doutsList[globalThreadIndex()*4+0] = (GPU_RAND % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES))+NUM_INPUT_STATES;
	doutsList[globalThreadIndex()*4+1] = (GPU_RAND % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES))+NUM_INPUT_STATES;
	doutsList[globalThreadIndex()*4+2] = (GPU_RAND % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES))+NUM_INPUT_STATES;
	doutsList[globalThreadIndex()*4+3] = (GPU_RAND % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES))+NUM_INPUT_STATES;
	dgateType[globalThreadIndex()] = gateType;
	/// initialize gateType-dependent values
	uint total(0);
	switch( gateType ) {
		case GATE_PROBABILISTIC: /// probabilistic gates
			#pragma unroll
			for (auto row(0); row<PGDIM; row++) {
				total = 0;
				#pragma unroll
				for (auto col(0); col<PGDIM; col++) {
					dProb[ globalThreadIndex()*PGDIM*PGDIM+row*PGDIM+col ] = GPU_RAND % 256;
					total += dProb[ globalThreadIndex()*PGDIM*PGDIM+row*PGDIM+col ];
				}
				rowMarginals[row] = total;
			}
			break;
		case GATE_DETERMINISTIC: /// deterministic gates
			#pragma unroll
			for (auto row(0); row<DGDIM1; row++) {
				for (auto col(0); col<DGDIM2; col++) {
					dDetData[globalThreadIndex()*DGDIM1*DGDIM2+row*DGDIM2+col] = GPU_RAND & 0b1;
				}
			}
			break;
	};
}

// KERNEL Update Prob Gates
__global__
void KernelUpdateProbGates(curandState* randstates, uint8 *dnumIns, uint8 *dnumOuts, uint *dinsList, uint *doutsList, uint8 *dgateType, uint8 *dProbData, uint *dProbRowMarginals, uint *statest, uint *statest1) {
	if (dgateType[globalThreadIndex()] != GATE_PROBABILISTIC) return; // only continue if prob gate
	register uint8 numIns(dnumIns[globalThreadIndex()]);
	register uint8 numOuts(dnumOuts[globalThreadIndex()]);
	register uint8 gateType(dgateType[globalThreadIndex()]);
	uint *insList(dinsList+4*globalThreadIndex());
	uint *outsList(doutsList+4*globalThreadIndex());
	uint *rowMarginals(dProbRowMarginals+PGDIM*globalThreadIndex());
	uint8 numOutPatterns(1<<numOuts);

    // Description of a Probabilistic Markov Gate
    // The original idea (easy version) is to use a table of floats that are row-normalized (rows sum to 1)
    // each row number corresponds to a unique bit representation of the inputs (up to 4 input bits set, 16 reprs)
    // each col number corresponds to a unique bit representation of the outputs (again, up to 16 representations)
    // the input row is determined deterministically by combining the input bits into a single integer (STEP A)
    // then the output col is determined by generating a random float [0,1] and summing
    // from left-to-right elements in the row until their total is greater than the random float. That position
    // is the output column (STEP B).
    // Reading and writing inputs and outputs means reading and writing from and to a bit buffer
    // one for time t (current time) that is a buffer of concatenated input bits, hidden state bits, and output bits
    // (outputs are effectively ignored)
    // the other is for time t+1 that is the output buffer, identical to the t buffer.
    // As gates are limited here to 4-in 4-out, then then writing to all outputs happens if
    // column 15 is selected by the process above (0b1111). In this case all 4 outputs fire.
    // The way this bit pattern is translated to positions in the output buffer to modify is
    // performed by indexing into outsList, which contains 4 numbers indicating positions in
    // the buffer (STEP C).
    //
    // Changes for This Implementation of Probabilistic Markov Gates
    // The below implementation of the above algorithm is optimized for GPU by by eliminating
    // floating point numbers, and eliminating conditional branching.
    // For these modifications we use bytes instead of floats so are limited to 256 levels of
    // precision, which works okay. Secondly, we keep track of the row marginals so we don't
    // have to normalize the rows and instead generate a random number between 0 and rowMarginal,
    // for instance. Branching is avoided by using a form of 'masking,' where the naive version
    // is: `for (auto i(0); i<numIns; i++) {...}` we instead iterate over all 4 or 16 positions
    // and appropriately multiply the accumulated number by a boolean, such as `*(i<numIns)`

    // STEP A: combine input bits into single integer representation
	register uint8 in(0);
	in |= statest[ insList[ globalThreadIndex()*4+0 ] ]*(0<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+1 ] ]*(1<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+2 ] ]*(2<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+3 ] ]*(3<numIns);
	register uint accumulation(0);
	register uint threshold(GPU_RAND % (rowMarginals[in]));

    // STEP B: march along the `in`-associated row governed by a random number
    // to determine the output bit pattern
	register uint8 out(0); /// keeps track of which combination of output bits will be set
	#pragma unroll
	for (register uint8 col(0); col<PGDIM; col++) {
		accumulation += dProbData[ globalThreadIndex()*(PGDIM*PGDIM)+PGDIM*in+col ]*(col<(1<<numOuts));
		out += (out<(1<<numOuts)); // NOTE: possibly increments `out` too many by 1
	}
	out -= (out==(1<<numOuts)) && numOuts; // account for earlier possibility of over-increment past last column

    // STEP C: write to the output buffer
	#pragma unroll
	for (register uint8 i=0; i<4; i++)
		atomicOr( &statest1[outsList[i]], (out>>i)&0b1 ); // unfortunately have to read/write whole 32 bits
}

// KERNEL Update Deterministic Gates
__global__
void KernelUpdateDetGates(curandState* randstates, uint8 *dnumIns, uint8 *dnumOuts, uint *dinsList, uint *doutsList, uint8 *dgateType, uint8 *dDetData, uint *statest, uint *statest1) {
	if (dgateType[globalThreadIndex()] != GATE_DETERMINISTIC) return; // only continue if prob gate
	register uint8 numIns(dnumIns[globalThreadIndex()]);
	register uint8 numOuts(dnumOuts[globalThreadIndex()]);
	uint *insList(dinsList+4*globalThreadIndex());
	uint *outsList(doutsList+4*globalThreadIndex());
	uint8 *detData(dDetData+DGDIM1*DGDIM2*globalThreadIndex());
	uint8 numOutPatterns(1<<numOuts);

	register uint8 in(0);
	in |= statest[ insList[ globalThreadIndex()*4+0 ] ]*(0<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+1 ] ]*(1<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+2 ] ]*(2<numIns);
	in |= statest[ insList[ globalThreadIndex()*4+3 ] ]*(3<numIns);

	/// set states_{t+1}
	/// for each bit determined set in the row's output mapping, set the associated bit in `statest1` mapped by the positions stored in `outsList`
	#pragma unroll
	for (register uint8 col=0; col<DGDIM2; col++) {
		atomicOr( &statest1[ outsList[col] ], detData[DGDIM2*in+col]*(col<numOuts) ); // must read/write whole 32 bits
	}
}

// PredictionWorld Update, to progress the state of the world
__global__
void KernelPredictionWorldUpdateOverPop(uint *worldState, uint **statest, uint **statest1) {
    // globalThreadIndex is now population ID (typically 100: 0-99)
    worldState[globalThreadIndex()] += 1; // CUDA should coalesce correctly here many-to-1...
    worldState[globalThreadIndex()] %= 16;
    //worldState[globalThreadIndex()] %= 8;
    statest[globalThreadIndex()][0] = (worldState[globalThreadIndex()*1]>>0)&0b1;
    statest[globalThreadIndex()][1] = (worldState[globalThreadIndex()*1]>>1)&0b1;
    statest[globalThreadIndex()][2] = (worldState[globalThreadIndex()*1]>>2)&0b1;
    statest[globalThreadIndex()][3] = (worldState[globalThreadIndex()*1]>>3)&0b1;
}

// PredictionWorld Evaluation, to evaluate each agent given the current state of the world
__global__
void KernelPredictionWorldEvaluateOverPop(uint *worldState, uint **statest1, uint *worldFitness) {
    // globalThreadIndex is now population ID (typically 100: 0-99)
    // tests for `+1%16` bit pattern prediction: 0->1, 1->2, ... 14->15, 15->0.
    // reads first 4 outputs of the outputs section of the buffer
    uint out(0);
    out |= statest1[globalThreadIndex()][NUM_INPUT_STATES+NUM_HIDDEN_STATES+0]<<0;
    out |= statest1[globalThreadIndex()][NUM_INPUT_STATES+NUM_HIDDEN_STATES+1]<<1;
    out |= statest1[globalThreadIndex()][NUM_INPUT_STATES+NUM_HIDDEN_STATES+2]<<2;
    out |= statest1[globalThreadIndex()][NUM_INPUT_STATES+NUM_HIDDEN_STATES+3]<<3;
    worldFitness[globalThreadIndex()] += (out == (worldState[globalThreadIndex()]+3)%16);
    //worldFitness[globalThreadIndex()] += ((out+8-1)%8) == worldState[globalThreadIndex()];
    //worldFitness[globalThreadIndex()] += ((out%16) == worldState[globalThreadIndex()]);
}

__global__
void KernelMaxOnesWorldUpdateOverPop(uint *worldState, uint **statest, uint **statest1) {
	// 1 constant input, 10 output states, 8 hidden,
	// like default TestWorld in our other framework, MABE, for comparison
	#pragma unroll
   for (uint o(0); o<NUM_INPUT_STATES; o++) statest1[globalThreadIndex()][o] = 1; // set all inputs to 1
}

__global__
void KernelMaxOnesWorldEvaluateOverPop(uint *worldState, uint **states1, uint *worldFitness) {
	// 1 constant input, 10 output states, 8 hidden,
	// like default TestWorld in our other framework, MABE, for comparison
	uint ones_count(0);
    #pragma unroll
	for (int i(NUM_INPUT_STATES+NUM_HIDDEN_STATES); i<NUM_STATES; i++)  {
		ones_count += states1[globalThreadIndex()][i];
    }
	worldFitness[globalThreadIndex()] += ones_count;
}

// Any organism run with this kernel will die and will be replaced with offspring from a parent
// This is also the 'mutation' function for the direct encoded structures
__global__
void KernelMoranProcess2(curandState *randstates,
									unsigned int dyingAgenti, unsigned int newParentIDPos,
									unsigned int *IDs,
									unsigned int gateCount,
                           uint8 *numIns_d, // take note! these pointers point to flat arrays of the entire population's gates
                           uint8 *numOuts_d,

                           uint *insList_d,
                           uint *outsList_d,

                           uint8 *gateType_d,
                           uint8 *detData_d,
                           uint8 *probData_d) {
    auto idx = globalThreadIndex();
	 auto outOffset = IDs[dyingAgenti]*gateCount;
	 auto inOffset = IDs[newParentIDPos]*gateCount;

    ////
    //// Number of Ins & Outs Mutation
    ////
    // modify parent's version by +1 or +0 or +(-1) [but only do so with p=0.002], then restrict number to original range
    // numIns_d and numOuts are single numbers per individual, so numIns_d is length popSize
    numIns_d[outOffset+idx] = (numIns_d[inOffset+idx] + 5 + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.002))% 5; // range 0,1,2,3,4
    numOuts_d[outOffset+idx] = (numOuts_d[inOffset+idx] + 5 + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.002)) % 5; // range 0,1,2,3,4

    ////
    //// Ins List Mutation
    ////
    // insList and outsList are groups of 4 numbers per individual, so insList is length popSize*4
    #pragma unroll // copy from parent
    for (int i(0); i<4; i++)
    insList_d[outOffset*4+idx*4+i] = (insList_d[inOffset*4+idx*4+i] + (NUM_INPUT_STATES+NUM_HIDDEN_STATES) + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.02)) % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);

    ////
    //// Outs List Mutation
    ////
    #pragma unroll // copy from parent with optional mutation (add a large random number)
    for (int i(0); i<4; i++)
    outsList_d[outOffset*4+idx*4+i] = ((outsList_d[inOffset*4+idx*4+i] + (GPU_RAND_FLOAT < 0.002)*(GPU_RAND&0b11111111111111)) % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES)) + NUM_INPUT_STATES;

    ////
    //// Gate Type Mutation
    ////
    // gateType are single numbers per individual, 0 det, 1 prob, etc.
    gateType_d[outOffset+idx] = (gateType_d[inOffset+idx] + NUM_GATE_TYPES + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.0002)) % NUM_GATE_TYPES;
    //detData are 16x4 values per individual
    #pragma unroll // copy from parent
    for (int i(0); i<DGDIM1*DGDIM2; i++) { detData_d[outOffset*DGDIM1*DGDIM2+idx*DGDIM1*DGDIM2+i] = detData_d[inOffset*DGDIM1*DGDIM2+idx*DGDIM1*DGDIM2+i]; }

    ////
    //// Deterministic Gate Logic Table Mutation
    ////
    // mutate up to 32 locations in the deterministic logic table
    #pragma unroll
	for (int i(0); i<16; i++) {
		auto selectedValue = GPU_RAND&0b111111; // now get ready to mutation one element of PGDIM*PGDIM (0-255)
	    detData_d[outOffset*DGDIM1*DGDIM2+idx*DGDIM1*DGDIM2+selectedValue] = (detData_d[inOffset*DGDIM1*DGDIM2+idx*DGDIM1*DGDIM2+selectedValue] + (GPU_RAND&0b1)*(GPU_RAND_FLOAT < 0.015)) % 2;
	}

    ////
    //// Probabilistic Gate Logic Table Mutation
    ////
    //probData are 16x16 values per individual
    #pragma unroll
    for (int i(0); i<PGDIM*PGDIM; i++) { probData_d[outOffset*PGDIM*PGDIM+idx*PGDIM*PGDIM+i] = probData_d[inOffset*PGDIM*PGDIM+idx*PGDIM*PGDIM+i]; }
    // mutate up to 64 locations in the probabilistic logic table
	for (int i(0); i<64; i++) {
		 auto selectedValue = GPU_RAND&0b11111111; // now get ready to mutation one element of PGDIM*PGDIM (0-255)
		 probData_d[outOffset*PGDIM*PGDIM+idx*PGDIM*PGDIM+selectedValue] = probData_d[inOffset*PGDIM*PGDIM+idx*PGDIM*PGDIM+selectedValue] + (GPU_RAND&0b11111111*(GPU_RAND_FLOAT < 0.02));
	}
}

// Any organism run with this kernel will die and will be replaced with offspring from a parent
// This is also the 'mutation' function for the direct encoded structures
__global__
void KernelMoranProcess(curandState *randstates,
                           uint8 *numIns,
                           uint8 *numOuts,

                           uint *insList,
                           uint *outsList,

                           uint8 *gateType,
                           uint8 *detData,
                           uint8 *probData,

                           uint8 *parentNumIns,
                           uint8 *parentNumOuts,
									
                           uint *parentInsList,
                           uint *parentOutsList,

                           uint8 *parentGateType,
                           uint8 *parentDetData,
                           uint8 *parentProbData) {
    auto idx = globalThreadIndex();

    ////
    //// Number of Ins & Outs Mutation
    ////
    // modify parent's version by +1 or +0 or +(-1) [but only do so with p=0.002], then restrict number to original range
    // numIns and numOuts are single numbers per individual, so numIns is length popSize
    numIns[idx] = (parentNumOuts[idx] + 5 + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.002))% 5; // range 0,1,2,3,4
    numOuts[idx] = (parentNumOuts[idx] + 5 + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.002)) % 5; // range 0,1,2,3,4

    ////
    //// Ins List Mutation
    ////
    // insList and outsList are groups of 4 numbers per individual, so insList is length popSize*4
    #pragma unroll // copy from parent
    for (int i(0); i<4; i++)
    insList[idx*4+i] = (parentInsList[idx*4+i] + (NUM_INPUT_STATES+NUM_HIDDEN_STATES) + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.02)) % (NUM_INPUT_STATES+NUM_HIDDEN_STATES);

    ////
    //// Outs List Mutation
    ////
    #pragma unroll // copy from parent with optional mutation (add a large random number)
    for (int i(0); i<4; i++)
    outsList[idx*4+i] = ((parentOutsList[idx*4+i] + (GPU_RAND_FLOAT < 0.002)*(GPU_RAND&0b11111111111111)) % (NUM_HIDDEN_STATES+NUM_OUTPUT_STATES)) + NUM_INPUT_STATES;

    ////
    //// Gate Type Mutation
    ////
    // gateType are single numbers per individual, 0 det, 1 prob, etc.
    gateType[idx] = (parentGateType[idx] + NUM_GATE_TYPES + (GPU_RAND&0b10 - 1)*(GPU_RAND_FLOAT < 0.0002)) % NUM_GATE_TYPES;
    //detData are 16x4 values per individual
    #pragma unroll // copy from parent
    for (int i(0); i<DGDIM1*DGDIM2; i++) { detData[idx*DGDIM1*DGDIM2+i] = parentDetData[idx*DGDIM1*DGDIM2+i]; }

    ////
    //// Deterministic Gate Logic Table Mutation
    ////
    // mutate up to 32 locations in the deterministic logic table
    #pragma unroll
	for (int i(0); i<16; i++) {
		auto selectedValue = GPU_RAND&0b111111; // now get ready to mutation one element of PGDIM*PGDIM (0-255)
	    detData[idx*DGDIM1*DGDIM2+selectedValue] = (parentDetData[idx*DGDIM1*DGDIM2+selectedValue] + (GPU_RAND&0b1)*(GPU_RAND_FLOAT < 0.015)) % 2;
	}

    ////
    //// Probabilistic Gate Logic Table Mutation
    ////
    //probData are 16x16 values per individual
    #pragma unroll
    for (int i(0); i<PGDIM*PGDIM; i++) { probData[idx*PGDIM*PGDIM+i] = parentProbData[idx*PGDIM*PGDIM+i]; }
    // mutate up to 64 locations in the probabilistic logic table
	for (int i(0); i<64; i++) {
		 auto selectedValue = GPU_RAND&0b11111111; // now get ready to mutation one element of PGDIM*PGDIM (0-255)
		 probData[idx*PGDIM*PGDIM+selectedValue] = parentProbData[idx*PGDIM*PGDIM+selectedValue] + (GPU_RAND&0b11111111*(GPU_RAND_FLOAT < 0.02));
	}
}

__global__
void KernelFitnessResetOverPop(uint *worldFitness) {
    worldFitness[globalThreadIndex()] = 0;
}

__global__
void KernelStatesResetOverPop(uint **states) {
    #pragma unroll
    for (int i(0); i<NUM_STATES; i++) states[globalThreadIndex()][i]=0;
}

__global__
void KernelIDResetOverPop(uint *ID) {
   ID[globalThreadIndex()] = globalThreadIndex();
}

__global__
void KernelPredictionWorldReset(uint *worldState) {
    worldState[globalThreadIndex()] = 0;
}

volatile sig_atomic_t userExitFlag = 0;
void catchCtrlC(int signalID) {
  if (userExitFlag==1) {
      printf("Early termination requested. Results may be incomplete.\n");
		cudaDeviceReset();
      raise(SIGTERM);
  }
  userExitFlag = 1;
  printf("\nQuitting after current update. (ctrl-c again to force quit)\n");
}

// Using this kernel to debug some memory weirdness with 2d array passing cpu<->device
__global__
void KernelArrayTest(uint **s, uint *totals) {
    totals[globalThreadIndex()] = 0;
    for (int i(0); i<10; i++) { totals[globalThreadIndex()]+=s[globalThreadIndex()][i]; }
}

#define TIMERS_ENABLED

inline void timerOn(Timer &timer) {
#ifdef TIMERS_ENABLED
    timer.start();
#endif
}
inline void timerOff(Timer &timer) {
#ifdef TIMERS_ENABLED
    timer.stop();
#endif
}

__global__
void testInitKernel(int *dflatdata) {
	auto id = globalThreadIndex();
	dflatdata[id] = id+1;
}

__global__
void testZeroKernel(int *dflatdata) {
	auto id = globalThreadIndex();
	dflatdata[id] = 0;
}

int main(int argc, char* argv[]) {
    Timer timerMain,
          timerResets,
          timerUpdateWorld,
          timerEvalWorld,
          timerUpdateGates,
          timerSort,
          timerMoran;

    signal(SIGINT, catchCtrlC);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("capturing device #%d\n",deviceCount);
	cudaSetDevice(deviceCount-1); // capture first device (change this on a shared testing node w/ multiple cards)

	//// DEBUGGING CODE
	//POP(popnums, int, 3, 2);
	//testInitKernel<<<2,3>>>(popnums_d);
	//cudaDeviceSynchronize();
	//popnums[0].memcpyD2H();
	//popnums[1].memcpyD2H();
	//printf("%d %d %d %d %d %d\n",popnums[0].host()[0],popnums[0].host()[1],popnums[0].host()[2],popnums[1].host()[0],popnums[1].host()[1],popnums[1].host()[2]);
	//for (int i=0; i<3; i++) for (int p=0; p<2; p++) popnums[p].host()[i]+=1;
	//popnums[0].memcpyH2D();
	//popnums[1].memcpyH2D();
	//testZeroKernel<<<2,3>>>(popnums_d);
	//cudaDeviceSynchronize();
	//popnums[0].memcpyD2H();
	//popnums[1].memcpyD2H();
	//printf("%d %d %d %d %d %d\n",popnums[0].host()[0],popnums[0].host()[1],popnums[0].host()[2],popnums[1].host()[0],popnums[1].host()[1],popnums[1].host()[2]);
	//exit(0);
	
	//// DEBUGGING CODE
	////int hdata1[3], hdata2[3];
	//ccu::array<int> data1(3), data2(3);
	//int hflatdata[6];
	//int *dflatdata; cudaMalloc( (void**)&dflatdata, sizeof(int)*6 );
	//data1.setDeviceData(dflatdata); data2.setDeviceData(dflatdata+3);
	//testInitKernel<<<1,6>>>(dflatdata); cudaShowErrors();
	//cudaDeviceSynchronize(); cudaShowErrors();
	////cudaMemcpy(hflatdata, dflatdata, sizeof(int)*6, cudaMemcpyDeviceToHost); cudaShowErrors();
	////cudaMemcpy(hdata1, dflatdata, sizeof(int)*3, cudaMemcpyDeviceToHost); cudaShowErrors();
	////cudaMemcpy(hdata2, dflatdata+3, sizeof(int)*3, cudaMemcpyDeviceToHost); cudaShowErrors();
	////printf("%d %d %d %d %d %d\n",hflatdata[0],hflatdata[1],hflatdata[2],hflatdata[3],hflatdata[4],hflatdata[5]);
	////printf("%d %d %d\n",hdata1[0],hdata1[1],hdata1[2]);
	////printf("%d %d %d\n",hdata2[0],hdata2[1],hdata2[2]);
	//data1.memcpyD2H(); data2.memcpyD2H();
	//printf("%d %d %d\n", data1.host()[0], data1.host()[1], data1.host()[2]);
	//printf("%d %d %d\n", data2.host()[0], data2.host()[1], data2.host()[2]);
	//printf("ran okay\n");
	//exit(0);


	const unsigned int BLOCK_DIM(100); // ideally: 128 or 256
	const unsigned int THREADS(BLOCK_DIM*20); /// how many threads (gates) the device can run concurrently. ideally scale up to 2496 on a K20
	const unsigned int POPSIZE(100);
	
    //CommandLineArgs args(argc, argv); // used for cub library
    //CubDebugExit(args.DeviceInit()); // used for cub library
    //// for cub library radix testing
    //ccu::array<uint> buffer(10);
    //std::iota( buffer.host(), buffer.host()+buffer.size(), 0 );
    //exit(0); // TODO paste radix test code here for reverse-sort

    // make vectors of size POPSIZE of type ccu::array<uint8> of size THREADS, for example.
	 POP(popGateType,			uint8,	THREADS,		POPSIZE);
	 POP(popNumIns,			uint8,	THREADS,		POPSIZE); // name, type, size per individual, num of individuals
	 POP(popNumOuts,			uint8,	THREADS,		POPSIZE);
	 POP(popInsList,			uint,	THREADS*4,	POPSIZE);
	 POP(popOutsList,		uint,	THREADS*4,	POPSIZE);
	 POP(popProbData,		uint8,	THREADS*PGDIM*PGDIM,	POPSIZE);
	 POP(popProbMarginals,	uint,	THREADS*PGDIM,			POPSIZE);
	 // TODO: account for marginal change when mutating
	 POP(popDetData,			uint8,	THREADS*DGDIM1*DGDIM2, POPSIZE);
	 POP_LITE(popStatest,			uint,	NUM_STATES, POPSIZE);
	 POP_LITE(popStatest1,		uint,    NUM_STATES, POPSIZE);
	 //POP(popPredictionWorldState, uint,	THREADS, POPSIZE); // Prediction Fitness Fn requires only 1 int state
	 //POP(popMNISTWorldState, uint,	THREADS*2, POPSIZE); // MNIST Fitness Fn requires x,y int states (roving eye)

    ccu::array<curandState> randstates(THREADS);
    ccu::array<uint> predictionWorldState(POPSIZE*1); // only 1 number for state in this world fn
    ccu::array<uint> maxOnesWorldState(POPSIZE*1); // only 1 number for state in this world fn
    //ccu::array<uint> MNISTWorldState(POPSIZE*2,false); // MNIST roving eye requires x and y states (thus `*2`)
    //ccu::array<uint> AsteroidWorldState(POPSIZE*?,false); // Unfinished, likely to be x,y,z
	 ccu::array<uint> fitness(POPSIZE);
	 ccu::array<uint> fitnessSorted(POPSIZE);
	 ccu::array<uint> ID(POPSIZE);
	 ccu::array<uint> IDSorted(POPSIZE);
    ccu::array<uint> indices(POPSIZE);

    ccu::array<uint*> popStatestPtrs(POPSIZE);
    ccu::array<uint*> popStatest1Ptrs(POPSIZE);
    for (int i(0); i<POPSIZE; i++) {
       // store device-pointers to each organism state buffer in a passable array
       // so that we can operate on the entire population using a kernel
       popStatestPtrs.host()[i] = popStatest[i].device();
       popStatest1Ptrs.host()[i] = popStatest1[i].device();
       popStatest[i].sync();
       popStatest1[i].sync();
    }

	 ExecutionPolicy epOrganism;
	 epOrganism.blockSize = BLOCK_DIM;
	 epOrganism.gridSize = THREADS/BLOCK_DIM;
	 ExecutionPolicy epPopulation;
	 epPopulation.blockSize = POPSIZE;
	 epPopulation.gridSize = 1;

	 // Determine temporary device storage requirements
	 void     *d_temp_storage = NULL;
	 size_t   temp_storage_bytes = 0;
	 cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
	 			fitness.device(), fitnessSorted.device(),
	 			ID.device(), IDSorted.device(), POPSIZE);
	 // Allocate temporary storage
	 cudaMalloc(&d_temp_storage, temp_storage_bytes);

	 launch(epOrganism,KernelRandInit,randstates.device(),0);
	 cudaDeviceSynchronize();
	 for (auto agenti(0); agenti<POPSIZE; agenti++) {
	     launch(epOrganism,KernelInitGates,
	   				    randstates.device(),
	   				    popNumIns[agenti].device(),
	   				    popNumOuts[agenti].device(),
	   				    popInsList[agenti].device(),
	   				    popOutsList[agenti].device(),
	   				    popGateType[agenti].device(),
	   				    popProbData[agenti].device(),
	   				    popProbMarginals[agenti].device(),
	   				    popDetData[agenti].device()
	   				    );
    }
	 cudaDeviceSynchronize();
     // commented code is for benchmarking
	 //showElapsedTime("update loop", [&] {
        //for (int updates(0); updates<2; updates++) {
        timerOn(timerMain);
        uint update(0);
        while(userExitFlag == false) {
            timerOn(timerResets);
            launch(epPopulation,KernelFitnessResetOverPop,fitness.device());
            //for (int worldState(0); worldState<16; worldState++) { // try each of the 0-15 numbers 4 times (for prediction world)
            for (int worldState(0); worldState<1; worldState++) { // do only once (for max ones world)
                timerOn(timerResets);
                launch(epPopulation,KernelIDResetOverPop,ID.device());
                launch(epPopulation,KernelStatesResetOverPop,popStatestPtrs.device());
                launch(epPopulation,KernelStatesResetOverPop,popStatest1Ptrs.device());
                launch(epPopulation,KernelPredictionWorldReset,predictionWorldState.device());
                cudaDeviceSynchronize();
                timerOff(timerResets);
                timerOn(timerUpdateWorld);
                //launch(epPopulation,KernelPredictionWorldUpdateOverPop,
                //              predictionWorldState.device(),
                //              popStatestPtrs.device(), 
                //              popStatest1Ptrs.device()
                //              );
                launch(epPopulation,KernelMaxOnesWorldUpdateOverPop,
                              maxOnesWorldState.device(),
                              popStatestPtrs.device(), 
                              popStatest1Ptrs.device()
                              );
					 cudaDeviceSynchronize();
                timerOff(timerUpdateWorld);
                timerOn(timerUpdateGates);
                for (int networkUpdates(0); networkUpdates<1; networkUpdates++) {
                    for (int agenti(0); agenti<POPSIZE; agenti++) {
                        launch(epOrganism,KernelUpdateProbGates,
                                randstates.device(),
                                popNumIns[agenti].device(),
                                popNumOuts[agenti].device(),
                                popInsList[agenti].device(),
                                popOutsList[agenti].device(),
                                popGateType[agenti].device(),
                                popProbData[agenti].device(),
                                popProbMarginals[agenti].device(),
                                popStatest[agenti].device(),
                                popStatest1[agenti].device()
                              );
                        launch(epOrganism,KernelUpdateDetGates,
                                randstates.device(),
                                popNumIns[agenti].device(),
                                popNumOuts[agenti].device(),
                                popInsList[agenti].device(),
                                popOutsList[agenti].device(),
                                popGateType[agenti].device(),
                                popDetData[agenti].device(),
                                popStatest[agenti].device(),
                                popStatest1[agenti].device()
                              );
                        cudaDeviceSynchronize();
                        std::swap(popStatest[agenti],popStatest1[agenti]); 

                    } // end of individual level
                    std::swap(popStatestPtrs,popStatest1Ptrs);
                }
                timerOff(timerUpdateGates);
                // back to the population level
                timerOn(timerEvalWorld);
                //launch(epPopulation, KernelPredictionWorldEvaluateOverPop,
                //              predictionWorldState.device(),
                //              popStatestPtrs.device(),
                //              fitness.device()
                //              );
                launch(epPopulation,KernelMaxOnesWorldEvaluateOverPop,
                              predictionWorldState.device(),
                              popStatestPtrs.device(),
                              fitness.device()
                              );
                cudaDeviceSynchronize();
                timerOff(timerEvalWorld);
            }
            //} // end of world updates level
            // THIS BLOCK SHOULD NOT BE ON THE HOST (I can't get cub::radixSort to output anything but 0's...)
            // first, randomize the arrays, maintaining pairness
            // (randomizing to avoid a specific bias in a later step)
				// TODO: need to figure out correct way to randomize both arrays the same way
            //std::copy(ID.host(), ID.host()+POPSIZE, indices.host());
            //std::random_shuffle(indices.host(), indices.host()+POPSIZE);
            //std::sort(ID.host(), ID.host()+POPSIZE, [&] (uint &a, uint &b){return indices.host()[a]<indices.host()[b];});
            //std::sort(fitness.host(), fitness.host()+POPSIZE, [&] (uint &a, uint &b){return indices.host()[a]<indices.host()[b];});

            // sort ID by fitness
            //timerOn(timerSort);
            //std::sort(ID.host(), ID.host()+POPSIZE, [&] (uint &a, uint &b){return fitness.host()[a]>fitness.host()[b];});
            //std::sort(fitness.host(), fitness.host()+POPSIZE, [&] (uint &a, uint &b){return a>b;});
            //timerOff(timerSort);

				// sort ID by fitness (GPU VERSION)
            timerOn(timerSort);
				cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
							fitness.device(), fitnessSorted.device(),
							ID.device(), IDSorted.device(), POPSIZE);
				cudaShowErrors();
            timerOff(timerSort);

            timerOff(timerMain);
            ///////////////////////////
            // DISPLAY FITNESS PROGRESS
            ///////////////////////////
				//printf("\n[%u] (%x,%x) ",update,popStatest[0].device(),popStatest1[0].device());
				if ((update&0b11111111) == 0b11111111) {
					 printf("[%u] ",update);
				    for (uint p(0); p<POPSIZE; p++) {
				  		printf("%u ",fitness.host()[p]);
				    }
				    printf("\n");
				}
				//printf("\n");
            timerOn(timerMain);

            timerOn(timerMoran);
            unsigned int newParentIDPos(0);
            //for (int i(0); i<POPSIZE; i++) {
            //   //printf("host [%d] device [%d]\n",popGateType[ID.host()[i]].host(),popGateType[ID.host()[i]].device());
            //}
				uint limit(POPSIZE-0.1*POPSIZE);
            #pragma unroll
            for (unsigned int dyingAgenti(limit); dyingAgenti<POPSIZE; dyingAgenti++) {
               newParentIDPos = static_cast<unsigned int>( (rand()/(float)RAND_MAX * rand()/(float)RAND_MAX * rand()/(float)RAND_MAX * rand()/(float)RAND_MAX * rand()/(float)RAND_MAX) * (limit-1) ); // for scale up to popsize=100
					launch(epOrganism, KernelMoranProcess2,
							randstates.device(),
							dyingAgenti, newParentIDPos, // unsigned int
							IDSorted.device(), // uint
							THREADS, // unsigned int
							popNumIns_d, popNumOuts_d, // uint8
							popInsList_d, popOutsList_d, // uint8
							popGateType_d, popDetData_d, popProbData_d); // uint8

               //launch(epOrganism, KernelMoranProcess,
               //       andstates.device(),
               //       popNumIns[IDSorted.device()[dyingAgenti]].device(), // dying organism section start // uint8
               //       popNumOuts[IDSorted.device()[dyingAgenti]].device(), // uint8

               //       popInsList[IDSorted.device()[dyingAgenti]].device(), // uint
               //       popOutsList[IDSorted.device()[dyingAgenti]].device(), // uint

               //       popGateType[IDSorted.device()[dyingAgenti]].device(), // uint8
               //       popDetData[IDSorted.device()[dyingAgenti]].device(), // uint8
               //       popProbData[IDSorted.device()[dyingAgenti]].device(), // uint8

               //       popNumIns[IDSorted.device()[newParentIDPos]].device(), // new parent data section start // uint8
               //       popNumOuts[IDSorted.device()[newParentIDPos]].device(), // uint8

               //       popInsList[IDSorted.device()[newParentIDPos]].device(), // uint
               //       popOutsList[IDSorted.device()[newParentIDPos]].device(), // uint

               //       popGateType[IDSorted.device()[newParentIDPos]].device(), // uint8
               //       popDetData[IDSorted.device()[newParentIDPos]].device(), // uint8
               //       popProbData[IDSorted.device()[newParentIDPos]].device()); // uint8

					// skip copying from ID.host to IDSorted.host and just use ID.host
               //       randstates.device(),
               //       popNumIns[ID.host()[dyingAgenti]].device(), // dying organism section start // uint8
               //       popNumOuts[ID.host()[dyingAgenti]].device(), // uint8

               //       popInsList[ID.host()[dyingAgenti]].device(), // uint
               //       popOutsList[ID.host()[dyingAgenti]].device(), // uint

               //       popGateType[ID.host()[dyingAgenti]].device(), // uint8
               //       popDetData[ID.host()[dyingAgenti]].device(), // uint8
               //       popProbData[ID.host()[dyingAgenti]].device(), // uint8

               //       popNumIns[ID.host()[newParentIDPos]].device(), // new parent data section start // uint8
               //       popNumOuts[ID.host()[newParentIDPos]].device(), // uint8

               //       popInsList[ID.host()[newParentIDPos]].device(), // uint
               //       popOutsList[ID.host()[newParentIDPos]].device(), // uint

               //       popGateType[ID.host()[newParentIDPos]].device(), // uint8
               //       popDetData[ID.host()[newParentIDPos]].device(), // uint8
               //       popProbData[ID.host()[newParentIDPos]].device()); // uint8
            }
            cudaDeviceSynchronize();
            timerOff(timerMoran);
            //auto minmax = std::minmax_element(fitnessSorted.host(), fitnessSorted.host()+POPSIZE);
            //printf("[ %d %d ]\n",*minmax.first, *minmax.second);
            //auto highestFitLoc = std::distance(fitnessSorted.host(),minmax.second);
            //auto highestFitID = IDSorted.host()[highestFitLoc];
            ++update;
        }
    timerOff(timerMain);
    //});
	 cudaShowErrors();
	 cudaDeviceReset();

#ifdef TIMERS_ENABLED
     printf("\n");
     printf("main: %.3f\n",timerMain.getTime());
     printf("resets: %.3f\n",timerResets.getTime());
     printf("worldUpdate: %.3f\n",timerUpdateWorld.getTime());
     printf("worldEval: %.3f\n",timerEvalWorld.getTime());
     printf("gates: %.3f\n",timerUpdateGates.getTime());
     printf("sort: %.3f\n",timerSort.getTime());
     printf("moran: %.3f\n",timerMoran.getTime());
#endif
    return 0;
}
