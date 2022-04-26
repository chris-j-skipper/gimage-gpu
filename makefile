#------------------------------------------------------------------------------

# Update this constant to the casacore folder path.
CASACORE=/home/chris/casacore-2.0.3

# Nothing after this point should have to be altered.
MYPROGRAM=cuGridder_polarisation
CODEFOLDER=/home/chris/Dropbox/programming/gridding

PHASECORRECTION=cppPhaseCorrection
CASACOREINTERFACE=cppCasacoreInterface
REPROJECTION=cuReprojectionNotComplex
DATA=cuData_polarisation
FUNCTIONS=cuFunctions
KERNELSET=cuKernelSet
KERNELCACHE=cuKernelCache
PARAMETERS=cuParameters

# the libraries themselves.
CASA=$(CASACORE)/build/casa/libcasa_casa.so
LATTICES=$(CASACORE)/build/lattices/libcasa_lattices.so
COORDINATES=$(CASACORE)/build/coordinates/libcasa_coordinates.so
IMAGES=$(CASACORE)/build/images/libcasa_images.so
TABLES=$(CASACORE)/build/tables/libcasa_tables.so

FORTRAN=/usr/local/casa-release-4.7.2-el6/lib/libgfortran.so.3
CC=g++
NVCC=nvcc

#------------------------------------------------------------------------------

all: $(MYPROGRAM)

$(MYPROGRAM): $(MYPROGRAM).o $(DATA).o $(FUNCTIONS).o $(REPROJECTION).o $(KERNELSET).o $(KERNELCACHE).o $(PARAMETERS).o link.o $(PHASECORRECTION).o $(CASACOREINTERFACE).o $(CASA) $(LATTICES) $(COORDINATES) $(IMAGES) $(TABLES)
	$(CC) $^ -l cudart -l cufft -l fftw3f -L /usr/local/cuda/lib64 -o $@

# place: -fsanitize=address	before the -o in the line above to get lots of extra information about seg faults
# and memory leaks.

link.o: $(MYPROGRAM).o $(FUNCTIONS).o $(DATA).o $(REPROJECTION).o $(KERNELSET).o $(KERNELCACHE).o $(PARAMETERS).o
	$(NVCC) -arch=sm_35 -dlink -o $@ $^

$(MYPROGRAM).o: $(CODEFOLDER)/$(MYPROGRAM).cu
	$(NVCC) -arch=sm_35 -c $^ -D__INTEL_COMPILER -o $@ -rdc=true

$(FUNCTIONS).o: $(CODEFOLDER)/$(FUNCTIONS).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

$(DATA).o: $(CODEFOLDER)/$(DATA).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

$(KERNELSET).o: $(CODEFOLDER)/$(KERNELSET).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

$(KERNELCACHE).o: $(CODEFOLDER)/$(KERNELCACHE).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

$(PARAMETERS).o: $(CODEFOLDER)/$(PARAMETERS).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

#$(REPROJECTION)_link.o: $(REPROJECTION).o
#        $(NVCC) -arch=sm_35 -dlink -o $@ $^

$(REPROJECTION).o: $(CODEFOLDER)/$(REPROJECTION).cu
	$(NVCC) -arch=sm_35 -c $^ -o $@ -rdc=true

$(PHASECORRECTION).o: $(CODEFOLDER)/$(PHASECORRECTION).cpp
	$(CC) -c $^ -o $@

$(CASACOREINTERFACE).o: $(CODEFOLDER)/$(CASACOREINTERFACE).cpp
	$(CC) -c $^ -o $@

clean:
	rm -f $(MYPROGRAM)
