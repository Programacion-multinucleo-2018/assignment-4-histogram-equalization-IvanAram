C_C = g++
CUDA_C = nvcc

CFLAGS = -std=c++11 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

EXE1 = bin/restoreImage
EXE2 = bin/restoreImageCuda

PROG1 = restoreImage.cpp
PROG2 = restoreImage.cu

all:
	$(C_C) -o $(EXE1) $(PROG1) $(CFLAGS)
	$(CUDA_C) -o $(EXE2) $(PROG2) $(CFLAGS)

rebuild: clean all

clean:
	rm -f ./bin/*
