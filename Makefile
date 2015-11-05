CFLAGS=-DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_DEBUG -O3 \
			 -fopenmp -Wall -funroll-loops -fomit-frame-pointer \
			 -flto -fwhole-program
SOURCE=src/fastLDA.cpp src/LDA.cpp 

all: src/fastLDA.cpp src/LDA.cpp
	g++ -o fastLDA $(CFLAGS) -I include -L lib $(SOURCE)

clean: 
	rm fastLDA
