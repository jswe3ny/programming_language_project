# run as make -j20
CXX       := g++
CXXFLAGS  := -std=c++17 -Wall -Wextra -O3 -march=native -pipe -MMD -MP
LDFLAGS   := -fopenmp -Wl,--as-needed

SRC := program.cpp \
       utils/utils.cpp utils/metrics.cpp \
       algos/linear_regression.cpp algos/logistic_regression.cpp \
       algos/knn.cpp algos/decision_tree.cpp algos/naive_bayes.cpp

OBJ := $(SRC:.cpp=.o)
DEP := $(OBJ:.o=.d)

all: program

program: $(OBJ)
	$(CXX) $(OBJ) $(LDFLAGS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f program $(OBJ) $(DEP)

-include $(DEP)
