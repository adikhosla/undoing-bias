all: learn

learn: internal/learn.cc
	g++ -O3 -w -o internal/learn internal/learn.cc

clean:
	rm internal/learn
