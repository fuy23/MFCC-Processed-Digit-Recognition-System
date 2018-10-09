main: main.o
	gcc -Wall -L/usr/local/lib -std=c11 -o main main.o -lgsl -lgslcblas -lm
main.o: main.c
	gcc -Wall -I/usr/local/include -c main.c
clean:
	rm *.o