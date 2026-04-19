CC = cc
CFLAGS = -std=c11 -Wall -Wextra -Werror -pedantic

.PHONY: all clean

all: main_simple

main_simple: animate.o main_simple.o
	$(CC) $(CFLAGS) -o $@ animate.o main_simple.o

animate.o: animate.c animate.h
	$(CC) $(CFLAGS) -c animate.c

main_simple.o: main_simple.c animate.h
	$(CC) $(CFLAGS) -c main_simple.c

clean:
	rm -f main_simple *.o simple.dat
