#include "animate.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OUTPUT_FILE "simple.dat"

int main(void) {
    struct canvas* canvas;
    struct sprite* rect;
    struct sprite_placement* prect1;
    struct sprite_placement* prect2;
    size_t frame_size_bytes;
    void* data;
    FILE* fp;
    size_t bytes_written;

    canvas = animate_create_canvas(10, 10, animate_color_argb(0, 0, 0, 0));
    rect = animate_create_rectangle(3, 6, animate_color_argb(0, 255, 255, 0), true);

    if (canvas == NULL || rect == NULL) {
        fprintf(stderr, "failed to create canvas or sprite\n");
        animate_destroy_canvas(canvas);
        animate_destroy_sprite(rect);
        return 1;
    }

    prect1 = animate_place_sprite(canvas, rect, 0, 0);
    prect2 = animate_place_sprite(canvas, rect, 2, 1);
    if (prect1 == NULL || prect2 == NULL) {
        fprintf(stderr, "failed to place sprite\n");
        animate_destroy_canvas(canvas);
        animate_destroy_sprite(rect);
        return 1;
    }

    frame_size_bytes = animate_frame_size_bytes(canvas);
    data = malloc(frame_size_bytes);
    if (data == NULL) {
        fprintf(stderr, "failed to allocate frame buffer\n");
        animate_destroy_canvas(canvas);
        animate_destroy_sprite(rect);
        return 1;
    }

    animate_generate_frame(canvas, 1, 25, data);

    fp = fopen(OUTPUT_FILE, "wb");
    if (fp == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", OUTPUT_FILE, strerror(errno));
        free(data);
        animate_destroy_canvas(canvas);
        animate_destroy_sprite(rect);
        return 1;
    }

    bytes_written = fwrite(data, 1, frame_size_bytes, fp);
    if (bytes_written != frame_size_bytes) {
        fprintf(stderr, "failed to write buffer (%zu/%zu): %s\n",
                bytes_written, frame_size_bytes, strerror(errno));
        fclose(fp);
        free(data);
        animate_destroy_canvas(canvas);
        animate_destroy_sprite(rect);
        return 1;
    }

    fclose(fp);
    free(data);
    animate_destroy_canvas(canvas);
    animate_destroy_sprite(rect);
    return 0;
}
