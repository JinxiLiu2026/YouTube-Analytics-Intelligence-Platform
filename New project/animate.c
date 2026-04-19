#include "animate.h"

#include <stdio.h>
#include <stdlib.h>

struct sprite {
    size_t width;
    size_t height;
    color_t* pixels;
    bool* mask;
    size_t refcount;
};

struct sprite_placement {
    struct canvas* canvas;
    struct sprite* sprite;
    ssize_t x;
    ssize_t y;
    ssize_t vx;
    ssize_t vy;
    ssize_t ax;
    ssize_t ay;
    animate_fn fn;
    void* priv;
    struct sprite_placement* prev;
    struct sprite_placement* next;
};

struct canvas {
    size_t height;
    size_t width;
    color_t background_color;
    struct sprite_placement* bottom;
    struct sprite_placement* top;
};

struct bitmap_header {
    uint8_t magic[2];
    uint32_t size_bytes;
    uint16_t reserved[2];
    uint32_t pixel_offset;
} __attribute__((packed));

struct bitmapv5_header {
    uint32_t bV5Size;
    uint32_t bV5Width;
    uint32_t bV5Height;
    uint16_t bV5Planes;
    uint16_t bV5BitCount;
    uint32_t bV5Compression;
    uint32_t bV5SizeImage;
    uint32_t bV5XPelsPerMeter;
    uint32_t bV5YPelsPerMeter;
    uint32_t bV5ClrUsed;
    uint32_t bV5ClrImportant;
    uint32_t bV5RedMask;
    uint32_t bV5GreenMask;
    uint32_t bV5BlueMask;
    uint32_t bV5AlphaMask;
    uint32_t bV5CSType;
    uint8_t bV5Endpoints[36];
    uint32_t bV5GammaRed;
    uint32_t bV5GammaGreen;
    uint32_t bV5GammaBlue;
    uint32_t bV5Intent;
    uint32_t bV5ProfileData;
    uint32_t bV5ProfileSize;
    uint32_t bV5Reserved;
};

static bool mul_overflow(size_t a, size_t b, size_t* out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static struct sprite* sprite_alloc(size_t width, size_t height) {
    size_t pixel_count;
    struct sprite* sprite;

    if (width == 0 || height == 0) {
        return NULL;
    }
    if (mul_overflow(width, height, &pixel_count)) {
        return NULL;
    }

    sprite = calloc(1, sizeof(*sprite));
    if (sprite == NULL) {
        return NULL;
    }

    sprite->pixels = calloc(pixel_count, sizeof(*sprite->pixels));
    sprite->mask = calloc(pixel_count, sizeof(*sprite->mask));
    if (sprite->pixels == NULL || sprite->mask == NULL) {
        free(sprite->pixels);
        free(sprite->mask);
        free(sprite);
        return NULL;
    }

    sprite->width = width;
    sprite->height = height;
    return sprite;
}

static void sprite_free(struct sprite* sprite) {
    if (sprite == NULL) {
        return;
    }
    free(sprite->pixels);
    free(sprite->mask);
    free(sprite);
}

static void detach_placement(struct sprite_placement* placement) {
    struct canvas* canvas = placement->canvas;

    if (placement->prev != NULL) {
        placement->prev->next = placement->next;
    } else {
        canvas->bottom = placement->next;
    }

    if (placement->next != NULL) {
        placement->next->prev = placement->prev;
    } else {
        canvas->top = placement->prev;
    }

    placement->prev = NULL;
    placement->next = NULL;
}

static void append_top(struct canvas* canvas, struct sprite_placement* placement) {
    placement->prev = canvas->top;
    placement->next = NULL;

    if (canvas->top != NULL) {
        canvas->top->next = placement;
    } else {
        canvas->bottom = placement;
    }

    canvas->top = placement;
}

static void prepend_bottom(struct canvas* canvas, struct sprite_placement* placement) {
    placement->prev = NULL;
    placement->next = canvas->bottom;

    if (canvas->bottom != NULL) {
        canvas->bottom->prev = placement;
    } else {
        canvas->top = placement;
    }

    canvas->bottom = placement;
}

static void swap_with_next(struct sprite_placement* placement) {
    struct canvas* canvas;
    struct sprite_placement* next;
    struct sprite_placement* before;
    struct sprite_placement* after;

    if (placement == NULL || placement->next == NULL) {
        return;
    }

    canvas = placement->canvas;
    next = placement->next;
    before = placement->prev;
    after = next->next;

    if (before != NULL) {
        before->next = next;
    } else {
        canvas->bottom = next;
    }

    next->prev = before;
    next->next = placement;

    placement->prev = next;
    placement->next = after;

    if (after != NULL) {
        after->prev = placement;
    } else {
        canvas->top = placement;
    }
}

static ssize_t animated_axis(ssize_t start, ssize_t v, ssize_t a, double t) {
    double position = (double)start + (double)v * t + 0.5 * (double)a * t * t;
    return (ssize_t)position;
}

struct canvas* animate_create_canvas(size_t height, size_t width,
                                     color_t background_color) {
    struct canvas* canvas = calloc(1, sizeof(*canvas));

    if (canvas == NULL) {
        return NULL;
    }

    canvas->height = height;
    canvas->width = width;
    canvas->background_color = background_color;
    return canvas;
}

struct sprite* animate_create_sprite(const char* file) {
    FILE* fp;
    struct bitmap_header bmp_header;
    struct bitmapv5_header dib_header;
    struct sprite* sprite;
    size_t y;

    if (file == NULL) {
        return NULL;
    }

    fp = fopen(file, "rb");
    if (fp == NULL) {
        return NULL;
    }

    if (fread(&bmp_header, sizeof(bmp_header), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    if (bmp_header.magic[0] != 'B' || bmp_header.magic[1] != 'M') {
        fclose(fp);
        return NULL;
    }

    if (fread(&dib_header, sizeof(dib_header), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    if (dib_header.bV5Width == 0 || dib_header.bV5Height == 0) {
        fclose(fp);
        return NULL;
    }

    if (dib_header.bV5BitCount != 32) {
        fclose(fp);
        return NULL;
    }

    sprite = sprite_alloc(dib_header.bV5Width, dib_header.bV5Height);
    if (sprite == NULL) {
        fclose(fp);
        return NULL;
    }

    if (fseek(fp, (long)bmp_header.pixel_offset, SEEK_SET) != 0) {
        sprite_free(sprite);
        fclose(fp);
        return NULL;
    }

    for (y = 0; y < sprite->height; y++) {
        size_t target_row = sprite->height - 1 - y;
        size_t row_start = target_row * sprite->width;
        size_t x;

        if (fread(&sprite->pixels[row_start], sizeof(color_t), sprite->width, fp)
            != sprite->width) {
            sprite_free(sprite);
            fclose(fp);
            return NULL;
        }

        for (x = 0; x < sprite->width; x++) {
            sprite->mask[row_start + x] = true;
        }
    }

    fclose(fp);
    return sprite;
}

struct sprite* animate_create_circle(size_t radius, color_t c, bool filled) {
    size_t diameter = radius * 2 + 1;
    struct sprite* sprite = sprite_alloc(diameter, diameter);
    size_t x;
    size_t y;

    (void)filled;

    if (sprite == NULL) {
        return NULL;
    }

    for (y = 0; y < diameter; y++) {
        for (x = 0; x < diameter; x++) {
            ssize_t dx = (ssize_t)x - (ssize_t)radius;
            ssize_t dy = (ssize_t)y - (ssize_t)radius;
            size_t idx = y * diameter + x;

            if (dx * dx + dy * dy <= (ssize_t)(radius * radius)) {
                sprite->pixels[idx] = c;
                sprite->mask[idx] = true;
            }
        }
    }

    return sprite;
}

struct sprite* animate_create_rectangle(size_t width, size_t height,
                                        color_t c, bool filled) {
    struct sprite* sprite = sprite_alloc(width, height);
    size_t x;
    size_t y;

    if (sprite == NULL) {
        return NULL;
    }

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            bool draw = filled || x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            size_t idx = y * width + x;

            if (draw) {
                sprite->pixels[idx] = c;
                sprite->mask[idx] = true;
            }
        }
    }

    return sprite;
}

bool animate_destroy_sprite(struct sprite* sprite) {
    if (sprite == NULL) {
        return false;
    }

    if (sprite->refcount != 0) {
        return false;
    }

    sprite_free(sprite);
    return true;
}

struct sprite_placement* animate_place_sprite(struct canvas* canvas,
                                              struct sprite* sprite,
                                              ssize_t x, ssize_t y) {
    struct sprite_placement* placement;

    if (canvas == NULL || sprite == NULL) {
        return NULL;
    }

    placement = calloc(1, sizeof(*placement));
    if (placement == NULL) {
        return NULL;
    }

    placement->canvas = canvas;
    placement->sprite = sprite;
    placement->x = x;
    placement->y = y;

    append_top(canvas, placement);
    sprite->refcount++;

    return placement;
}

void animate_placement_up(struct sprite_placement* sprite_placement) {
    swap_with_next(sprite_placement);
}

void animate_placement_down(struct sprite_placement* sprite_placement) {
    if (sprite_placement != NULL && sprite_placement->prev != NULL) {
        swap_with_next(sprite_placement->prev);
    }
}

void animate_placement_top(struct sprite_placement* sprite_placement) {
    struct canvas* canvas;

    if (sprite_placement == NULL || sprite_placement->canvas == NULL) {
        return;
    }

    canvas = sprite_placement->canvas;
    if (canvas->top == sprite_placement) {
        return;
    }

    detach_placement(sprite_placement);
    append_top(canvas, sprite_placement);
}

void animate_placement_bottom(struct sprite_placement* sprite_placement) {
    struct canvas* canvas;

    if (sprite_placement == NULL || sprite_placement->canvas == NULL) {
        return;
    }

    canvas = sprite_placement->canvas;
    if (canvas->bottom == sprite_placement) {
        return;
    }

    detach_placement(sprite_placement);
    prepend_bottom(canvas, sprite_placement);
}

void animate_destroy_placement(struct sprite_placement* sprite_placement) {
    if (sprite_placement == NULL) {
        return;
    }

    if (sprite_placement->canvas != NULL) {
        detach_placement(sprite_placement);
    }

    if (sprite_placement->sprite != NULL && sprite_placement->sprite->refcount > 0) {
        sprite_placement->sprite->refcount--;
    }

    free(sprite_placement);
}

void animate_set_animation_params(struct sprite_placement* sprite_placement,
                                  ssize_t vx, ssize_t vy,
                                  ssize_t ax, ssize_t ay) {
    if (sprite_placement == NULL) {
        return;
    }

    sprite_placement->vx = vx;
    sprite_placement->vy = vy;
    sprite_placement->ax = ax;
    sprite_placement->ay = ay;
    sprite_placement->fn = NULL;
    sprite_placement->priv = NULL;
}

void animate_set_animation_function(struct sprite_placement* sprite_placement,
                                    animate_fn fn, void* priv) {
    if (sprite_placement == NULL) {
        return;
    }

    sprite_placement->fn = fn;
    sprite_placement->priv = priv;
}

void animate_destroy_canvas(struct canvas* canvas) {
    struct sprite_placement* placement;
    struct sprite_placement* next;

    if (canvas == NULL) {
        return;
    }

    placement = canvas->bottom;
    while (placement != NULL) {
        next = placement->next;
        if (placement->sprite != NULL && placement->sprite->refcount > 0) {
            placement->sprite->refcount--;
        }
        free(placement);
        placement = next;
    }

    free(canvas);
}

size_t animate_frame_size_bytes(struct canvas* canvas) {
    size_t pixel_count;

    if (canvas == NULL) {
        return 0;
    }

    if (mul_overflow(canvas->width, canvas->height, &pixel_count)) {
        return 0;
    }

    if (pixel_count > SIZE_MAX / sizeof(color_t)) {
        return 0;
    }

    return pixel_count * sizeof(color_t);
}

void animate_generate_frame(const struct canvas* canvas, size_t frame,
                            size_t frame_rate, void* buf) {
    color_t* out = buf;
    size_t pixel_count;
    double t = 0.0;
    struct sprite_placement* placement;

    if (canvas == NULL || buf == NULL) {
        return;
    }

    if (mul_overflow(canvas->width, canvas->height, &pixel_count)) {
        return;
    }

    for (size_t i = 0; i < pixel_count; i++) {
        out[i] = canvas->background_color;
    }

    if (frame_rate != 0) {
        t = (double)frame / (double)frame_rate;
    }

    for (placement = canvas->bottom; placement != NULL; placement = placement->next) {
        ssize_t draw_x = placement->x;
        ssize_t draw_y = placement->y;
        struct sprite* sprite = placement->sprite;

        if (sprite == NULL) {
            continue;
        }

        if (placement->fn != NULL) {
            placement->fn(placement->priv, &draw_x, &draw_y, (float)t);
        } else {
            draw_x = animated_axis(placement->x, placement->vx, placement->ax, t);
            draw_y = animated_axis(placement->y, placement->vy, placement->ay, t);
        }

        for (size_t sy = 0; sy < sprite->height; sy++) {
            for (size_t sx = 0; sx < sprite->width; sx++) {
                size_t sprite_idx = sy * sprite->width + sx;
                ssize_t cx;
                ssize_t cy;

                if (!sprite->mask[sprite_idx]) {
                    continue;
                }

                cx = draw_x + (ssize_t)sx;
                cy = draw_y + (ssize_t)sy;

                if (cx < 0 || cy < 0) {
                    continue;
                }
                if ((size_t)cx >= canvas->width || (size_t)cy >= canvas->height) {
                    continue;
                }

                out[(size_t)cy * canvas->width + (size_t)cx] = sprite->pixels[sprite_idx];
            }
        }
    }
}
