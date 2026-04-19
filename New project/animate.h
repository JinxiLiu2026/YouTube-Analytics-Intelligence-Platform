#ifndef ANIMATE_H
#define ANIMATE_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

struct sprite;
struct canvas;
struct sprite_placement;

typedef uint32_t color_t;

typedef void (*animate_fn)(void* priv, ssize_t* x, ssize_t* y, float t);

static inline color_t animate_color_rgb(unsigned r, unsigned g, unsigned b) {
    return ((r & 0xff) << 16) | ((g & 0xff) << 8) | ((b & 0xff) << 0);
}

static inline color_t animate_color_argb(unsigned a,
                                         unsigned r, unsigned g, unsigned b) {
    return animate_color_rgb(r, g, b) | ((a & 0xff) << 24);
}

struct canvas* animate_create_canvas(size_t height, size_t width,
                                     color_t background_color);

void animate_destroy_canvas(struct canvas* canvas);

struct sprite* animate_create_sprite(const char* file);

struct sprite* animate_create_rectangle(size_t width, size_t height, color_t c,
                                        bool filled);

struct sprite* animate_create_circle(size_t radius, color_t c, bool filled);

bool animate_destroy_sprite(struct sprite* sprite);

struct sprite_placement* animate_place_sprite(struct canvas* canvas,
                                              struct sprite* sprite,
                                              ssize_t x, ssize_t y);

void animate_placement_up(struct sprite_placement* sprite_placement);

void animate_placement_down(struct sprite_placement* sprite_placement);

void animate_placement_top(struct sprite_placement* sprite_placement);

void animate_placement_bottom(struct sprite_placement* sprite_placement);

void animate_destroy_placement(struct sprite_placement* sprite_placement);

void animate_set_animation_params(struct sprite_placement* sprite_placement,
                                  ssize_t vx, ssize_t vy,
                                  ssize_t ax, ssize_t ay);

void animate_set_animation_function(struct sprite_placement* sprite_placement,
                                    animate_fn fn, void* priv);

size_t animate_frame_size_bytes(struct canvas* canvas);

void animate_generate_frame(const struct canvas* canvas, size_t frame,
                            size_t frame_rate, void* buf);

#endif
