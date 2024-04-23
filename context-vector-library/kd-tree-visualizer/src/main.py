import random

import pygame

WIDTH = 1500
HEIGHT = 1500
MARGIN = 25

X_RANGE = (2 * MARGIN, WIDTH - 2 * MARGIN)
Y_RANGE = (2 * MARGIN, HEIGHT - 2 * MARGIN)
MOVE = 10


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    running = 1

    points = make_points()
    make_target = lambda: (
        random.randint(*X_RANGE),
        random.randint(*Y_RANGE),
    )
    target = make_target()

    kd_tree = make_kd_tree(points)
    state = {
        "kd_tree": kd_tree,
        "target": target,
        "root": kd_tree,
        "stack": [],
        "leaf": find_leaf(kd_tree, target),
        "closest": find_closest(kd_tree, target),
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = 0
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    if state["root"][0] is not None:
                        state["stack"].append(state["root"])
                        state["root"] = state["root"][0]
                elif event.key == pygame.K_RIGHT:
                    if state["root"][2] is not None:
                        state["stack"].append(state["root"])
                        state["root"] = state["root"][2]
                elif event.key == pygame.K_UP:
                    if len(state["stack"]) > 0:
                        state["root"] = state["stack"].pop()
                elif event.key == pygame.K_r:
                    points = make_points()
                    kd_tree = make_kd_tree(points)
                    assert kd_tree is not None
                    state["root"] = kd_tree
                    state["kd_tree"] = kd_tree
                    state["stack"] = []
                    state["leaf"] = find_leaf(kd_tree, target)
                    state["closest"] = find_closest(kd_tree, target)
                elif event.key == pygame.K_t:
                    target = make_target()
                    state["target"] = target
                    state["leaf"] = find_leaf(kd_tree, target)
                    state["closest"] = find_closest(kd_tree, target)
                elif event.key == pygame.K_w:
                    target = state["target"]
                    if target[1] > Y_RANGE[0]:
                        state["target"] = (target[0], target[1] - MOVE)
                        state["leaf"] = find_leaf(kd_tree, target)
                        state["closest"] = find_closest(kd_tree, target)
                elif event.key == pygame.K_s:
                    target = state["target"]
                    if target[1] < Y_RANGE[1]:
                        state["target"] = (target[0], target[1] + MOVE)
                        state["leaf"] = find_leaf(kd_tree, target)
                        state["closest"] = find_closest(kd_tree, target)
                elif event.key == pygame.K_a:
                    target = state["target"]
                    if target[0] > X_RANGE[0]:
                        state["target"] = (target[0] - MOVE, target[1])
                        state["leaf"] = find_leaf(kd_tree, target)
                        state["closest"] = find_closest(kd_tree, target)
                elif event.key == pygame.K_d:
                    target = state["target"]
                    if target[0] < X_RANGE[1]:
                        state["target"] = (target[0] + MOVE, target[1])
                        state["leaf"] = find_leaf(kd_tree, target)
                        state["closest"] = find_closest(kd_tree, target)

        draw(screen, state)
        pygame.display.flip()

    pygame.quit()


def make_points():
    points = []
    N = 50
    while len(points) < N:
        too_close = False
        threshold = ((min(WIDTH, HEIGHT) - 4 * MARGIN) // N) / 2
        x = random.randint(*X_RANGE)
        y = random.randint(*Y_RANGE)
        for point in points:
            if abs(point[0] - x) < threshold:
                too_close = True
                break
            if abs(point[1] - y) < threshold:
                too_close = True
                break

        if too_close:
            continue
        points.append((x, y))
    return points


def make_kd_tree(points, depth=0):
    if len(points) == 0:
        return None
    axis = depth % 2
    sorted_points = sort_by_axis(points, axis)
    median_idx = len(sorted_points) // 2
    median = sorted_points[median_idx]
    left_items = sorted_points[:median_idx]
    right_items = sorted_points[median_idx + 1 :]

    return (
        make_kd_tree(left_items, depth + 1),
        median,
        make_kd_tree(right_items, depth + 1),
    )


def find_leaf(root, target, depth=0):
    axis = depth % 2
    left, value, right = root

    if value == target:
        return value

    if target[axis] < value[axis]:
        if left is None:
            return value
        return find_leaf(left, target, depth + 1)
    else:
        if right is None:
            return value
        return find_leaf(right, target, depth + 1)


def find_closest(root, target, depth=0):
    axis = depth % 2
    left, value, right = root

    if value == target:
        return value

    closest = value
    if target[axis] < value[axis]:
        if left is not None:
            ret = find_closest(left, target, depth + 1)
            if distance(ret, target) < distance(closest, target):
                closest = ret
        if right is not None and (target[axis] - closest[axis]) ** 2 < distance(
            closest, target
        ):
            ret = find_closest(right, target, depth + 1)
            if distance(ret, target) < distance(closest, target):
                closest = ret
    else:
        if right is not None:
            ret = find_closest(right, target, depth + 1)
            if distance(ret, target) < distance(closest, target):
                closest = ret
        if left is not None and (target[axis] - closest[axis]) ** 2 < distance(
            closest, target
        ):
            ret = find_closest(left, target, depth + 1)
            if distance(ret, target) < distance(closest, target):
                closest = ret

    return closest


def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def sort_by_axis(l, axis):
    return list(sorted(l, key=lambda p: p[axis]))


def draw(screen: pygame.Surface, state):
    screen.fill((0, 0, 0))
    draw_base(
        state["kd_tree"],
        state["root"][1],
        screen,
        (MARGIN, WIDTH - MARGIN),
        (MARGIN, HEIGHT - MARGIN),
    )
    pygame.draw.lines(
        screen,
        "white",
        True,
        [
            (MARGIN, MARGIN),
            (WIDTH - MARGIN, MARGIN),
            (WIDTH - MARGIN, HEIGHT - MARGIN),
            (MARGIN, HEIGHT - MARGIN),
        ],
    )
    pygame.draw.circle(screen, "orange", state["root"][1], 15.0)
    pygame.draw.circle(screen, "purple", state["target"], 10.0)
    pygame.draw.circle(screen, "darkgreen", state["leaf"], 10.0)
    pygame.draw.circle(screen, "white", state["closest"], 10.0)
    draw_kd_tree(
        state["kd_tree"], screen, (MARGIN, WIDTH - MARGIN), (MARGIN, HEIGHT - MARGIN)
    )


def draw_kd_tree(root, screen: pygame.Surface, x_bounds, y_bounds, depth=0):
    if root == None:
        return
    axis = depth % 2
    left, value, right = root

    if axis == 0:
        pygame.draw.line(
            screen, (100, 0, 0), (value[0], y_bounds[0]), (value[0], y_bounds[1])
        )
    else:
        pygame.draw.line(
            screen, (0, 0, 100), (x_bounds[0], value[1]), (x_bounds[1], value[1])
        )

    color = "blue" if axis == 1 else "red"
    pygame.draw.circle(screen, color, value, 5.0)

    if axis == 0:
        left_x_bounds = (x_bounds[0], value[0])
        right_x_bounds = (value[0], x_bounds[1])
        left_y_bounds = y_bounds
        right_y_bounds = y_bounds
    else:
        left_x_bounds = x_bounds
        right_x_bounds = x_bounds
        left_y_bounds = (y_bounds[0], value[1])
        right_y_bounds = (value[1], y_bounds[1])

    draw_kd_tree(left, screen, left_x_bounds, left_y_bounds, depth + 1)
    draw_kd_tree(right, screen, right_x_bounds, right_y_bounds, depth + 1)


def draw_base(root, target, screen: pygame.Surface, x_bounds, y_bounds, depth=0):
    if root == None:
        return
    axis = depth % 2
    left, value, right = root

    color = (0, 0, 30) if axis == 1 else (30, 0, 0)
    if value[0] == target[0] and value[1] == target[1]:
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(
                x_bounds[0],
                y_bounds[0],
                x_bounds[1] - x_bounds[0],
                y_bounds[1] - y_bounds[0],
            ),
        )
        return

    if axis == 0:
        left_x_bounds = (x_bounds[0], value[0])
        right_x_bounds = (value[0], x_bounds[1])
        left_y_bounds = y_bounds
        right_y_bounds = y_bounds
    else:
        left_x_bounds = x_bounds
        right_x_bounds = x_bounds
        left_y_bounds = (y_bounds[0], value[1])
        right_y_bounds = (value[1], y_bounds[1])

    draw_base(left, target, screen, left_x_bounds, left_y_bounds, depth + 1)
    draw_base(right, target, screen, right_x_bounds, right_y_bounds, depth + 1)


if __name__ == "__main__":
    main()
