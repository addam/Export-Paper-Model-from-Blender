def get_nester(method):
    return rect_pack_custom if method == 'CUSTOM' else rect_pack_bpy


class Page:
    """Container for several Islands"""
    __slots__ = ('islands', 'name', 'image_path')

    def __init__(self, num=1):
        self.islands = list()
        self.name = "page{}".format(num)  # note: this is only used in svg files naming
        self.image_path = None


def try_emplace(island, cage_size, page_islands, stops_x, stops_y, occupied_cache):
    """Tries to put island to each pair from stops_x, stops_y
    and checks if it overlaps with any islands present on the page.
    Returns True and positions the given island on success."""
    bbox_x, bbox_y = island.bounding_box.xy
    for x in stops_x:
        if x + bbox_x > cage_size.x:
            continue
        for y in stops_y:
            if y + bbox_y > cage_size.y or (x, y) in occupied_cache:
                continue
            for i, obstacle in enumerate(page_islands):
                # if this obstacle overlaps with the island, try another stop
                if (x + bbox_x > obstacle.pos.x and
                        obstacle.pos.x + obstacle.bounding_box.x > x and
                        y + bbox_y > obstacle.pos.y and
                        obstacle.pos.y + obstacle.bounding_box.y > y):
                    if x >= obstacle.pos.x and y >= obstacle.pos.y:
                        occupied_cache.add((x, y))
                    # just a stupid heuristic to make subsequent searches faster
                    if i > 0:
                        page_islands[1:i+1] = page_islands[:i]
                        page_islands[0] = obstacle
                    break
            else:
                # if no obstacle called break, this position is okay
                island.pos.xy = x, y
                page_islands.append(island)
                stops_x.append(x + bbox_x)
                stops_y.append(y + bbox_y)
                return True
    return False


def drop_portion(stops, border, divisor):
    stops.sort()
    # distance from left neighbor to the right one, excluding the first stop
    distances = [right - left for left, right in zip(stops, chain(stops[2:], [border]))]
    quantile = sorted(distances)[len(distances) // divisor]
    return [stop for stop, distance in zip(stops, chain([quantile], distances)) if distance >= quantile]


def rect_pack_custom(islands, cage_size):
    pages = list()
    # sort islands by their diagonal... just a guess
    remaining_islands = sorted(islands, reverse=True, key=lambda island: island.bounding_box.length_squared)
    while remaining_islands:
        # create a new page and try to fit as many islands onto it as possible
        page = Page(len(pages) + 1)
        occupied_cache = set()
        stops_x, stops_y = [0], [0]
        for island in remaining_islands:
            try_emplace(island, cage_size, page.islands, stops_x, stops_y, occupied_cache)
            # if overwhelmed with stops, drop a quarter of them
            if len(stops_x)**2 > 4 * len(islands) + 100:
                stops_x = drop_portion(stops_x, cage_size.x, 4)
                stops_y = drop_portion(stops_y, cage_size.y, 4)
        remaining_islands = [island for island in remaining_islands if island not in page.islands]
        pages.append(page)
    return pages


def rect_pack_bpy(islands, cage_size):
    return []
