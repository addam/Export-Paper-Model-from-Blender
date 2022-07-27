from mathutils.geometry import box_pack_2d

def get_nester(method):
    return rect_pack_custom if method == 'CCOR' else rect_pack_bpy


class Page:
    """Container for several Islands"""
    __slots__ = ('islands', 'name', 'image_path')

    def __init__(self, num=1, islands=None):
        self.islands = islands or list()
        self.name = "page{}".format(num)  # note: this is only used in svg files naming
        self.image_path = None


def can_pack_bpy(islands, cage_size):
    if len(islands) <= 1:
        return True
    aspect = cage_size.y / cage_size.x
    boxes = [[0, 0, isle.bounding_box.x, isle.bounding_box.y / aspect] for isle in islands]
    width, height = box_pack_2d(boxes)
    if width < cage_size.x > height:
        for (x, y, *_), isle in zip(boxes, islands):
            isle.pos.xy = x, y * aspect
        return True
    return False


def branch_and_bound(can_pack_fn):
    # exhaustive search for minimal number of pages
    def result(islands, cage_size):
        best = [i for i, _ in enumerate(islands)]
        path = [0]
        while True:
            if path[-1] > max(path[:-1], default=0) + 1 or max(path) >= max(best):
                # invalid step
                path.pop()
                if len(path) <= 1:
                    break
                path[-1] += 1
                continue
            page = [isle for i, isle in zip(path, islands) if i == path[-1]]
            if can_pack_fn(page, cage_size):
                # good step
                if len(path) == len(islands):
                    best = list(path)
                    path.pop()
                    path[-1] += 1
                else:
                    path.append(0)
            else:
                # bad step
                path[-1] += 1
        pages = [list() for _ in range(max(best) + 1)]
        for i, isle in zip(best, islands):
            pages[i].append(isle)
        for page in pages:
            can_pack_fn(page, cage_size)
        return [Page(i, page) for i, page in enumerate(pages)]
    return result

rect_pack_bpy = branch_and_bound(can_pack_bpy)
