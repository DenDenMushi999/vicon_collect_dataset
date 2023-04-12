from trimesh.exchange.ply import _parse_header, _ply_binary

def load_ply(path):
    with open(path, 'rb') as f:
        elements, is_ascii, image_name = _parse_header(f)
        _ply_binary(elements, f)
    return elements

path = "0.ply"
el = load_ply(path)
print(el)
