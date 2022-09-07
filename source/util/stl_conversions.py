import trimesh

def convert_stl_to_ply(path):
    mesh = trimesh.load(path)
    outputpath = path.rsplit(".", 1)[0] + ".ply"
    ply = trimesh.exchange.ply.export_ply(mesh, encoding='binary', include_attributes=False)
    with open (outputpath, "wb+") as output:
        output.write(ply)
    return outputpath