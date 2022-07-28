import drjit as dr
import mitsuba as mi
import normal_depth_integrator
from mitsuba.scalar_rgb import Transform4f as T
from matplotlib import pyplot as plt

mi.set_variant('cuda_ad_rgb')

def offsetVerts(params, opt):
    trafo = mi.Transform4f.translate(opt['deform_verts'])

    params['sphere.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()

aov_integrator = {
    'type': 'normal_depth',
    'aovs': "nn:sh_normal,dd.y:depth"
}

integrator = {
        'type': 'normal_depth'
}

# refscene
ref_scene = mi.load_dict({
    'type': 'scene',
    'sensor':  {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },
    'bunny': {
        'type': 'ply',
        'filename': '../../resources/meshes/bunny.ply',
        'to_world': T.scale(6.5),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
        },
    },
    'light': {
        'type': 'obj',
        'filename': '../../resources/meshes/sphere.obj',
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'rgb', 'value': [1e3, 1e3, 1e3]}
        },
        'to_world': T.translate([2.5, 2.5, 7.0]).scale(0.25)
    }
})

# object
scene = mi.load_dict({
    'type': 'scene',
    'sensor':  {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=(0, 0, 2),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },
    'sphere': {
        'type': 'obj',
        'filename': '../../resources/meshes/sphere.obj',
        'to_world': T.scale(0.5),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
        },
    },
    'light': {
        'type': 'obj',
        'filename': '../../resources/meshes/sphere.obj',
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'rgb', 'value': [1e3, 1e3, 1e3]}
        },
        'to_world': T.translate([2.5, 2.5, 7.0]).scale(0.25)
    }
})


img_ref = mi.render(ref_scene, seed=0, spp=1024, integrator=mi.load_dict(integrator))
mi.util.convert_to_bitmap(img_ref)
#bitmap = mi.Bitmap(img_ref_all, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
#channels = dict(bitmap.split())
#img_ref = channels['pos']

params = mi.traverse(scene)
print(params)
initial_vertex_positions = dr.unravel(mi.Point3f, params['sphere.vertex_positions'])
initial_vertex_normals = dr.unravel(mi.Point3f, params['sphere.vertex_normals'])

opt = mi.ad.Adam(lr=0.025)
vertex_count = params['sphere.vertex_count']
opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

img_init = mi.render(scene, seed=0, spp=1024, integrator=mi.load_dict(integrator))

loss_hist = []
for it in range(20):
    offsetVerts(params, opt)

    img = mi.render(scene, params, integrator=mi.load_dict(integrator), seed=it, spp=16)

    loss = dr.sum(dr.sqr(img - img_ref)) / len(img)

    # use differentiated normals for that later on
    # maybe don't use this loss, it is also based on the alpha channel of the normal values
    ## predicted_vertex_normals = dr.unravel(mi.Point3f, params['sphere.vertex_normals'])
    ## d_initial = 2*initial_vertex_normals - 1
    ## d_predicted = 2*predicted_vertex_normals - 1
    ## angular_loss = dr.sum(dr.sqrt(1 - d_predicted * d_initial * (d_predicted * d_initial).A)) / len(img)

    dr.backward(loss)

    opt.step()
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0][0].plot(loss_hist)
    axs[0][0].set_xlabel('iteration');
    axs[0][0].set_ylabel('Loss');
    axs[0][0].set_title('Parameter error plot');
    axs[1][1].imshow(mi.util.convert_to_bitmap(img_ref))
    axs[1][1].axis('off')
    axs[1][1].set_title('Reference Image');
    axs[1][0].imshow(mi.util.convert_to_bitmap(img))
    axs[1][0].axis('off')
    axs[1][0].set_title('Optimized image')
    axs[0][1].imshow(mi.util.convert_to_bitmap(img_init))
    axs[0][1].axis('off')
    axs[0][1].set_title('Initial Image')
    plt.savefig('../../output/test'+str(it)+".png")

    loss_hist.append(loss)
    #with open('../../output/deformverts_opt'+str(it)+'.txt', 'w') as f:
    #    x = opt['deform_verts'].x
    #    y = opt['deform_verts'].y
    #    z = opt['deform_verts'].z
    #    f.writelines(' '.join(str(e) for e in x))
    #    f.writelines('\n\n')
    #    f.writelines(' '.join(str(e) for e in y))
    #    f.writelines('\n\n')
    #    f.writelines(' '.join(str(e) for e in z))
    #with open('../../output/normals'+str(it)+'.txt', 'w') as f:
    #    f.writelines(' '.join(str(e) for e in params['sphere.vertex_normals']))
    print(f"Iteration {it:02d}: error={loss[0]:6f}", end='\r')

mesh = mi.Mesh(
    "deformed_sphere",
    vertex_count=vertex_count,
    face_count=params['sphere.face_count'],
    has_vertex_normals=True,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = dr.ravel(params['sphere.vertex_positions'])
mesh_params["faces"] = dr.ravel(params['sphere.faces'])
mesh_params.update()
mesh.write_ply("../../output/deformed_sphere.ply")