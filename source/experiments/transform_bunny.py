# original code base https://mitsuba.readthedocs.io/en/latest/src/inverse_rendering/object_pose_estimation.html
# used to get familiar with mitsuba3 and test integrators
import drjit as dr
import mitsuba as mi
import source.mesh_generation.normal_reparam_integrator
import source.mesh_generation.depth_reparam_integrator

mi.set_variant('cuda_ad_rgb')

from mitsuba.scalar_rgb import Transform4f as T
from matplotlib import pyplot as plt

def apply_transformation(params, opt):
    opt['trans'] = dr.clamp(opt['trans'], -0.5, 0.5)
    opt['angle'] = dr.clamp(opt['angle'], -0.5, 0.5)

    trafo = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, 0.0]).rotate([0, 1, 0], opt['angle'] * 100.0)

    params['bunny.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()

def plot_figures(img_init, img_ref, img_curr, name, it):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0][0].plot(loss_hist)
    axs[0][0].set_xlabel('iteration');
    axs[0][0].set_ylabel('Loss');
    axs[0][0].set_title('Parameter error plot');

    axs[0][1].imshow(mi.util.convert_to_bitmap(img_init))
    axs[0][1].axis('off')
    axs[0][1].set_title('Initial Image')

    axs[1][0].imshow(mi.util.convert_to_bitmap(img_curr))
    axs[1][0].axis('off')
    axs[1][0].set_title('Optimized image')

    axs[1][1].imshow(mi.util.convert_to_bitmap(img_ref))
    axs[1][1].axis('off')
    axs[1][1].set_title('Reference Image');
    plt.savefig('../../output/'+name+str(it)+".png")

depth_integrator = {
        'type': 'depth_reparam'
}
normal_integrator = {
        'type': 'normal_reparam'
}

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

depth_integrator_lodaded=mi.load_dict(depth_integrator)
normal_integrator_lodaded=mi.load_dict(normal_integrator)
depth_img_ref = mi.render(scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
normal_img_ref = mi.render(scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)

params = mi.traverse(scene)
initial_vertex_positions = dr.unravel(mi.Point3f, params['bunny.vertex_positions'])

opt = mi.ad.Adam(lr=0.025)
opt['angle'] = mi.Float(0.25)
opt['trans'] = mi.Point2f(0.1, -0.25)

apply_transformation(params, opt)
depth_img_init = mi.render(scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
normal_img_init = mi.render(scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)

loss_hist = []
for it in range(20):
    apply_transformation(params, opt)

    depth_img = mi.render(scene, params, seed=it, spp=16, integrator=depth_integrator_lodaded)
    normal_img = mi.render(scene, seed=it, spp=16, integrator=normal_integrator_lodaded)

    depth_loss = dr.sum(dr.sqr(depth_img - depth_img_ref)) / len(depth_img)
    normal_loss = dr.sum(dr.sqr(normal_img - normal_img_ref)) / len(normal_img)
    loss = depth_loss*0.5 + normal_loss*0.5

    dr.backward(loss)

    opt.step()

    loss_hist.append(loss)
    print(f"Iteration {it:02d}: error={loss[0]:6f}, angle={opt['angle'][0]:.4f}, trans=[{opt['trans'].x[0]:.4f}, {opt['trans'].y[0]:.4f}]",
        end='\r')

    plot_figures(depth_img_init, depth_img_ref, depth_img, 'depth', it)
    plot_figures(normal_img_init, normal_img_ref, normal_img, 'normal', it)