import argparse
from source.render.render_direct import Direct
from source.render.render_aov import AOV
from source.render.line_generation import LineGen

def run(render_type, line_gen, input_path, output_dirs, output_name, aovs=[]):
    renders = []
    if render_type == "aov" or render_type == "combined":
        renders.append(AOV(output_dirs, aovs))
    elif render_type == "rendering" or render_type == "combined":
        renders.append(Direct(output_dirs))
    else:
        raise Exception("Given type not known!")
    for renderer in renders:
        scene = renderer.create_scene(input_path, output_name)
        renderer.render(scene, input_path)

    if line_gen:
        line_gen = LineGen(output_dirs)
        line_gen.create_lines(input_path, output_name)


def diff_ars(args):
    run(args.render_type, args.line_gen, args.input_path, args.output_dirs, args.output_name, args.aovs)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--render_type", type=str, default='combined', help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--line_gen", type=bool, default=True, help="if sketch should be generated")
    parser.add_argument("--input_path", default='..\\..\\resources\\thingi10k\\0_499\\32770.ply', type=str, help="path to input model")
    parser.add_argument("--output_dirs", type=dict, default={'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'rendering': '..\\..\\output', 'sketch': '..\\..\\output'}, help="directories to save renderings")
    parser.add_argument("--output_name", type=str, default="test", help="Name of output images")
    parser.add_argument("--aovs", type=dir, default={"nn": "sh_normal", "dd.y": "depth"}, help="Types of AOV; use \"sh_normal\" or \"depth\"")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'rendering': '..\\..\\output'}
    params = [
        '--render_type', 'aov',
        '--input_path', '..\\..\\resources\\thingi10k\\0_499\\32770.ply',
        ]
    main(params)