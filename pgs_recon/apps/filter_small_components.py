import argparse
import math

import pgs_recon.utils.wavefront as wobj
from pgs_recon.utils import geometry as geom

def parse_filter_cc(arg: str):
    arg = arg.lower()
    if arg == 'largest':
        return -1
    else:
        try:
            return int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f'{arg} is not \'largest\' or an integer')


def parse_filter_cc_area(arg: str):
    try:
        area = float(arg)
    except ValueError:
        area = math.nan
    # nan compares False against every component, so it would save an empty
    # mesh; a negative would silently mean 'keep everything'
    if not math.isfinite(area) or area < 0:
        raise argparse.ArgumentTypeError(f'{arg} is not an area >= 0')
    return area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input mesh file')
    parser.add_argument('--output-file', '-o', required=True,
                        help='Output mesh file')
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument('--filter-cc', default='largest', type=parse_filter_cc,
                         help="Filter the mesh's connected components:\n"
                              " - 'largest': keep only the largest connected component\n"
                              " - N: remove all connected components with fewer than N faces")
    filters.add_argument('--filter-cc-area', metavar='AREA',
                         type=parse_filter_cc_area,
                         help="Filter the mesh's connected components by "
                              "surface area instead, keeping those of at least "
                              "AREA in the mesh's units squared (cm^2 on an "
                              "autoscaled reconstruction). 0 keeps everything "
                              "and reports the inventory. An area means the "
                              "same thing from one scan to the next; a face "
                              "count does not.")
    args = parser.parse_args()

    # Load the mesh
    print('Loading mesh...')
    obj = wobj.load_obj(args.input_file)
    mesh = geom.wavefront_to_mesh(obj)

    if args.filter_cc_area is not None:
        print(f'Filtering components by area '
              f'(>= {args.filter_cc_area} units^2)...')
        print(geom.remove_connected_components_by_area(
            mesh, min_area=args.filter_cc_area))
    elif args.filter_cc > 0:
        print(f'Removing connected components smaller than '
              f'{args.filter_cc} faces...')
        print(geom.remove_connected_components_by_size(
            mesh, num_faces=args.filter_cc))
    elif args.filter_cc < 0:
        print('Keeping largest connected component...')
        print(geom.keep_largest_connected_component(
            mesh, filter_vertices=True))

    print('Saving mesh...')
    obj = geom.mesh_to_wavefront(mesh, obj)
    wobj.save_obj(obj, args.output_file)


if __name__ == '__main__':
    main()
