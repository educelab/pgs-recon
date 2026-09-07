import argparse
import math
import sys

import pgs_recon.utils.wavefront as wobj
from pgs_recon.utils import geometry as geom

def parse_filter_cc(arg: str):
    arg = arg.lower()
    if arg == 'none':
        return 0
    elif arg == 'largest':
        return -1
    else:
        return int(arg)


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
    parser.add_argument('--scale', type=float,
                        help='Scale mesh before processing')
    parser.add_argument('--distance-threshold', type=float, default=0.02,
                        help='During ground estimation, points are considered '
                             'inliers if their distance from the fitted '
                             'surface is less than this. Too small and the '
                             'band is the mesh noise rather than the ground; '
                             'too large and it reaches the object where that '
                             'meets the ground. On EduceLab PGS captures '
                             'anything in 0.01-0.05 works.')
    parser.add_argument('--surface-degree', type=int, default=2,
                        help='Degree of the polynomial surface fit to the '
                             'ground. A scan bed is usually bowed by more than '
                             'the distance threshold, in which case a plane (0) '
                             'only removes the strip where the two coincide.')
    parser.add_argument('--drop-below-ground', action='store_true',
                        help='Also drop the connected components that lie '
                             'entirely below the fitted ground surface. The '
                             "scan bed's fiducial squares reconstruct as "
                             'shallow recesses under the bed, which ground '
                             'removal leaves behind because they are below '
                             'its band rather than inside it. The 20-22 of '
                             'them a scan delivers run 0.51-2.84 cm^2, the '
                             'size of the fragments the area filter exists to '
                             'keep, so only their being under the bed tells '
                             'them apart -- and nothing real is under the bed.')
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument('--filter-cc', default='largest', type=parse_filter_cc,
                         help="Filter the mesh's connected components after "
                              "removing the ground:\n"
                              " - 'none': No filtering\n"
                              " - 'largest': keep only the largest connected component\n"
                              " - N: remove all connected components with fewer than N faces")
    filters.add_argument('--filter-cc-area', metavar='AREA',
                         type=parse_filter_cc_area,
                         help="Filter the mesh's connected components after "
                              "removing the ground by surface area instead, "
                              "keeping those of at least AREA in the mesh's "
                              "units squared (cm^2 on an autoscaled "
                              "reconstruction). 0 keeps everything and "
                              "reports the inventory. An area means the same "
                              "thing from one scan to the next; a face count "
                              "does not.")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load the mesh
    print('Loading mesh...')
    obj = wobj.load_obj(args.input_file)
    mesh = geom.wavefront_to_mesh(obj)

    if args.scale is not None:
        print('Scaling mesh...')
        mesh.vertices *= args.scale

    print('Fitting ground surface...')
    try:
        surface, ground = geom.segment_ground_surface(
            mesh, dist_threshold=args.distance_threshold,
            degree=args.surface_degree, seed=args.seed)
    except ValueError as e:
        # Writing the mesh anyway would hand on a silently mangled one
        sys.exit(f'{args.input_file}: {e}')
    print(f'Removing ground ({len(ground)} of {mesh.vertices.shape[0]} '
          f'vertices, warp {surface.warp:.4f}, fit rms {surface.rms:.5f} = '
          f'{surface.rms / args.distance_threshold:.2f} of the threshold)...')
    geom.remove_vertices_by_index(mesh, ground)

    # Ahead of the area/size filters, not instead of them: the two catch
    # different things, speckle and the bed's own fiducial recesses
    if args.drop_below_ground:
        print('Dropping components below the ground surface...')
        print(geom.remove_connected_components_below_surface(mesh, surface))

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
