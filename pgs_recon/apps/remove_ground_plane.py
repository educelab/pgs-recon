import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', required=True,
                        help='Input mesh file')
    parser.add_argument('--output-file', '-o', required=True,
                        help='Output mesh file')
    parser.add_argument('--scale', type=float,
                        help='Scale mesh before processing')
    parser.add_argument('--distance-threshold', type=float, default=0.1,
                        help='During ground plane estimation, points are '
                             'considered inliers if their distance from the '
                             'plane is less than the distance threshold. This '
                             'should be tuned based on the point density.')
    parser.add_argument('--filter-cc', default='largest', type=parse_filter_cc,
                        help="Filter the mesh's connected components after "
                             "removing the ground plane:\n"
                             " - 'none': No filtering\n"
                             " - 'largest': keep only the largest connected component\n"
                             " - N: remove all connected components with fewer than N faces")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load the mesh
    print('Loading mesh...')
    obj = wobj.load_obj(args.input_file)
    mesh = geom.wavefront_to_mesh(obj)

    if args.scale is not None:
        print('Scaling mesh...')
        mesh.vertices *= args.scale

    print('Fitting plane...')
    _, plane_inliers = geom.segment_plane(mesh,
                                          dist_threshold=args.distance_threshold,
                                          seed=args.seed)
    print('Removing plane...')
    geom.remove_vertices_by_index(mesh, plane_inliers)

    if args.filter_cc > 0:
        print(f'Removing connected components smaller than {args.filter_cc} faces...')
        geom.remove_connected_components_by_size(mesh, num_faces=args.filter_cc)
    elif args.filter_cc < 0:
        print('Keeping largest connected component...')
        geom.keep_largest_connected_component(mesh, filter_vertices=True)

    print('Saving mesh...')
    obj = geom.mesh_to_wavefront(mesh, obj)
    wobj.save_obj(obj, args.output_file)


if __name__ == '__main__':
    main()
