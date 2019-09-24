import argparse
import logging
from enums import Index, StretchTypes

from indexes import NormalizedDifferentialVegetation, BuiltUp, NormalizedDifferentialVegetationRedEdge, Polymer1, \
    Polymer2, Soil, WorldViewWater, WV3Carbonate, AIOHGroupContent, AluniteKaolinite


def run(args: argparse.Namespace) -> None:
    index = None
    which_algorithm = Index[args.index]

    if which_algorithm == Index.NDVI:
        index = NormalizedDifferentialVegetation(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.WORLD_VIEW_WATER:
        index = WorldViewWater(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.POLYMER_1:
        index = Polymer1(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.POLYMER_2:
        index = Polymer2(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.SOIL:
        index = Soil(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.BUILT_UP:
        index = BuiltUp(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.NDVI_RE:
        index = NormalizedDifferentialVegetationRedEdge(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.WV3_CARBONATE:
        index = WV3Carbonate(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.AIOH_GROUP_CONTENT:
        index = AIOHGroupContent(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)
    elif which_algorithm == Index.ALUNITE_KAOLINITE:
        index = AluniteKaolinite(args.image_path, StretchTypes.LINEAR_PERCENT_STRETCH)

    if index:
        with index:
            output = index.process_index()
            logging.info('Processed file saved to {}'.format(output))


def build_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load WV3 16 band image and run index on it.')
    parser.add_argument('--verbose', help='Output extra logging at DEBUG level', action='store_true')
    parser.add_argument('--index', help='Which index to run on the WV3 image', required=True,
                        choices=['NDVI', 'WORLD_VIEW_WATER', 'POLYMER_1', 'POLYMER_2', 'SOIL', 'BUILT_UP', 'NDVI_RE',
                                 'WV3_CARBONATE', 'ALUNITE_KAOLINITE', 'AIOH_GROUP_CONTENT'])
    parser.add_argument('--image-path', help='Path to WV3 16 band image', required=True, type=str)

    return parser.parse_args()


if __name__ == '__main__':
    _args = build_cmd_line_args()

    if _args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(threadName)s::%(asctime)s::%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(threadName)s::%(asctime)s::%(message)s")
    logger = logging.getLogger('wv3_index_processing')

    run(_args)
