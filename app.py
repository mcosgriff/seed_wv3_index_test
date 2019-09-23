from algorithms import Index, process_image
import argparse
import logging


def run(args: argparse.Namespace) -> None:
    output = process_image(Index[args.index], args.image_path)
    logging.info('Processed file saved to {}'.format(output))


def build_cmd_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Load WV3 16 band image and run index on it.')
    parser.add_argument('--verbose', help='Output extra logging at DEBUG level', action='store_true')
    parser.add_argument('--index', help='Which index to run on the WV3 image', required=True,
                        choices=['NDVI', 'WORLD_VIEW_WATER', 'POLYMER_1', 'POLYMER_2',
                                 'SOIL', 'BUILT_UP', 'NDVI_RE'])
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
